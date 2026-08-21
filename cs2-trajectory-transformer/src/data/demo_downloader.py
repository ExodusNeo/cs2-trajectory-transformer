"""
Automated CS2 Demo Replay Scraper and Ingestion Engine.
Supports:
1. Faceit Match Replays (Open API & Public CDN downloads).
2. HLTV Pro Tournament Replays (Archive decompression .gz, .zip, .bz2, .tar.gz).
3. Automated Background Decompression & File Organization into data/raw_demos/.
"""

import os
import sys
import json
import gzip
import bz2
import zipfile
import tarfile
import logging
import time
import random
import urllib.request
import urllib.error
from typing import List, Optional, Dict, Tuple
from tqdm import tqdm

try:
    import zstandard
except ImportError:
    zstandard = None

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

# Default storage path on D: drive if available, otherwise relative project path
DEFAULT_STORAGE_DIR = r"D:\cs2_replay_data\raw_demos" if os.path.exists("D:\\") else "data/raw_demos"


class DownloadProgressBar(tqdm):
    """Provides live download progress bar in terminal."""
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def polite_request(req: urllib.request.Request, max_retries: int = 3, initial_delay: float = 0.5) -> bytes:
    """
    Executes an HTTP request with polite server-friendly exponential backoff
    to prevent overwhelming Faceit API servers or triggering rate limits.
    """
    for attempt in range(max_retries):
        try:
            # Polite jitter delay before querying
            time.sleep(initial_delay + random.uniform(0.1, 0.3))
            with urllib.request.urlopen(req, timeout=30) as response:
                return response.read()
        except urllib.error.HTTPError as e:
            if e.code == 429 or e.code >= 500:
                # Rate limit hit or server busy -> Exponential backoff with jitter
                wait_time = (2 ** attempt) + random.uniform(0.5, 1.5)
                logging.warning(f"Server returned HTTP {e.code}. Backing off politely for {wait_time:.1f}s (Attempt {attempt+1}/{max_retries})...")
                time.sleep(wait_time)
            elif e.code == 404:
                logging.warning(f"Resource not found (HTTP 404): {req.full_url}")
                return b""
            else:
                logging.error(f"HTTP Error {e.code} for {req.full_url}: {e.reason}")
                return b""
        except Exception as e:
            logging.warning(f"Network error on attempt {attempt+1}: {e}")
            time.sleep(1.0)
            
    return b""


class CS2ReplayDownloader:
    """
    Automated replay downloader and decompressor for CS2 .dem match files.
    Optimized for polite server interactions, automatic deduplication, and D: drive storage.
    """
    def __init__(self, base_dir: Optional[str] = None):
        self.base_dir = base_dir or DEFAULT_STORAGE_DIR
        self.clean_dir = os.path.join(self.base_dir, "clean")
        self.cheater_dir = os.path.join(self.base_dir, "cheaters")
        os.makedirs(self.clean_dir, exist_ok=True)
        os.makedirs(self.cheater_dir, exist_ok=True)

    def decompress_archive(self, file_path: str, destination_dir: str) -> List[str]:
        """
        Decompresses .gz, .zst, .bz2, .zip, or .tar.gz files and extracts all .dem files.
        """
        extracted_dems = []
        bname = os.path.basename(file_path)
        base_name, ext = os.path.splitext(bname)
        ext = ext.lower()

        try:
            if ext == '.gz' and not file_path.endswith('.tar.gz'):
                out_path = os.path.join(destination_dir, base_name if base_name.endswith('.dem') else f"{base_name}.dem")
                with gzip.open(file_path, 'rb') as f_in, open(out_path, 'wb') as f_out:
                    f_out.write(f_in.read())
                extracted_dems.append(out_path)

            elif ext == '.zst':
                if zstandard is None:
                    raise ImportError("zstandard package is required to decompress .zst files. Run 'pip install zstandard'")
                out_path = os.path.join(destination_dir, base_name if base_name.endswith('.dem') else f"{base_name}.dem")
                dctx = zstandard.ZstdDecompressor()
                with open(file_path, 'rb') as f_in, open(out_path, 'wb') as f_out:
                    dctx.copy_stream(f_in, f_out)
                extracted_dems.append(out_path)

            elif ext == '.bz2':
                out_path = os.path.join(destination_dir, base_name if base_name.endswith('.dem') else f"{base_name}.dem")
                with bz2.open(file_path, 'rb') as f_in, open(out_path, 'wb') as f_out:
                    f_out.write(f_in.read())
                extracted_dems.append(out_path)

            elif ext == '.zip':
                with zipfile.ZipFile(file_path, 'r') as zip_ref:
                    for member in zip_ref.namelist():
                        if member.lower().endswith('.dem'):
                            zip_ref.extract(member, destination_dir)
                            extracted_dems.append(os.path.join(destination_dir, member))

            elif file_path.endswith('.tar.gz') or ext == '.tar':
                with tarfile.open(file_path, 'r:*') as tar_ref:
                    for member in tar_ref.getmembers():
                        if member.name.lower().endswith('.dem'):
                            tar_ref.extract(member, destination_dir)
                            extracted_dems.append(os.path.join(destination_dir, member.name))

            elif ext == '.dem':
                extracted_dems.append(file_path)

            logging.info(f"Extracted {len(extracted_dems)} CS2 replay(s) from {bname}")
            return extracted_dems

        except Exception as e:
            logging.error(f"Failed decompressing {file_path}: {e}")
            return []

    def download_url(
        self, 
        url: str, 
        is_cheater: bool = False, 
        custom_filename: Optional[str] = None
    ) -> List[str]:
        """
        Downloads a match replay archive from direct Backblaze CDN with progress bar,
        decompresses it, and automatically purges the compressed archive to save drive space.
        """
        target_dir = self.cheater_dir if is_cheater else self.clean_dir
        bname = custom_filename or url.split('/')[-1].split('?')[0]
        if not bname:
            bname = "downloaded_match.dem.zst"
            
        final_dem_name = bname.replace('.zst', '').replace('.gz', '').replace('.bz2', '')
        if not final_dem_name.endswith('.dem'):
            final_dem_name += '.dem'
            
        final_dem_path = os.path.join(target_dir, final_dem_name)

        # Optimization: Skip download if already present locally (Deduplication)
        if os.path.exists(final_dem_path) and os.path.getsize(final_dem_path) > 1024 * 1024:
            logging.info(f"[CACHE HIT] Replay already exists locally on D: drive: {final_dem_name} (Skipping CDN download).")
            return [final_dem_path]

        temp_download_path = os.path.join(target_dir, bname)
        logging.info(f"Connecting to CDN stream: {url}")
        
        try:
            req = urllib.request.Request(
                url, 
                headers={'User-Agent': 'CS2TrajectoryTransformer/1.0 (Thesis Research; Academic Ingestion)'}
            )
            with urllib.request.urlopen(req, timeout=45) as response:
                total_size = int(response.info().get('Content-Length', 0))
                with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=f"D: -> {bname}") as pbar:
                    with open(temp_download_path, 'wb') as out_file:
                        while True:
                            chunk = response.read(1024 * 128)
                            if not chunk:
                                break
                            out_file.write(chunk)
                            pbar.update(len(chunk))

            # Automatically decompress to .dem
            extracted = self.decompress_archive(temp_download_path, target_dir)
            
            # Immediately delete compressed .zst archive to preserve drive space
            if extracted and temp_download_path not in extracted and os.path.exists(temp_download_path):
                os.remove(temp_download_path)
                
            return extracted

        except Exception as e:
            logging.error(f"Download failed for {url}: {e}")
            if os.path.exists(temp_download_path):
                os.remove(temp_download_path)
            return []

    def fetch_faceit_match_demo(
        self, 
        match_id: str, 
        api_key: Optional[str] = None, 
        is_cheater: bool = False
    ) -> List[str]:
        """
        Queries Faceit API politely with retry backoff to fetch and download a CS2 match replay.
        """
        headers = {'User-Agent': 'CS2TrajectoryTransformer/1.0 (Thesis Research; Academic Ingestion)'}
        if api_key:
            headers['Authorization'] = f'Bearer {api_key}'
            
        api_url = f"https://open.faceit.com/data/v4/matches/{match_id}"
        req = urllib.request.Request(api_url, headers=headers)
        raw_body = polite_request(req, max_retries=3, initial_delay=0.4)
        
        if not raw_body:
            return []
            
        try:
            data = json.loads(raw_body.decode('utf-8'))
            demo_url = data.get('demo_url', [])
            if isinstance(demo_url, list) and len(demo_url) > 0:
                demo_url = demo_url[0]
                
            if demo_url and isinstance(demo_url, str):
                logging.info(f"Found Backblaze CDN demo stream for match {match_id}")
                ext_suffix = ".dem.zst" if demo_url.endswith(".zst") else ".dem.gz"
                return self.download_url(demo_url, is_cheater=is_cheater, custom_filename=f"faceit_{match_id}{ext_suffix}")
            else:
                logging.warning(f"No demo URL in Faceit match payload for {match_id}")
                return []
        except Exception as e:
            logging.error(f"Error parsing match payload for {match_id}: {e}")
            return []

    def fetch_banned_cheater_matches(
        self, 
        banned_steam_or_nicknames: List[str], 
        api_key: Optional[str] = None,
        matches_per_player: int = 2
    ) -> List[str]:
        """Queries Faceit API politely for confirmed banned cheaters and downloads their match replays."""
        headers = {'User-Agent': 'CS2TrajectoryTransformer/1.0 (Thesis Research; Academic Ingestion)'}
        if api_key:
            headers['Authorization'] = f'Bearer {api_key}'
            
        all_downloaded = []
        for player in banned_steam_or_nicknames:
            try:
                url = f"https://open.faceit.com/data/v4/players?nickname={player}"
                req = urllib.request.Request(url, headers=headers)
                raw = polite_request(req, max_retries=3, initial_delay=0.5)
                if not raw:
                    continue
                p_data = json.loads(raw.decode('utf-8'))
                p_id = p_data.get('player_id')
                if not p_id:
                    continue
                    
                hist_url = f"https://open.faceit.com/data/v4/players/{p_id}/history?game=cs2&limit={matches_per_player}"
                req2 = urllib.request.Request(hist_url, headers=headers)
                raw_hist = polite_request(req2, max_retries=3, initial_delay=0.5)
                if not raw_hist:
                    continue
                hist_data = json.loads(raw_hist.decode('utf-8'))
                items = hist_data.get('items', [])
                for match in items:
                    m_id = match.get('match_id')
                    if m_id:
                        dems = self.fetch_faceit_match_demo(m_id, api_key=api_key, is_cheater=True)
                        all_downloaded.extend(dems)
            except Exception as e:
                logging.error(f"Error querying banned account {player}: {e}")
                
        return all_downloaded

    def list_downloaded_demos(self) -> Dict[str, List[str]]:
        """Returns inventory of all available .dem files on D: drive."""
        clean_dems = [os.path.join(self.clean_dir, f) for f in os.listdir(self.clean_dir) if f.endswith('.dem')]
        cheater_dems = [os.path.join(self.cheater_dir, f) for f in os.listdir(self.cheater_dir) if f.endswith('.dem')]
        return {
            'clean': clean_dems,
            'cheaters': cheater_dems,
            'total': len(clean_dems) + len(cheater_dems),
            'base_dir': self.base_dir
        }

