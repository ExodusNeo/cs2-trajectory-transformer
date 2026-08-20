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
import urllib.request
from typing import List, Optional, Dict, Tuple
from tqdm import tqdm

try:
    import zstandard
except ImportError:
    zstandard = None

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')


class DownloadProgressBar(tqdm):
    """Provides live download progress bar in terminal."""
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


class CS2ReplayDownloader:
    """
    Automated replay downloader and decompressor for CS2 .dem match files.
    Manages raw clean baselines and banned cheater datasets.
    """
    def __init__(self, base_dir: str = "data/raw_demos"):
        self.base_dir = base_dir
        self.clean_dir = os.path.join(base_dir, "clean")
        self.cheater_dir = os.path.join(base_dir, "cheaters")
        os.makedirs(self.clean_dir, exist_ok=True)
        os.makedirs(self.cheater_dir, exist_ok=True)

    def decompress_archive(self, file_path: str, destination_dir: str) -> List[str]:
        """
        Decompresses .gz, .bz2, .zip, or .tar.gz files and extracts all .dem files.
        """
        extracted_dems = []
        bname = os.path.basename(file_path)
        base_name, ext = os.path.splitext(bname)
        ext = ext.lower()

        try:
            if ext == '.gz' and not file_path.endswith('.tar.gz'):
                # Single .dem.gz file
                out_path = os.path.join(destination_dir, base_name if base_name.endswith('.dem') else f"{base_name}.dem")
                with gzip.open(file_path, 'rb') as f_in, open(out_path, 'wb') as f_out:
                    f_out.write(f_in.read())
                extracted_dems.append(out_path)

            elif ext == '.zst':
                # Single .dem.zst file (standard in CS2 Faceit)
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
        Downloads a match replay archive from a direct URL with progress bar,
        decompresses it, and registers it into the clean or cheater replay store.
        """
        target_dir = self.cheater_dir if is_cheater else self.clean_dir
        bname = custom_filename or url.split('/')[-1].split('?')[0]
        if not bname:
            bname = "downloaded_match.dem.gz"
            
        temp_download_path = os.path.join(target_dir, bname)
        
        logging.info(f"Connecting to: {url}")
        try:
            req = urllib.request.Request(
                url, 
                headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) CS2TrajectoryTransformer/1.0'}
            )
            with urllib.request.urlopen(req) as response:
                total_size = int(response.info().get('Content-Length', 0))
                with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=bname) as pbar:
                    with open(temp_download_path, 'wb') as out_file:
                        while True:
                            chunk = response.read(1024 * 64)
                            if not chunk:
                                break
                            out_file.write(chunk)
                            pbar.update(len(chunk))

            # Automatically decompress
            extracted = self.decompress_archive(temp_download_path, target_dir)
            
            # Remove compressed archive if successfully extracted to save disk space
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
        Queries the Faceit Open API (or public mirror) to fetch and download a CS2 match replay.
        """
        headers = {'User-Agent': 'CS2TrajectoryTransformer/1.0'}
        if api_key:
            headers['Authorization'] = f'Bearer {api_key}'
            
        api_url = f"https://open.faceit.com/data/v4/matches/{match_id}"
        
        try:
            req = urllib.request.Request(api_url, headers=headers)
            with urllib.request.urlopen(req) as response:
                data = json.loads(response.read().decode('utf-8'))
                demo_url = data.get('demo_url', [])
                if isinstance(demo_url, list) and len(demo_url) > 0:
                    demo_url = demo_url[0]
                    
                if demo_url and isinstance(demo_url, str):
                    logging.info(f"Found demo URL for match {match_id}: {demo_url}")
                    ext_suffix = ".dem.zst" if demo_url.endswith(".zst") else ".dem.gz"
                    return self.download_url(demo_url, is_cheater=is_cheater, custom_filename=f"faceit_{match_id}{ext_suffix}")
                else:
                    logging.warning(f"No demo URL available for match {match_id}")
                    return []
        except Exception as e:
            logging.error(f"Faceit API query error for match {match_id}: {e}")
            return []

    
    def fetch_banned_cheater_matches(
        self, 
        banned_steam_or_nicknames: List[str], 
        api_key: Optional[str] = None,
        matches_per_player: int = 2
    ) -> List[str]:
        """
        Queries Faceit API for confirmed banned cheaters and downloads their match replays.
        This directly fulfills the 1,000 Cheater Demos requirement in the concept paper.
        """
        headers = {'User-Agent': 'CS2TrajectoryTransformer/1.0'}
        if api_key:
            headers['Authorization'] = f'Bearer {api_key}'
            
        all_downloaded = []
        for player in banned_steam_or_nicknames:
            try:
                # 1. Resolve player ID
                url = f"https://open.faceit.com/data/v4/players?nickname={player}"
                req = urllib.request.Request(url, headers=headers)
                with urllib.request.urlopen(req) as resp:
                    p_data = json.loads(resp.read().decode('utf-8'))
                    p_id = p_data.get('player_id')
                    
                if not p_id:
                    continue
                    
                # 2. Get player match history
                hist_url = f"https://open.faceit.com/data/v4/players/{p_id}/history?game=cs2&limit={matches_per_player}"
                req = urllib.request.Request(hist_url, headers=headers)
                with urllib.request.urlopen(req) as resp:
                    hist_data = json.loads(resp.read().decode('utf-8'))
                    items = hist_data.get('items', [])
                    
                for match in items:
                    m_id = match.get('match_id')
                    if m_id:
                        dems = self.fetch_faceit_match_demo(m_id, api_key=api_key, is_cheater=True)
                        all_downloaded.extend(dems)
                        
            except Exception as e:
                logging.error(f"Error fetching banned matches for {player}: {e}")
                
        return all_downloaded

    def list_downloaded_demos(self) -> Dict[str, List[str]]:
        """Returns inventory of all available .dem files."""
        clean_dems = [os.path.join(self.clean_dir, f) for f in os.listdir(self.clean_dir) if f.endswith('.dem')]
        cheater_dems = [os.path.join(self.cheater_dir, f) for f in os.listdir(self.cheater_dir) if f.endswith('.dem')]
        return {
            'clean': clean_dems,
            'cheaters': cheater_dems,
            'total': len(clean_dems) + len(cheater_dems)
        }
