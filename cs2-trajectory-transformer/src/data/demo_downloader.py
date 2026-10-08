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
        self.staging_dir = os.path.join(self.base_dir, "staging")
        self.manifest_path = os.path.join(self.staging_dir, "staging_manifest.json")
        os.makedirs(self.clean_dir, exist_ok=True)
        os.makedirs(self.cheater_dir, exist_ok=True)
        os.makedirs(self.staging_dir, exist_ok=True)

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
                            # Prevent Zip Slip directory traversal attacks
                            safe_name = os.path.basename(member)
                            if not safe_name:
                                continue
                            out_path = os.path.join(destination_dir, safe_name)
                            with zip_ref.open(member) as source, open(out_path, 'wb') as target:
                                target.write(source.read())
                            extracted_dems.append(out_path)

            elif file_path.endswith('.tar.gz') or ext == '.tar':
                with tarfile.open(file_path, 'r:*') as tar_ref:
                    for member in tar_ref.getmembers():
                        if member.name.lower().endswith('.dem'):
                            # Prevent Tar Slip directory traversal attacks
                            safe_name = os.path.basename(member.name)
                            if not safe_name:
                                continue
                            out_path = os.path.join(destination_dir, safe_name)
                            extracted_file = tar_ref.extractfile(member)
                            if extracted_file:
                                with open(out_path, 'wb') as target:
                                    target.write(extracted_file.read())
                                extracted_dems.append(out_path)

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
        if not url.lower().startswith("https://"):
            logging.error(f"Insecure or invalid URL scheme rejected: {url}. Only HTTPS URLs are allowed.")
            return []

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

    def get_player_bans(self, player_id: str, api_key: Optional[str] = None) -> List[dict]:
        """Queries the official Faceit API for a player's ban records."""
        headers = {'User-Agent': 'CS2TrajectoryTransformer/1.0 (Thesis Research; Academic Ingestion)'}
        if api_key:
            headers['Authorization'] = f'Bearer {api_key}'
            
        ban_url = f"https://open.faceit.com/data/v4/players/{player_id}/bans"
        req = urllib.request.Request(ban_url, headers=headers)
        raw = polite_request(req, max_retries=3, initial_delay=0.3)
        if not raw:
            return []
            
        try:
            ban_data = json.loads(raw.decode('utf-8'))
            return ban_data.get('items', [])
        except Exception as e:
            logging.error(f"Error parsing bans for player {player_id}: {e}")
            return []

    def is_banned_for_cheating(self, player_id: str, api_key: Optional[str] = None) -> Tuple[bool, Optional[int], Optional[str]]:
        """
        Verifies whether a player account has an active or permanent ban for cheating.
        Returns: (is_cheater_banned: bool, starts_at_timestamp: Optional[int], reason: Optional[str])
        """
        bans = self.get_player_bans(player_id, api_key=api_key)
        for b in bans:
            reason = str(b.get('reason', '')).lower()
            # Faceit reasons: 'cheating', 'cheat', 'ban evasion', 'smurfing', etc.
            if 'cheat' in reason or 'aim' in reason:
                starts_at = b.get('starts_at')
                logging.info(f"[CONFIRMED CHEATER] Player {player_id} has confirmed ban: reason='{b.get('reason')}', starts_at={starts_at}")
                return True, starts_at, b.get('reason')
        return False, None, None

    def fetch_banned_cheater_matches(
        self, 
        banned_steam_or_nicknames: List[str], 
        api_key: Optional[str] = None,
        matches_per_player: int = 2,
        verify_ban: bool = True
    ) -> List[str]:
        """
        Queries Faceit API for banned cheaters, mathematically verifies their cheating ban,
        identifies the match played right before the ban timestamp, and downloads the replay.
        """
        headers = {'User-Agent': 'CS2TrajectoryTransformer/1.0 (Thesis Research; Academic Ingestion)'}
        if api_key:
            headers['Authorization'] = f'Bearer {api_key}'
            
        all_downloaded = []
        for player in banned_steam_or_nicknames:
            player = str(player).strip()
            if not player or player.startswith('#'):
                continue
                
            try:
                # Support both SteamID64 (numeric) and Faceit Nicknames
                if player.isdigit() and len(player) >= 16:
                    url = f"https://open.faceit.com/data/v4/players?game_player_id={player}&game=cs2"
                else:
                    url = f"https://open.faceit.com/data/v4/players?nickname={player}"
                    
                req = urllib.request.Request(url, headers=headers)
                raw = polite_request(req, max_retries=3, initial_delay=0.4)
                if not raw:
                    logging.warning(f"Could not resolve player '{player}' on Faceit.")
                    continue
                    
                p_data = json.loads(raw.decode('utf-8'))
                p_id = p_data.get('player_id')
                nickname = p_data.get('nickname', player)
                if not p_id:
                    continue
                    
                # 1. Verify ban status
                starts_at_ts = None
                if verify_ban:
                    is_cheater, starts_at_ts, ban_reason = self.is_banned_for_cheating(p_id, api_key=api_key)
                    if not is_cheater:
                        logging.warning(f"Player '{nickname}' does not have an active cheating ban on record. Skipping.")
                        continue
                    logging.info(f"Verified cheating ban for '{nickname}' (Reason: {ban_reason})")
                    
                # 2. Fetch match history (reverse chronological order)
                hist_url = f"https://open.faceit.com/data/v4/players/{p_id}/history?game=cs2&limit=10"
                req2 = urllib.request.Request(hist_url, headers=headers)
                raw_hist = polite_request(req2, max_retries=3, initial_delay=0.4)
                if not raw_hist:
                    continue
                    
                hist_data = json.loads(raw_hist.decode('utf-8'))
                items = hist_data.get('items', [])
                
                # 3. Select matches played before ban was enacted
                target_matches = []
                for m in items:
                    m_id = m.get('match_id')
                    s_at = m.get('started_at', 0)
                    f_at = m.get('finished_at', s_at)
                    
                    if starts_at_ts is not None:
                        # Allow up to 1-hour window if ban was issued shortly after match concluded
                        if s_at <= (starts_at_ts + 3600):
                            target_matches.append(m_id)
                    else:
                        target_matches.append(m_id)
                        
                    if len(target_matches) >= matches_per_player:
                        break
                        
                # If no timestamp-filtered match found, take the most recent finished match
                if not target_matches and items:
                    target_matches.append(items[0].get('match_id'))
                    
                logging.info(f"Selected {len(target_matches)} pre-ban match(es) for cheater '{nickname}': {target_matches}")
                for mid in target_matches:
                    dems = self.fetch_faceit_match_demo(mid, api_key=api_key, is_cheater=True)
                    all_downloaded.extend(dems)
                    
            except Exception as e:
                logging.error(f"Error querying banned account '{player}': {e}")
                
        return all_downloaded

    def scan_matches_for_cheaters(
        self, 
        match_ids: List[str], 
        api_key: Optional[str] = None,
        max_cheater_matches: int = 5
    ) -> List[str]:
        """
        Automated Match Spider / Ban Scanner:
        Iterates through match lobbies, inspects all 10 players via GET /players/{id}/bans,
        and automatically downloads the replay if any player in the match was banned for cheating.
        """
        headers = {'User-Agent': 'CS2TrajectoryTransformer/1.0 (Thesis Research; Academic Ingestion)'}
        if api_key:
            headers['Authorization'] = f'Bearer {api_key}'
            
        confirmed_cheater_dems = []
        logging.info(f"[*] Scanning {len(match_ids)} match lobbies for confirmed banned cheaters...")
        
        for idx, mid in enumerate(match_ids, 1):
            if len(confirmed_cheater_dems) >= max_cheater_matches:
                break
                
            try:
                # 1. Fetch match details
                api_url = f"https://open.faceit.com/data/v4/matches/{mid}"
                req = urllib.request.Request(api_url, headers=headers)
                raw_body = polite_request(req, max_retries=2, initial_delay=0.3)
                if not raw_body:
                    continue
                    
                m_data = json.loads(raw_body.decode('utf-8'))
                teams = m_data.get('teams', {})
                players_to_check = []
                
                for t_name in ['faction1', 'faction2']:
                    roster = teams.get(t_name, {}).get('roster', [])
                    for p in roster:
                        pid = p.get('player_id')
                        p_nick = p.get('nickname')
                        if pid:
                            players_to_check.append((pid, p_nick))
                            
                # 2. Check each player's ban status
                match_has_cheater = False
                cheater_name = None
                for pid, p_nick in players_to_check:
                    is_cheater, _, reason = self.is_banned_for_cheating(pid, api_key=api_key)
                    if is_cheater:
                        match_has_cheater = True
                        cheater_name = p_nick
                        logging.info(f"[ALERT] Match {mid} contains confirmed banned cheater: '{cheater_name}' (Reason: {reason})")
                        break
                        
                # 3. If match had a cheater, download demo
                if match_has_cheater:
                    demo_url = m_data.get('demo_url', [])
                    if isinstance(demo_url, list) and len(demo_url) > 0:
                        demo_url = demo_url[0]
                    if demo_url:
                        dems = self.download_url(demo_url, is_cheater=True, custom_filename=f"cheater_{cheater_name}_{mid}.dem.zst")
                        confirmed_cheater_dems.extend(dems)
                        logging.info(f"[+] Downloaded confirmed cheater replay: {dems}")
                        
            except Exception as e:
                logging.error(f"Error scanning match {mid}: {e}")
                
        return confirmed_cheater_dems

    def list_downloaded_demos(self) -> Dict[str, List[str]]:
        """Returns inventory of all available .dem files on D: drive."""
        clean_dems = [os.path.join(self.clean_dir, f) for f in os.listdir(self.clean_dir) if f.endswith('.dem')]
        cheater_dems = [os.path.join(self.cheater_dir, f) for f in os.listdir(self.cheater_dir) if f.endswith('.dem')]
        staged_dems = [os.path.join(self.staging_dir, f) for f in os.listdir(self.staging_dir) if f.endswith('.dem')] if os.path.exists(self.staging_dir) else []
        return {
            'clean': clean_dems,
            'cheaters': cheater_dems,
            'staging': staged_dems,
            'total': len(clean_dems) + len(cheater_dems) + len(staged_dems),
            'base_dir': self.base_dir
        }

    def load_staging_manifest(self) -> Dict[str, dict]:
        """Loads the current staging manifest JSON tracking candidate matches."""
        if os.path.exists(self.manifest_path):
            try:
                with open(self.manifest_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logging.error(f"Error loading staging manifest: {e}")
                return {}
        return {}

    def save_staging_manifest(self, manifest: Dict[str, dict]):
        """Persists the staging manifest JSON."""
        try:
            with open(self.manifest_path, 'w', encoding='utf-8') as f:
                json.dump(manifest, f, indent=2)
        except Exception as e:
            logging.error(f"Error saving staging manifest: {e}")

    def stage_match(self, match_id: str, api_key: Optional[str] = None) -> List[str]:
        """
        Stages a freshly completed match into local buffer storage before its 30-day CDN expiration.
        Records match timestamp and player roster for deferred ban auditing.
        """
        headers = {'User-Agent': 'CS2TrajectoryTransformer/1.0 (Thesis Research; Academic Ingestion)'}
        if api_key:
            headers['Authorization'] = f'Bearer {api_key}'

        manifest = self.load_staging_manifest()
        if match_id in manifest and manifest[match_id].get('status') != 'error':
            logging.info(f"Match {match_id} is already in staging (Status: {manifest[match_id].get('status')}). Skipping download.")
            return []

        try:
            url = f"https://open.faceit.com/data/v4/matches/{match_id}"
            req = urllib.request.Request(url, headers=headers)
            raw = polite_request(req, max_retries=3, initial_delay=0.3)
            if not raw:
                return []

            m_data = json.loads(raw.decode('utf-8'))
            demo_url = m_data.get('demo_url', [])
            if isinstance(demo_url, list) and len(demo_url) > 0:
                demo_url = demo_url[0]

            if not demo_url:
                logging.warning(f"No demo_url found for match {match_id}.")
                return []

            # Extract 10 players
            teams = m_data.get('teams', {})
            players_roster = []
            for t_name in ['faction1', 'faction2']:
                roster = teams.get(t_name, {}).get('roster', [])
                for p in roster:
                    pid = p.get('player_id')
                    p_nick = p.get('nickname')
                    steam_id = p.get('game_player_id', '')
                    if pid:
                        p_elo = float(p.get('faceit_elo', p.get('elo', 1500.0)))
                        players_roster.append({
                            'player_id': pid,
                            'nickname': p_nick,
                            'steam_id': steam_id,
                            'elo': p_elo
                        })

            # Download into staging directory
            bname = f"staging_{match_id}.dem.zst"
            extracted_dems = self.download_url(demo_url, is_cheater=False, custom_filename=bname)
            # Ensure moved into staging_dir
            import shutil
            staged_dems = []
            for dem in extracted_dems:
                dest = os.path.join(self.staging_dir, os.path.basename(dem))
                if os.path.abspath(dem) != os.path.abspath(dest):
                    shutil.move(dem, dest)
                staged_dems.append(dest)

            manifest[match_id] = {
                'match_id': match_id,
                'staged_at': int(time.time()),
                'finished_at': m_data.get('finished_at', int(time.time())),
                'demo_files': [os.path.basename(d) for d in staged_dems],
                'players': players_roster,
                'status': 'pending_audit',
                'last_checked': int(time.time())
            }
            self.save_staging_manifest(manifest)
            logging.info(f"[+] Successfully staged match {match_id} with {len(players_roster)} players tracked into {self.staging_dir}")
            return staged_dems

        except Exception as e:
            logging.error(f"Error staging match {match_id}: {e}")
            return []

    def audit_staging(
        self, 
        api_key: Optional[str] = None, 
        graduation_days: int = 21,
        auto_extract: bool = False,
        parquet_dir: Optional[str] = None,
        max_workers: int = 4
    ) -> Dict[str, int]:
        """
        Deferred Ban Auditor:
        Examines all staged match candidates via GET /players/{id}/bans.
        1. If ANY player has received a cheating ban: promotes match to cheaters/.
        2. If match exceeds graduation_days with zero infractions: graduates match to clean/.
        """
        manifest = self.load_staging_manifest()
        import shutil

        stats = {'cheaters_detected': 0, 'clean_graduated': 0, 'pending': 0}
        logging.info(f"[*] Auditing {len(manifest)} staged candidate matches in buffer...")

        for mid, info in manifest.items():
            if info.get('status') != 'pending_audit':
                continue

            demo_files = info.get('demo_files', [])
            cheater_found = False
            cheater_nickname = None
            cheater_reason = None

            # 1. Audit players in match
            for p in info.get('players', []):
                pid = p.get('player_id')
                is_cheat, _, reason = self.is_banned_for_cheating(pid, api_key=api_key)
                if is_cheat:
                    cheater_found = True
                    cheater_nickname = p.get('nickname')
                    cheater_reason = reason
                    break

            if cheater_found:
                logging.info(f"[CONFIRMED CHEATER DETECTED] Match {mid} has banned player '{cheater_nickname}'! Promoting to cheater corpus.")
                
                # Persist banned SteamID to central registry
                ban_file = os.path.join("data", "banned_steamids.json")
                detected_steamids = []
                try:
                    os.makedirs("data", exist_ok=True)
                    existing_bans = set()
                    if os.path.exists(ban_file):
                        with open(ban_file, 'r', encoding='utf-8') as f:
                            existing_bans = set(json.load(f))
                    for p in info.get('players', []):
                        if (p.get('nickname') == cheater_nickname or p.get('player_id') == cheater_nickname) and p.get('steam_id'):
                            existing_bans.add(str(p.get('steam_id')))
                            detected_steamids.append(str(p.get('steam_id')))
                    with open(ban_file, 'w', encoding='utf-8') as f:
                        json.dump(list(existing_bans), f, indent=2)
                except Exception as e:
                    logging.warning(f"Could not persist banned SteamID to {ban_file}: {e}")

                for fname in demo_files:
                    src = os.path.join(self.staging_dir, fname)
                    dst = os.path.join(self.cheater_dir, fname)
                    if os.path.exists(src):
                        shutil.move(src, dst)

                if auto_extract and parquet_dir:
                    from data.batch_processor import batch_process_demos
                    batch_process_demos(
                        self.cheater_dir, 
                        os.path.join(parquet_dir, "cheaters"), 
                        is_cheater_dataset=True, 
                        banned_steamids=detected_steamids or None,
                        max_workers=max_workers
                    )

                info['status'] = 'cheater_detected'
                info['cheater'] = cheater_nickname
                info['ban_reason'] = cheater_reason
                stats['cheaters_detected'] += 1

            else:
                # 2. Check observation age
                finished_at = info.get('finished_at', info.get('staged_at', time.time()))
                days_elapsed = (time.time() - finished_at) / 86400.0

                if days_elapsed >= graduation_days:
                    logging.info(f"[GRADUATION] Match {mid} passed {days_elapsed:.1f} days without infractions. Promoting to clean baseline.")
                    for fname in demo_files:
                        src = os.path.join(self.staging_dir, fname)
                        dst = os.path.join(self.clean_dir, fname)
                        if os.path.exists(src):
                            shutil.move(src, dst)

                    if auto_extract and parquet_dir:
                        from data.batch_processor import batch_process_demos
                        batch_process_demos(self.clean_dir, os.path.join(parquet_dir, "clean"), is_cheater_dataset=False, max_workers=max_workers)

                    info['status'] = 'clean_graduated'
                    stats['clean_graduated'] += 1
                else:
                    info['last_checked'] = int(time.time())
                    stats['pending'] += 1

        self.save_staging_manifest(manifest)
        logging.info(f"[OK] Staging audit complete: {stats['cheaters_detected']} cheaters promoted, {stats['clean_graduated']} clean graduated, {stats['pending']} still pending.")
        return stats


