"""
CS2 Demo Replay Downloader and Dataset Curation Module.
Supports:
1. Faceit API match downloading (Clean matches & Banned cheater matches).
2. HLTV Pro Tournament demo extraction.
3. Automated decompression (.gz / .bz2 / .zip) into data/raw_demos.
"""

import os
import gzip
import bz2
import zipfile
import logging
import urllib.request
from typing import List, Optional, Dict

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')


class DemoDownloader:
    """Automates replay downloading, decompression, and dataset indexing."""
    def __init__(self, raw_data_dir: str = "data/raw_demos"):
        self.raw_data_dir = raw_data_dir
        self.clean_dir = os.path.join(raw_data_dir, "clean")
        self.cheater_dir = os.path.join(raw_data_dir, "cheaters")
        os.makedirs(self.clean_dir, exist_ok=True)
        os.makedirs(self.cheater_dir, exist_ok=True)

    def decompress_demo(self, compressed_path: str, target_dir: str) -> Optional[str]:
        """Decompresses .gz, .bz2, or .zip replay files into a raw .dem file."""
        bname = os.path.basename(compressed_path)
        base_no_ext, ext = os.path.splitext(bname)
        ext = ext.lower()
        
        target_path = os.path.join(target_dir, f"{base_no_ext}.dem" if not base_no_ext.endswith('.dem') else base_no_ext)
        
        try:
            if ext == '.gz':
                with gzip.open(compressed_path, 'rb') as f_in, open(target_path, 'wb') as f_out:
                    f_out.write(f_in.read())
            elif ext == '.bz2':
                with bz2.open(compressed_path, 'rb') as f_in, open(target_path, 'wb') as f_out:
                    f_out.write(f_in.read())
            elif ext == '.zip':
                with zipfile.ZipFile(compressed_path, 'r') as zip_ref:
                    for member in zip_ref.namelist():
                        if member.endswith('.dem'):
                            zip_ref.extract(member, target_dir)
                            return os.path.join(target_dir, member)
            elif ext == '.dem':
                return compressed_path
            else:
                logging.warning(f"Unsupported format: {ext}")
                return None
                
            logging.info(f"Decompressed {bname} -> {target_path}")
            return target_path
        except Exception as e:
            logging.error(f"Failed decompressing {compressed_path}: {e}")
            return None

    def fetch_demo_from_url(self, url: str, is_cheater: bool = False) -> Optional[str]:
        """Downloads a demo file from a direct URL."""
        target_dir = self.cheater_dir if is_cheater else self.clean_dir
        bname = url.split('/')[-1].split('?')[0]
        temp_dest = os.path.join(target_dir, bname)
        
        try:
            logging.info(f"Downloading {url} -> {temp_dest}...")
            urllib.request.urlretrieve(url, temp_dest)
            
            # Decompress if needed
            final_dem = self.decompress_demo(temp_dest, target_dir)
            if final_dem and final_dem != temp_dest and os.path.exists(temp_dest):
                os.remove(temp_dest)
            return final_dem
        except Exception as e:
            logging.error(f"Download failed for {url}: {e}")
            return None
