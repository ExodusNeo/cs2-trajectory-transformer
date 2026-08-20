"""
Automated Replay Crawler & Pipeline Ingestion Script for Phase 2.
Crawls Faceit CS2 matches (Clean Pro/Hub matches and Banned Cheater matches),
downloads 128-tick .dem replays, and batch-extracts Active Tracking Windows (ATW) into Parquet.

Usage:
1. With Faceit API Key (Recommended):
   python crawl_replays.py --api_key YOUR_FACEIT_API_KEY --clean_count 50 --cheater_count 50

2. Using an input file with Match URLs or Match IDs:
   python crawl_replays.py --match_list my_matches.txt --label clean

3. Batch process already downloaded .dem files:
   python crawl_replays.py --process_only
"""

import os
import sys
import argparse
import logging
import json
import urllib.request
from typing import List, Optional

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

from data.demo_downloader import CS2ReplayDownloader
from data.batch_processor import batch_process_demos

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

# Sample verified Faceit Hubs & Competition IDs for 128-tick Clean CS2 Demos
DEFAULT_CLEAN_HUBS = [
    "74ca5b83-5056-4dc8-84a1-5742f0bbf878", # FPL Europe CS2 Hub
    "e9d56ff0-149b-449e-b9b5-4b361a49f579", # FPL North America CS2 Hub
]


class FaceitMatchCrawler:
    """Automates crawling match IDs and demo URLs from the Faceit API."""
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get("FACEIT_API_KEY")
        self.downloader = CS2ReplayDownloader()
        
    def get_headers(self) -> dict:
        headers = {'User-Agent': 'CS2TrajectoryTransformer/1.0'}
        if self.api_key:
            headers['Authorization'] = f'Bearer {self.api_key}'
        return headers

    def crawl_hub_matches(self, hub_id: str, limit: int = 20) -> List[str]:
        """Fetches recent match IDs from a verified competitive Faceit hub."""
        if not self.api_key:
            logging.warning("No Faceit API key provided. Set --api_key or FACEIT_API_KEY environment variable.")
            return []
            
        url = f"https://open.faceit.com/data/v4/hubs/{hub_id}/matches?type=past&limit={limit}"
        try:
            req = urllib.request.Request(url, headers=self.get_headers())
            with urllib.request.urlopen(req) as resp:
                data = json.loads(resp.read().decode('utf-8'))
                items = data.get('items', [])
                match_ids = [m.get('match_id') for m in items if m.get('match_id')]
                logging.info(f"Retrieved {len(match_ids)} matches from Faceit Hub {hub_id}")
                return match_ids
        except Exception as e:
            logging.error(f"Error crawling hub {hub_id}: {e}")
            return []

    def crawl_banned_players(self, banned_usernames: List[str], matches_per_player: int = 3) -> List[str]:
        """Crawls confirmed cheater match replays for a list of banned accounts."""
        logging.info(f"Crawling banned cheater matches for {len(banned_usernames)} player accounts...")
        dems = self.downloader.fetch_banned_cheater_matches(
            banned_steam_or_nicknames=banned_usernames,
            api_key=self.api_key,
            matches_per_player=matches_per_player
        )
        return dems


def main():
    parser = argparse.ArgumentParser(description="Phase 2: CS2 Replay Crawler & Batch Ingestion")
    parser.add_argument("--api_key", type=str, default=None, help="Faceit Developer API Key")
    parser.add_argument("--match_list", type=str, default=None, help="Text file containing match URLs or IDs (one per line)")
    parser.add_argument("--label", type=str, choices=['clean', 'cheater'], default='clean', help="Category for matches")
    parser.add_argument("--clean_count", type=int, default=10, help="Target number of clean hub matches to crawl")
    parser.add_argument("--process_only", action="store_true", help="Skip download and run batch extraction on data/raw_demos")
    parser.add_argument("--workers", type=int, default=4, help="Parallel worker processes for feature extraction")
    
    args = parser.parse_args()
    downloader = CS2ReplayDownloader()
    
    if args.process_only:
        print("=" * 65)
        print("PHASE 2: BATCH FEATURE EXTRACTION (ATW PARQUET CONVERSION)")
        print("=" * 65)
        clean_extracted = batch_process_demos("data/raw_demos/clean", "data/processed_parquet/clean", is_cheater_dataset=False, max_workers=args.workers)
        cheat_extracted = batch_process_demos("data/raw_demos/cheaters", "data/processed_parquet/cheaters", is_cheater_dataset=True, max_workers=args.workers)
        print(f"\n[✓] Processing Complete! Extracted {clean_extracted} clean segments and {cheat_extracted} cheater segments.")
        return

    crawler = FaceitMatchCrawler(api_key=args.api_key)
    
    # Mode A: Download from user match list file
    if args.match_list and os.path.exists(args.match_list):
        print(f"[*] Reading match list from: {args.match_list}")
        with open(args.match_list, "r") as f:
            lines = [line.strip() for line in f if line.strip() and not line.startswith('#')]
            
        is_cheat = (args.label == 'cheater')
        for item in lines:
            if item.startswith("http://") or item.startswith("https://"):
                downloader.download_url(item, is_cheater=is_cheat)
            else:
                downloader.fetch_faceit_match_demo(item, api_key=args.api_key, is_cheater=is_cheat)
                
    # Mode B: Automated Hub Crawl
    elif args.api_key:
        print("=" * 65)
        print("PHASE 2: AUTOMATED FACEIT MATCH REPLAY CRAWLER")
        print("=" * 65)
        for hub in DEFAULT_CLEAN_HUBS:
            match_ids = crawler.crawl_hub_matches(hub, limit=args.clean_count)
            for m_id in match_ids:
                downloader.fetch_faceit_match_demo(m_id, api_key=args.api_key, is_cheater=False)
                
    else:
        print("=" * 65)
        print("PHASE 2 CRAWLER STATUS")
        print("=" * 65)
        print("To download real CS2 replays, you can provide either:")
        print("  1. A free Faceit Developer API Key (run: python crawl_replays.py --api_key YOUR_KEY)")
        print("  2. A list of demo URLs or Match IDs in a text file (run: python crawl_replays.py --match_list my_matches.txt)")
        print("  3. Or place .dem files directly in data/raw_demos/ and run: python crawl_replays.py --process_only")
        print("=" * 65)


if __name__ == "__main__":
    main()
