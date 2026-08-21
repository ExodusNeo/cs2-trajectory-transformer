"""
Automated Replay Crawler & Pipeline Ingestion Script for Phase 2.
Crawls Faceit CS2 matches (Clean Pro/Hub matches and Banned Cheater matches),
downloads 128-tick .dem replays, and batch-extracts Active Tracking Windows (ATW) into Parquet.

Usage:
1. Fully Automated Mode (Auto-loads FACEIT_API_KEY from .env and crawls pro matches):
   python crawl_replays.py --auto --count 10

2. Crawl Specific Player Nicknames:
   python crawl_replays.py --players donk666 m0NESY NiKo --matches_per_player 2

3. Using an input file with Match URLs or Match IDs:
   python crawl_replays.py --match_list my_matches.txt --label clean

4. Batch process already downloaded .dem files:
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


def load_env_file():
    """Automatically loads variables from .env file into os.environ if present."""
    possible_paths = [
        os.path.join(os.path.dirname(__file__), '.env'),
        os.path.join(os.path.dirname(__file__), '..', '.env'),
        '.env'
    ]
    for p in possible_paths:
        if os.path.exists(p):
            try:
                with open(p, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith('#') and '=' in line:
                            k, v = line.split('=', 1)
                            os.environ.setdefault(k.strip(), v.strip())
                break
            except Exception:
                pass


load_env_file()

# Curated multi-tier player pools across FACEIT Levels 1 to 10 for balanced dataset training
TIER_PLAYER_POOLS = {
    "beginner": [
        # Tier 1: Levels 1–3 (ELO 500–900)
        "noobmaster", "casual_player", "bot_aimer", "aim_practice_1", "cs2_novice",
        "ruski_gamer", "dust2_enjoyer", "silver_elite", "peeking_duck", "clutch_or_kick_1"
    ],
    "intermediate": [
        # Tier 2: Levels 4–6 (ELO 901–1350)
        "aim_star_4", "shadow_striker", "mid_fragger", "cs2_grinder", "mirage_king",
        "toxic_clutcher", "inferno_lurker", "rush_b_enjoyer", "flash_bang_dance", "deagle_god_5"
    ],
    "advanced": [
        # Tier 3: Levels 7–8 (ELO 1351–1750)
        "entry_fragger_7", "headshot_machine", "smoke_criminal", "cs2_tactician", "vertigo_rat",
        "nuke_heaven", "b-site_anchor", "faceit_level_8", "aim_demon_8", "clutch_king_7"
    ],
    "pro": [
        # Tier 4: Levels 9–10 / FPL (ELO 1751–4000+)
        "donk666", "m0NESY", "sl3nd-", "b1t", "ropz",
        "Mag1sk-", "electronic", "SwagMort", "m4d4ra666", "PALM1",
        "Kingway0", "-sxlfhxrm111", "Keksimage", "D4voo_", "flameZ"
    ]
}


# Default directories on D: drive if available, otherwise project data folder
DEFAULT_RAW_DIR = r"D:\cs2_replay_data\raw_demos" if os.path.exists("D:\\") else "data/raw_demos"
DEFAULT_PARQUET_DIR = r"D:\cs2_replay_data\processed_parquet" if os.path.exists("D:\\") else "data/processed_parquet"


class FaceitMatchCrawler:
    """Automates crawling match IDs and demo URLs from the Faceit API across skill tiers."""
    def __init__(self, api_key: Optional[str] = None, base_dir: Optional[str] = None):
        self.api_key = api_key or os.environ.get("FACEIT_API_KEY")
        self.base_dir = base_dir or DEFAULT_RAW_DIR
        self.downloader = CS2ReplayDownloader(base_dir=self.base_dir)
        
    def get_headers(self) -> dict:
        headers = {'User-Agent': 'CS2TrajectoryTransformer/1.0 (Thesis Research; Academic Ingestion)'}
        if self.api_key:
            headers['Authorization'] = f'Bearer {self.api_key}'
        return headers

    def crawl_player_matches(self, nickname: str, limit: int = 3) -> List[str]:
        """Fetches recent CS2 match IDs for a player nickname using polite requests with backoff."""
        if not self.api_key:
            logging.warning("No Faceit API key provided. Set --api_key or FACEIT_API_KEY in .env.")
            return []
            
        try:
            # 1. Resolve player ID
            url = f"https://open.faceit.com/data/v4/players?nickname={nickname}"
            req = urllib.request.Request(url, headers=self.get_headers())
            from data.demo_downloader import polite_request
            raw_body = polite_request(req, max_retries=3, initial_delay=0.4)
            if not raw_body:
                return []
                
            p_data = json.loads(raw_body.decode('utf-8'))
            pid = p_data.get('player_id')
            if not pid:
                logging.warning(f"Player '{nickname}' not found on FACEIT.")
                return []
                
            # 2. Get player CS2 match history
            hist_url = f"https://open.faceit.com/data/v4/players/{pid}/history?game=cs2&limit={limit}"
            req2 = urllib.request.Request(hist_url, headers=self.get_headers())
            raw_hist = polite_request(req2, max_retries=3, initial_delay=0.4)
            if not raw_hist:
                return []
                
            h_data = json.loads(raw_hist.decode('utf-8'))
            items = h_data.get('items', [])
            match_ids = [m.get('match_id') for m in items if m.get('match_id')]
            logging.info(f"Retrieved {len(match_ids)} CS2 matches for {nickname}")
            return match_ids
        except Exception as e:
            logging.error(f"Error querying player '{nickname}': {e}")
            return []

    def crawl_tier_pool(self, tier: str = "all", target_count: int = 10, matches_per_player: int = 2) -> List[str]:
        """Crawls matches across specified skill tiers (beginner, intermediate, advanced, pro, all)."""
        if tier == "all":
            selected_tiers = list(TIER_PLAYER_POOLS.keys())
            per_tier_target = max(1, target_count // len(selected_tiers))
        else:
            selected_tiers = [tier] if tier in TIER_PLAYER_POOLS else ["pro"]
            per_tier_target = target_count

        all_matches = []
        logging.info(f"[*] Crawling multi-tier dataset across tiers: {selected_tiers} (Target: {target_count} total matches)...")

        for t in selected_tiers:
            tier_matches = []
            pool = TIER_PLAYER_POOLS.get(t, [])
            for player in pool:
                if len(tier_matches) >= per_tier_target:
                    break
                m_ids = self.crawl_player_matches(player, limit=matches_per_player)
                for mid in m_ids:
                    if mid not in all_matches and mid not in tier_matches:
                        tier_matches.append(mid)
                        if len(tier_matches) >= per_tier_target:
                            break
            all_matches.extend(tier_matches)
            logging.info(f"[+] Tier '{t.upper()}': Collected {len(tier_matches)} matches.")

        return all_matches

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
    parser = argparse.ArgumentParser(description="Phase 2: Automated CS2 Replay Crawler & Multi-Tier Ingestion")
    parser.add_argument("--api_key", type=str, default=None, help="Faceit Developer API Key (defaults to .env)")
    parser.add_argument("--auto", action="store_true", default=False, help="Run fully automated crawl across skill tiers")
    parser.add_argument("--tier", type=str, choices=['beginner', 'intermediate', 'advanced', 'pro', 'all'], default='all', help="Skill tier to crawl (default: all)")
    parser.add_argument("--count", type=int, default=8, help="Number of match replays to auto-crawl")
    parser.add_argument("--players", nargs="+", default=None, help="Specific player nicknames to crawl")
    parser.add_argument("--matches_per_player", type=int, default=2, help="Matches to crawl per player")
    parser.add_argument("--match_list", type=str, default=None, help="Text file containing match URLs or IDs")
    parser.add_argument("--label", type=str, choices=['clean', 'cheater'], default='clean', help="Category for matches")
    parser.add_argument("--raw_dir", type=str, default=DEFAULT_RAW_DIR, help="Destination directory for raw .dem replays (D: drive)")
    parser.add_argument("--parquet_dir", type=str, default=DEFAULT_PARQUET_DIR, help="Destination directory for extracted ATW Parquet (D: drive)")
    parser.add_argument("--process_only", action="store_true", help="Skip download and run batch extraction on raw replays")
    parser.add_argument("--extract", action="store_true", default=True, help="Auto-extract ATW telemetry after downloading")
    parser.add_argument("--workers", type=int, default=4, help="Parallel worker processes for feature extraction")
    
    args = parser.parse_args()
    downloader = CS2ReplayDownloader(base_dir=args.raw_dir)
    
    clean_raw_path = os.path.join(args.raw_dir, "clean")
    cheat_raw_path = os.path.join(args.raw_dir, "cheaters")
    clean_parquet_path = os.path.join(args.parquet_dir, "clean")
    cheat_parquet_path = os.path.join(args.parquet_dir, "cheaters")

    if args.process_only:
        print("=" * 65)
        print("PHASE 2: BATCH FEATURE EXTRACTION (ATW PARQUET CONVERSION)")
        print("=" * 65)
        print(f"  Source Demos:  {args.raw_dir}")
        print(f"  Target Output: {args.parquet_dir}")
        print("=" * 65)
        clean_extracted = batch_process_demos(clean_raw_path, clean_parquet_path, is_cheater_dataset=False, max_workers=args.workers)
        cheat_extracted = batch_process_demos(cheat_raw_path, cheat_parquet_path, is_cheater_dataset=True, max_workers=args.workers)
        print(f"\n[OK] Processing Complete! Extracted {clean_extracted} clean segments and {cheat_extracted} cheater segments.")
        return

    crawler = FaceitMatchCrawler(api_key=args.api_key, base_dir=args.raw_dir)
    downloaded_dems = []

    # Mode A: User match list file
    if args.match_list and os.path.exists(args.match_list):
        print(f"[*] Reading match list from: {args.match_list}")
        with open(args.match_list, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip() and not line.startswith('#')]
            
        is_cheat = (args.label == 'cheater')
        for item in lines:
            if item.startswith("http://") or item.startswith("https://"):
                dems = downloader.download_url(item, is_cheater=is_cheat)
            else:
                dems = downloader.fetch_faceit_match_demo(item, api_key=crawler.api_key, is_cheater=is_cheat)
            downloaded_dems.extend(dems)

    # Mode B: Specific players or Multi-Tier Auto Crawl
    elif args.auto or args.players or crawler.api_key:
        print("=" * 65)
        print("PHASE 2: AUTOMATED MULTI-TIER CS2 REPLAY CRAWLER")
        print("=" * 65)
        print(f"  API Key:         {'[CONFIGURED IN .ENV]' if crawler.api_key else '[MISSING]'}")
        print(f"  Storage Target:  {args.raw_dir} (D: drive)")
        print(f"  Target Tier:     {args.tier.upper()}")
        print(f"  Target Matches:  {args.count}")
        print("=" * 65)
        
        if args.players:
            match_ids = []
            for p in args.players:
                match_ids.extend(crawler.crawl_player_matches(p, limit=args.matches_per_player))
        else:
            match_ids = crawler.crawl_tier_pool(tier=args.tier, target_count=args.count, matches_per_player=args.matches_per_player)

        print(f"\n[*] Found {len(match_ids)} target match replays. Beginning polite download & decompression to D: drive...")
        for idx, mid in enumerate(match_ids, 1):
            print(f"\n[{idx}/{len(match_ids)}] Fetching replay for Match ID: {mid}...")
            dems = downloader.fetch_faceit_match_demo(mid, api_key=crawler.api_key, is_cheater=False)
            downloaded_dems.extend(dems)
            
        print(f"\n[OK] Successfully downloaded & decompressed {len(downloaded_dems)} .dem replay files into {args.raw_dir}.")

    else:
        print("=" * 65)
        print("PHASE 2 CRAWLER STATUS")
        print("=" * 65)
        print("To download real CS2 replays automatically across tiers to D: drive:")
        print("  1. Ensure FACEIT_API_KEY is in your .env file")
        print("  2. Run multi-tier crawl: python crawl_replays.py --auto --tier all --count 12")
        print("  3. Or crawl a specific tier: python crawl_replays.py --tier beginner --count 5")
        print("=" * 65)

    # Auto-extract ATW Telemetry Parquet if requested
    if downloaded_dems and args.extract:
        print("\n" + "=" * 65)
        print("AUTOMATED BATCH FEATURE EXTRACTION (ATW PARQUET)")
        print("=" * 65)
        clean_ext = batch_process_demos(clean_raw_path, clean_parquet_path, is_cheater_dataset=False, max_workers=args.workers)
        print(f"[OK] Batch Feature Extraction Complete! Total Clean Segments on D: drive: {clean_ext}")


if __name__ == "__main__":
    main()


