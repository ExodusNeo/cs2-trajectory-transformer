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
from data.batch_processor import batch_process_demos, audit_clean_atw_quota

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
                with open(p, 'r', encoding='utf-8-sig') as f:
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
    parser.add_argument("--banned_file", type=str, default=None, help="Text file containing suspected/banned player nicknames or SteamIDs")
    parser.add_argument("--scan_cheaters", action="store_true", default=False, help="Run automated Match Spider to scan lobbies for confirmed banned cheaters")
    parser.add_argument("--autonomous", action="store_true", default=False, help="Run fully autonomous ingestion: stages fresh matches and audits buffer for bans")
    parser.add_argument("--stage_recent", action="store_true", default=False, help="Stage fresh matches (< 48h old) to buffer before 30-day CDN deletion")
    parser.add_argument("--audit_staging", action="store_true", default=False, help="Audit staged matches against GET /players/{id}/bans and promote cheaters/clean")
    parser.add_argument("--graduation_days", type=int, default=21, help="Days without cheating infractions before graduating staged match to clean (default: 21)")
    parser.add_argument("--extract", action="store_true", default=True, help="Auto-extract ATW telemetry after downloading")
    parser.add_argument("--workers", type=int, default=4, help="Parallel worker processes for feature extraction")
    parser.add_argument("--target_clean_atws", type=int, default=30000, help="Target clean ATW quota for empirical stopping condition (default: 30000)")
    parser.add_argument("--audit_clean_quota", action="store_true", default=False, help="Audit current processed Parquet inventory against clean ATW quota")
    parser.add_argument("--crawl_until_quota", action="store_true", default=False, help="Continuously crawl and extract matches until target clean ATW quota is satisfied")
    
    args = parser.parse_args()
    downloader = CS2ReplayDownloader(base_dir=args.raw_dir)
    
    clean_raw_path = os.path.join(args.raw_dir, "clean")
    cheat_raw_path = os.path.join(args.raw_dir, "cheaters")
    clean_parquet_path = os.path.join(args.parquet_dir, "clean")
    cheat_parquet_path = os.path.join(args.parquet_dir, "cheaters")

    if args.audit_clean_quota:
        print("=" * 65)
        print("EMPIRICAL STOPPING CONDITION AUDIT (TEST SPLIT QUOTA)")
        print("=" * 65)
        print(f"  Processed Directory:             {args.parquet_dir}")
        print(f"  Target Clean ATWs (Test Split):  {args.target_clean_atws:,}")
        print("=" * 65)
        audit_res = audit_clean_atw_quota(args.parquet_dir, target_clean=args.target_clean_atws)
        print(f"  [Overall Corpus]")
        print(f"    Total Clean ATWs:              {audit_res['overall_clean_count']:,}")
        print(f"    Total Cheater ATWs:            {audit_res['overall_cheater_count']:,}")
        print(f"    Total Parquet Segments:        {audit_res['overall_total_count']:,}")
        print(f"  [Held-Out Test Partition (Nominal ~10% Split)]")
        print(f"    Verified Clean Test ATWs:      {audit_res['test_clean_count']:,} / {audit_res['target_clean']:,} ({audit_res['progress_pct']:.2f}%)")
        print(f"    Cheater Test ATWs:             {audit_res['test_cheater_count']:,}")
        print(f"    Total Test Segments:           {audit_res['test_total_count']:,}")
        print("-" * 65)
        print(f"  Test Quota Satisfied:            {'YES [STOPPING CONDITION MET]' if audit_res['quota_met'] else 'NO [INGESTION REQUIRED]'}")
        if not audit_res['quota_met']:
            print(f"  Remaining Clean Gap (Test):      {audit_res['deficit']:,} clean ATWs")
        print("=" * 65)
        return

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
        audit_res = audit_clean_atw_quota(args.parquet_dir, target_clean=args.target_clean_atws)
        print(f"[*] Stopping Condition Audit: Test Clean={audit_res['test_clean_count']:,} / {audit_res['target_clean']:,} ({audit_res['progress_pct']:.2f}%) | Overall Clean={audit_res['overall_clean_count']:,}. Quota Met: {audit_res['quota_met']}")
        return

    crawler = FaceitMatchCrawler(api_key=args.api_key, base_dir=args.raw_dir)
    downloaded_dems = []
    is_cheater_crawl = False

    # Mode E: Fully Autonomous Rolling Ingestion & Deferred Ban Audit
    if args.autonomous or args.stage_recent or args.audit_staging:
        print("=" * 65)
        print("PHASE 2: AUTONOMOUS ROLLING BUFFER & DEFERRED BAN AUDITOR")
        print("=" * 65)
        print(f"  Storage Target:    {args.raw_dir} (D: drive)")
        print(f"  Staging Cache:     {downloader.staging_dir}")
        print(f"  Graduation Window: {args.graduation_days} days without infraction")
        print("=" * 65)

        # 1. Stage fresh matches if requested
        if args.autonomous or args.stage_recent:
            print(f"\n[*] Harvesting fresh candidate matches across {args.tier.upper()} tier (Target: {args.count} matches)...")
            fresh_matches = crawler.crawl_tier_pool(tier=args.tier, target_count=args.count, matches_per_player=args.matches_per_player)
            print(f"[*] Found {len(fresh_matches)} candidate matches. Streaming .dem.zst to local staging buffer...")
            staged_count = 0
            for idx, mid in enumerate(fresh_matches, 1):
                print(f"  [{idx}/{len(fresh_matches)}] Staging match {mid}...")
                staged = downloader.stage_match(mid, api_key=crawler.api_key)
                if staged:
                    staged_count += 1
            print(f"[OK] Buffered {staged_count} fresh replay files into {downloader.staging_dir} before 30-day CDN expiration.")

        # 2. Audit existing staged matches for bans
        if args.autonomous or args.audit_staging:
            print(f"\n[*] Executing Deferred Ban Audit across all buffered matches...")
            stats = downloader.audit_staging(
                api_key=crawler.api_key,
                graduation_days=args.graduation_days,
                auto_extract=args.extract,
                parquet_dir=args.parquet_dir,
                max_workers=args.workers
            )
            print("\n" + "=" * 65)
            print("AUTONOMOUS BUFFER AUDIT SUMMARY")
            print("=" * 65)
            print(f"  Confirmed Cheaters Detected: {stats['cheaters_detected']} (Promoted to cheaters/)")
            print(f"  Clean Matches Graduated:     {stats['clean_graduated']} (Promoted to clean/)")
            print(f"  Pending Ongoing Observation: {stats['pending']} (Awaiting ban maturation)")
            print("=" * 65)
        return

    # Mode A: Banned Cheater List Ingestion with Ban Verification
    elif args.banned_file or args.banned_players:

        is_cheater_crawl = True
        banned_pool = []
        if args.banned_file and os.path.exists(args.banned_file):
            with open(args.banned_file, "r", encoding="utf-8") as f:
                banned_pool.extend([line.strip() for line in f if line.strip() and not line.startswith('#')])
        if args.banned_players:
            banned_pool.extend(args.banned_players)

        print("=" * 65)
        print("PHASE 2: CONFIRMED BANNED CHEATER INGESTION & BAN VERIFICATION")
        print("=" * 65)
        print(f"  Candidate Accounts: {len(banned_pool)}")
        print(f"  API Verification:   ENABLED (GET /players/{{id}}/bans)")
        print(f"  Destination:        {cheat_raw_path}")
        print("=" * 65)
        dems = downloader.fetch_banned_cheater_matches(
            banned_steam_or_nicknames=banned_pool,
            api_key=crawler.api_key,
            matches_per_player=args.matches_per_player,
            verify_ban=True
        )
        downloaded_dems.extend(dems)
        print(f"\n[OK] Downloaded {len(downloaded_dems)} pre-ban cheater replay files into {cheat_raw_path}.")

    # Mode B: Automated Match Spider / Ban Scanner
    elif args.scan_cheaters:
        is_cheater_crawl = True
        print("=" * 65)
        print("PHASE 2: AUTOMATED MATCH SPIDER & CHEATER BAN SCANNER")
        print("=" * 65)
        print(f"  Scanning recent match lobbies across skill tiers...")
        print(f"  Target Confirmed Cheater Demos: {args.count}")
        print("=" * 65)
        # Gather candidate matches from recent tier pool
        candidate_matches = crawler.crawl_tier_pool(tier="all", target_count=args.count * 4, matches_per_player=2)
        dems = downloader.scan_matches_for_cheaters(
            match_ids=candidate_matches,
            api_key=crawler.api_key,
            max_cheater_matches=args.count
        )
        downloaded_dems.extend(dems)
        print(f"\n[OK] Spider scan complete! Found and downloaded {len(downloaded_dems)} verified cheater match replays.")

    # Mode C: User match list file
    elif args.match_list and os.path.exists(args.match_list):
        print(f"[*] Reading match list from: {args.match_list}")
        with open(args.match_list, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip() and not line.startswith('#')]
            
        is_cheat = (args.label == 'cheater')
        is_cheater_crawl = is_cheat
        for item in lines:
            if item.startswith("http://") or item.startswith("https://"):
                dems = downloader.download_url(item, is_cheater=is_cheat)
            else:
                dems = downloader.fetch_faceit_match_demo(item, api_key=crawler.api_key, is_cheater=is_cheat)
            downloaded_dems.extend(dems)

    # Mode D1: Continuous Ingestion until Clean ATW Quota is Satisfied
    elif args.crawl_until_quota:
        print("=" * 65)
        print("PHASE 2: EMPIRICAL STOPPING CONDITION INGESTION LOOP")
        print("=" * 65)
        print(f"  Target Quota (Test Split): {args.target_clean_atws:,} Clean ATWs")
        print(f"  Storage Target:            {args.raw_dir} (D: drive)")
        print(f"  Batch Increment:           {args.count} matches per iteration")
        print("=" * 65)
        
        cycle = 1
        while True:
            audit_res = audit_clean_atw_quota(args.parquet_dir, target_clean=args.target_clean_atws)
            print(f"\n[*] Iteration {cycle}: Test Split Clean = {audit_res['test_clean_count']:,} / {audit_res['target_clean']:,} ({audit_res['progress_pct']:.2f}%) | Overall Clean = {audit_res['overall_clean_count']:,}")
            if audit_res['quota_met']:
                print(f"[OK] Stopping condition satisfied! Held-out test clean count ({audit_res['test_clean_count']:,}) >= Target ({audit_res['target_clean']:,}).")
                break
                
            match_ids = crawler.crawl_tier_pool(tier=args.tier, target_count=args.count, matches_per_player=args.matches_per_player)
            if not match_ids:
                print("[!] No additional matches discovered in tier pool. Stopping ingestion loop.")
                break
                
            cycle_dems = []
            for mid in match_ids:
                dems = downloader.fetch_faceit_match_demo(mid, api_key=crawler.api_key, is_cheater=False)
                cycle_dems.extend(dems)
                
            if cycle_dems and args.extract:
                batch_process_demos(clean_raw_path, clean_parquet_path, is_cheater_dataset=False, max_workers=args.workers)
                
            cycle += 1
        return

    # Mode D2: Specific players or Multi-Tier Auto Crawl (Clean)
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
        print("PHASE 2 CRAWLER STATUS & USAGE")
        print("=" * 65)
        print("1. Fully Autonomous Mode (Stages Fresh Matches & Audits Buffer for Bans):")
        print("   python crawl_replays.py --autonomous --count 8")
        print("2. Audit Staged Matches Only:")
        print("   python crawl_replays.py --audit_staging")
        print("3. Clean Replays (Multi-Tier Direct Ingestion):")
        print("   python crawl_replays.py --auto --tier all --count 12")
        print("4. Confirmed Cheater Replays (from Banned List with Verification):")
        print("   python crawl_replays.py --banned_file data/banned_cheaters.txt --matches_per_player 1")
        print("5. Automated Match Spider (Scan Lobbies for Cheaters):")
        print("   python crawl_replays.py --scan_cheaters --count 5")
        print("6. Audit Clean ATW Quota (Empirical Stopping Condition on Test Split):")
        print("   python crawl_replays.py --audit_clean_quota --target_clean_atws 30000")
        print("7. Autonomous Ingestion Loop Until Test Split Quota is Satisfied:")
        print("   python crawl_replays.py --crawl_until_quota --target_clean_atws 30000 --count 10")
        print("=" * 65)

    # Auto-extract ATW Telemetry Parquet if requested
    if downloaded_dems and args.extract:
        print("\n" + "=" * 65)
        print("AUTOMATED BATCH FEATURE EXTRACTION (ATW PARQUET)")
        print("=" * 65)
        if is_cheater_crawl:
            cheat_ext = batch_process_demos(cheat_raw_path, cheat_parquet_path, is_cheater_dataset=True, max_workers=args.workers)
            print(f"[OK] Batch Feature Extraction Complete! Total Cheater Segments on D: drive: {cheat_ext}")
        else:
            clean_ext = batch_process_demos(clean_raw_path, clean_parquet_path, is_cheater_dataset=False, max_workers=args.workers)
            print(f"[OK] Batch Feature Extraction Complete! Total Clean Segments on D: drive: {clean_ext}")
            
        # Report progress toward empirical stopping condition
        audit_res = audit_clean_atw_quota(args.parquet_dir, target_clean=args.target_clean_atws)
        print("\n" + "=" * 65)
        print("EMPIRICAL STOPPING CONDITION AUDIT (TEST SPLIT)")
        print("=" * 65)
        print(f"  Overall Clean ATWs:              {audit_res['overall_clean_count']:,}")
        print(f"  Held-Out Test Clean ATWs:        {audit_res['test_clean_count']:,} / {audit_res['target_clean']:,} ({audit_res['progress_pct']:.2f}%)")
        print(f"  Stopping Condition Met (Test):   {'YES [QUOTA SATISFIED]' if audit_res['quota_met'] else 'NO [INGESTION SHORTFALL]'}")
        if not audit_res['quota_met']:
            print(f"  Remaining Clean Gap (Test):      {audit_res['deficit']:,} ATWs")
        print("=" * 65)


if __name__ == "__main__":
    main()


