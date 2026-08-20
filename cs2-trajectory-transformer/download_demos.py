"""
CLI Utility to Download and Manage CS2 Replay Datasets for Thesis Experiments.

Usage Examples:
1. List downloaded replays:
   python download_demos.py --list

2. Download clean match from URL:
   python download_demos.py --url https://example.com/match123.dem.gz --label clean

3. Download cheater match from URL:
   python download_demos.py --url https://example.com/cheater_match.dem.gz --label cheater

4. Download Faceit match by Match ID:
   python download_demos.py --faceit_match_id 1-abc-123 --api_key YOUR_FACEIT_KEY --label clean
"""

import argparse
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

from data.demo_downloader import CS2ReplayDownloader


def main():
    parser = argparse.ArgumentParser(description="CS2 Replay Downloader & Dataset Manager")
    parser.add_argument("--url", type=str, help="Direct URL to a CS2 .dem / .dem.gz / .zip replay")
    parser.add_argument("--faceit_match_id", type=str, help="Faceit match ID to query via Open API")
    parser.add_argument("--api_key", type=str, default=None, help="Faceit API key (optional for public CDN)")
    parser.add_argument("--label", type=str, choices=['clean', 'cheater'], default='clean', help="Replay category")
    parser.add_argument("--list", action="store_true", help="List all downloaded replay files on disk")
    
    args = parser.parse_args()
    downloader = CS2ReplayDownloader()
    
    if args.list:
        inventory = downloader.list_downloaded_demos()
        print("=" * 65)
        print("CS2 MATCH REPLAY INVENTORY (data/raw_demos/)")
        print("=" * 65)
        print(f"Clean Matches:   {len(inventory['clean'])} files")
        for f in inventory['clean'][:5]:
            print(f"  > [Clean]   {os.path.basename(f)}")
        print(f"Cheater Matches: {len(inventory['cheaters'])} files")
        for f in inventory['cheaters'][:5]:
            print(f"  > [Cheater] {os.path.basename(f)}")
        print(f"Total Demos:     {inventory['total']} match files")
        print("=" * 65)
        return

    is_cheater = (args.label == 'cheater')
    
    if args.url:
        print(f"[*] Downloading {args.label} replay from: {args.url}")
        dems = downloader.download_url(args.url, is_cheater=is_cheater)
        print(f"[✓] Extracted {len(dems)} .dem file(s): {dems}")
        
    elif args.faceit_match_id:
        print(f"[*] Fetching Faceit match: {args.faceit_match_id}")
        dems = downloader.fetch_faceit_match_demo(args.faceit_match_id, api_key=args.api_key, is_cheater=is_cheater)
        print(f"[✓] Extracted {len(dems)} .dem file(s): {dems}")
        
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
