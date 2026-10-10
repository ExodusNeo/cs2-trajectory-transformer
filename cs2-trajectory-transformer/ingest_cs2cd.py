"""
Download (optional) and ingest CS2CD matches into ATW Parquet windows.

Examples:
    # Pilot: 40 random matches per folder (seeded), then extract features
    python ingest_cs2cd.py --download_per_folder 40 --seed 0
    # Process whatever is already downloaded
    python ingest_cs2cd.py

Requires CS2_PSEUDONYMIZATION_SALT (see .env.example). Output windows carry source='cs2cd' and
are written to a separate store so results can be reported per data source.
"""

import argparse
import logging
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

from data.cs2cd_adapter import download_cs2cd_subset, process_cs2cd_directory


def main() -> None:
    ap = argparse.ArgumentParser(description="Ingest the CS2CD dataset")
    ap.add_argument("--raw_root", default="data/raw_demos/cs2cd", help="where CS2CD folders live / are downloaded")
    ap.add_argument("--output_dir", default="data/processed_parquet_cs2cd", help="ATW Parquet output (kept separate from FACEIT)")
    ap.add_argument("--download_per_folder", type=int, default=0, help="download N random matches per folder first (0 = skip)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    if args.download_per_folder > 0:
        chosen = download_cs2cd_subset(args.raw_root, args.download_per_folder, seed=args.seed)
        logging.info(f"Selected matches: { {k: len(v) for k, v in chosen.items()} }")
    totals = process_cs2cd_directory(args.raw_root, args.output_dir, max_workers=args.workers)
    print(totals)
    if totals.get('errors'):
        sys.exit(1)


if __name__ == "__main__":
    main()
