"""
Batch Demo Processing Pipeline for CS2 Replays.
Scans raw .dem files, extracts active combat tracking trajectories,
computes 8D micro-kinematic feature vectors, and saves structured Parquet datasets.
"""

import os
import glob
import logging
import numpy as np
import pandas as pd
from typing import List, Optional, Dict
from concurrent.futures import ProcessPoolExecutor, as_completed

from .demo_parser import CS2DemoParser
from .atw_filter import extract_active_tracking_windows
from features.kinematics import compute_kinematic_features

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')


FEATURE_COLUMNS = [
    'yaw', 
    'pitch', 
    'angular_velocity', 
    'angular_accel', 
    'angular_jerk', 
    'trajectory_curvature', 
    'curvature_entropy', 
    'tremor_power_8_12hz'
]


def process_single_demo(
    demo_path: str, 
    output_dir: str, 
    is_cheater_demo: bool = False,
    default_elo: float = 1500.0,
    min_window_len: int = 32
) -> int:
    """
    Processes a single CS2 .dem replay:
    1. Parses player ticks and weapon events.
    2. Identifies all players and teams.
    3. Extracts Active Tracking Windows for each player.
    4. Computes kinematic & tremor features.
    5. Saves trajectory segments to Parquet.
    
    Returns the count of extracted ATW trajectory segments.
    """
    match_name = os.path.splitext(os.path.basename(demo_path))[0]
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        parser = CS2DemoParser(demo_path)
        ticks_df = parser.parse_ticks()
        if ticks_df.empty:
            logging.warning(f"No tick data in {demo_path}")
            return 0
            
        events_dict = parser.parse_events()
        fire_events = events_dict.get('weapon_fire', pd.DataFrame())
        
        steam_ids = ticks_df['steamid'].dropna().unique()
        total_segments = 0
        
        for steamid in steam_ids:
            p_df = ticks_df[ticks_df['steamid'] == steamid].sort_values('tick').reset_index(drop=True)
            if len(p_df) < min_window_len:
                continue
                
            # Filter weapon fires by this player
            if not fire_events.empty and 'user_steamid' in fire_events.columns:
                p_fires = fire_events[fire_events['user_steamid'] == steamid]
                event_ticks = p_fires['tick'].tolist() if 'tick' in p_fires.columns else []
            else:
                event_ticks = []
                
            # Extract Active Tracking Windows (ATW)
            atw_segments = extract_active_tracking_windows(
                player_df=p_df,
                event_ticks=event_ticks,
                tick_buffer=64,
                min_window_len=min_window_len
            )
            
            for seg_idx, seg_df in enumerate(atw_segments):
                # Compute 8D biomechanical features
                featured_df = compute_kinematic_features(seg_df, tick_rate=128.0, extract_tremor=True)
                
                # Metadata tags
                featured_df['match_id'] = match_name
                featured_df['steamid'] = steamid
                featured_df['segment_id'] = seg_idx
                featured_df['is_aimbot'] = int(is_cheater_demo)
                featured_df['player_elo'] = default_elo
                
                # Export to Parquet
                out_filename = f"{match_name}_p{steamid}_seg{seg_idx}.parquet"
                out_path = os.path.join(output_dir, out_filename)
                featured_df.to_parquet(out_path, index=False)
                total_segments += 1
                
        logging.info(f"Processed {match_name}: {total_segments} ATW segments extracted.")
        return total_segments
        
    except Exception as e:
        logging.error(f"Error processing {demo_path}: {e}")
        return 0


def batch_process_demos(
    demo_dir: str, 
    output_dir: str, 
    is_cheater_dataset: bool = False,
    max_workers: int = 4
) -> int:
    """Processes an entire directory of .dem files in parallel."""
    demo_files = glob.glob(os.path.join(demo_dir, "*.dem"))
    logging.info(f"Found {len(demo_files)} demos in {demo_dir}")
    
    total_extracted = 0
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(process_single_demo, demo, output_dir, is_cheater_dataset): demo 
            for demo in demo_files
        }
        for future in as_completed(futures):
            demo_path = futures[future]
            try:
                count = future.result()
                total_extracted += count
            except Exception as e:
                logging.error(f"Failed {demo_path}: {e}")
                
    logging.info(f"Batch completed: {total_extracted} total trajectory segments saved.")
    return total_extracted
