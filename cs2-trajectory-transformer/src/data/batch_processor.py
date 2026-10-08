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
    banned_steamids: Optional[Iterable[Union[str, int]]] = None,
    player_elos: Optional[Dict[Union[str, int], float]] = None,
    default_elo: float = 1500.0,
    min_window_len: int = 32,
    tick_rate: float = 64.0,
    use_fov_filter: bool = True
) -> int:
    """
    Processes a single CS2 .dem replay:
    1. Parses player ticks and weapon events.
    2. Identifies all players and teams.
    3. Extracts Active Tracking Windows (combining 30-deg FOV cones & weapon fire buffers).
    4. Computes 8D kinematic & tremor features at native 64 Hz.
    5. Correctly labels ONLY verified banned cheaters (never false-labeling clean players).
    6. Saves trajectory segments to Parquet.
    
    Returns the count of extracted ATW trajectory segments.
    """
    match_name = os.path.splitext(os.path.basename(demo_path))[0]
    os.makedirs(output_dir, exist_ok=True)
    
    banned_set = {str(b) for b in banned_steamids} if banned_steamids else set()
    
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
                p_fires = fire_events[fire_events['user_steamid'].astype(str) == str(steamid)]
                event_ticks = p_fires['tick'].tolist() if 'tick' in p_fires.columns else []
            else:
                event_ticks = []
                
            # Extract enemy telemetry for 30-degree FOV cone encounters
            enemy_df = None
            if use_fov_filter and 'team_num' in p_df.columns:
                p_team = p_df['team_num'].iloc[0]
                enemy_df = ticks_df[ticks_df['team_num'] != p_team]
                
            # Extract Active Tracking Windows (ATW)
            atw_segments = extract_active_tracking_windows(
                player_df=p_df,
                enemy_df=enemy_df,
                event_ticks=event_ticks,
                fov_deg=30.0,
                tick_buffer=64,
                min_window_len=min_window_len
            )
            
            # Ground-truth cheater labeling:
            # Tag ONLY verified banned accounts. Never blindly label all 10 players in a match.
            if banned_set:
                is_player_cheater = int(str(steamid) in banned_set)
            elif is_cheater_demo:
                # If demo filename includes the specific cheater steamid
                is_player_cheater = int(str(steamid) in match_name)
            else:
                is_player_cheater = 0
                
            p_elo = default_elo
            if player_elos:
                p_elo = player_elos.get(str(steamid), player_elos.get(int(steamid) if str(steamid).isdigit() else 0, default_elo))
            
            for seg_idx, seg_df in enumerate(atw_segments):
                # Compute 8D biomechanical features at native 64 Hz
                featured_df = compute_kinematic_features(seg_df, tick_rate=tick_rate, extract_tremor=True)
                
                # Metadata tags
                featured_df['match_id'] = match_name
                featured_df['steamid'] = steamid
                featured_df['segment_id'] = seg_idx
                featured_df['is_aimbot'] = is_player_cheater
                featured_df['player_elo'] = float(p_elo)
                
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
    banned_steamids_map: Optional[Dict[str, List[Union[str, int]]]] = None,
    player_elos_map: Optional[Dict[str, Dict[Union[str, int], float]]] = None,
    tick_rate: float = 64.0,
    max_workers: int = 4
) -> int:
    """Processes an entire directory of .dem files in parallel."""
    demo_files = glob.glob(os.path.join(demo_dir, "*.dem"))
    logging.info(f"Found {len(demo_files)} demos in {demo_dir}")
    
    total_extracted = 0
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        for demo in demo_files:
            m_name = os.path.splitext(os.path.basename(demo))[0]
            b_ids = banned_steamids_map.get(m_name) if banned_steamids_map else None
            p_elos = player_elos_map.get(m_name) if player_elos_map else None
            fut = executor.submit(
                process_single_demo, demo, output_dir, is_cheater_dataset, b_ids, p_elos, 1500.0, 32, tick_rate, True
            )
            futures[fut] = demo
            
        for future in as_completed(futures):
            demo_path = futures[future]
            try:
                count = future.result()
                total_extracted += count
            except Exception as e:
                logging.error(f"Failed {demo_path}: {e}")
                
    logging.info(f"Batch completed: {total_extracted} total trajectory segments saved.")
    return total_extracted
