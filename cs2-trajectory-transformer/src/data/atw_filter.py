"""
Active Tracking Window (ATW) Extractor Module.
Filters out passive navigation noise and isolates high-signal combat engagement windows:
1. Spatial FOV Cone: Enemy player within a 30-degree visual cone and line of sight.
2. Temporal Event Buffer: Pre/post window around weapon fire and damage events (+/- 64 ticks).
3. Window Merging & Duration Pruning: Merges adjacent triggers and drops micro-fragments (< 32 ticks).
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Optional


def calculate_relative_angle(
    player_pos: np.ndarray, 
    pitch_deg: float, 
    yaw_deg: float, 
    target_pos: np.ndarray
) -> float:
    """
    Computes angular deviation (in degrees) between the player's 3D look vector and target position.
    
    CS2 Coordinate System:
    - Yaw: 0 deg = +X axis, 90 deg = +Y axis, 180/-180 = -X axis, -90 = -Y axis.
    - Pitch: -89 deg (looking straight up) to +89 deg (looking straight down) or standard spherical.
    """
    # Relative position vector from player to target
    rel_vec = target_pos - player_pos
    dist = np.linalg.norm(rel_vec)
    if dist < 1e-6:
        return 0.0
        
    rel_unit = rel_vec / dist
    
    # Player 3D sight unit vector from pitch and yaw
    pitch_rad = np.radians(pitch_deg)
    yaw_rad = np.radians(yaw_deg)
    
    # In CS2, forward direction:
    vx = np.cos(pitch_rad) * np.cos(yaw_rad)
    vy = np.cos(pitch_rad) * np.sin(yaw_rad)
    vz = -np.sin(pitch_rad)  # CS2 positive pitch looks down (+Z is up in world space)
    
    look_vec = np.array([vx, vy, vz], dtype=np.float64)
    look_norm = np.linalg.norm(look_vec)
    if look_norm > 1e-6:
        look_vec /= look_norm
        
    dot_product = np.clip(np.dot(look_vec, rel_unit), -1.0, 1.0)
    angle_rad = np.arccos(dot_product)
    return float(np.degrees(angle_rad))


def find_fov_encounters(
    player_df: pd.DataFrame, 
    enemy_df: pd.DataFrame, 
    fov_threshold_deg: float = 30.0, 
    max_distance: float = 3500.0
) -> List[Tuple[int, int]]:
    """
    Scans aligned player and enemy telemetry to find continuous tick ranges where
    the enemy is within the player's FOV cone and maximum engagement distance.
    """
    # Merge on tick
    merged = pd.merge(
        player_df[['tick', 'X', 'Y', 'Z', 'pitch', 'yaw']],
        enemy_df[['tick', 'X', 'Y', 'Z']],
        on='tick',
        suffixes=('_p', '_e')
    ).sort_values('tick')
    
    if merged.empty:
        return []
        
    p_pos = merged[['X_p', 'Y_p', 'Z_p']].values
    e_pos = merged[['X_e', 'Y_e', 'Z_e']].values
    pitches = merged['pitch'].values
    yaws = merged['yaw'].values
    ticks = merged['tick'].values
    
    in_fov_mask = np.zeros(len(merged), dtype=bool)
    
    for i in range(len(merged)):
        dist = np.linalg.norm(e_pos[i] - p_pos[i])
        if dist <= max_distance:
            angle = calculate_relative_angle(p_pos[i], pitches[i], yaws[i], e_pos[i])
            if angle <= fov_threshold_deg:
                in_fov_mask[i] = True
                
    # Extract continuous tick intervals where in_fov_mask is True
    windows = []
    in_window = False
    start_tick = None
    
    for i in range(len(in_fov_mask)):
        if in_fov_mask[i] and not in_window:
            in_window = True
            start_tick = ticks[i]
        elif not in_fov_mask[i] and in_window:
            in_window = False
            end_tick = ticks[i - 1]
            windows.append((int(start_tick), int(end_tick)))
            
    if in_window:
        windows.append((int(start_tick), int(ticks[-1])))
        
    return windows


def find_combat_event_windows(
    event_ticks: List[int], 
    tick_buffer: int = 64
) -> List[Tuple[int, int]]:
    """
    Generates [event_tick - tick_buffer, event_tick + tick_buffer] windows around weapon fires or hits.
    """
    windows = []
    for t in event_ticks:
        start_t = max(0, int(t - tick_buffer))
        end_t = int(t + tick_buffer)
        windows.append((start_t, end_t))
    return windows


def merge_overlapping_windows(
    windows: List[Tuple[int, int]], 
    min_duration: int = 32
) -> List[Tuple[int, int]]:
    """
    Merges overlapping or adjacent tick windows and prunes windows shorter than min_duration.
    """
    if not windows:
        return []
        
    # Sort windows by start tick
    sorted_windows = sorted(windows, key=lambda w: w[0])
    merged = []
    curr_start, curr_end = sorted_windows[0]
    
    for next_start, next_end in sorted_windows[1:]:
        if next_start <= curr_end:
            # Overlap or contiguous -> extend
            curr_end = max(curr_end, next_end)
        else:
            # Disjoint -> commit previous if long enough
            if (curr_end - curr_start + 1) >= min_duration:
                merged.append((curr_start, curr_end))
            curr_start, curr_end = next_start, next_end
            
    # Commit last window
    if (curr_end - curr_start + 1) >= min_duration:
        merged.append((curr_start, curr_end))
        
    return merged


def extract_active_tracking_windows(
    player_df: pd.DataFrame, 
    enemy_df: Optional[pd.DataFrame] = None, 
    event_ticks: Optional[List[int]] = None, 
    fov_deg: float = 30.0, 
    tick_buffer: int = 64, 
    min_window_len: int = 32
) -> List[pd.DataFrame]:
    """
    High-level extractor: combines FOV encounters and combat event buffers to slice
    player telemetry into Active Tracking Window (ATW) DataFrame segments.
    """
    raw_windows = []
    
    # 1. FOV encounters if enemy data is provided
    if enemy_df is not None and not enemy_df.empty:
        fov_wins = find_fov_encounters(player_df, enemy_df, fov_threshold_deg=fov_deg)
        raw_windows.extend(fov_wins)
        
    # 2. Combat event buffers (weapon fire / damage)
    if event_ticks:
        evt_wins = find_combat_event_windows(event_ticks, tick_buffer=tick_buffer)
        raw_windows.extend(evt_wins)
        
    # 3. Merge and prune
    merged_windows = merge_overlapping_windows(raw_windows, min_duration=min_window_len)
    
    # 4. Extract DataFrame slices
    atw_slices = []
    for start_t, end_t in merged_windows:
        slice_df = player_df[(player_df['tick'] >= start_t) & (player_df['tick'] <= end_t)].copy()
        if len(slice_df) >= min_window_len:
            slice_df.reset_index(drop=True, inplace=True)
            atw_slices.append(slice_df)
            
    return atw_slices
