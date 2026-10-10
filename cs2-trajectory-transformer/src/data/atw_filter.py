"""
Active Tracking Window (ATW) Extractor Module.
Filters out passive navigation noise and isolates high-signal combat engagement windows:
1. Spatial FOV Cone: Living enemy within a 30-degree visual cone and 3500 units (no occlusion check).
2. Temporal Event Buffer: Pre/post window around weapon fire and damage events (+/- 64 ticks).
3. Window Merging & Duration Pruning: Merges adjacent triggers and drops fragments shorter than
   L_min = 64 ticks (1.0 s), the length of one tremor FFT window.
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Optional

from features.kinematics import view_target_angles_rad


def calculate_relative_angle(
    player_pos: np.ndarray, 
    pitch_deg: float, 
    yaw_deg: float, 
    target_pos: np.ndarray
) -> float:
    """
    Angular deviation (degrees) between the player's look vector and a target position.

    Thesis Reference: Chapter 3, Equation (3). CS2 convention: yaw 0 deg = +X, 90 deg = +Y;
    positive pitch looks down (+Z is up in world space).
    """
    player_pos = np.asarray(player_pos, dtype=np.float64)
    target_pos = np.asarray(target_pos, dtype=np.float64)
    if np.linalg.norm(target_pos - player_pos) < 1e-6:
        return 0.0
    angle = view_target_angles_rad(
        player_pos[None, :], np.array([pitch_deg]), np.array([yaw_deg]), target_pos[None, :]
    )
    return float(np.degrees(angle[0]))


def find_fov_encounters(
    player_df: pd.DataFrame, 
    enemy_df: pd.DataFrame, 
    fov_threshold_deg: float = 30.0, 
    max_distance: float = 3500.0
) -> List[Tuple[int, int]]:
    """
    Finds contiguous tick ranges where one enemy is inside the player's FOV cone and range.

    Vectorized over ticks. Dead enemies are ignored when an 'is_alive' column is present.
    Returns inclusive (start_tick, end_tick) pairs.
    """
    if 'is_alive' in enemy_df.columns:
        enemy_df = enemy_df[enemy_df['is_alive'].astype(bool)]
    merged = pd.merge(
        player_df[['tick', 'X', 'Y', 'Z', 'pitch', 'yaw']],
        enemy_df[['tick', 'X', 'Y', 'Z']],
        on='tick',
        suffixes=('_p', '_e')
    ).sort_values('tick')
    
    if merged.empty:
        return []
        
    p_pos = merged[['X_p', 'Y_p', 'Z_p']].to_numpy(dtype=np.float64)
    e_pos = merged[['X_e', 'Y_e', 'Z_e']].to_numpy(dtype=np.float64)
    ticks = merged['tick'].to_numpy()
    
    dist = np.linalg.norm(e_pos - p_pos, axis=-1)
    angles_deg = np.degrees(view_target_angles_rad(
        p_pos, merged['pitch'].to_numpy(), merged['yaw'].to_numpy(), e_pos
    ))
    in_fov = (dist > 1.0) & (dist <= max_distance) & (angles_deg <= fov_threshold_deg)
    if not in_fov.any():
        return []

    # Run boundaries: a run breaks when the mask flips or the tick sequence has a gap.
    idx = np.flatnonzero(in_fov)
    breaks = np.flatnonzero((np.diff(idx) != 1) | (np.diff(ticks[idx]) != 1)) + 1
    runs = np.split(idx, breaks)
    return [(int(ticks[r[0]]), int(ticks[r[-1]])) for r in runs]


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
    min_window_len: int = 64,
    max_window_len: int = 512
) -> List[pd.DataFrame]:
    """
    High-level extractor: combines FOV encounters and combat event buffers to slice
    player telemetry into Active Tracking Window (ATW) DataFrame segments.
    Enforces minimum length L_min (64 ticks = 1.0 s, one full FFT window) and cap L_max (512 ticks).
    """
    raw_windows = []
    
    # 1. FOV encounters if enemy data is provided
    if enemy_df is not None and not enemy_df.empty:
        if 'steamid' in enemy_df.columns:
            for _, e_group in enemy_df.groupby('steamid'):
                fov_wins = find_fov_encounters(player_df, e_group, fov_threshold_deg=fov_deg)
                raw_windows.extend(fov_wins)
        else:
            fov_wins = find_fov_encounters(player_df, enemy_df, fov_threshold_deg=fov_deg)
            raw_windows.extend(fov_wins)
        
    # 2. Combat event buffers (weapon fire / damage)
    if event_ticks:
        evt_wins = find_combat_event_windows(event_ticks, tick_buffer=tick_buffer)
        raw_windows.extend(evt_wins)
        
    # 3. Merge and prune
    merged_windows = merge_overlapping_windows(raw_windows, min_duration=min_window_len)
    
    # 4. Extract DataFrame slices and enforce L_max duration capping (Sec 3.2.3, Table 4)
    atw_slices = []
    for start_t, end_t in merged_windows:
        slice_df = player_df[(player_df['tick'] >= start_t) & (player_df['tick'] <= end_t)].copy()
        if len(slice_df) < min_window_len:
            continue
        slice_df.reset_index(drop=True, inplace=True)
        
        if len(slice_df) <= max_window_len:
            atw_slices.append(slice_df)
        else:
            # Chunk long tracking duels into max_window_len sub-windows (with 50% stride)
            stride = max_window_len // 2
            for c_start in range(0, len(slice_df), stride):
                chunk = slice_df.iloc[c_start:c_start + max_window_len].copy().reset_index(drop=True)
                if len(chunk) >= min_window_len:
                    atw_slices.append(chunk)
                if c_start + max_window_len >= len(slice_df):
                    break
            
    return atw_slices
