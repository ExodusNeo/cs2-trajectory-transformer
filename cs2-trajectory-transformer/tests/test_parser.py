"""
Unit Tests for CS2 Demo Parser & Active Tracking Window (ATW) Extractor.
Verifies 3D Vector FOV Angle Geometry, Event Window Merging, and Trajectory Slicing.
"""

import sys
import os
import numpy as np
import pandas as pd
import pytest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from data.atw_filter import (
    calculate_relative_angle,
    find_combat_event_windows,
    merge_overlapping_windows,
    extract_active_tracking_windows
)


def test_relative_fov_geometry():
    """Test relative angle calculation between look vector and target position."""
    player_pos = np.array([0.0, 0.0, 64.0])
    
    # 1. Target directly in front along +X axis (yaw = 0 deg, pitch = 0 deg)
    target_front = np.array([1000.0, 0.0, 64.0])
    angle_front = calculate_relative_angle(player_pos, pitch_deg=0.0, yaw_deg=0.0, target_pos=target_front)
    assert np.isclose(angle_front, 0.0, atol=1e-3), f"Expected 0 deg, got {angle_front}"
    
    # 2. Target 90 degrees to the right (+Y axis in CS2 with yaw = 0 looking +X)
    target_right = np.array([0.0, 1000.0, 64.0])
    angle_right = calculate_relative_angle(player_pos, pitch_deg=0.0, yaw_deg=0.0, target_pos=target_right)
    assert np.isclose(angle_right, 90.0, atol=1e-3), f"Expected 90 deg, got {angle_right}"
    
    # 3. Target behind (180 degrees)
    target_behind = np.array([-1000.0, 0.0, 64.0])
    angle_behind = calculate_relative_angle(player_pos, pitch_deg=0.0, yaw_deg=0.0, target_pos=target_behind)
    assert np.isclose(angle_behind, 180.0, atol=1e-3), f"Expected 180 deg, got {angle_behind}"
    
    # 4. Target within 30-degree FOV cone
    target_cone = np.array([1000.0, 200.0, 64.0])
    angle_cone = calculate_relative_angle(player_pos, pitch_deg=0.0, yaw_deg=0.0, target_pos=target_cone)
    assert angle_cone < 30.0, f"Expected < 30 deg, got {angle_cone}"


def test_window_merging_and_pruning():
    """Test merging overlapping intervals and dropping short fragments."""
    # Windows: [100, 200] and [150, 250] should merge to [100, 250] (len=151 >= 32)
    # Window: [300, 310] (len=11 < 32) should be dropped
    windows = [(100, 200), (150, 250), (300, 310)]
    merged = merge_overlapping_windows(windows, min_duration=32)
    
    assert len(merged) == 1, f"Expected 1 merged window, got {len(merged)}"
    assert merged[0] == (100, 250), f"Expected (100, 250), got {merged[0]}"


def test_extract_active_tracking_windows():
    """Test end-to-end ATW extraction from player DataFrame and combat shot events."""
    n_ticks = 1000
    df = pd.DataFrame({
        'tick': np.arange(n_ticks),
        'steamid': [76561198000000000] * n_ticks,
        'yaw': np.linspace(0, 360, n_ticks),
        'pitch': np.zeros(n_ticks),
        'X': np.linspace(0, 2000, n_ticks),
        'Y': np.zeros(n_ticks),
        'Z': np.full(n_ticks, 64.0)
    })
    
    # Weapon fire at tick 200 and tick 600
    event_ticks = [200, 600]
    
    slices = extract_active_tracking_windows(
        player_df=df, 
        event_ticks=event_ticks, 
        tick_buffer=64, 
        min_window_len=32
    )
    
    assert len(slices) == 2, f"Expected 2 ATW slices, got {len(slices)}"
    assert len(slices[0]) == 129, f"Expected 129 ticks (+/- 64 around 200), got {len(slices[0])}"
    assert slices[0]['tick'].min() == 200 - 64
    assert slices[0]['tick'].max() == 200 + 64
