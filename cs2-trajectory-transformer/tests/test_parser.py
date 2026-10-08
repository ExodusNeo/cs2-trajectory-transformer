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


def test_extract_active_tracking_windows_max_capping():
    """Test that engagements exceeding max_window_len (512 ticks) are properly chunked."""
    n_ticks = 1500
    df = pd.DataFrame({
        'tick': np.arange(n_ticks),
        'steamid': [76561198000000000] * n_ticks,
        'yaw': np.zeros(n_ticks),
        'pitch': np.zeros(n_ticks),
        'X': np.zeros(n_ticks),
        'Y': np.zeros(n_ticks),
        'Z': np.full(n_ticks, 64.0)
    })
    
    # Continuous events creating an extended ~900-tick window
    event_ticks = list(range(100, 900, 50))
    slices = extract_active_tracking_windows(
        player_df=df,
        event_ticks=event_ticks,
        tick_buffer=64,
        min_window_len=32,
        max_window_len=512
    )
    assert len(slices) >= 2, "Expected extended engagement to be split into chunks"
    for s in slices:
        assert len(s) <= 512, f"Expected slice <= 512 ticks, got {len(s)}"


def test_pseudonymization_with_salt():
    """Test deterministic cryptographic pseudonymization under Section 3.3.1 (RA 10173)."""
    from data.batch_processor import pseudonymize_steamid, pseudonymize_match_id
    
    steamid = 76561198012345678
    match_id = "test_match_uuid_12345"
    salt = "custom_test_salt_987"
    
    anon_id1 = pseudonymize_steamid(steamid, salt=salt)
    anon_id2 = pseudonymize_steamid(steamid, salt=salt)
    assert anon_id1 == anon_id2, "Pseudonymization must be deterministic with same salt"
    assert isinstance(anon_id1, int), "Pseudonymized SteamID must be an integer"
    assert anon_id1 > 0, "Pseudonymized SteamID must be positive"
    assert anon_id1 < (1 << 60), "Pseudonymized SteamID must fit in 60-bit integer range"
    
    anon_match1 = pseudonymize_match_id(match_id, salt=salt)
    anon_match2 = pseudonymize_match_id(match_id, salt=salt)
    assert anon_match1 == anon_match2, "Pseudonymized match must be deterministic"
    assert anon_match1.startswith("match_"), "Pseudonymized match must start with match_ prefix"


def test_pseudonymization_missing_salt_raises():
    """Test that missing managed secret salt raises ValueError under RA 10173 governance."""
    from data.batch_processor import pseudonymize_steamid, pseudonymize_match_id
    
    old_env = os.environ.pop("CS2_PSEUDONYMIZATION_SALT", None)
    try:
        with pytest.raises(ValueError, match="managed secret salt"):
            pseudonymize_steamid(76561198012345678, salt=None)
            
        with pytest.raises(ValueError, match="managed secret salt"):
            pseudonymize_match_id("match_123", salt=None)
    finally:
        if old_env is not None:
            os.environ["CS2_PSEUDONYMIZATION_SALT"] = old_env
