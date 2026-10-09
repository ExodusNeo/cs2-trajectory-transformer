"""
Unit Tests for CS2 Trajectory Dataset and DataLoader Pipeline.
Verifies zero data leakage partitioning, attention mask creation, and tensor shapes.
"""

import sys
import os
import shutil
import tempfile
import numpy as np
import pandas as pd
import torch
import pytest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from data.dataset import (
    CS2TrajectoryDataset,
    collate_trajectory_batch,
    create_partitioned_dataloaders,
    FEATURE_COLUMNS
)


@pytest.fixture
def sample_parquet_dir():
    """Generates a temporary directory with synthetic parquet trajectory segments across 3 matches."""
    temp_dir = tempfile.mkdtemp()
    
    # Create 3 independent matches with distinct player rosters
    players_by_match = {
        "match0": [76561198000000001, 76561198000000002],
        "match1": [76561198000000003, 76561198000000004],
        "match2": [76561198000000005, 76561198000000006]
    }
    
    for match_id, players in players_by_match.items():
        for p in players:
            for seg in range(2):
                n_ticks = np.random.randint(64, 128)
                data = {
                    'yaw': np.random.uniform(0, 360, n_ticks),
                    'pitch': np.random.uniform(-45, 45, n_ticks),
                    'angular_velocity': np.random.uniform(0, 10, n_ticks),
                    'angular_accel': np.random.uniform(-50, 50, n_ticks),
                    'angular_jerk': np.random.uniform(-500, 500, n_ticks),
                    'trajectory_curvature': np.random.uniform(0, 10, n_ticks),
                    'curvature_entropy': np.random.uniform(1.0, 3.0, n_ticks),
                    'tremor_power_8_12hz': np.random.uniform(0.1, 0.8, n_ticks),
                    'match_id': match_id,
                    'steamid': p,
                    'segment_id': seg,
                    'is_aimbot': int(p % 2 == 0),
                    'player_elo': 1800.0
                }
                df = pd.DataFrame(data)
                fpath = os.path.join(temp_dir, f"{match_id}_p{p}_seg{seg}.parquet")
                df.to_parquet(fpath, index=False)
                
    yield temp_dir
    shutil.rmtree(temp_dir)


def test_dataset_item_loading(sample_parquet_dir):
    """Verify individual sample loading and standardization."""
    import glob
    files = glob.glob(os.path.join(sample_parquet_dir, "*.parquet"))
    ds = CS2TrajectoryDataset(files)
    
    assert len(ds) == 12  # 3 matches * 2 players * 2 segments

    sample = ds[0]
    
    assert 'features' in sample
    assert 'seq_len' in sample
    assert 'aimbot_label' in sample
    assert 'elo_label' in sample
    assert sample['features'].shape[-1] == len(FEATURE_COLUMNS)


def test_batch_collate_and_masks(sample_parquet_dir):
    """Verify variable-length batch padding and attention mask generation."""
    import glob
    files = glob.glob(os.path.join(sample_parquet_dir, "*.parquet"))
    ds = CS2TrajectoryDataset(files)
    
    batch_samples = [ds[0], ds[1], ds[2], ds[3]]
    batch = collate_trajectory_batch(batch_samples)
    
    features = batch['features']
    masks = batch['attention_mask']
    
    assert features.ndim == 3  # [batch, max_len, feature_dim]
    assert masks.shape == features.shape[:2]  # [batch, max_len]
    assert masks.dtype == torch.bool
    
    # Check that mask matches sequence lengths
    for i, s in enumerate(batch_samples):
        slen = s['seq_len'].item()
        assert masks[i, :slen].all()  # True for valid ticks
        if slen < features.shape[1]:
            assert (~masks[i, slen:]).all()  # False for padded ticks


def test_zero_data_leakage_splits(sample_parquet_dir):
    """Verify that no player ID appears in more than one partition (Train/Val/Test)."""
    train_loader, val_loader, test_loader = create_partitioned_dataloaders(
        data_dir=sample_parquet_dir,
        train_ratio=0.50,
        val_ratio=0.25,
        test_ratio=0.25,
        batch_size=2,
        seed=123
    )
    
    train_players = set()
    for batch in train_loader:
        train_players.update(batch['player_ids'].tolist())
        
    val_players = set()
    for batch in val_loader:
        val_players.update(batch['player_ids'].tolist())
        
    test_players = set()
    for batch in test_loader:
        test_players.update(batch['player_ids'].tolist())
        
    # Check pairwise disjointness (Zero Data Leakage)
    assert train_players.isdisjoint(val_players), "Data leakage between Train and Val!"
    assert train_players.isdisjoint(test_players), "Data leakage between Train and Test!"
    assert val_players.isdisjoint(test_players), "Data leakage between Val and Test!"


def test_zero_data_leakage_matches_and_players_clusters():
    """Verify that both Match IDs and Player IDs are disjoint when independent matches exist."""
    temp_dir = tempfile.mkdtemp()
    try:
        # 3 independent matches with disjoint players
        match_configs = [
            ("matchA", [101, 102]),
            ("matchB", [201, 202]),
            ("matchC", [301, 302])
        ]
        for m_id, players in match_configs:
            for p in players:
                df = pd.DataFrame({
                    'yaw': [0.0]*32, 'pitch': [0.0]*32, 'angular_velocity': [0.0]*32,
                    'angular_accel': [0.0]*32, 'angular_jerk': [0.0]*32, 'trajectory_curvature': [0.0]*32,
                    'curvature_entropy': [0.0]*32, 'tremor_power_8_12hz': [0.0]*32,
                    'match_id': m_id, 'steamid': p, 'segment_id': 0, 'is_aimbot': 0, 'player_elo': 1500.0
                })
                df.to_parquet(os.path.join(temp_dir, f"{m_id}_p{p}_seg0.parquet"), index=False)
                
        train_l, val_l, test_l = create_partitioned_dataloaders(
            data_dir=temp_dir, train_ratio=0.34, val_ratio=0.33, test_ratio=0.33, batch_size=2, seed=42
        )
        
        def get_matches_and_players(loader):
            matches, players = set(), set()
            for b in loader:
                players.update(b['player_ids'].tolist())
                matches.update(b['match_ids'])
            return matches, players
            
        tr_m, tr_p = get_matches_and_players(train_l)
        va_m, va_p = get_matches_and_players(val_l)
        te_m, te_p = get_matches_and_players(test_l)
        
        assert tr_p.isdisjoint(va_p)
        assert tr_p.isdisjoint(te_p)
        assert va_p.isdisjoint(te_p)
        
        assert tr_m.isdisjoint(va_m)
        assert tr_m.isdisjoint(te_m)
        assert va_m.isdisjoint(te_m)
    finally:
        shutil.rmtree(temp_dir)


def test_scaler_save_and_load(tmp_path):
    """Verify that global normalization statistics serialize and deserialize accurately."""
    from data.dataset import save_scaler_stats, load_scaler_stats
    scaler_file = str(tmp_path / "scaler.npz")
    mean = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], dtype=np.float32)
    std = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5], dtype=np.float32)
    
    save_scaler_stats(scaler_file, mean, std)
    loaded = load_scaler_stats(scaler_file)
    
    assert loaded is not None
    l_mean, l_std, l_cols = loaded
    np.testing.assert_allclose(l_mean, mean)
    np.testing.assert_allclose(l_std, std)
    assert l_cols == FEATURE_COLUMNS


def test_stratified_cluster_partitioning_minority_cheaters(tmp_path):
    """
    Verify that partition_dataset_files stratifies clean and cheater clusters even when
    the cheater minority class has fewer than 3 clusters (e.g. exactly 2 cheater clusters).
    Guarantees that test_files contains both classes, preventing degenerate evaluation metrics.
    """
    from data.dataset import partition_dataset_files, CS2TrajectoryDataset
    
    # Create 10 independent clean matches (each 1 player)
    for i in range(10):
        m_id = f"match_c{i}"
        p_id = 76561198000000100 + i
        df = pd.DataFrame({
            'yaw': [0.0]*32, 'pitch': [0.0]*32, 'angular_velocity': [0.0]*32,
            'angular_accel': [0.0]*32, 'angular_jerk': [0.0]*32, 'trajectory_curvature': [0.0]*32,
            'curvature_entropy': [0.0]*32, 'tremor_power_8_12hz': [0.0]*32,
            'match_id': m_id, 'steamid': p_id, 'segment_id': 0, 'is_aimbot': 0, 'player_elo': 1500.0
        })
        df.to_parquet(str(tmp_path / f"{m_id}_p{p_id}_seg0.parquet"), index=False)
        
    # Create exactly 2 independent cheater matches (< 3 clusters)
    for i in range(2):
        m_id = f"match_x{i}"
        p_id = 76561198000000900 + i
        df = pd.DataFrame({
            'yaw': [0.0]*32, 'pitch': [0.0]*32, 'angular_velocity': [0.0]*32,
            'angular_accel': [0.0]*32, 'angular_jerk': [0.0]*32, 'trajectory_curvature': [0.0]*32,
            'curvature_entropy': [0.0]*32, 'tremor_power_8_12hz': [0.0]*32,
            'match_id': m_id, 'steamid': p_id, 'segment_id': 0, 'is_aimbot': 1, 'player_elo': 2400.0
        })
        df.to_parquet(str(tmp_path / f"{m_id}_p{p_id}_seg0.parquet"), index=False)
        
    train_files, val_files, test_files = partition_dataset_files(
        str(tmp_path), train_ratio=0.80, val_ratio=0.10, test_ratio=0.10, seed=42
    )
    
    # Train files must contain both classes
    train_labels = [pd.read_parquet(f)['is_aimbot'].iloc[0] for f in train_files]
    assert 0 in train_labels, "Train split missing clean class!"
    assert 1 in train_labels, "Train split missing cheater class!"
    
    # Test files MUST contain both classes (even though cheater class only had 2 clusters)
    test_labels = [pd.read_parquet(f)['is_aimbot'].iloc[0] for f in test_files]
    assert 0 in test_labels, "Test split missing clean class!"
    assert 1 in test_labels, "Test split missing cheater class! Unstratified split dropped minority class."
    
    # Check disjoint zero-leakage guarantee across matches and players
    def extract_m_and_p(fpaths):
        ms, ps = set(), set()
        for f in fpaths:
            df = pd.read_parquet(f)
            ms.update(df['match_id'].unique())
            ps.update(df['steamid'].unique())
        return ms, ps
        
    tr_m, tr_p = extract_m_and_p(train_files)
    va_m, va_p = extract_m_and_p(val_files)
    te_m, te_p = extract_m_and_p(test_files)
    
    assert tr_m.isdisjoint(te_m), "Match leakage between train and test!"
    assert tr_p.isdisjoint(te_p), "Player leakage between train and test!"
    assert tr_m.isdisjoint(va_m), "Match leakage between train and val!"
    assert va_m.isdisjoint(te_m), "Match leakage between val and test!"
