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
    """Generates a temporary directory with synthetic parquet trajectory segments."""
    temp_dir = tempfile.mkdtemp()
    
    # Create files for 4 distinct players across 2 matches
    players = [76561198000000001, 76561198000000002, 76561198000000003, 76561198000000004]
    
    for match_idx in range(2):
        match_id = f"match{match_idx}"
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
    
    assert len(ds) == 16  # 2 matches * 4 players * 2 segments
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
