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


def _create_synthetic_parquet(tmp_path, match_id: str, player_id: int, is_aimbot: int):
    """Helper to write a valid synthetic Parquet ATW file."""
    df = pd.DataFrame({
        'yaw': [0.0] * 32,
        'pitch': [0.0] * 32,
        'angular_velocity': [0.0] * 32,
        'angular_accel': [0.0] * 32,
        'angular_jerk': [0.0] * 32,
        'trajectory_curvature': [0.0] * 32,
        'curvature_entropy': [0.0] * 32,
        'tremor_power_8_12hz': [0.0] * 32,
        'match_id': match_id,
        'steamid': player_id,
        'segment_id': 0,
        'is_aimbot': is_aimbot,
        'player_elo': 1500.0 if is_aimbot == 0 else 2400.0
    })
    fpath = tmp_path / f"{match_id}_p{player_id}_seg0.parquet"
    df.to_parquet(str(fpath), index=False)
    return fpath


def test_partition_dataset_files_two_and_two_infeasible_raises_value_error(tmp_path):
    """
    Verify that exactly 2 clean and 2 cheater components (C=2, X=2, total=4) raises a
    descriptive ValueError explaining that populating Train and Test with both classes
    leaves Validation empty under strict zero-leakage constraints.
    """
    from data.dataset import partition_dataset_files
    
    # 2 clean matches, 2 cheater matches (4 disjoint clusters)
    _create_synthetic_parquet(tmp_path, "match_c0", 101, is_aimbot=0)
    _create_synthetic_parquet(tmp_path, "match_c1", 102, is_aimbot=0)
    _create_synthetic_parquet(tmp_path, "match_x0", 201, is_aimbot=1)
    _create_synthetic_parquet(tmp_path, "match_x1", 202, is_aimbot=1)
    
    with pytest.raises(ValueError) as exc_info:
        partition_dataset_files(str(tmp_path), train_ratio=0.80, val_ratio=0.10, test_ratio=0.10, seed=42)
        
    err_msg = str(exc_info.value)
    assert "2 clean" in err_msg or "clean=2" in err_msg
    assert "2 cheater" in err_msg or "cheater=2" in err_msg
    assert "Validation empty" in err_msg


def test_partition_dataset_files_three_clean_two_cheaters_succeeds(tmp_path):
    """
    Verify that 3 clean and 2 cheater components (C=3, X=2, total=5) succeeds without
    leaving Validation empty: Train has both classes, Test has both classes, and
    Validation has clean samples.
    """
    from data.dataset import partition_dataset_files
    
    # 3 clean matches, 2 cheater matches (5 disjoint clusters)
    _create_synthetic_parquet(tmp_path, "match_c0", 101, is_aimbot=0)
    _create_synthetic_parquet(tmp_path, "match_c1", 102, is_aimbot=0)
    _create_synthetic_parquet(tmp_path, "match_c2", 103, is_aimbot=0)
    _create_synthetic_parquet(tmp_path, "match_x0", 201, is_aimbot=1)
    _create_synthetic_parquet(tmp_path, "match_x1", 202, is_aimbot=1)
    
    train_files, val_files, test_files = partition_dataset_files(
        str(tmp_path), train_ratio=0.80, val_ratio=0.10, test_ratio=0.10, seed=42
    )
    
    # All splits must be non-empty
    assert len(train_files) > 0, "Train split is empty!"
    assert len(val_files) > 0, "Validation split is empty!"
    assert len(test_files) > 0, "Test split is empty!"
    
    # Train must have both classes
    train_labels = [pd.read_parquet(f)['is_aimbot'].iloc[0] for f in train_files]
    assert 0 in train_labels and 1 in train_labels, "Train missing one class!"
    
    # Test must have both classes
    test_labels = [pd.read_parquet(f)['is_aimbot'].iloc[0] for f in test_files]
    assert 0 in test_labels and 1 in test_labels, "Test missing one class!"
    
    # Validation must have clean samples (non-empty)
    val_labels = [pd.read_parquet(f)['is_aimbot'].iloc[0] for f in val_files]
    assert 0 in val_labels, "Validation missing clean samples!"
    
    # Strict zero-leakage checks
    def get_meta(files):
        ms, ps = set(), set()
        for f in files:
            df = pd.read_parquet(f)
            ms.update(df['match_id'].unique())
            ps.update(df['steamid'].unique())
        return ms, ps
        
    tr_m, tr_p = get_meta(train_files)
    va_m, va_p = get_meta(val_files)
    te_m, te_p = get_meta(test_files)
    
    assert tr_m.isdisjoint(va_m) and tr_m.isdisjoint(te_m) and va_m.isdisjoint(te_m)
    assert tr_p.isdisjoint(va_p) and tr_p.isdisjoint(te_p) and va_p.isdisjoint(te_p)


def test_partition_dataset_files_two_clean_three_cheaters_succeeds(tmp_path):
    """
    Verify that 2 clean and 3 cheater components (C=2, X=3, total=5) succeeds:
    Train and Test have both classes, and Validation has cheater samples.
    """
    from data.dataset import partition_dataset_files
    
    # 2 clean matches, 3 cheater matches (5 disjoint clusters)
    _create_synthetic_parquet(tmp_path, "match_c0", 101, is_aimbot=0)
    _create_synthetic_parquet(tmp_path, "match_c1", 102, is_aimbot=0)
    _create_synthetic_parquet(tmp_path, "match_x0", 201, is_aimbot=1)
    _create_synthetic_parquet(tmp_path, "match_x1", 202, is_aimbot=1)
    _create_synthetic_parquet(tmp_path, "match_x2", 203, is_aimbot=1)
    
    train_files, val_files, test_files = partition_dataset_files(
        str(tmp_path), train_ratio=0.80, val_ratio=0.10, test_ratio=0.10, seed=42
    )
    
    assert len(train_files) > 0 and len(val_files) > 0 and len(test_files) > 0
    train_labels = [pd.read_parquet(f)['is_aimbot'].iloc[0] for f in train_files]
    test_labels = [pd.read_parquet(f)['is_aimbot'].iloc[0] for f in test_files]
    val_labels = [pd.read_parquet(f)['is_aimbot'].iloc[0] for f in val_files]
    assert 0 in train_labels and 1 in train_labels
    assert 0 in test_labels and 1 in test_labels
    assert 1 in val_labels and 0 not in val_labels, "When C=2, X=3, validation must contain cheater samples only"


def test_partition_dataset_files_three_clean_three_cheaters_both_in_val(tmp_path):
    """
    Verify that 3 clean and 3 cheater components (C=3, X=3, total=6) guarantees
    that Train, Test, AND Validation each contain both clean and cheater samples.
    """
    from data.dataset import partition_dataset_files
    
    for i in range(3):
        _create_synthetic_parquet(tmp_path, f"match_c{i}", 100 + i, is_aimbot=0)
        _create_synthetic_parquet(tmp_path, f"match_x{i}", 200 + i, is_aimbot=1)
        
    train_files, val_files, test_files = partition_dataset_files(
        str(tmp_path), train_ratio=0.34, val_ratio=0.33, test_ratio=0.33, seed=42
    )
    
    for split_name, f_list in [("Train", train_files), ("Val", val_files), ("Test", test_files)]:
        assert len(f_list) > 0, f"{split_name} split is empty!"
        labels = [pd.read_parquet(f)['is_aimbot'].iloc[0] for f in f_list]
        assert 0 in labels, f"{split_name} missing clean class!"
        assert 1 in labels, f"{split_name} missing cheater class!"


def test_partition_dataset_files_insufficient_single_class_raises_value_error(tmp_path):
    """Verify that fewer than 3 single-class components raises a descriptive ValueError."""
    from data.dataset import partition_dataset_files
    
    _create_synthetic_parquet(tmp_path, "match_c0", 101, is_aimbot=0)
    _create_synthetic_parquet(tmp_path, "match_c1", 102, is_aimbot=0)
    
    with pytest.raises(ValueError) as exc_info:
        partition_dataset_files(str(tmp_path), seed=42)
        
    assert "at least 3 connected components" in str(exc_info.value)
    assert "clean=2" in str(exc_info.value)
    assert "cheater=0" in str(exc_info.value)


def test_partition_dataset_files_one_clean_two_cheaters_raises_value_error(tmp_path):
    """Verify that C=1, X=2 (total 3) raises a descriptive ValueError."""
    from data.dataset import partition_dataset_files
    
    _create_synthetic_parquet(tmp_path, "match_c0", 101, is_aimbot=0)
    _create_synthetic_parquet(tmp_path, "match_x0", 201, is_aimbot=1)
    _create_synthetic_parquet(tmp_path, "match_x1", 202, is_aimbot=1)
    
    with pytest.raises(ValueError) as exc_info:
        partition_dataset_files(str(tmp_path), seed=42)
        
    assert "clean=1" in str(exc_info.value)
    assert "cheater=2" in str(exc_info.value)


# ==============================================================================
# Threshold Calibration Unit Tests (Clean-Only, Cheater-Only, Dual-Class, Empty)
# ==============================================================================

def test_calibrate_operating_threshold_clean_only_validation():
    """
    Verify threshold calibration on clean-only validation data (e.g. C>=3, X=2):
    Calibrates to achieve zero false positives on clean samples, while explicitly
    marking validation TPR as unmeasured.
    """
    from evaluate import calibrate_operating_threshold
    
    y_val = np.array([0] * 50)
    # Predicted probabilities for clean duels: range [0.01, 0.42]
    np.random.seed(42)
    y_val_prob = np.random.uniform(0.01, 0.42, size=50)
    
    tau, info = calibrate_operating_threshold(y_val, y_val_prob, target_fpr=0.0001, return_info=True)
    
    assert info['is_calibrated'] is True
    assert info['calibration_mode'] == 'clean_only_zero_fp'
    assert info['clean_count'] == 50
    assert info['cheater_count'] == 0
    assert info['empirical_fpr'] == 0.0
    assert info['empirical_tpr'] is None  # TPR cannot be evaluated without validation cheaters
    assert tau >= float(np.max(y_val_prob))  # Threshold set to achieve 0 false alarms
    assert info['nominal_fpr_95_upper'] == pytest.approx(3.0 / 50, rel=1e-3)


def test_calibrate_operating_threshold_cheater_only_validation():
    """
    Verify threshold calibration on cheater-only validation data (e.g. C=2, X>=3):
    Because 0 clean validation samples exist, FPR calibration is unavailable.
    Must return default fallback threshold (0.5000) and explicitly set is_calibrated=False.
    """
    from evaluate import calibrate_operating_threshold
    
    y_val = np.array([1] * 20)
    y_val_prob = np.random.uniform(0.60, 0.99, size=20)
    
    tau, info = calibrate_operating_threshold(y_val, y_val_prob, target_fpr=0.0001, return_info=True)
    
    assert info['is_calibrated'] is False
    assert info['calibration_mode'] == 'unavailable'
    assert info['clean_count'] == 0
    assert info['cheater_count'] == 20
    assert info['empirical_fpr'] is None
    assert tau == 0.5  # Uncalibrated fallback
    assert "0 clean samples" in info['status_message']


def test_calibrate_operating_threshold_dual_class_validation():
    """
    Verify threshold calibration on dual-class validation data (e.g. C>=3, X>=3):
    Calibrates via empirical ROC curve, achieving validation FPR <= target_fpr and
    measuring validation TPR.
    """
    from evaluate import calibrate_operating_threshold
    
    # 100 clean samples (low scores) and 20 cheater samples (high scores)
    np.random.seed(42)
    y_clean = np.zeros(100, dtype=int)
    y_clean_prob = np.random.uniform(0.01, 0.25, size=100)
    
    y_cheat = np.ones(20, dtype=int)
    y_cheat_prob = np.random.uniform(0.70, 0.99, size=20)
    
    y_val = np.concatenate([y_clean, y_cheat])
    y_val_prob = np.concatenate([y_clean_prob, y_cheat_prob])
    
    tau, info = calibrate_operating_threshold(y_val, y_val_prob, target_fpr=0.0001, return_info=True)
    
    assert info['is_calibrated'] is True
    assert info['calibration_mode'] == 'dual_class_roc'
    assert info['clean_count'] == 100
    assert info['cheater_count'] == 20
    assert info['empirical_fpr'] <= 0.0001
    assert info['empirical_tpr'] is not None
    assert info['empirical_tpr'] > 0.0  # High TPR achieved because classes are separable
    assert 0.0 <= tau <= 1.0


def test_calibrate_operating_threshold_empty_validation():
    """Verify empty validation data returns is_calibrated=False and fallback 0.5000."""
    from evaluate import calibrate_operating_threshold
    
    tau, info = calibrate_operating_threshold(np.array([]), np.array([]), return_info=True)
    assert info['is_calibrated'] is False
    assert tau == 0.5
    assert info['calibration_mode'] == 'unavailable'


def test_calibrate_operating_threshold_tied_scores():
    """
    Verify threshold calibration with tied clean scores:
    When 10 samples are tied at 0.80 and allowed FP = 5 (target_fpr=0.05, N=100),
    setting tau=0.80 yields 10 false positives (FPR=0.10 > 0.05) due to score >= tau.
    The tie-aware calibrator must reject 0.80 and select a threshold strictly greater
    than 0.80, guaranteeing empirical validation FPR <= target_fpr.
    """
    from evaluate import calibrate_operating_threshold

    y_val = np.array([0] * 100)
    # 90 samples at 0.10, 10 samples tied at 0.80
    y_val_prob = np.array([0.10] * 90 + [0.80] * 10)

    tau, info = calibrate_operating_threshold(y_val, y_val_prob, target_fpr=0.05, return_info=True)

    assert info['is_calibrated'] is True
    assert info['clean_sample_count'] == 100
    assert info['observed_fp_count'] <= 5  # Must not exceed allowed 5 false positives
    assert info['empirical_fpr'] <= 0.05
    assert info['empirical_tpr'] is None  # TPR unavailable on clean-only split
    assert tau > 0.80  # Must reject 0.80 because of ties
    assert info['observed_fp_count'] == 0
    assert info['nominal_fpr_95_upper'] == pytest.approx(3.0 / 100, rel=1e-3)
    assert "does NOT certify" in info['statistical_evidence']


def test_calibrate_operating_threshold_equal_to_max_score():
    """
    Verify threshold calibration when threshold equals the maximum score:
    When 3 samples are tied at 0.80 and allowed FP = 5 (target_fpr=0.05, N=100),
    setting tau=0.80 yields 3 false positives (FPR=0.03 <= 0.05).
    The minimal valid threshold selected is exactly equal to max(clean_probs).
    """
    from evaluate import calibrate_operating_threshold

    y_val = np.array([0] * 100)
    # 97 samples at 0.10, 3 samples at 0.80 (maximum score)
    y_val_prob = np.array([0.10] * 97 + [0.80] * 3)

    tau, info = calibrate_operating_threshold(y_val, y_val_prob, target_fpr=0.05, return_info=True)

    assert info['is_calibrated'] is True
    assert tau == pytest.approx(0.80)
    assert tau == float(np.max(y_val_prob))
    assert info['observed_fp_count'] == 3
    assert info['clean_sample_count'] == 100
    assert info['empirical_fpr'] == pytest.approx(0.03)
    assert info['empirical_tpr'] is None
    assert info['nominal_fpr_95_upper'] is not None
    assert info['nominal_fpr_95_upper'] > 0.03
    assert "does NOT certify" in info['statistical_evidence']


def test_calibrate_operating_threshold_scores_at_one():
    """
    Verify threshold calibration when clean scores are at 1.0:
    When clean samples have scores at 1.0, any valid threshold tau in [0.0, 1.0]
    classifies them as positive (score >= tau is true).
    If the count of 1.0 scores exceeds allowed false positives, calibration
    must be marked unmet (is_calibrated=False, calibration_mode='unmet')
    and return fallback 0.5000.
    """
    from evaluate import calibrate_operating_threshold

    y_val = np.array([0] * 50)
    y_val_prob = np.array([1.0] * 50)

    tau, info = calibrate_operating_threshold(y_val, y_val_prob, target_fpr=0.0001, return_info=True)

    assert info['is_calibrated'] is False
    assert info['calibration_mode'] == 'unmet'
    assert tau == 0.5  # Uncalibrated fallback
    assert info['observed_fp_count'] == 50
    assert info['clean_sample_count'] == 50
    assert info['empirical_fpr'] == 1.0
    assert info['empirical_tpr'] is None
    assert "could not be met" in info['status_message']


def test_calibrate_operating_threshold_target_cannot_be_met():
    """
    Verify threshold calibration when empirical target cannot be met:
    Target allows at most 1 false positive (N=100, target_fpr=0.01), but
    5 clean samples have score 1.0. Minimum achievable FPR is 0.05 > 0.01.
    Must mark calibration unmet and return fallback 0.5000.
    """
    from evaluate import calibrate_operating_threshold

    y_val = np.array([0] * 100)
    # 95 samples at 0.10, 5 samples at 1.0
    y_val_prob = np.array([0.10] * 95 + [1.0] * 5)

    tau, info = calibrate_operating_threshold(y_val, y_val_prob, target_fpr=0.01, return_info=True)

    assert info['is_calibrated'] is False
    assert info['calibration_mode'] == 'unmet'
    assert tau == 0.5
    assert info['observed_fp_count'] == 5
    assert info['clean_sample_count'] == 100
    assert info['empirical_fpr'] == 0.05
    assert "could not be met" in info['status_message']



