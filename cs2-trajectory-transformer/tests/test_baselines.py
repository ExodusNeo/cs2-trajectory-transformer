"""
Unit Tests for Baseline Models and Downloader Modules.
"""

import sys
import os
import torch
import numpy as np
import pandas as pd
import pytest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from models.baselines import BiLSTMBaseline, ClassicalBaselines


def test_bilstm_baseline_forward():
    """Test Bi-LSTM baseline forward pass with masking."""
    batch_size = 4
    seq_len = 64
    feat_dim = 8
    
    model = BiLSTMBaseline(feature_dim=feat_dim, hidden_dim=32, num_layers=1)
    x = torch.randn(batch_size, seq_len, feat_dim)
    mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
    
    out = model(x, attention_mask=mask)
    assert out.shape == (batch_size, 1)
    assert (out >= 0.0).all() and (out <= 1.0).all()


def test_classical_baselines_fit_predict():
    """Test Random Forest, Gradient Boosting, and MLP baselines."""
    baselines = ClassicalBaselines()
    
    X_train = np.random.randn(50, 16)
    y_train = np.random.randint(0, 2, 50)
    X_test = np.random.randn(10, 16)
    
    baselines.fit_all(X_train, y_train)
    preds = baselines.predict_probabilities(X_test)
    
    assert 'Random Forest' in preds
    assert ('Gradient Boosting' in preds or 'XGBoost' in preds)
    assert 'MLP' in preds
    assert len(preds['Random Forest']) == 10


def test_cluster_bootstrap_and_design_effect():
    """Test cluster bootstrap and design effect statistical calculations in evaluate.py."""
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from evaluate import compute_cluster_bootstrap_bounds, compute_design_effect
    
    cluster_ids = ['m1', 'm1', 'm1', 'm2', 'm2', 'm3', 'm3']
    y_true = np.array([0, 0, 0, 0, 0, 0, 0])
    y_pred = np.array([0.01, 0.02, 0.01, 0.03, 0.02, 0.01, 0.02])
    
    # Zero false positives case (score-derived ICC)
    boot_upper = compute_cluster_bootstrap_bounds(cluster_ids, y_true, y_pred, threshold=0.5, n_bootstraps=100)
    assert 0.0 < boot_upper <= 1.0
    
    rho, deff, n_eff, adj_upper = compute_design_effect(cluster_ids, y_true, y_pred, threshold=0.5)
    assert 0.0 <= rho <= 1.0
    assert deff >= 1.0
    assert 0.0 < n_eff <= 7.0
    assert 0.0 < adj_upper <= 1.0
    
    # Case with clustered false positives
    y_pred_with_fp = np.array([0.8, 0.9, 0.85, 0.01, 0.02, 0.01, 0.02])
    rho2, deff2, n_eff2, adj_upper2 = compute_design_effect(cluster_ids, y_true, y_pred_with_fp, threshold=0.5)
    assert rho2 > 0.0
    assert deff2 > 1.0
    assert n_eff2 < 7.0


def test_audit_clean_atw_quota():
    """Test empirical stopping condition quota auditing function on test split and overall corpus."""
    from data.batch_processor import audit_clean_atw_quota
    
    # Audit on non-existent directory
    res_empty = audit_clean_atw_quota("non_existent_directory_xyz", target_clean=30000)
    assert res_empty['clean_count'] == 0
    assert res_empty['test_clean_count'] == 0
    assert res_empty['overall_clean_count'] == 0
    assert res_empty['quota_met'] is False
    assert res_empty['deficit'] == 30000
    
    # Audit on repository processed_parquet directory
    p_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'processed_parquet'))
    if os.path.exists(p_dir):
        # When target is 50, test split has 60 clean windows -> quota met
        res = audit_clean_atw_quota(p_dir, target_clean=50)
        assert res['is_test_split_audited'] is True
        assert res['overall_clean_count'] >= 300
        assert res['test_clean_count'] >= 50
        assert res['clean_count'] == res['test_clean_count']
        assert res['target_clean'] == 50
        assert res['quota_met'] is True
        assert res['progress_pct'] >= 100.0
        
        # When target is 30,000, test split has 60 clean windows -> quota not met
        res_30k = audit_clean_atw_quota(p_dir, target_clean=30000)
        assert res_30k['quota_met'] is False
        assert res_30k['deficit'] == 30000 - res_30k['test_clean_count']


def test_audit_clean_atw_quota_fail_closed(tmp_path):
    """Verify that audit_clean_atw_quota strictly fails closed when partitioning cannot be formed."""
    from data.batch_processor import audit_clean_atw_quota
    
    # Create an unpartitionable dataset: only 1 match (1 cluster), but 100 clean trajectory segments
    for seg in range(10):
        df = pd.DataFrame({
            'yaw': [0.0]*32, 'pitch': [0.0]*32, 'angular_velocity': [0.0]*32,
            'angular_accel': [0.0]*32, 'angular_jerk': [0.0]*32, 'trajectory_curvature': [0.0]*32,
            'curvature_entropy': [0.0]*32, 'tremor_power_8_12hz': [0.0]*32,
            'match_id': "match_solo", 'steamid': 1001, 'segment_id': seg, 'is_aimbot': 0, 'player_elo': 1500.0
        })
        df.to_parquet(str(tmp_path / f"match_solo_p1001_seg{seg}.parquet"), index=False)
        
    # With target_clean=5: overall corpus has 10 clean ATWs, but test partition cannot be formed (< 3 matches)
    res_partitioned = audit_clean_atw_quota(str(tmp_path), target_clean=5, partition_test_split=True)
    assert res_partitioned['overall_clean_count'] == 10
    assert res_partitioned['is_test_split_audited'] is False
    assert res_partitioned['test_clean_count'] == 0
    assert res_partitioned['clean_count'] == 0
    # Must fail closed: quota_met MUST be False even though overall_clean >= target
    assert res_partitioned['quota_met'] is False
    assert res_partitioned['deficit'] == 5
    assert res_partitioned['progress_pct'] == 0.0
    assert res_partitioned['partition_error'] is not None

    # When unpartitioned corpus audit is explicitly requested (partition_test_split=False):
    res_overall = audit_clean_atw_quota(str(tmp_path), target_clean=5, partition_test_split=False)
    assert res_overall['overall_clean_count'] == 10
    assert res_overall['is_test_split_audited'] is False
    assert res_overall['clean_count'] == 10
    assert res_overall['quota_met'] is True


def test_session_player_and_match_clustering():
    """Test session clustering logic across both match and player clustering axes."""
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from evaluate import compute_cluster_bootstrap_bounds, compute_design_effect
    
    # Simulate session-level data with 5 matches and 4 unique players across multiple matches
    session_match_ids = ['m1', 'm1', 'm2', 'm2', 'm3', 'm4', 'm5']
    session_player_ids = ['p1', 'p2', 'p1', 'p3', 'p2', 'p4', 'p1']
    s_y_true = np.array([0, 0, 0, 0, 0, 0, 0])
    s_y_pred = np.array([0.02, 0.01, 0.03, 0.01, 0.02, 0.01, 0.02])
    
    # Match axis
    m_boot = compute_cluster_bootstrap_bounds(session_match_ids, s_y_true, s_y_pred, threshold=0.5, n_bootstraps=50)
    m_rho, m_deff, m_neff, m_upper = compute_design_effect(session_match_ids, s_y_true, s_y_pred, threshold=0.5)
    assert 0.0 < m_boot <= 1.0
    assert 0.0 < m_neff <= 7.0
    
    # Player axis
    p_boot = compute_cluster_bootstrap_bounds(session_player_ids, s_y_true, s_y_pred, threshold=0.5, n_bootstraps=50)
    p_rho, p_deff, p_neff, p_upper = compute_design_effect(session_player_ids, s_y_true, s_y_pred, threshold=0.5)
    assert 0.0 < p_boot <= 1.0
    assert 0.0 < p_neff <= 7.0
    
    # Conservative combined bounds
    cons_neff = min(m_neff, p_neff)
    cons_upper = max(m_upper, p_upper)
    assert cons_neff <= max(m_neff, p_neff)
    assert cons_upper >= min(m_upper, p_upper)

