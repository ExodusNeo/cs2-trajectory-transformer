"""
Unit Tests for Baseline Models and Downloader Modules.
"""

import sys
import os
import torch
import numpy as np
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
    assert 'Gradient Boosting' in preds
    assert 'MLP' in preds
    assert len(preds['Random Forest']) == 10
