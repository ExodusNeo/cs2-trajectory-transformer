"""
Unit Tests for ST-Trans Architecture, Dual Heads, InfoNCE, and Focal Loss.
"""

import sys
import os
import torch
import pytest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from models.st_transformer import STTrajectoryTransformer
from models.losses import SupervisedInfoNCELoss, FocalLoss


def test_st_transformer_forward_with_mask():
    """Verify forward pass output shapes and masking."""
    batch_size = 4
    seq_len = 128
    feature_dim = 8
    
    model = STTrajectoryTransformer(
        feature_dim=feature_dim, 
        d_model=64, 
        nhead=4, 
        num_layers=2, 
        embed_dim=32
    )
    
    x = torch.randn(batch_size, seq_len, feature_dim)
    mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
    # Mask last 32 ticks of last sample
    mask[3, 96:] = False
    
    aimbot_prob, smurf_emb, elo_pred = model(x, attention_mask=mask)
    
    assert aimbot_prob.shape == (batch_size, 1)
    assert (aimbot_prob >= 0.0).all() and (aimbot_prob <= 1.0).all()
    
    assert smurf_emb.shape == (batch_size, 32)
    # Check L2 unit normalization
    norms = torch.norm(smurf_emb, p=2, dim=-1)
    assert torch.isclose(norms, torch.ones_like(norms), atol=1e-4).all()
    
    assert elo_pred.shape == (batch_size, 1)


def test_infonce_contrastive_loss():
    """Verify Supervised InfoNCE pulls same-player pairs and pushes different players."""
    criterion = SupervisedInfoNCELoss(temperature=0.1)
    
    # 4 samples: player 1 has 2 samples (idx 0, 1), player 2 has 2 samples (idx 2, 3)
    labels = torch.tensor([1, 1, 2, 2], dtype=torch.long)
    
    # Perfect clustering: player 1 along x-axis, player 2 along y-axis
    emb_good = torch.tensor([
        [1.0, 0.0, 0.0, 0.0],
        [0.99, 0.01, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.99, 0.01, 0.0]
    ], dtype=torch.float32)
    
    # Poor clustering: all random
    emb_bad = torch.tensor([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0]
    ], dtype=torch.float32)
    
    loss_good = criterion(emb_good, labels)
    loss_bad = criterion(emb_bad, labels)
    
    assert loss_good.item() < loss_bad.item(), "InfoNCE should give much lower loss for well-separated player embeddings."


def test_focal_loss():
    """Verify Focal Loss down-weights easy confident predictions."""
    criterion = FocalLoss(gamma=2.0)
    
    # Confident correct predictions
    preds_easy = torch.tensor([[0.99], [0.01]], dtype=torch.float32)
    targets = torch.tensor([[1.0], [0.0]], dtype=torch.float32)
    
    # Uncertain predictions
    preds_hard = torch.tensor([[0.55], [0.45]], dtype=torch.float32)
    
    loss_easy = criterion(preds_easy, targets)
    loss_hard = criterion(preds_hard, targets)
    
    assert loss_easy.item() < loss_hard.item(), "Focal loss should heavily penalize hard/uncertain errors over confident correct ones."
