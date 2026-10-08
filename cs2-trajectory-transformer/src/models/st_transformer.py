"""
Spatial-Temporal Trajectory Transformer (ST-Trans) PyTorch Module.
Dual-head architecture for Aimbot Binary Classification and Contrastive Smurf Embedding.
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, Union


class PositionalEncoding(nn.Module):
    """Sinusoidal Positional Encoding for variable-length telemetry sequences."""
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [batch_size, seq_len, d_model]"""
        return x + self.pe[:, :x.size(1)]


class STTrajectoryTransformer(nn.Module):
    """
    Spatial-Temporal Trajectory Transformer (ST-Trans).
    
    Processes 8D kinematic sequences [batch, seq_len, 8]:
    - yaw, pitch, angular_velocity, angular_accel, angular_jerk,
      spherical_curvature, curvature_entropy, tremor_power_8_12hz.
      
    Outputs:
    1. Aimbot Probability (Sigmoid binary classification: organic vs algorithmic aim).
    2. Player Latent Embedding (32-dim unit-normalized vector for InfoNCE Contrastive Smurf Detection).
    3. Calibrated Skill ELO (Linear projection from player latent embedding).
    """
    def __init__(
        self, 
        feature_dim: int = 8, 
        d_model: int = 128, 
        nhead: int = 8, 
        num_layers: int = 4,
        embed_dim: int = 32,
        dim_feedforward: int = 512,
        dropout: float = 0.1
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.d_model = d_model
        self.embed_dim = embed_dim
        
        # 1. Input Linear Projection & Positional Encoding
        self.input_projection = nn.Linear(feature_dim, d_model)
        self.layer_norm_in = nn.LayerNorm(d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        self.dropout = nn.Dropout(dropout)
        
        # 2. Transformer Encoder Layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=dim_feedforward, 
            dropout=dropout,
            batch_first=True,
            activation='gelu'
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 3. Task Heads
        # Head A: Aimbot Detection
        self.aimbot_head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        # Head B: Smurf Latent Embedding (InfoNCE Contrastive Space)
        self.smurf_embedding_head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, embed_dim)
        )
        
        # Head C: Calibrated Skill ELO Estimator
        self.elo_head = nn.Sequential(
            nn.Linear(embed_dim, 32),
            nn.GELU(),
            nn.Linear(32, 1)
        )

    def forward(
        self, 
        x: torch.Tensor, 
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        x: [batch_size, seq_len, feature_dim]
        attention_mask: [batch_size, seq_len] (True for valid ticks, False for padding)
        
        Returns:
            - aimbot_prob: [batch_size, 1]
            - smurf_embedding: [batch_size, embed_dim] (L2 normalized)
            - predicted_elo: [batch_size, 1]
        """
        # Project and encode
        x_proj = self.input_projection(x)
        x_proj = self.layer_norm_in(x_proj)
        x_emb = self.pos_encoder(x_proj)
        x_emb = self.dropout(x_emb)
        
        # Key padding mask for PyTorch Transformer (True = Ignore)
        src_key_padding_mask = ~attention_mask if attention_mask is not None else None
        
        # Transformer representation
        encoded = self.transformer_encoder(x_emb, src_key_padding_mask=src_key_padding_mask)
        
        # Mask-aware temporal average pooling
        if attention_mask is not None:
            mask_expanded = attention_mask.unsqueeze(-1).float()
            pooled = (encoded * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1.0)
        else:
            pooled = torch.mean(encoded, dim=1)
            
        # Task outputs
        aimbot_prob = self.aimbot_head(pooled)
        
        # Latent projection & L2 normalization for InfoNCE
        raw_embed = self.smurf_embedding_head(pooled)
        smurf_embedding = F.normalize(raw_embed, p=2, dim=-1)
        
        predicted_elo = self.elo_head(smurf_embedding)
        
        return aimbot_prob, smurf_embedding, predicted_elo


class SmurfDetector:
    """
    Thesis Reference: Chapter 1 & Chapter 3, Section 3.2.6 — Biometric Skill Profiler & Smurf Detector
    
    Evaluates player motor proficiency against reported account rank to expose rank spoofing:
    1. Continuous ELO Discrepancy: Delta_ELO = predicted_elo - reported_elo
    2. Threshold Decision Rule: Flags smurf if Delta_ELO >= threshold_elo (default: 400.0, ~2 FACEIT tiers).
    3. Biometric Embedding Alignment: Cosine similarity matching to skill-tier cluster centroids.
    """
    def __init__(self, threshold_elo: float = 400.0):
        self.threshold_elo = threshold_elo

    def evaluate_player(
        self, 
        predicted_elo: float, 
        reported_elo: float, 
        player_embedding: Optional[np.ndarray] = None,
        tier_centroids: Optional[Dict[str, np.ndarray]] = None
    ) -> Dict[str, Union[float, bool, str, Dict[str, float]]]:
        """
        Evaluates a single player's predicted motor skill against nominal account rating.
        """
        delta_elo = float(predicted_elo - reported_elo)
        is_smurf = delta_elo >= self.threshold_elo
        
        # Categorize discrepancy severity
        if delta_elo >= 800.0:
            severity = "CRITICAL_SMURF"
        elif delta_elo >= 400.0:
            severity = "HIGH_CONFIDENCE_SMURF"
        elif delta_elo >= 200.0:
            severity = "SUSPICIOUS_UNDER_RANKED"
        elif delta_elo <= -400.0:
            severity = "BOOSTED_OVER_RANKED"
        else:
            severity = "RANK_ALIGNED"

        centroid_similarities = {}
        if player_embedding is not None and tier_centroids is not None:
            norm_emb = player_embedding / (np.linalg.norm(player_embedding) + 1e-6)
            for tier_name, centroid in tier_centroids.items():
                norm_c = centroid / (np.linalg.norm(centroid) + 1e-6)
                sim = float(np.dot(norm_emb, norm_c))
                centroid_similarities[tier_name] = sim

        return {
            'predicted_elo': float(predicted_elo),
            'reported_elo': float(reported_elo),
            'delta_elo': delta_elo,
            'is_smurf': is_smurf,
            'discrepancy_tier': severity,
            'centroid_similarities': centroid_similarities
        }
