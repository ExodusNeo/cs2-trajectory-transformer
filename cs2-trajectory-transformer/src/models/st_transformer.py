"""
Spatial-Temporal Trajectory Transformer (ST-Trans) PyTorch Module.
Dual-head architecture for Aimbot Classification and Skill ELO Estimation.
"""

import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """Sinusoidal Positional Encoding for temporal sequence dynamics."""
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]


class STTrajectoryTransformer(nn.Module):
    """
    Spatial-Temporal Trajectory Transformer (ST-Trans).
    Processes kinematic sequences [batch, seq_len, feature_dim] and predicts:
    1. Aimbot Probability (Organic vs Algorithmic Aim Assistance)
    2. Player Intrinsic ELO Skill Level
    """
    def __init__(self, feature_dim: int = 6, d_model: int = 128, nhead: int = 8, num_layers: int = 4):
        super().__init__()
        self.input_projection = nn.Linear(feature_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=512, 
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Dual-Head Classifiers
        self.aimbot_head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        self.smurf_head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1)
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        x: Tensor of shape (batch_size, sequence_length, feature_dim)
        """
        x_emb = self.input_projection(x)
        x_emb = self.pos_encoder(x_emb)
        
        encoded = self.transformer_encoder(x_emb)
        
        # Global Temporal Average Pooling
        pooled = torch.mean(encoded, dim=1)
        
        aimbot_prob = self.aimbot_head(pooled)
        predicted_elo = self.smurf_head(pooled)
        
        return aimbot_prob, predicted_elo
