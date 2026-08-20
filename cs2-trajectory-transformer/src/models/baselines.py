"""
Classical Machine Learning Baselines for Benchmark Comparison:
1. Random Forest Classifier
2. Gradient Boosting / XGBoost
3. Multi-Layer Perceptron (MLP)
4. Bidirectional LSTM (Bi-LSTM)
"""

import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from typing import Tuple, Dict
import numpy as np


class BiLSTMBaseline(nn.Module):
    """Bidirectional LSTM baseline for sequential trajectory classification."""
    def __init__(self, feature_dim: int = 8, hidden_dim: int = 64, num_layers: int = 2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=feature_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor = None) -> torch.Tensor:
        """x: [batch, seq_len, feature_dim]"""
        lstm_out, _ = self.lstm(x)
        # Average pooling over valid sequence
        if attention_mask is not None:
            mask_exp = attention_mask.unsqueeze(-1).float()
            pooled = (lstm_out * mask_exp).sum(dim=1) / mask_exp.sum(dim=1).clamp(min=1.0)
        else:
            pooled = torch.mean(lstm_out, dim=1)
        return self.classifier(pooled)


class ClassicalBaselines:
    """Wraps tabular / flattened feature classifiers."""
    def __init__(self):
        self.rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
        self.gb = GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42)
        self.mlp = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=200, random_state=42)

    def fit_all(self, X_train: np.ndarray, y_train: np.ndarray):
        """Fits all classical models on flattened statistical feature vectors."""
        self.rf.fit(X_train, y_train)
        self.gb.fit(X_train, y_train)
        self.mlp.fit(X_train, y_train)

    def predict_probabilities(self, X_test: np.ndarray) -> Dict[str, np.ndarray]:
        """Returns predicted probabilities for each baseline."""
        return {
            'Random Forest': self.rf.predict_proba(X_test)[:, 1],
            'Gradient Boosting': self.gb.predict_proba(X_test)[:, 1],
            'MLP': self.mlp.predict_proba(X_test)[:, 1]
        }
