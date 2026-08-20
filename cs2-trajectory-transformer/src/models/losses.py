"""
Loss Functions for CS2 Trajectory Transformer:
1. InfoNCE Contrastive Loss (Supervised Contrastive for Smurf/Player Biometrics).
2. Focal Loss (for Aimbot Detection under Class Imbalance).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SupervisedInfoNCELoss(nn.Module):
    """
    Supervised InfoNCE (SupCon) Loss for player trajectory representations.
    Pulls embeddings of the same player together in latent space while pushing different players apart.
    """
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        embeddings: [batch_size, embed_dim] (L2 normalized)
        labels: [batch_size] player IDs
        """
        device = embeddings.device
        batch_size = embeddings.shape[0]
        if batch_size < 2:
            return torch.tensor(0.0, device=device, requires_grad=True)

        # Normalize embeddings
        embeddings = F.normalize(embeddings, p=2, dim=-1)
        
        # Cosine similarity matrix [batch_size, batch_size]
        sim_matrix = torch.matmul(embeddings, embeddings.T) / self.temperature

        # Create positive mask (same player, but not self)
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask  # Positive pairs excluding self

        # If no positive pairs in batch, return 0
        pos_counts = mask.sum(1)
        if (pos_counts == 0).all():
            return torch.tensor(0.0, device=device, requires_grad=True)

        # Numerical stability: subtract max logit
        logits_max, _ = torch.max(sim_matrix, dim=1, keepdim=True)
        logits = sim_matrix - logits_max.detach()

        # Denominator: sum over all non-self elements
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-9)

        # Mean of log-likelihood over positive pairs
        mean_log_prob_pos = (mask * log_prob).sum(1) / (pos_counts + 1e-9)
        loss = -mean_log_prob_pos[pos_counts > 0].mean()
        return loss


class FocalLoss(nn.Module):
    """
    Binary Focal Loss for addressing extreme class imbalance in aimbot detection.
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    """
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        inputs: [batch_size, 1] (probabilities from Sigmoid)
        targets: [batch_size, 1] (0.0 or 1.0)
        """
        inputs = inputs.clamp(min=1e-6, max=1.0 - 1e-6)
        bce_loss = - (targets * torch.log(inputs) + (1.0 - targets) * torch.log(1.0 - inputs))
        
        p_t = targets * inputs + (1.0 - targets) * (1.0 - inputs)
        alpha_t = targets * self.alpha + (1.0 - targets) * (1.0 - self.alpha)
        
        focal_weight = alpha_t * ((1.0 - p_t) ** self.gamma)
        loss = (focal_weight * bce_loss).mean()
        return loss
