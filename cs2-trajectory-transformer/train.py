"""
Training and Validation Script for Spatial-Temporal Trajectory Transformer (ST-Trans).
Optimizes Dual-Task Objectives:
- Aimbot Detection (Focal Loss / Binary Cross-Entropy)
- Smurf Identification (Supervised InfoNCE Contrastive Loss + ELO Regression)
"""

import os
import argparse
import logging
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from sklearn.metrics import roc_auc_score, f1_score

from models.st_transformer import STTrajectoryTransformer
from models.losses import SupervisedInfoNCELoss, FocalLoss
from data.dataset import create_partitioned_dataloaders

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')


def train_one_epoch(
    model: nn.Module, 
    dataloader, 
    optimizer, 
    criterion_aimbot, 
    criterion_contrastive,
    criterion_elo,
    device: torch.device,
    lambda_contrastive: float = 0.5,
    lambda_elo: float = 0.2
) -> dict:
    model.train()
    total_loss = 0.0
    total_aimbot_loss = 0.0
    total_con_loss = 0.0
    
    for batch in dataloader:
        features = batch['features'].to(device)
        mask = batch['attention_mask'].to(device)
        aimbot_labels = batch['aimbot_labels'].to(device)
        elo_labels = batch['elo_labels'].to(device)
        player_ids = batch['player_ids'].to(device)
        
        optimizer.zero_grad()
        
        aimbot_prob, embeddings, predicted_elo = model(features, attention_mask=mask)
        
        loss_aim = criterion_aimbot(aimbot_prob, aimbot_labels)
        loss_con = criterion_contrastive(embeddings, player_ids)
        loss_elo = criterion_elo(predicted_elo, elo_labels)
        
        loss = loss_aim + lambda_contrastive * loss_con + lambda_elo * loss_elo
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        total_aimbot_loss += loss_aim.item()
        total_con_loss += loss_con.item()
        
    n_batches = len(dataloader)
    return {
        'loss': total_loss / n_batches,
        'aimbot_loss': total_aimbot_loss / n_batches,
        'contrastive_loss': total_con_loss / n_batches
    }


def evaluate(
    model: nn.Module, 
    dataloader, 
    criterion_aimbot, 
    device: torch.device
) -> dict:
    model.eval()
    all_preds = []
    all_targets = []
    total_loss = 0.0
    
    with torch.no_grad():
        for batch in dataloader:
            features = batch['features'].to(device)
            mask = batch['attention_mask'].to(device)
            aimbot_labels = batch['aimbot_labels'].to(device)
            
            aimbot_prob, _, _ = model(features, attention_mask=mask)
            loss_aim = criterion_aimbot(aimbot_prob, aimbot_labels)
            
            total_loss += loss_aim.item()
            all_preds.extend(aimbot_prob.cpu().numpy().flatten())
            all_targets.extend(aimbot_labels.cpu().numpy().flatten())
            
    n_batches = len(dataloader)
    auroc = roc_auc_score(all_targets, all_preds) if len(set(all_targets)) > 1 else 0.5
    binary_preds = [1 if p >= 0.5 else 0 for p in all_preds]
    f1 = f1_score(all_targets, binary_preds, zero_division=0)
    
    return {
        'eval_loss': total_loss / n_batches,
        'auroc': auroc,
        'f1': f1
    }
