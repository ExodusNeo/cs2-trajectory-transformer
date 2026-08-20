"""
Training and Validation Script for Spatial-Temporal Trajectory Transformer (ST-Trans).
Optimizes Dual-Task Objectives:
- Aimbot Detection (Focal Loss / Binary Cross-Entropy)
- Smurf Identification (Supervised InfoNCE Contrastive Loss + ELO Regression)
"""

import os
import sys
import argparse
import logging
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from sklearn.metrics import roc_auc_score, f1_score

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

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


def main():
    parser = argparse.ArgumentParser(description="Train ST-Trans CS2 Trajectory Transformer")
    parser.add_argument("--data_dir", type=str, default="data/processed_parquet", help="Directory with Parquet telemetry")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--d_model", type=int, default=64, help="Transformer hidden dimension")
    parser.add_argument("--nhead", type=int, default=4, help="Number of attention heads")
    parser.add_argument("--num_layers", type=int, default=4, help="Number of transformer layers")
    parser.add_argument("--save_path", type=str, default="models/checkpoints/best_model.pt", help="Checkpoint save path")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Training ST-Trans on {device} (Epochs: {args.epochs}, Batch Size: {args.batch_size})")

    # Dataloaders
    train_loader, val_loader, test_loader = create_partitioned_dataloaders(args.data_dir, batch_size=args.batch_size)
    print(f"[*] Dataset split: {len(train_loader.dataset)} Train, {len(val_loader.dataset)} Val, {len(test_loader.dataset)} Test samples.")

    # Model
    model = STTrajectoryTransformer(
        feature_dim=8,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dim_feedforward=args.d_model * 4
    ).to(device)

    # Loss functions & Optimizer
    criterion_aimbot = FocalLoss(alpha=0.25, gamma=2.0)
    criterion_contrastive = SupervisedInfoNCELoss(temperature=0.1)
    criterion_elo = nn.MSELoss()

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2)

    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    best_auroc = 0.0

    print("\n" + "=" * 65)
    print(f"{'Epoch':<8}{'Train Loss':<14}{'Aimbot Loss':<14}{'Val AUROC':<12}{'Val F1':<10}")
    print("=" * 65)

    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            criterion_aimbot=criterion_aimbot,
            criterion_contrastive=criterion_contrastive,
            criterion_elo=criterion_elo,
            device=device
        )
        scheduler.step()

        val_metrics = evaluate(model, val_loader, criterion_aimbot, device)

        print(f"{epoch:<8}{train_metrics['loss']:<14.4f}{train_metrics['aimbot_loss']:<14.4f}{val_metrics['auroc']:<12.4f}{val_metrics['f1']:<10.4f}")

        if val_metrics['auroc'] >= best_auroc:
            best_auroc = val_metrics['auroc']
            torch.save(model.state_dict(), args.save_path)

    print("=" * 65)
    print(f"[SUCCESS] Training Complete! Best Validation AUROC: {best_auroc:.4f}")
    print(f"[SUCCESS] Model saved to {args.save_path}")


if __name__ == "__main__":
    main()
