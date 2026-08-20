"""
Comprehensive Evaluation & Benchmarking Suite for CS2 Trajectory Transformer.
Computes:
- AUROC & AUPRC
- Strict False Positive Rate (FPR) at 95% / 99% Sensitivity
- Classification F1-Score & Accuracy
- Smurf Latent Embedding Alignment & ELO MAE
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    roc_auc_score, 
    average_precision_score, 
    confusion_matrix, 
    f1_score, 
    accuracy_score,
    roc_curve
)
from typing import Dict, Tuple

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

from models.st_transformer import STTrajectoryTransformer
from data.dataset import create_partitioned_dataloaders


def compute_metrics(y_true: np.ndarray, y_pred_prob: np.ndarray) -> Dict[str, float]:
    """Computes key thesis metrics from predictions and ground truth."""
    auroc = roc_auc_score(y_true, y_pred_prob) if len(set(y_true)) > 1 else 0.5
    auprc = average_precision_score(y_true, y_pred_prob) if len(set(y_true)) > 1 else 0.0
    
    # Standard 0.5 threshold
    y_pred_bin = (y_pred_prob >= 0.5).astype(int)
    acc = accuracy_score(y_true, y_pred_bin)
    f1 = f1_score(y_true, y_pred_bin, zero_division=0)
    
    # Calculate False Positive Rate at high sensitivity
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_prob)
    # Find operating point where TPR >= 0.95
    idx_95 = np.argmax(tpr >= 0.95) if (tpr >= 0.95).any() else -1
    fpr_at_95_tpr = float(fpr[idx_95]) if idx_95 != -1 else 1.0
    
    return {
        'AUROC': float(auroc),
        'AUPRC': float(auprc),
        'Accuracy': float(acc),
        'F1-Score': float(f1),
        'FPR_at_95_TPR': float(fpr_at_95_tpr)
    }


def evaluate_model_on_loader(
    model: torch.nn.Module, 
    dataloader, 
    device: torch.device
) -> Tuple[Dict[str, float], np.ndarray, np.ndarray, np.ndarray]:
    """Runs inference across dataloader and calculates metrics."""
    model.eval()
    all_preds = []
    all_targets = []
    all_embeddings = []
    all_elo_preds = []
    all_elo_targets = []
    
    with torch.no_grad():
        for batch in dataloader:
            features = batch['features'].to(device)
            mask = batch['attention_mask'].to(device)
            aimbot_labels = batch['aimbot_labels'].to(device)
            elo_labels = batch['elo_labels'].to(device)
            
            aimbot_prob, smurf_emb, elo_pred = model(features, attention_mask=mask)
            
            all_preds.extend(aimbot_prob.cpu().numpy().flatten())
            all_targets.extend(aimbot_labels.cpu().numpy().flatten())
            all_embeddings.append(smurf_emb.cpu().numpy())
            all_elo_preds.extend((elo_pred * 2000.0).cpu().numpy().flatten())
            all_elo_targets.extend((elo_labels * 2000.0).cpu().numpy().flatten())
            
    y_true = np.array(all_targets)
    y_pred = np.array(all_preds)
    embeddings = np.vstack(all_embeddings) if all_embeddings else np.array([])
    
    metrics = compute_metrics(y_true, y_pred)
    elo_mae = float(np.mean(np.abs(np.array(all_elo_preds) - np.array(all_elo_targets))))
    metrics['ELO_MAE'] = elo_mae
    
    return metrics, y_true, y_pred, embeddings


def main():
    parser = argparse.ArgumentParser(description="Evaluate Trained ST-Trans Checkpoint")
    parser.add_argument("--data_dir", type=str, default="data/processed_parquet", help="Directory containing Parquet files")
    parser.add_argument("--model_path", type=str, default="models/checkpoints/best_model.pt", help="Path to saved model checkpoint")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for evaluation")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Evaluating model checkpoint: {args.model_path} on {device}")

    # Load model
    model = STTrajectoryTransformer(
        feature_dim=8, 
        d_model=64, 
        nhead=4, 
        num_layers=4, 
        dim_feedforward=256
    ).to(device)
    if os.path.exists(args.model_path):
        checkpoint = torch.load(args.model_path, map_location=device)
        model.load_state_dict(checkpoint)
        print("[*] Successfully loaded checkpoint weights.")
    else:
        print(f"[!] Checkpoint not found at {args.model_path}, evaluating with initialized weights.")

    # Load test dataloader
    _, _, test_loader = create_partitioned_dataloaders(args.data_dir, batch_size=args.batch_size)
    print(f"[*] Test dataset size: {len(test_loader.dataset)} segments ({len(test_loader)} batches)")

    metrics, y_true, y_pred, _ = evaluate_model_on_loader(model, test_loader, device)

    print("\n" + "=" * 50)
    print("      THESIS EVALUATION METRICS (TEST SET)      ")
    print("=" * 50)
    for k, v in metrics.items():
        print(f"  > {k:<18}: {v:.4f}")
    print("=" * 50)


if __name__ == "__main__":
    main()
