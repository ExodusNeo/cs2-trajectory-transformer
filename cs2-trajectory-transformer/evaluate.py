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


def calibrate_operating_threshold(
    y_val: np.ndarray, 
    y_val_prob: np.ndarray, 
    target_fpr: float = 0.001
) -> float:
    """
    Calibrates operational decision threshold tau* on the validation partition.
    Ensures empirical validation FPR <= target_fpr while maximizing sensitivity.
    """
    if len(set(y_val)) <= 1:
        return 0.5
    fpr, tpr, thresholds = roc_curve(y_val, y_val_prob)
    valid_indices = np.where(fpr <= target_fpr)[0]
    if len(valid_indices) > 0:
        best_idx = valid_indices[-1]
        return float(thresholds[best_idx])
    return 0.95


def compute_metrics(
    y_true: np.ndarray, 
    y_pred_prob: np.ndarray,
    operating_threshold: float = 0.5
) -> Dict[str, float]:
    """Computes key thesis metrics from predictions and ground truth."""
    auroc = roc_auc_score(y_true, y_pred_prob) if len(set(y_true)) > 1 else 0.5
    auprc = average_precision_score(y_true, y_pred_prob) if len(set(y_true)) > 1 else 0.0
    
    # Calibrated decision threshold classification
    y_pred_bin = (y_pred_prob >= operating_threshold).astype(int)
    acc = accuracy_score(y_true, y_pred_bin)
    f1 = f1_score(y_true, y_pred_bin, zero_division=0)
    
    # Calculate False Positive Rate at high sensitivity (95% TPR)
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_prob)
    idx_95 = np.argmax(tpr >= 0.95) if (tpr >= 0.95).any() else -1
    fpr_at_95_tpr = float(fpr[idx_95]) if idx_95 != -1 else 1.0
    
    # Calculate True Positive Rate at low False Positive Rate (FPR <= 0.001 / 0.1%)
    idx_low_fpr = np.where(fpr <= 0.001)[0]
    tpr_at_low_fpr = float(tpr[idx_low_fpr[-1]]) if len(idx_low_fpr) > 0 else 0.0
    
    # Calculate True Positive Rate at strict operational target (FPR <= 0.0001 / 0.01%)
    idx_strict_fpr = np.where(fpr <= 0.0001)[0]
    tpr_at_strict_fpr = float(tpr[idx_strict_fpr[-1]]) if len(idx_strict_fpr) > 0 else 0.0
    
    # Statistical Confidence Bound on False Positive Rate at operating threshold
    n_neg = int(np.sum(y_true == 0))
    fp_at_tau = int(np.sum((y_true == 0) & (y_pred_bin == 1)))
    fpr_at_tau = float(fp_at_tau / max(1, n_neg))
    if n_neg > 0:
        if fp_at_tau == 0:
            # Rule of Three: -ln(0.05) / N ~ 3 / N
            fpr_95_ci_upper = float(3.0 / n_neg)
        else:
            from scipy.stats import beta
            fpr_95_ci_upper = float(beta.ppf(0.95, fp_at_tau + 1, n_neg - fp_at_tau))
    else:
        fpr_95_ci_upper = 1.0
    
    return {
        'AUROC': float(auroc),
        'AUPRC': float(auprc),
        'Accuracy': float(acc),
        'F1-Score': float(f1),
        'Operating_Threshold': float(operating_threshold),
        'FPR_at_Operating_Threshold': fpr_at_tau,
        'FPR_95_Upper_Bound': fpr_95_ci_upper,
        'FPR_at_95_TPR': float(fpr_at_95_tpr),
        'TPR_at_0.1%_FPR': float(tpr_at_low_fpr),
        'TPR_at_0.01%_FPR': float(tpr_at_strict_fpr),
        'Negative_Samples': n_neg
    }




def evaluate_model_on_loader(
    model: torch.nn.Module, 
    dataloader, 
    device: torch.device,
    operating_threshold: float = 0.5
) -> Tuple[Dict[str, float], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Runs inference across dataloader and calculates metrics."""
    model.eval()
    all_preds = []
    all_targets = []
    all_embeddings = []
    all_elo_preds = []
    all_elo_targets = []
    
    all_player_ids = []
    
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
            if 'player_ids' in batch:
                all_player_ids.extend(batch['player_ids'].cpu().numpy().flatten())
            
    y_true = np.array(all_targets)
    y_pred = np.array(all_preds)
    embeddings = np.vstack(all_embeddings) if all_embeddings else np.array([])
    player_ids = np.array(all_player_ids)
    
    metrics = compute_metrics(y_true, y_pred, operating_threshold=operating_threshold)
    elo_mae = float(np.mean(np.abs(np.array(all_elo_preds) - np.array(all_elo_targets))))
    metrics['ELO_MAE'] = elo_mae
    
    # Spearman's Rank Correlation (Table 8)
    try:
        from scipy.stats import spearmanr
        if len(all_elo_preds) > 1 and len(set(all_elo_targets)) > 1:
            corr, _ = spearmanr(all_elo_preds, all_elo_targets)
            metrics['Spearman_Correlation'] = float(corr)
        else:
            metrics['Spearman_Correlation'] = 0.0
    except Exception:
        metrics['Spearman_Correlation'] = 0.0
    
    # Biometric Identification Retrieval (Table 8 / Proposal Section 3.2.9)
    if len(embeddings) > 1 and len(player_ids) == len(embeddings):
        try:
            from sklearn.metrics.pairwise import cosine_similarity
            sim_matrix = cosine_similarity(embeddings)
            np.fill_diagonal(sim_matrix, -1.0)  # Exclude self-match
            top1_correct = 0
            top5_correct = 0
            valid_queries = 0
            for i in range(len(embeddings)):
                pid = player_ids[i]
                same_player_mask = (player_ids == pid)
                same_player_mask[i] = False
                if np.any(same_player_mask):
                    valid_queries += 1
                    ranked = np.argsort(sim_matrix[i])[::-1]
                    if player_ids[ranked[0]] == pid:
                        top1_correct += 1
                    if pid in player_ids[ranked[:5]]:
                        top5_correct += 1
            if valid_queries > 0:
                metrics['Biometric_P@1'] = float(top1_correct / valid_queries)
                metrics['Biometric_P@5'] = float(top5_correct / valid_queries)
            else:
                metrics['Biometric_P@1'] = 0.0
                metrics['Biometric_P@5'] = 0.0
        except Exception:
            metrics['Biometric_P@1'] = 0.0
            metrics['Biometric_P@5'] = 0.0

    # Cluster-aware session evaluation (Match-Player unit)
    if len(all_player_ids) == len(all_preds) and len(all_player_ids) > 0:
        session_map = {}
        for idx in range(len(all_preds)):
            pid = all_player_ids[idx]
            if pid not in session_map:
                session_map[pid] = {'targets': [], 'preds': []}
            session_map[pid]['targets'].append(all_targets[idx])
            session_map[pid]['preds'].append(all_preds[idx])
            
        session_targets = []
        session_peak_preds = []
        for pid, s_data in session_map.items():
            session_targets.append(int(max(s_data['targets'])))
            session_peak_preds.append(float(max(s_data['preds'])))
            
        s_y_true = np.array(session_targets)
        s_y_pred = np.array(session_peak_preds)
        if len(set(s_y_true)) > 1:
            metrics['Session_AUROC'] = float(roc_auc_score(s_y_true, s_y_pred))
        s_neg = int(np.sum(s_y_true == 0))
        s_fp = int(np.sum((s_y_true == 0) & (s_y_pred >= operating_threshold)))
        metrics['Session_Total_Count'] = float(len(session_map))
        metrics['Session_Clean_Count'] = float(s_neg)
        metrics['Session_FPR'] = float(s_fp / max(1, s_neg))
        if s_neg > 0:
            if s_fp == 0:
                metrics['Session_FPR_95_Upper'] = float(3.0 / s_neg)
            else:
                from scipy.stats import beta
                metrics['Session_FPR_95_Upper'] = float(beta.ppf(0.95, s_fp + 1, s_neg - s_fp))
        else:
            metrics['Session_FPR_95_Upper'] = 1.0
    
    return metrics, y_true, y_pred, embeddings, player_ids




def profile_inference_latency(
    model: torch.nn.Module, 
    device: torch.device, 
    seq_len: int = 256, 
    n_runs: int = 100
) -> Dict[str, float]:
    """
    Thesis Reference: Chapter 1 & Chapter 3, Section 3.2.8 — Server-Side Latency Profiling
    Profiles inference throughput with rigorous CUDA synchronization.
    """
    import time
    model.eval()
    dummy_input_1 = torch.randn(1, seq_len, 8, device=device)
    dummy_mask_1 = torch.ones(1, seq_len, dtype=torch.bool, device=device)
    
    # Warmup
    with torch.no_grad():
        for _ in range(15):
            _ = model(dummy_input_1, attention_mask=dummy_mask_1)
    if device.type == 'cuda':
        torch.cuda.synchronize()
            
    latencies_ms = []
    with torch.no_grad():
        for _ in range(n_runs):
            if device.type == 'cuda':
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = model(dummy_input_1, attention_mask=dummy_mask_1)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            latencies_ms.append((time.perf_counter() - t0) * 1000.0)
            
    latencies = np.array(latencies_ms)
    mean_lat = float(np.mean(latencies))
    p50_lat = float(np.percentile(latencies, 50))
    p95_lat = float(np.percentile(latencies, 95))
    p99_lat = float(np.percentile(latencies, 99))
    
    # Batch=32 test for full match evaluation
    dummy_input_32 = torch.randn(32, seq_len, 8, device=device)
    dummy_mask_32 = torch.ones(32, seq_len, dtype=torch.bool, device=device)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(30):
            _ = model(dummy_input_32, attention_mask=dummy_mask_32)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    batch_lat = ((time.perf_counter() - t0) / 30) * 1000.0
    
    # Match audit throughput estimation (~100 ATWs per match)
    match_est_ms = (batch_lat / 32.0) * 100.0
    
    return {
        'single_window_mean_ms': mean_lat,
        'single_window_p50_ms': p50_lat,
        'single_window_p95_ms': p95_lat,
        'single_window_p99_ms': p99_lat,
        'batch_32_latency_ms': float(batch_lat),
        'est_100_atw_match_audit_ms': float(match_est_ms)
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate Trained ST-Trans Checkpoint")
    parser.add_argument("--data_dir", type=str, default="data/processed_parquet", help="Directory containing Parquet files")
    parser.add_argument("--model_path", type=str, default="models/checkpoints/best_model.pt", help="Path to saved model checkpoint")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for evaluation")
    parser.add_argument("--d_model", type=int, default=128, help="Transformer hidden dimension")
    parser.add_argument("--nhead", type=int, default=8, help="Number of attention heads")
    parser.add_argument("--num_layers", type=int, default=4, help="Number of transformer layers")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Evaluating model checkpoint: {args.model_path} on {device}")

    # Load model
    model = STTrajectoryTransformer(
        feature_dim=8, 
        d_model=args.d_model, 
        nhead=args.nhead, 
        num_layers=args.num_layers, 
        dim_feedforward=args.d_model * 4
    ).to(device)
    if os.path.exists(args.model_path):
        checkpoint = torch.load(args.model_path, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint)
        print("[*] Successfully loaded checkpoint weights.")
    else:
        print(f"[!] Checkpoint not found at {args.model_path}, evaluating with initialized weights.")

    # Load dataloaders
    train_loader, val_loader, test_loader = create_partitioned_dataloaders(args.data_dir, batch_size=args.batch_size)
    print(f"[*] Partitions: Val={len(val_loader.dataset)} segments, Test={len(test_loader.dataset)} segments")

    # 1. Calibrate operational decision threshold tau* on validation partition
    print("[*] Calibrating operational decision threshold on validation set...")
    val_metrics, val_true, val_pred, _, _ = evaluate_model_on_loader(model, val_loader, device)
    calibrated_tau = calibrate_operating_threshold(val_true, val_pred, target_fpr=0.001)
    print(f"[*] Calibrated operating threshold tau*: {calibrated_tau:.4f}")

    # 2. Evaluate on held-out test partition using calibrated threshold
    print(f"[*] Evaluating on held-out test dataset with tau*={calibrated_tau:.4f}...")
    metrics, y_true, y_pred, embeddings, player_ids = evaluate_model_on_loader(
        model, test_loader, device, operating_threshold=calibrated_tau
    )

    print("\n" + "=" * 50)
    print("      THESIS EVALUATION METRICS (TEST SET)      ")
    print("=" * 50)
    for k, v in metrics.items():
        print(f"  > {k:<28}: {v:.4f}")
    print("=" * 50)


    # Inference Latency Profiling
    print("\n[*] Profiling Server-Side Inference Latency...")
    lat_metrics = profile_inference_latency(model, device)
    print("=" * 50)
    print("      LATENCY & THROUGHPUT BENCHMARK (ms)       ")
    print("=" * 50)
    for k, v in lat_metrics.items():
        print(f"  > {k:<28}: {v:.2f} ms")
    print("=" * 50)


if __name__ == "__main__":
    main()
