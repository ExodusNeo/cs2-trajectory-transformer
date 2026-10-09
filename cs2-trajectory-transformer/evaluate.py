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
from typing import Dict, Tuple, List, Optional
from scipy.stats import beta, spearmanr

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

from models.st_transformer import STTrajectoryTransformer
from data.dataset import create_partitioned_dataloaders


def calibrate_operating_threshold(
    y_val: np.ndarray, 
    y_val_prob: np.ndarray, 
    target_fpr: float = 0.0001
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
    if len(set(y_true)) > 1:
        fpr, tpr, thresholds = roc_curve(y_true, y_pred_prob)
        idx_95 = np.argmax(tpr >= 0.95) if (tpr >= 0.95).any() else -1
        fpr_at_95_tpr = float(fpr[idx_95]) if idx_95 != -1 else 1.0
        
        # Calculate True Positive Rate at low False Positive Rate (FPR <= 0.001 / 0.1%)
        idx_low_fpr = np.where(fpr <= 0.001)[0]
        tpr_at_low_fpr = float(tpr[idx_low_fpr[-1]]) if len(idx_low_fpr) > 0 else 0.0
        
        # Calculate True Positive Rate at strict operational target (FPR <= 0.0001 / 0.01%)
        idx_strict_fpr = np.where(fpr <= 0.0001)[0]
        tpr_at_strict_fpr = float(tpr[idx_strict_fpr[-1]]) if len(idx_strict_fpr) > 0 else 0.0
    else:
        fpr_at_95_tpr = 1.0
        tpr_at_low_fpr = 0.0
        tpr_at_strict_fpr = 0.0
    
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
        'Window_Total_Count': float(len(y_true)),
        'Window_Clean_Count': float(n_neg),
        'Window_FP_Count': float(fp_at_tau),
        'Window_FPR_at_Tau': fpr_at_tau,
        'Window_Nominal_FPR_95_Upper': fpr_95_ci_upper,
        'Window_FPR_95_Upper': fpr_95_ci_upper,
        'FPR_at_Operating_Threshold': fpr_at_tau,
        'FPR_95_Upper_Bound': fpr_95_ci_upper,
        'FPR_at_95_TPR': float(fpr_at_95_tpr),
        'TPR_at_0.1%_FPR': float(tpr_at_low_fpr),
        'TPR_at_0.01%_FPR': float(tpr_at_strict_fpr),
        'Negative_Samples': n_neg
    }




def compute_cluster_bootstrap_bounds(
    cluster_ids: List[str],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    threshold: float,
    n_bootstraps: int = 1000,
    random_state: int = 42
) -> float:
    """
    Computes dependence-aware 95% upper confidence bound on False Positive Rate
    using cluster bootstrap over independent clusters (e.g. match IDs),
    accounting for intra-cluster correlation and overlapping ATW segments.
    """
    rng = np.random.RandomState(random_state)
    c_arr = np.array(cluster_ids)
    unique_clusters = np.unique(c_arr)
    
    clean_counts = []
    fp_counts = []
    for cid in unique_clusters:
        mask = (c_arr == cid)
        c_true = y_true[mask]
        c_pred = y_pred[mask]
        neg_mask = (c_true == 0)
        c_clean = int(np.sum(neg_mask))
        c_fp = int(np.sum(neg_mask & (c_pred >= threshold)))
        clean_counts.append(c_clean)
        fp_counts.append(c_fp)
        
    clean_arr = np.array(clean_counts)
    fp_arr = np.array(fp_counts)
    total_clean = np.sum(clean_arr)
    
    if total_clean == 0:
        return 1.0
        
    n_c = len(unique_clusters)
    if n_c <= 1:
        # Fall back to nominal Rule of Three if only 1 cluster
        return float(3.0 / total_clean) if np.sum(fp_arr) == 0 else float(np.sum(fp_arr) / total_clean)
        
    boot_indices = rng.choice(n_c, size=(n_bootstraps, n_c), replace=True)
    boot_clean = np.sum(clean_arr[boot_indices], axis=1)
    boot_fp = np.sum(fp_arr[boot_indices], axis=1)
    
    valid_mask = boot_clean > 0
    boot_fpr = np.zeros(n_bootstraps, dtype=np.float32)
    boot_fpr[valid_mask] = boot_fp[valid_mask] / boot_clean[valid_mask]
    
    percentile_95 = float(np.percentile(boot_fpr, 95))
    if percentile_95 == 0.0:
        # When 0 false positives observed across all cluster resamples,
        # apply cluster-level Rule of Three bound: 3 / N_clean_clusters
        clean_clusters = int(np.sum(clean_arr > 0))
        percentile_95 = float(3.0 / max(1, clean_clusters))
        
    return percentile_95


def compute_design_effect(
    cluster_ids: List[str],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    threshold: float
) -> Tuple[float, float, float, float]:
    """
    Thesis Reference: Chapter 3, Section 3.2.9 — Design-Effect Adjustment
    Computes intra-cluster correlation (rho_hat), average cluster size (m_bar),
    design effect (Deff = 1 + (m_bar - 1) * rho_hat), effective sample size (N_eff),
    and adjusted 95% upper bound on False Positive Rate.
    
    Note on Zero-False-Alarm Regime:
    When total_fp == 0, sample variance of binary classifications is zero, making standard
    binary ANOVA ICC degenerate (0/0). Rather than ignoring clustering and falling back to
    unadjusted independence (which would assume completely uncorrelated observations), we estimate
    the latent intra-cluster correlation (rho_hat) via one-way ANOVA over the continuous model anomaly
    scores of clean negative windows across clusters. This serves as an empirical sensitivity analysis
    proxy capturing latent model calibration clustering, yielding a penalizing design effect
    (Deff = 1 + (m_bar - 1) * rho_hat) and conservative effective sample size (N_eff = N_clean / Deff).
    
    Returns:
    --------
    (rho_hat, deff, n_eff, adjusted_fpr_95_upper)
    """
    c_arr = np.array(cluster_ids)
    unique_clusters = np.unique(c_arr)
    
    clean_counts = []
    fp_counts = []
    for cid in unique_clusters:
        mask = (c_arr == cid)
        c_true = y_true[mask]
        c_pred = y_pred[mask]
        neg_mask = (c_true == 0)
        c_clean = int(np.sum(neg_mask))
        c_fp = int(np.sum(neg_mask & (c_pred >= threshold)))
        clean_counts.append(c_clean)
        fp_counts.append(c_fp)
        
    clean_arr = np.array(clean_counts)
    fp_arr = np.array(fp_counts)
    total_clean = int(np.sum(clean_arr))
    total_fp = int(np.sum(fp_arr))
    
    if total_clean <= 0 or len(unique_clusters) <= 1:
        return 0.0, 1.0, float(total_clean), (3.0 / max(1, total_clean) if total_fp == 0 else 1.0)
        
    K = len(unique_clusters)
    p_hat = total_fp / total_clean
    m_bar = float(total_clean / K)
    
    # Effective cluster size weighting constant for unequal cluster sizes (Donner & Klar, 2000)
    m_0 = float((total_clean - np.sum(clean_arr ** 2) / total_clean) / max(1, K - 1))
    
    # When 0 binary false alarms are observed, binary sample variance is zero.
    # Rather than falling back to independence, estimate latent intra-cluster correlation (rho_hat)
    # as an empirical sensitivity analysis proxy from continuous model anomaly scores of clean negative windows across clusters.
    if total_fp == 0 or p_hat == 0.0 or p_hat >= 1.0:
        neg_mask_all = (y_true == 0)
        clean_scores = y_pred[neg_mask_all]
        clean_c_arr = c_arr[neg_mask_all]
        
        grand_mean = float(np.mean(clean_scores)) if len(clean_scores) > 0 else 0.0
        ssb_s = 0.0
        ssw_s = 0.0
        for cid in unique_clusters:
            c_mask = (clean_c_arr == cid)
            c_s = clean_scores[c_mask]
            if len(c_s) > 0:
                c_m = float(np.mean(c_s))
                ssb_s += len(c_s) * (c_m - grand_mean) ** 2
                ssw_s += float(np.sum((c_s - c_m) ** 2))
                
        msb_s = ssb_s / max(1, K - 1)
        msw_s = ssw_s / max(1, total_clean - K)
        denom_s = msb_s + (m_0 - 1.0) * msw_s
        if denom_s > 1e-9:
            rho_hat = float(max(0.0, min(1.0, (msb_s - msw_s) / denom_s)))
        else:
            rho_hat = 0.0
            
        deff = float(max(1.0, 1.0 + (m_bar - 1.0) * rho_hat))
        n_eff = float(max(1.0, total_clean / deff))
        adj_upper = float(3.0 / n_eff)
        return rho_hat, deff, n_eff, adj_upper
        
    # ANOVA estimate of intra-cluster correlation coefficient (ICC) on binary false alarms
    cluster_rates = np.zeros(K, dtype=np.float64)
    valid_c = clean_arr > 0
    cluster_rates[valid_c] = fp_arr[valid_c] / clean_arr[valid_c]
    
    ssb = float(np.sum(clean_arr * (cluster_rates - p_hat) ** 2))
    msb = ssb / max(1, K - 1)
    
    sst = float(total_clean * p_hat * (1.0 - p_hat))
    ssw = max(0.0, sst - ssb)
    msw = ssw / max(1, total_clean - K)
    
    denom = msb + (m_0 - 1.0) * msw
    if denom > 1e-9:
        rho_hat = float(max(0.0, min(1.0, (msb - msw) / denom)))
    else:
        rho_hat = 0.0
        
    deff = float(max(1.0, 1.0 + (m_bar - 1.0) * rho_hat))
    n_eff = float(max(1.0, total_clean / deff))
    adj_upper = float(beta.ppf(0.95, total_fp + 1, max(1.0, n_eff - total_fp)))
    return rho_hat, deff, n_eff, adj_upper


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
    all_match_ids = []
    
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
            if 'match_ids' in batch:
                all_match_ids.extend(batch['match_ids'])
            else:
                all_match_ids.extend([''] * len(aimbot_prob))
            
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
            mid = all_match_ids[idx] if idx < len(all_match_ids) and all_match_ids[idx] else ''
            session_key = (mid, pid) if mid else pid
            if session_key not in session_map:
                session_map[session_key] = {'targets': [], 'preds': [], 'match_id': mid, 'player_id': pid}
            session_map[session_key]['targets'].append(all_targets[idx])
            session_map[session_key]['preds'].append(all_preds[idx])
            
        session_targets = []
        session_peak_preds = []
        session_match_ids = []
        session_player_ids = []
        for skey, s_data in session_map.items():
            session_targets.append(int(max(s_data['targets'])))
            session_peak_preds.append(float(max(s_data['preds'])))
            session_match_ids.append(s_data.get('match_id', ''))
            session_player_ids.append(str(s_data.get('player_id', '')))
            
        s_y_true = np.array(session_targets)
        s_y_pred = np.array(session_peak_preds)
        if len(set(s_y_true)) > 1:
            metrics['Session_AUROC'] = float(roc_auc_score(s_y_true, s_y_pred))
        s_neg = int(np.sum(s_y_true == 0))
        s_fp = int(np.sum((s_y_true == 0) & (s_y_pred >= operating_threshold)))
        metrics['Session_Total_Count'] = float(len(session_map))
        metrics['Session_Clean_Count'] = float(s_neg)
        metrics['Session_FP_Count'] = float(s_fp)
        metrics['Session_FPR'] = float(s_fp / max(1, s_neg))
        if s_neg > 0:
            if s_fp == 0:
                s_nominal_upper = float(3.0 / s_neg)
            else:
                from scipy.stats import beta
                s_nominal_upper = float(beta.ppf(0.95, s_fp + 1, s_neg - s_fp))
        else:
            s_nominal_upper = 1.0
        metrics['Session_Nominal_FPR_95_Upper'] = s_nominal_upper
        metrics['Session_FPR_95_Upper'] = s_nominal_upper

        # Dependence-Aware Cluster Bootstrap & Design-Effect Bounds
        if any(m for m in all_match_ids):
            # 1. Match Cluster Bootstrap (Windows and Sessions)
            metrics['Window_Match_Cluster_Bootstrap_FPR_95_Upper'] = compute_cluster_bootstrap_bounds(
                all_match_ids, y_true, y_pred, operating_threshold
            )
            metrics['Window_Cluster_Bootstrap_FPR_95_Upper'] = metrics['Window_Match_Cluster_Bootstrap_FPR_95_Upper']
            metrics['Session_Match_Cluster_Bootstrap_FPR_95_Upper'] = compute_cluster_bootstrap_bounds(
                session_match_ids, s_y_true, s_y_pred, operating_threshold
            )
            metrics['Session_Cluster_Bootstrap_FPR_95_Upper'] = metrics['Session_Match_Cluster_Bootstrap_FPR_95_Upper']
            
            # 2. Match Design-Effect Adjustment (ICC rho_hat, Deff, N_eff)
            w_m_rho, w_m_deff, w_m_neff, w_m_adj_upper = compute_design_effect(
                all_match_ids, y_true, y_pred, operating_threshold
            )
            metrics['Window_Match_ICC_Rho'] = w_m_rho
            metrics['Window_Match_Design_Effect'] = w_m_deff
            metrics['Window_Match_Effective_Sample_Size'] = w_m_neff
            metrics['Window_Match_Adjusted_FPR_95_Upper'] = w_m_adj_upper
            metrics['Window_Design_Effect_Adjusted_FPR_95_Upper'] = w_m_adj_upper
            
            s_m_rho, s_m_deff, s_m_neff, s_m_adj_upper = compute_design_effect(
                session_match_ids, s_y_true, s_y_pred, operating_threshold
            )
            metrics['Session_Match_ICC_Rho'] = s_m_rho
            metrics['Session_Match_Design_Effect'] = s_m_deff
            metrics['Session_Match_Effective_Sample_Size'] = s_m_neff
            metrics['Session_Match_Adjusted_FPR_95_Upper'] = s_m_adj_upper
            metrics['Session_ICC_Rho'] = s_m_rho
            metrics['Session_Design_Effect'] = s_m_deff
            metrics['Session_Effective_Sample_Size'] = s_m_neff
            metrics['Session_Design_Effect_Adjusted_FPR_95_Upper'] = s_m_adj_upper

        # 3. Player-Level Clustering (accounting for repeated player accounts across matches)
        if len(all_player_ids) > 0 and len(np.unique(all_player_ids)) > 1:
            pid_list = [str(p) for p in all_player_ids]
            metrics['Window_Player_Bootstrap_FPR_95_Upper'] = compute_cluster_bootstrap_bounds(
                pid_list, y_true, y_pred, operating_threshold
            )
            w_p_rho, w_p_deff, w_p_neff, w_p_adj_upper = compute_design_effect(
                pid_list, y_true, y_pred, operating_threshold
            )
            metrics['Window_Player_ICC_Rho'] = w_p_rho
            metrics['Window_Player_Design_Effect'] = w_p_deff
            metrics['Window_Player_Effective_Sample_Size'] = w_p_neff
            metrics['Window_Player_Adjusted_FPR_95_Upper'] = w_p_adj_upper
            
            # Session Player Clustering
            if len(session_player_ids) > 0 and len(np.unique(session_player_ids)) > 1:
                metrics['Session_Player_Cluster_Bootstrap_FPR_95_Upper'] = compute_cluster_bootstrap_bounds(
                    session_player_ids, s_y_true, s_y_pred, operating_threshold
                )
                s_p_rho, s_p_deff, s_p_neff, s_p_adj_upper = compute_design_effect(
                    session_player_ids, s_y_true, s_y_pred, operating_threshold
                )
                metrics['Session_Player_ICC_Rho'] = s_p_rho
                metrics['Session_Player_Design_Effect'] = s_p_deff
                metrics['Session_Player_Effective_Sample_Size'] = s_p_neff
                metrics['Session_Player_Adjusted_FPR_95_Upper'] = s_p_adj_upper
                
                if any(m for m in all_match_ids):
                    s_cons_neff = min(s_m_neff, s_p_neff)
                    metrics['Session_Conservative_Effective_Sample_Size'] = s_cons_neff
                    metrics['Session_Conservative_Adjusted_FPR_95_Upper'] = max(s_m_adj_upper, s_p_adj_upper)
            
            # Conservative combined effective sample size between match and player clustering axes
            if any(m for m in all_match_ids):
                cons_neff = min(w_m_neff, w_p_neff)
                metrics['Window_Conservative_Effective_Sample_Size'] = cons_neff
                metrics['Window_Conservative_Adjusted_FPR_95_Upper'] = max(w_m_adj_upper, w_p_adj_upper)

    # 4. Match-Level Evaluation (Match cluster unit — Rule of Three over independent clean matches)
    if any(m for m in all_match_ids):
        match_map = {}
        for idx in range(len(all_preds)):
            mid = all_match_ids[idx] if idx < len(all_match_ids) and all_match_ids[idx] else ''
            if not mid:
                continue
            if mid not in match_map:
                match_map[mid] = {'targets': [], 'preds': []}
            match_map[mid]['targets'].append(all_targets[idx])
            match_map[mid]['preds'].append(all_preds[idx])
            
        if len(match_map) > 0:
            m_targets = []
            m_peak_preds = []
            for mid, m_data in match_map.items():
                m_targets.append(int(max(m_data['targets'])))
                m_peak_preds.append(float(max(m_data['preds'])))
                
            m_y_true = np.array(m_targets)
            m_y_pred = np.array(m_peak_preds)
            m_neg = int(np.sum(m_y_true == 0))
            m_fp = int(np.sum((m_y_true == 0) & (m_y_pred >= operating_threshold)))
            
            metrics['Match_Total_Count'] = float(len(match_map))
            metrics['Match_Clean_Count'] = float(m_neg)
            metrics['Match_FP_Count'] = float(m_fp)
            metrics['Match_FPR'] = float(m_fp / max(1, m_neg))
            if m_neg > 0:
                if m_fp == 0:
                    m_nominal_upper = float(3.0 / m_neg)
                else:
                    m_nominal_upper = float(beta.ppf(0.95, m_fp + 1, m_neg - m_fp))
            else:
                m_nominal_upper = 1.0
            metrics['Match_Nominal_FPR_95_Upper'] = m_nominal_upper
    
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
    parser.add_argument("--target_fpr", type=float, default=0.0001, help="Operational target false positive rate (default: 0.0001 = 0.01%%)")
    parser.add_argument("--use_global_norm", action="store_true", help="Use global dataset standardization")
    parser.add_argument("--allow_untrained", action="store_true", help="Allow evaluation with initialized random weights if checkpoint is missing")
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
    model_path = args.model_path
    if not os.path.exists(model_path) and os.path.exists(os.path.join("cs2-trajectory-transformer", model_path)):
        model_path = os.path.join("cs2-trajectory-transformer", model_path)

    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint)
        print(f"[*] Successfully loaded checkpoint weights from: {model_path}")
    elif args.allow_untrained:
        print(f"[!] Warning: Checkpoint not found at {args.model_path}, evaluating with initialized weights (--allow_untrained).")
    else:
        raise FileNotFoundError(
            f"Model checkpoint not found: '{args.model_path}'. "
            f"A trained checkpoint is required to evaluate model performance. "
            f"Train the model using train.py or pass --allow_untrained to test execution without trained weights."
        )

    # Load dataloaders (checking for saved training scaler)
    scaler_path = model_path.replace('.pt', '_scaler.npz')
    scaler_load = scaler_path if os.path.exists(scaler_path) else None
    if scaler_load:
        print(f"[*] Found training scaler statistics at: {scaler_load}")

    data_dir = args.data_dir
    if not os.path.exists(data_dir) and os.path.exists(os.path.join("cs2-trajectory-transformer", data_dir)):
        data_dir = os.path.join("cs2-trajectory-transformer", data_dir)

    train_loader, val_loader, test_loader = create_partitioned_dataloaders(
        data_dir, 
        batch_size=args.batch_size,
        use_global_norm=args.use_global_norm,
        scaler_load_path=scaler_load
    )
    print(f"[*] Partitions: Val={len(val_loader.dataset)} segments, Test={len(test_loader.dataset)} segments")

    # 1. Calibrate operational decision threshold tau* on validation partition
    print(f"[*] Calibrating operational decision threshold on validation set (target FPR <= {args.target_fpr*100:.3f}%)...")
    val_metrics, val_true, val_pred, _, _ = evaluate_model_on_loader(model, val_loader, device)
    calibrated_tau = calibrate_operating_threshold(val_true, val_pred, target_fpr=args.target_fpr)
    print(f"[*] Calibrated operating threshold tau*: {calibrated_tau:.4f}")

    # 2. Evaluate on held-out test partition using calibrated threshold
    print(f"[*] Evaluating on held-out test dataset with tau*={calibrated_tau:.4f}...")
    metrics, y_true, y_pred, embeddings, player_ids = evaluate_model_on_loader(
        model, test_loader, device, operating_threshold=calibrated_tau
    )

    print("\n" + "=" * 65)
    print("           THESIS EVALUATION METRICS (TEST SET)           ")
    print("=" * 65)
    for k, v in metrics.items():
        if isinstance(v, (int, np.integer)) or (('_Count' in k or '_Size' in k or 'Negative_Samples' in k) and isinstance(v, (int, float)) and v == int(v)):
            print(f"  > {k:<45}: {int(v)}")
        elif 'FPR' in k:
            print(f"  > {k:<45}: {v:.6f} ({v*100:.4f}%)")
        else:
            print(f"  > {k:<45}: {v:.4f}")
    print("=" * 65)


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
