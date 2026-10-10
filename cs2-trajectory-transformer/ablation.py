"""
Feature Ablation Study Pipeline for Spatial-Temporal Trajectory Transformer (ST-Trans).

Answers the central research question: do physics-informed and target-relative channels
improve leakage-free cheat detection over a transformer fed only raw view angles?

Configurations (identical architecture, split and seed):
1. Full model (9 channels, features.kinematics.MODEL_FEATURE_COLUMNS)
2. w/o tremor band power
3. w/o speed-derived jerk
4. w/o geodesic curvature & curvature entropy
5. w/o target-relative channels (self-kinematics only)
6. Target-relative only (aim error + rate)
7. Raw angles only (yaw, pitch): a raw-input baseline in the spirit of AntiCheatPT

A negative result (e.g. tremor adds nothing on real data) is a valid finding: report it.

Outputs:
- reports/ablation_study_summary.csv
- reports/ablation_study.png
"""

import os
import sys
import argparse
import logging
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import matplotlib.pyplot as plt

# Ensure src modules are resolvable
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

from models.st_transformer import STTrajectoryTransformer
from models.losses import SupervisedInfoNCELoss, FocalLoss, masked_smooth_l1_loss
from data.dataset import create_partitioned_dataloaders, FEATURE_COLUMNS
from evaluate import evaluate_model_on_loader, compute_metrics

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')


# ==============================================================================
# Thesis Reference: Chapter 3, Table 5 & Section 3.2.4 — Feature Space
# Definition of ablation configurations targeting specific biomechanical invariants
# ==============================================================================
def _without(*drop: str) -> List[str]:
    return [c for c in FEATURE_COLUMNS if c not in drop]


ABLATION_CONFIGS = {
    'Full ST-Trans (9 channels)': {
        'features': list(FEATURE_COLUMNS),
        'hypothesis': 'All candidate kinematic + target-relative channels'
    },
    'w/o tremor band power': {
        'features': _without('tremor_power_8_12hz'),
        'hypothesis': 'Does 8-12 Hz band power add signal on count-quantized 64 Hz angles? (Eq 11)'
    },
    'w/o speed-derived jerk': {
        'features': _without('angular_jerk'),
        'hypothesis': 'Contribution of the minimum-jerk smoothness proxy (Eq 7)'
    },
    'w/o curvature & entropy': {
        'features': _without('trajectory_curvature', 'curvature_entropy'),
        'hypothesis': 'Contribution of geodesic curvature and its entropy (Eq 9, 10)'
    },
    'w/o target-relative channels': {
        'features': _without('aim_error', 'aim_error_rate'),
        'hypothesis': 'Self-kinematics only: can motion shape alone expose aim assistance?'
    },
    'Target-relative only': {
        'features': ['aim_error', 'aim_error_rate'],
        'hypothesis': 'Crosshair-to-enemy relationship alone (Fitts-style acquisition)'
    },
    'Raw angles only (yaw, pitch)': {
        'features': ['yaw', 'pitch'],
        'hypothesis': 'Raw-input transformer baseline without engineered channels'
    }
}


def train_ablation_model(
    train_loader,
    val_loader,
    feature_dim: int,
    epochs: int = 12,
    lr: float = 1e-4,
    device: torch.device = None
) -> STTrajectoryTransformer:
    """
    Trains an ST-Trans model variant for a specific feature dimensionality.
    Uses identical architecture and optimization to ensure rigorous comparability.
    """
    model = STTrajectoryTransformer(
        feature_dim=feature_dim,
        d_model=128,
        nhead=8,
        num_layers=4,
        embed_dim=32,
        dim_feedforward=512,
        dropout=0.1
    ).to(device)

    # Thesis Reference: Chapter 3, Equation (16) & (17) — Focal & InfoNCE Losses
    criterion_aim = FocalLoss(alpha=0.25, gamma=2.0)
    criterion_con = SupervisedInfoNCELoss(temperature=0.07)

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_auroc = -1.0
    best_state_dict = None

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0

        for batch in train_loader:
            features = batch['features'].to(device)
            mask = batch['attention_mask'].to(device)
            aim_labels = batch['aimbot_labels'].to(device)
            elo_labels = batch['elo_labels'].to(device)
            elo_mask = batch['elo_mask'].to(device)
            p_ids = batch['player_ids'].to(device)

            optimizer.zero_grad()
            aim_prob, emb, elo_pred = model(features, attention_mask=mask)

            loss_aim = criterion_aim(aim_prob, aim_labels)
            loss_con = criterion_con(emb, p_ids)
            loss_elo = masked_smooth_l1_loss(elo_pred, elo_labels, elo_mask)

            # Thesis Reference: Chapter 3, Equation (14) — Multi-Task Composite Loss
            loss = loss_aim + 0.5 * loss_con + 0.2 * loss_elo
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()

        scheduler.step()

        # Validate
        val_metrics, *_ = evaluate_model_on_loader(model, val_loader, device)
        if val_metrics['AUROC'] > best_val_auroc:
            best_val_auroc = val_metrics['AUROC']
            best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    # Restore best checkpoint
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    return model


def plot_ablation_results(df_results: pd.DataFrame, save_path: str = "reports/ablation_study.png"):
    """
    Renders a publication-ready 3-panel comparison figure visualizing the ablation findings.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), dpi=300)
    
    config_names = [name.replace(" (", "\n(") for name in df_results.index]
    x = np.arange(len(config_names))
    width = 0.35

    # Panel 1: AUROC & AUPRC Comparison
    ax1 = axes[0]
    rects1 = ax1.bar(x - width/2, df_results['AUROC'], width, label='AUROC', color='#1f77b4', alpha=0.9)
    rects2 = ax1.bar(x + width/2, df_results['AUPRC'], width, label='AUPRC', color='#2ca02c', alpha=0.9)
    ax1.set_ylabel('Metric Score [0.0 - 1.0]', fontsize=11, fontweight='bold')
    ax1.set_title('Detection Discrimination (AUROC vs. AUPRC)', fontsize=12, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(config_names, fontsize=8.5)
    ax1.set_ylim([0.0, 1.08])
    ax1.legend(loc='lower left', fontsize=10)
    ax1.grid(True, linestyle=':', alpha=0.6)

    # Add value annotations on bars
    for r in rects1:
        h = r.get_height()
        ax1.annotate(f'{h:.2f}', xy=(r.get_x() + r.get_width() / 2, h),
                     xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=7.5)
    for r in rects2:
        h = r.get_height()
        ax1.annotate(f'{h:.2f}', xy=(r.get_x() + r.get_width() / 2, h),
                     xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=7.5)

    # Panel 2: False Positive Rate @ 95% Sensitivity (Lower is Better)
    ax2 = axes[1]
    fpr_vals = df_results['FPR_at_95_TPR'] * 100.0  # Percentage
    bar_colors = ['#2ca02c' if f <= 5.0 else '#d62728' for f in fpr_vals]
    rects_fpr = ax2.bar(x, fpr_vals, width=0.5, color=bar_colors, alpha=0.85)
    ax2.axhline(5.0, color='darkorange', linestyle='--', lw=1.5, label='5% Max Acceptable False Ban Ceiling')
    ax2.set_ylabel('FPR @ 95% TPR (%)', fontsize=11, fontweight='bold')
    ax2.set_title('False Positive Rate at 95% Sensitivity\n(Competitive Esports False-Ban Risk)', fontsize=12, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(config_names, fontsize=8.5)
    ax2.legend(loc='upper left', fontsize=9.5)
    ax2.grid(True, linestyle=':', alpha=0.6)

    for r in rects_fpr:
        h = r.get_height()
        ax2.annotate(f'{h:.1f}%', xy=(r.get_x() + r.get_width() / 2, h),
                     xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8, fontweight='bold')

    # Panel 3: Relative Degradation Delta (AUROC Drop from Full Model)
    ax3 = axes[2]
    full_auroc = df_results['AUROC'].iloc[0]  # first row is the full model
    delta_auroc = (full_auroc - df_results['AUROC']) * 100.0  # Percentage points drop
    rects_delta = ax3.bar(x, delta_auroc, width=0.5, color='#e377c2', alpha=0.85)
    ax3.set_ylabel('AUROC Performance Drop (% pts)', fontsize=11, fontweight='bold')
    ax3.set_title('Biomechanical Contribution\n(AUROC Degradation when Invariant Removed)', fontsize=12, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(config_names, fontsize=8.5)
    ax3.grid(True, linestyle=':', alpha=0.6)

    for r in rects_delta:
        h = r.get_height()
        sign = "-" if h > 0 else ""
        ax3.annotate(f'{sign}{abs(h):.1f}%', xy=(r.get_x() + r.get_width() / 2, max(h, 0)),
                     xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8, fontweight='bold')

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    logging.info(f"Publication figure saved to: {save_path}")


def run_ablation_study(
    data_dir: str = "data/processed_parquet",
    epochs: int = 12,
    batch_size: int = 16,
    seed: int = 42
):
    """
    Executes the systematic ablation study across all configurations in ABLATION_CONFIGS.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Executing Feature Ablation Study on {device} (Epochs per config: {epochs}, Seed: {seed})")

    results = {}

    for config_name, conf in ABLATION_CONFIGS.items():
        feature_cols = conf['features']
        n_features = len(feature_cols)
        logging.info(f"\n{'='*70}\n[ABLATION] Testing: {config_name}\nFeatures ({n_features}): {feature_cols}\nHypothesis: {conf['hypothesis']}\n{'='*70}")

        # Strict reproducible split with identical seed across all configurations
        train_loader, val_loader, test_loader = create_partitioned_dataloaders(
            data_dir=data_dir,
            train_ratio=0.70,
            val_ratio=0.15,
            test_ratio=0.15,
            batch_size=batch_size,
            seed=seed,
            feature_cols=feature_cols,
            samples_per_player=4
        )

        model = train_ablation_model(
            train_loader=train_loader,
            val_loader=val_loader,
            feature_dim=n_features,
            epochs=epochs,
            lr=1e-4,
            device=device
        )

        # Evaluate on test set
        metrics, y_true, y_pred, *_ = evaluate_model_on_loader(model, test_loader, device)
        logging.info(
            f"--> Result for {config_name}: "
            f"AUROC={metrics['AUROC']:.4f} | AUPRC={metrics['AUPRC']:.4f} | "
            f"Acc={metrics['Accuracy']*100:.1f}% | F1={metrics['F1-Score']:.4f} | "
            f"FPR@95%={metrics['FPR_at_95_TPR']*100:.2f}% | ELO MAE={metrics['ELO_MAE']:.1f}"
        )

        results[config_name] = {
            'Channels': n_features,
            'AUROC': metrics['AUROC'],
            'AUPRC': metrics['AUPRC'],
            'Accuracy': metrics['Accuracy'] * 100.0,
            'F1-Score': metrics['F1-Score'],
            'FPR_at_95_TPR': metrics['FPR_at_95_TPR'],
            'ELO_MAE': metrics['ELO_MAE'],
            'Hypothesis': conf['hypothesis']
        }

    df_results = pd.DataFrame(results).T
    os.makedirs("reports", exist_ok=True)
    csv_path = "reports/ablation_study_summary.csv"
    df_results.to_csv(csv_path)

    print("\n" + "=" * 90)
    print("THESIS FEATURE ABLATION STUDY RESULTS (Task 4.2)")
    print("=" * 90)
    print(df_results[['Channels', 'AUROC', 'AUPRC', 'Accuracy', 'F1-Score', 'FPR_at_95_TPR', 'ELO_MAE']].to_string())
    print("=" * 90)
    print(f"[+] Detailed tabular report saved to: {csv_path}")

    # Generate multi-panel publication plot
    plot_ablation_results(df_results, save_path="reports/ablation_study.png")

    return df_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Feature Ablation Study for CS2 Trajectory Transformer")
    parser.add_argument("--data_dir", type=str, default="data/processed_parquet", help="Path to processed parquet data")
    parser.add_argument("--epochs", type=int, default=12, help="Number of epochs per ablation configuration")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible partitions")
    args = parser.parse_args()

    run_ablation_study(
        data_dir=args.data_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        seed=args.seed
    )
