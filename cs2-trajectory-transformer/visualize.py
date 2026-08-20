"""
Publication-Ready Visualizations for Thesis Manuscript:
1. ROC & Precision-Recall Curves
2. t-SNE & UMAP Latent Biometric Space Embeddings (Skill Clusters & Cheaters)
3. 8-12 Hz Physiological Hand Tremor Power Spectral Density (PSD)
4. Trajectory Curvature Heatmap
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve, auc
from sklearn.manifold import TSNE
from typing import Optional


def plot_roc_pr_curves(
    y_true: np.ndarray, 
    y_pred: np.ndarray, 
    save_path: Optional[str] = "reports/roc_pr_curve.png"
):
    """Generates dual-panel ROC and Precision-Recall publication plots."""
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)
    
    precision, recall, _ = precision_recall_curve(y_true, y_pred)
    pr_auc = auc(recall, precision)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), dpi=300)
    
    # 1. ROC Curve
    ax1.plot(fpr, tpr, color='#1f77b4', lw=2.5, label=f'ST-Trans (AUC = {roc_auc:.3f})')
    ax1.plot([0, 1], [0, 1], color='gray', linestyle='--', lw=1.5, label='Random Chance')
    ax1.set_xlim([0.0, 1.0])
    ax1.set_ylim([0.0, 1.05])
    ax1.set_xlabel('False Positive Rate (FPR)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('True Positive Rate (TPR)', fontsize=12, fontweight='bold')
    ax1.set_title('Receiver Operating Characteristic (ROC)', fontsize=13, fontweight='bold')
    ax1.grid(True, linestyle=':', alpha=0.6)
    ax1.legend(loc="lower right", fontsize=10)
    
    # 2. Precision-Recall Curve
    ax2.plot(recall, precision, color='#2ca02c', lw=2.5, label=f'ST-Trans (AUPRC = {pr_auc:.3f})')
    ax2.set_xlim([0.0, 1.0])
    ax2.set_ylim([0.0, 1.05])
    ax2.set_xlabel('Recall / Sensitivity', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Precision (PPV)', fontsize=12, fontweight='bold')
    ax2.set_title('Precision-Recall Curve', fontsize=13, fontweight='bold')
    ax2.grid(True, linestyle=':', alpha=0.6)
    ax2.legend(loc="lower left", fontsize=10)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    return fig


def plot_tsne_embeddings(
    embeddings: np.ndarray, 
    labels: np.ndarray, 
    label_names: dict = None,
    save_path: Optional[str] = "reports/tsne_latent_space.png"
):
    """Projects 32-dim InfoNCE latent embeddings into 2D via t-SNE."""
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
    tsne = TSNE(n_components=2, perplexity=min(30, max(5, len(embeddings)//4)), random_state=42)
    emb_2d = tsne.fit_transform(embeddings)
    
    plt.figure(figsize=(9, 7), dpi=300)
    unique_labels = np.unique(labels)
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
    
    for idx, lab in enumerate(unique_labels):
        mask = (labels == lab)
        name = label_names.get(lab, f"Class {lab}") if label_names else f"Class {lab}"
        plt.scatter(
            emb_2d[mask, 0], 
            emb_2d[mask, 1], 
            color=colors[idx], 
            label=name, 
            alpha=0.75, 
            s=40, 
            edgecolors='none'
        )
        
    plt.title('t-SNE Projection of 32-Dim InfoNCE Biometric Embeddings', fontsize=13, fontweight='bold')
    plt.xlabel('t-SNE Dimension 1', fontsize=11)
    plt.ylabel('t-SNE Dimension 2', fontsize=11)
    plt.grid(True, linestyle=':', alpha=0.5)
    plt.legend(loc='best', fontsize=10)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
