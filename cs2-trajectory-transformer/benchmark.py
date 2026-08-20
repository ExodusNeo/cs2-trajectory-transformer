"""
Automated Benchmark Suite for Thesis Research.
1. Trains ST-Trans Model & Baseline Models (Random Forest, Gradient Boosting, MLP, Bi-LSTM).
2. Generates Comparative Performance Tables (AUROC, AUPRC, Accuracy, F1, FPR@95%TPR, ELO MAE).
3. Generates Publication Figures in reports/ (ROC curves, PR curves, t-SNE latent space).
"""

import os
import sys
import glob
import logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))
from models.st_transformer import STTrajectoryTransformer
from models.losses import SupervisedInfoNCELoss, FocalLoss
from models.baselines import BiLSTMBaseline, ClassicalBaselines
from data.dataset import create_partitioned_dataloaders, FEATURE_COLUMNS
from evaluate import compute_metrics, evaluate_model_on_loader
from visualize import plot_roc_pr_curves, plot_tsne_embeddings

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')


def train_st_transformer(train_loader, val_loader, epochs: int = 15, device: torch.device = None):
    """Trains the ST-Trans model with Focal Loss and Supervised InfoNCE."""
    model = STTrajectoryTransformer(
        feature_dim=8, 
        d_model=64, 
        nhead=4, 
        num_layers=3, 
        embed_dim=32,
        dim_feedforward=256
    ).to(device)
    
    criterion_aim = FocalLoss(alpha=0.25, gamma=2.0)
    criterion_con = SupervisedInfoNCELoss(temperature=0.07)
    criterion_elo = nn.MSELoss()
    
    optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    
    best_auroc = 0.0
    best_weights = None
    
    logging.info(f"Starting ST-Trans training for {epochs} epochs...")
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        
        for batch in train_loader:
            features = batch['features'].to(device)
            mask = batch['attention_mask'].to(device)
            aim_labels = batch['aimbot_labels'].to(device)
            elo_labels = batch['elo_labels'].to(device)
            p_ids = batch['player_ids'].to(device)
            
            optimizer.zero_grad()
            aim_prob, emb, elo_pred = model(features, attention_mask=mask)
            
            loss_aim = criterion_aim(aim_prob, aim_labels)
            loss_con = criterion_con(emb, p_ids)
            loss_elo = criterion_elo(elo_pred, elo_labels)
            
            loss = loss_aim + 0.4 * loss_con + 0.2 * loss_elo
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            
        scheduler.step()
        
        # Validation
        val_metrics, _, _, _ = evaluate_model_on_loader(model, val_loader, device)
        logging.info(f"Epoch {epoch:02d}/{epochs:02d} | Train Loss: {total_loss/len(train_loader):.4f} | Val AUROC: {val_metrics['AUROC']:.4f} | Val Acc: {val_metrics['Accuracy']*100:.1f}%")
        
        if val_metrics['AUROC'] >= best_auroc:
            best_auroc = val_metrics['AUROC']
            os.makedirs("models/checkpoints", exist_ok=True)
            torch.save(model.state_dict(), "models/checkpoints/best_model.pt")
            
    # Load best checkpoint
    model.load_state_dict(torch.load("models/checkpoints/best_model.pt"))
    return model


def train_and_eval_bilstm(train_loader, test_loader, epochs: int = 15, device: torch.device = None):
    """Trains Bi-LSTM baseline."""
    model = BiLSTMBaseline(feature_dim=8, hidden_dim=32, num_layers=2).to(device)
    criterion = nn.BCELoss()
    optimizer = AdamW(model.parameters(), lr=1e-3)
    
    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            features = batch['features'].to(device)
            mask = batch['attention_mask'].to(device)
            aim_labels = batch['aimbot_labels'].to(device)
            
            optimizer.zero_grad()
            preds = model(features, attention_mask=mask)
            loss = criterion(preds, aim_labels)
            loss.backward()
            optimizer.step()
            
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for batch in test_loader:
            features = batch['features'].to(device)
            mask = batch['attention_mask'].to(device)
            aim_labels = batch['aimbot_labels'].to(device)
            preds = model(features, attention_mask=mask)
            all_preds.extend(preds.cpu().numpy().flatten())
            all_targets.extend(aim_labels.cpu().numpy().flatten())
            
    return compute_metrics(np.array(all_targets), np.array(all_preds))


def train_and_eval_tabular_baselines(train_loader, test_loader):
    """Extracts statistical summary features to train Random Forest, Gradient Boosting, and MLP."""
    def extract_tabular(loader):
        X_list, y_list = [], []
        for batch in loader:
            feats = batch['features'].numpy()
            masks = batch['attention_mask'].numpy()
            labels = batch['aimbot_labels'].numpy().flatten()
            
            for i in range(len(feats)):
                valid = feats[i, masks[i]]
                if len(valid) == 0:
                    continue
                # Feature statistics (mean, std, max, min for all 8 dims = 32 tabular features)
                f_mean = np.mean(valid, axis=0)
                f_std = np.std(valid, axis=0)
                f_max = np.max(valid, axis=0)
                f_min = np.min(valid, axis=0)
                stat_vec = np.concatenate([f_mean, f_std, f_max, f_min])
                X_list.append(stat_vec)
                y_list.append(labels[i])
        return np.array(X_list), np.array(y_list)
        
    X_train, y_train = extract_tabular(train_loader)
    X_test, y_test = extract_tabular(test_loader)
    
    baselines = ClassicalBaselines()
    baselines.fit_all(X_train, y_train)
    preds = baselines.predict_probabilities(X_test)
    
    results = {}
    for model_name, prob in preds.items():
        results[model_name] = compute_metrics(y_test, prob)
    return results


def run_full_benchmark():
    """Executes full comparative benchmark study and outputs tables and figures."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Running benchmark on device: {device}")
    
    data_dir = "data/processed_parquet"
    train_loader, val_loader, test_loader = create_partitioned_dataloaders(
        data_dir=data_dir,
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15,
        batch_size=16
    )
    
    # 1. Train & Evaluate ST-Trans
    st_model = train_st_transformer(train_loader, val_loader, epochs=15, device=device)
    st_metrics, y_test, y_pred, embeddings = evaluate_model_on_loader(st_model, test_loader, device)
    
    # 2. Train & Evaluate Bi-LSTM Baseline
    lstm_metrics = train_and_eval_bilstm(train_loader, test_loader, epochs=15, device=device)
    
    # 3. Train & Evaluate Tabular Baselines (RF, Gradient Boosting, MLP)
    tabular_results = train_and_eval_tabular_baselines(train_loader, test_loader)
    
    # 4. Compile Comparison Table
    all_results = {
        'Spatial-Temporal Transformer (ST-Trans, Ours)': st_metrics,
        'Bidirectional LSTM (Bi-LSTM)': lstm_metrics,
        'Gradient Boosting Classifier': tabular_results['Gradient Boosting'],
        'Random Forest Classifier': tabular_results['Random Forest'],
        'Multi-Layer Perceptron (MLP)': tabular_results['MLP']
    }
    
    df_results = pd.DataFrame(all_results).T[['AUROC', 'AUPRC', 'Accuracy', 'F1-Score', 'FPR_at_95_TPR']]
    df_results['Accuracy'] = df_results['Accuracy'] * 100.0
    
    os.makedirs("reports", exist_ok=True)
    csv_path = "reports/benchmark_summary.csv"
    df_results.to_csv(csv_path)
    
    print("\n" + "=" * 80)
    print("THESIS BENCHMARK RESULTS: ST-TRANS VS. CLASSICAL BASELINES")
    print("=" * 80)
    print(df_results.to_string())
    print("=" * 80)
    
    # 5. Generate Publication Plots
    plot_roc_pr_curves(y_test, y_pred, save_path="reports/roc_pr_curve.png")
    
    # Extract labels for t-SNE
    test_labels = y_test
    label_map = {0: 'Clean Human Motor Aim', 1: 'Algorithmic / DMA Aimbot'}
    plot_tsne_embeddings(embeddings, test_labels, label_names=label_map, save_path="reports/tsne_latent_space.png")
    
    logging.info("Benchmark complete! Figures and tables saved in reports/.")


if __name__ == "__main__":
    run_full_benchmark()
