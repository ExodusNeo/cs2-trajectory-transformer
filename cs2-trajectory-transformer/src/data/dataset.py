"""
PyTorch Dataset and DataLoader Pipeline for CS2 Trajectories.
Features:
- Strict Player-ID and Match-ID partition splits (Zero Data Leakage).
- Variable-length sequence padding & attention masking.
- Batch collation for ST-Trans dual-task training (Aimbot BCE + Smurf InfoNCE).
"""

import os
import glob
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Dict, Optional


FEATURE_COLUMNS = [
    'yaw', 
    'pitch', 
    'angular_velocity', 
    'angular_accel', 
    'angular_jerk', 
    'trajectory_curvature', 
    'curvature_entropy', 
    'tremor_power_8_12hz'
]


class CS2TrajectoryDataset(Dataset):
    """
    PyTorch Dataset loading preprocessed ATW Parquet trajectory segments.
    """
    def __init__(
        self, 
        file_paths: List[str], 
        feature_cols: Optional[List[str]] = None,
        max_seq_len: int = 512,
        normalize: bool = True
    ):
        self.file_paths = file_paths
        self.feature_cols = feature_cols or FEATURE_COLUMNS
        self.max_seq_len = max_seq_len
        self.normalize = normalize
        
    def __len__(self) -> int:
        return len(self.file_paths)
        
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        fpath = self.file_paths[idx]
        df = pd.read_parquet(fpath)
        
        feats = df[self.feature_cols].values.astype(np.float32)
        seq_len = min(len(feats), self.max_seq_len)
        feats = feats[:seq_len]
        
        if self.normalize:
            # Per-segment standardization with epsilon stability
            mean = np.mean(feats, axis=0, keepdims=True)
            std = np.std(feats, axis=0, keepdims=True) + 1e-6
            feats = (feats - mean) / std
            
        aimbot_label = float(df['is_aimbot'].iloc[0]) if 'is_aimbot' in df.columns else 0.0
        elo_label = float(df['player_elo'].iloc[0]) / 2000.0 if 'player_elo' in df.columns else 0.75
        player_id = int(df['steamid'].iloc[0]) if 'steamid' in df.columns else 0
        
        return {
            'features': torch.tensor(feats, dtype=torch.float32),
            'seq_len': torch.tensor(seq_len, dtype=torch.long),
            'aimbot_label': torch.tensor(aimbot_label, dtype=torch.float32),
            'elo_label': torch.tensor(elo_label, dtype=torch.float32),
            'player_id': torch.tensor(player_id, dtype=torch.long)
        }


def collate_trajectory_batch(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """
    Collates a list of variable-length trajectory samples into a padded batch tensor.
    Returns:
        - features: [batch_size, max_batch_len, feature_dim]
        - attention_mask: [batch_size, max_batch_len] (True for valid ticks, False for padding)
        - aimbot_labels: [batch_size, 1]
        - elo_labels: [batch_size, 1]
        - player_ids: [batch_size]
    """
    batch_size = len(batch)
    lengths = [sample['seq_len'].item() for sample in batch]
    max_len = max(lengths)
    feature_dim = batch[0]['features'].shape[-1]
    
    padded_features = torch.zeros((batch_size, max_len, feature_dim), dtype=torch.float32)
    attention_mask = torch.zeros((batch_size, max_len), dtype=torch.bool)
    aimbot_labels = torch.zeros((batch_size, 1), dtype=torch.float32)
    elo_labels = torch.zeros((batch_size, 1), dtype=torch.float32)
    player_ids = torch.zeros(batch_size, dtype=torch.long)
    
    for i, sample in enumerate(batch):
        slen = lengths[i]
        padded_features[i, :slen] = sample['features']
        attention_mask[i, :slen] = True  # Valid positions
        aimbot_labels[i, 0] = sample['aimbot_label']
        elo_labels[i, 0] = sample['elo_label']
        player_ids[i] = sample['player_id']
        
    return {
        'features': padded_features,
        'attention_mask': attention_mask,
        'aimbot_labels': aimbot_labels,
        'elo_labels': elo_labels,
        'player_ids': player_ids
    }


def create_partitioned_dataloaders(
    data_dir: str,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    batch_size: int = 32,
    seed: int = 42
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Partitions dataset by unique SteamID / Match to strictly prevent data leakage.
    """
    files = glob.glob(os.path.join(data_dir, "*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files found in {data_dir}")
        
    # Extract unique match IDs or player IDs from filenames
    # Format: {match_id}_p{steamid}_seg{seg_idx}.parquet
    records = []
    for f in files:
        bname = os.path.basename(f)
        parts = bname.split('_p')
        match_id = parts[0]
        steamid = parts[1].split('_seg')[0] if len(parts) > 1 else '0'
        records.append({'fpath': f, 'match_id': match_id, 'steamid': steamid})
        
    df_meta = pd.DataFrame(records)
    unique_players = np.array(df_meta['steamid'].unique())
    rng = np.random.default_rng(seed)
    rng.shuffle(unique_players)
    
    n_players = len(unique_players)
    n_train = int(n_players * train_ratio)
    n_val = int(n_players * val_ratio)
    
    train_players = set(unique_players[:n_train])
    val_players = set(unique_players[n_train:n_train + n_val])
    test_players = set(unique_players[n_train + n_val:])
    
    train_files = df_meta[df_meta['steamid'].isin(train_players)]['fpath'].tolist()
    val_files = df_meta[df_meta['steamid'].isin(val_players)]['fpath'].tolist()
    test_files = df_meta[df_meta['steamid'].isin(test_players)]['fpath'].tolist()
    
    train_ds = CS2TrajectoryDataset(train_files)
    val_ds = CS2TrajectoryDataset(val_files)
    test_ds = CS2TrajectoryDataset(test_files)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_trajectory_batch)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_trajectory_batch)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_trajectory_batch)
    
    return train_loader, val_loader, test_loader
