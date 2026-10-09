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
from typing import List, Tuple, Dict, Optional, Union, Any


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


def normalize_kinematic_features(
    feats: np.ndarray, 
    feature_cols: List[str], 
    global_mean: Optional[np.ndarray] = None, 
    global_std: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Standardizes or robustly scales kinematic features across ATW segments.
    Preserves absolute amplitude of superhuman jerk and near-zero tremor without per-window collapse.
    """
    feats = feats.copy()
    if global_mean is not None and global_std is not None:
        return (feats - global_mean) / (global_std + 1e-6)
    
    for col_idx, col_name in enumerate(feature_cols):
        if col_name in ['yaw', 'pitch']:
            feats[:, col_idx] = feats[:, col_idx] / np.pi
        elif col_name == 'angular_velocity':
            feats[:, col_idx] = np.log1p(np.maximum(0.0, feats[:, col_idx]))
        elif col_name in ['angular_accel', 'angular_jerk']:
            sign = np.sign(feats[:, col_idx])
            feats[:, col_idx] = sign * np.log1p(np.abs(feats[:, col_idx]))
        elif col_name == 'trajectory_curvature':
            feats[:, col_idx] = np.clip(feats[:, col_idx] / 10.0, 0.0, 5.0)
        elif col_name == 'curvature_entropy':
            feats[:, col_idx] = feats[:, col_idx] / 3.3219
        elif col_name == 'tremor_power_8_12hz':
            pass
    return feats


def save_scaler_stats(
    path: str, 
    global_mean: np.ndarray, 
    global_std: np.ndarray, 
    feature_cols: Optional[List[str]] = None
) -> None:
    """
    Saves global normalization statistics to an .npz file for deterministic inference reuse.
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    np.savez(
        path, 
        mean=global_mean.astype(np.float32), 
        std=global_std.astype(np.float32), 
        feature_cols=feature_cols or FEATURE_COLUMNS
    )


def load_scaler_stats(
    path: str
) -> Optional[Tuple[np.ndarray, np.ndarray, List[str]]]:
    """
    Loads global normalization statistics from an .npz file if present.
    """
    if not os.path.exists(path):
        return None
    try:
        data = np.load(path, allow_pickle=True)
        mean = data['mean'].astype(np.float32)
        std = data['std'].astype(np.float32)
        cols = data['feature_cols'].tolist() if 'feature_cols' in data else FEATURE_COLUMNS
        return mean, std, cols
    except Exception:
        return None


class CS2TrajectoryDataset(Dataset):
    """
    PyTorch Dataset loading preprocessed ATW Parquet trajectory segments.
    """
    def __init__(
        self, 
        file_paths: List[str], 
        feature_cols: Optional[List[str]] = None,
        max_seq_len: int = 512,
        normalize: bool = True,
        global_mean: Optional[np.ndarray] = None,
        global_std: Optional[np.ndarray] = None
    ):
        self.file_paths = file_paths
        self.feature_cols = feature_cols or FEATURE_COLUMNS
        self.max_seq_len = max_seq_len
        self.normalize = normalize
        self.global_mean = global_mean
        self.global_std = global_std
        
    def __len__(self) -> int:
        return len(self.file_paths)
        
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        fpath = self.file_paths[idx]
        df = pd.read_parquet(fpath)
        
        feats = df[self.feature_cols].values.astype(np.float32)
        seq_len = min(len(feats), self.max_seq_len)
        feats = feats[:seq_len]
        
        if self.normalize:
            feats = normalize_kinematic_features(feats, self.feature_cols, self.global_mean, self.global_std)
            
        aimbot_label = float(df['is_aimbot'].iloc[0]) if 'is_aimbot' in df.columns else 0.0

        elo_label = float(df['player_elo'].iloc[0]) / 2000.0 if 'player_elo' in df.columns else 0.75
        player_id = int(df['steamid'].iloc[0]) if 'steamid' in df.columns else 0
        match_id = str(df['match_id'].iloc[0]) if 'match_id' in df.columns else ""
        
        return {
            'features': torch.tensor(feats, dtype=torch.float32),
            'seq_len': torch.tensor(seq_len, dtype=torch.long),
            'aimbot_label': torch.tensor(aimbot_label, dtype=torch.float32),
            'elo_label': torch.tensor(elo_label, dtype=torch.float32),
            'player_id': torch.tensor(player_id, dtype=torch.long),
            'match_id': match_id
        }


def collate_trajectory_batch(batch: List[Dict[str, Any]]) -> Dict[str, Union[torch.Tensor, List[str]]]:
    """
    Collates a list of variable-length trajectory samples into a padded batch tensor.
    Returns:
        - features: [batch_size, max_batch_len, feature_dim]
        - attention_mask: [batch_size, max_batch_len] (True for valid ticks, False for padding)
        - aimbot_labels: [batch_size, 1]
        - elo_labels: [batch_size, 1]
        - player_ids: [batch_size]
        - match_ids: List[str] of length batch_size
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
    match_ids = []
    
    for i, sample in enumerate(batch):
        slen = lengths[i]
        padded_features[i, :slen] = sample['features']
        attention_mask[i, :slen] = True  # Valid positions
        aimbot_labels[i, 0] = sample['aimbot_label']
        elo_labels[i, 0] = sample['elo_label']
        player_ids[i] = sample['player_id']
        match_ids.append(str(sample.get('match_id', '')))
        
    return {
        'features': padded_features,
        'attention_mask': attention_mask,
        'aimbot_labels': aimbot_labels,
        'elo_labels': elo_labels,
        'player_ids': player_ids,
        'match_ids': match_ids
    }


def compute_dataset_statistics(
    file_paths: List[str], 
    feature_cols: Optional[List[str]] = None,
    max_samples: int = 1000
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes global mean and standard deviation across training trajectory files.
    """
    cols = feature_cols or FEATURE_COLUMNS
    all_feats = []
    sample_files = file_paths[:max_samples]
    for f in sample_files:
        try:
            df = pd.read_parquet(f, columns=cols)
            all_feats.append(df[cols].values)
        except Exception:
            continue
            
    if not all_feats:
        return np.zeros(len(cols), dtype=np.float32), np.ones(len(cols), dtype=np.float32)
        
    concat = np.concatenate(all_feats, axis=0)
    mean = np.mean(concat, axis=0).astype(np.float32)
    std = np.std(concat, axis=0).astype(np.float32)
    std = np.maximum(std, 1e-4)
    return mean, std


def partition_dataset_files(
    data_dir: str,
    train_ratio: float = 0.80,
    val_ratio: float = 0.10,
    test_ratio: float = 0.10,
    seed: int = 42
) -> Tuple[List[str], List[str], List[str]]:
    """
    Partitions Parquet files by connected components of Match-ID and Player-ID
    to strictly prevent data leakage across splits.
    Guarantees:
      P_train ∩ P_test = ∅
      M_train ∩ M_test = ∅
      
    Returns:
    --------
    (train_files, val_files, test_files)
    """
    files = glob.glob(os.path.join(data_dir, "**", "*.parquet"), recursive=True)
    if not files:
        files = glob.glob(os.path.join(data_dir, "*.parquet"))
    files = sorted(list(set(files)))
    if not files:
        raise FileNotFoundError(f"No parquet files found in {data_dir}")
        
    records = []
    for f in files:
        bname = os.path.basename(f)
        parts = bname.split('_p')
        match_id = parts[0]
        steamid = parts[1].split('_seg')[0] if len(parts) > 1 else '0'
        records.append({'fpath': f, 'match_id': match_id, 'steamid': steamid})
        
    df_meta = pd.DataFrame(records)
    
    # Build connected clusters of matches and players
    from collections import defaultdict
    match_to_players = defaultdict(set)
    player_to_matches = defaultdict(set)
    for _, row in df_meta.iterrows():
        match_to_players[row['match_id']].add(row['steamid'])
        player_to_matches[row['steamid']].add(row['match_id'])
        
    visited_matches = set()
    clusters = []
    all_matches = sorted(df_meta['match_id'].unique())
    for m in all_matches:
        if m in visited_matches:
            continue
        c_matches = set()
        c_players = set()
        queue = [m]
        while queue:
            curr_m = queue.pop(0)
            if curr_m in c_matches:
                continue
            c_matches.add(curr_m)
            visited_matches.add(curr_m)
            for p in match_to_players[curr_m]:
                c_players.add(p)
                for next_m in player_to_matches[p]:
                    if next_m not in c_matches and next_m not in visited_matches:
                        queue.append(next_m)
        clusters.append((c_matches, c_players))
        
    # Classify clusters by ground-truth match labels (clean vs cheater)
    match_labels = {}
    for m, group in df_meta.groupby('match_id'):
        sample_f = group['fpath'].iloc[0]
        try:
            import pyarrow.parquet as pq
            tbl = pq.read_table(sample_f, columns=['is_aimbot'])
            match_labels[m] = int(tbl['is_aimbot'][0].as_py())
        except Exception:
            match_labels[m] = 1 if ('cheater' in sample_f.lower() or 'cheat' in sample_f.lower()) else 0

    clean_clusters = [c for c in clusters if all(match_labels.get(m, 0) == 0 for m in c[0])]
    cheat_clusters = [c for c in clusters if any(match_labels.get(m, 0) == 1 for m in c[0])]

    def _split_cluster_group(group_list: List, tr_r: float, va_r: float, seed_val: int) -> Tuple[List, List, List]:
        if not group_list:
            return [], [], []
        r = np.random.default_rng(seed_val)
        indices = np.arange(len(group_list))
        r.shuffle(indices)
        n = len(group_list)
        if n >= 3:
            n_tr = max(1, int(n * tr_r))
            n_va = max(1, int(n * va_r))
            if n_tr + n_va >= n:
                n_tr = max(1, n - 2)
                n_va = 1
            return [group_list[i] for i in indices[:n_tr]], [group_list[i] for i in indices[n_tr:n_tr+n_va]], [group_list[i] for i in indices[n_tr+n_va:]]
        elif n == 2:
            return [group_list[indices[0]]], [], [group_list[indices[1]]]
        else:
            return [group_list[0]], [], []

    # Stratified cluster partitioning across classes
    if len(clean_clusters) >= 3 and len(cheat_clusters) >= 3:
        tr_clean, val_clean, test_clean = _split_cluster_group(clean_clusters, train_ratio, val_ratio, seed)
        tr_cheat, val_cheat, test_cheat = _split_cluster_group(cheat_clusters, train_ratio, val_ratio, seed + 1)
        train_clusters = tr_clean + tr_cheat
        val_clusters = val_clean + val_cheat
        test_clusters = test_clean + test_cheat
        
        train_matches = set().union(*[c[0] for c in train_clusters])
        val_matches = set().union(*[c[0] for c in val_clusters])
        test_matches = set().union(*[c[0] for c in test_clusters])
        
        train_files = df_meta[df_meta['match_id'].isin(train_matches)]['fpath'].tolist()
        val_files = df_meta[df_meta['match_id'].isin(val_matches)]['fpath'].tolist()
        test_files = df_meta[df_meta['match_id'].isin(test_matches)]['fpath'].tolist()
    elif len(clusters) >= 3:
        rng = np.random.default_rng(seed)
        cluster_indices = np.arange(len(clusters))
        rng.shuffle(cluster_indices)
        n_c = len(clusters)
        n_train = max(1, int(n_c * train_ratio))
        n_val = max(1, int(n_c * val_ratio))
        if n_train + n_val >= n_c:
            n_train = max(1, n_c - 2)
            n_val = 1
            
        train_clusters = [clusters[i] for i in cluster_indices[:n_train]]
        val_clusters = [clusters[i] for i in cluster_indices[n_train:n_train + n_val]]
        test_clusters = [clusters[i] for i in cluster_indices[n_train + n_val:]]
        
        train_matches = set().union(*[c[0] for c in train_clusters])
        val_matches = set().union(*[c[0] for c in val_clusters])
        test_matches = set().union(*[c[0] for c in test_clusters])
        
        train_files = df_meta[df_meta['match_id'].isin(train_matches)]['fpath'].tolist()
        val_files = df_meta[df_meta['match_id'].isin(val_matches)]['fpath'].tolist()
        test_files = df_meta[df_meta['match_id'].isin(test_matches)]['fpath'].tolist()
    else:
        # Fallback for datasets where graph connectivity creates fewer than 3 disjoint components:
        # Search match allocations that produce non-empty partitions while strictly enforcing:
        # M_train ∩ M_test = ∅ AND P_train ∩ P_test = ∅
        unique_matches = np.array(sorted(df_meta['match_id'].unique()))
        n_m = len(unique_matches)
        if n_m < 3:
            raise ValueError(
                f"Strict joint zero-leakage partitioning requires at least 3 distinct matches, "
                f"but found only {n_m}. Cannot create disjoint Train, Val, and Test partitions."
            )
            
        found_split = False
        for attempt in range(100):
            rng.shuffle(unique_matches)
            n_train = max(1, int(n_m * train_ratio))
            n_val = max(1, int(n_m * val_ratio))
            if n_train + n_val >= n_m:
                n_train = max(1, n_m - 2)
                n_val = 1
                
            train_m_set = set(unique_matches[:n_train])
            val_m_set = set(unique_matches[n_train:n_train + n_val])
            test_m_set = set(unique_matches[n_train + n_val:])
            
            # Enforce zero-leakage player filtering across match partitions
            train_df = df_meta[df_meta['match_id'].isin(train_m_set)]
            train_p = set(train_df['steamid'].unique())
            
            val_df = df_meta[df_meta['match_id'].isin(val_m_set) & (~df_meta['steamid'].isin(train_p))]
            val_p = set(val_df['steamid'].unique())
            
            test_df = df_meta[df_meta['match_id'].isin(test_m_set) & (~df_meta['steamid'].isin(train_p | val_p))]
            
            if len(val_df) > 0 and len(test_df) > 0:
                train_files = train_df['fpath'].tolist()
                val_files = val_df['fpath'].tolist()
                test_files = test_df['fpath'].tolist()
                found_split = True
                break
                
        if not found_split:
            raise ValueError(
                "Strict joint zero-leakage partitioning constraint violated: "
                "M_train ∩ M_test = ∅ AND P_train ∩ P_test = ∅. "
                "The match-player connectivity graph is too densely connected across the current matches "
                "to yield disjoint non-empty partitions. Additional independent matches are required."
            )

    return train_files, val_files, test_files


def create_partitioned_dataloaders(
    data_dir: str,
    train_ratio: float = 0.80,
    val_ratio: float = 0.10,
    test_ratio: float = 0.10,
    batch_size: int = 32,
    seed: int = 42,
    feature_cols: Optional[List[str]] = None,
    max_seq_len: int = 512,
    use_global_norm: bool = False,
    scaler_save_path: Optional[str] = None,
    scaler_load_path: Optional[str] = None
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Partitions dataset by connected components of Match-ID and Player-ID to strictly prevent data leakage.
    Guarantees:
      P_train ∩ P_test = ∅
      M_train ∩ M_test = ∅
    """
    train_files, val_files, test_files = partition_dataset_files(
        data_dir,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed
    )

        
    global_mean, global_std = None, None
    if scaler_load_path and os.path.exists(scaler_load_path):
        loaded = load_scaler_stats(scaler_load_path)
        if loaded is not None:
            global_mean, global_std, _ = loaded
    elif use_global_norm and train_files:
        global_mean, global_std = compute_dataset_statistics(train_files, feature_cols=feature_cols)
        if scaler_save_path:
            save_scaler_stats(scaler_save_path, global_mean, global_std, feature_cols=feature_cols)
        
    train_ds = CS2TrajectoryDataset(train_files, feature_cols=feature_cols, max_seq_len=max_seq_len, global_mean=global_mean, global_std=global_std)
    val_ds = CS2TrajectoryDataset(val_files, feature_cols=feature_cols, max_seq_len=max_seq_len, global_mean=global_mean, global_std=global_std)
    test_ds = CS2TrajectoryDataset(test_files, feature_cols=feature_cols, max_seq_len=max_seq_len, global_mean=global_mean, global_std=global_std)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_trajectory_batch)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_trajectory_batch)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_trajectory_batch)
    
    return train_loader, val_loader, test_loader

