"""
PyTorch Dataset and DataLoader Pipeline for CS2 Trajectories.
Features:
- Strict Player-ID and Match-ID partition splits (Zero Data Leakage).
- Variable-length sequence padding & attention masking.
- Batch collation for ST-Trans dual-task training (Aimbot BCE + Smurf InfoNCE).
"""

import os
import glob
import logging
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, Sampler
from typing import Iterator, List, Tuple, Dict, Optional, Union, Any

from features.kinematics import MODEL_FEATURE_COLUMNS


ELO_SCALE = 2000.0  # Thesis Eq (18): ELO targets are regressed as elo / 2000


FEATURE_COLUMNS = MODEL_FEATURE_COLUMNS


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
        elif col_name == 'aim_error':
            feats[:, col_idx] = feats[:, col_idx] / np.pi
        elif col_name == 'aim_error_rate':
            sign = np.sign(feats[:, col_idx])
            feats[:, col_idx] = sign * np.log1p(np.abs(feats[:, col_idx]))
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

        raw_elo = float(df['player_elo'].iloc[0]) if 'player_elo' in df.columns else float('nan')
        elo_valid = bool(np.isfinite(raw_elo))
        elo_label = raw_elo / ELO_SCALE if elo_valid else 0.0
        player_id = int(df['steamid'].iloc[0]) if 'steamid' in df.columns else 0
        match_id = str(df['match_id'].iloc[0]) if 'match_id' in df.columns else ""
        
        return {
            'features': torch.tensor(feats, dtype=torch.float32),
            'seq_len': torch.tensor(seq_len, dtype=torch.long),
            'aimbot_label': torch.tensor(aimbot_label, dtype=torch.float32),
            'elo_label': torch.tensor(elo_label, dtype=torch.float32),
            'elo_valid': torch.tensor(elo_valid, dtype=torch.bool),
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
        - elo_labels: [batch_size, 1] (elo / 2000; 0.0 where unknown)
        - elo_mask: [batch_size, 1] (True where the player's ELO is known)
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
    elo_mask = torch.zeros((batch_size, 1), dtype=torch.bool)
    player_ids = torch.zeros(batch_size, dtype=torch.long)
    match_ids = []
    
    for i, sample in enumerate(batch):
        slen = lengths[i]
        padded_features[i, :slen] = sample['features']
        attention_mask[i, :slen] = True  # Valid positions
        aimbot_labels[i, 0] = sample['aimbot_label']
        elo_labels[i, 0] = sample['elo_label']
        elo_mask[i, 0] = bool(sample.get('elo_valid', True))
        player_ids[i] = sample['player_id']
        match_ids.append(str(sample.get('match_id', '')))
        
    return {
        'features': padded_features,
        'attention_mask': attention_mask,
        'aimbot_labels': aimbot_labels,
        'elo_labels': elo_labels,
        'elo_mask': elo_mask,
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


def _read_file_label(fpath: str) -> int:
    """Reads the window-level 'is_aimbot' label (0/1) from a Parquet file; 0 if unreadable."""
    try:
        import pyarrow.parquet as pq
        tbl = pq.read_table(fpath, columns=['is_aimbot'])
        return int(tbl['is_aimbot'][0].as_py())
    except Exception as e:
        logging.warning(f"Could not read is_aimbot from {fpath} ({e}); treating as clean.")
        return 0


def _components_are_degenerate(
    clusters: List,
    clean_clusters: List,
    cheat_clusters: List,
    match_labels: Dict[str, int],
    match_file_counts: Dict[str, int],
    val_ratio: float,
    test_ratio: float
) -> bool:
    """
    True when component-level splitting cannot give a usable held-out split:
    (a) the largest component holds more than 1 - (val+test)/2 of all windows, or
    (b) a class has >= 3 matches but fewer than 3 components (e.g. one giant clean component).
    """
    total = sum(match_file_counts.values())
    largest = max(sum(match_file_counts.get(m, 0) for m in c[0]) for c in clusters)
    if len(clusters) == 1:
        return len(match_labels) >= 3
    if total > 0 and largest / total > 1.0 - (val_ratio + test_ratio) / 2.0:
        return True
    for label, comps in ((0, clean_clusters), (1, cheat_clusters)):
        n_matches = sum(1 for v in match_labels.values() if v == label)
        if 0 < len(comps) < 3 and n_matches >= 3:
            return True
    return False


def _partition_by_match_drop(
    df_meta: pd.DataFrame,
    match_labels: Dict[str, int],
    train_ratio: float,
    val_ratio: float,
    seed: int
) -> Tuple[List[str], List[str], List[str]]:
    """
    Zero-leakage split for densely connected match-player graphs.

    1. Matches are split per class (clean / cheater) into train/val/test by match count.
    2. Test keeps every window of its matches. Val drops windows of test players.
       Train drops windows of test and val players.
    Result: M_i and M_j disjoint and P_i and P_j disjoint for all splits; only windows are discarded.
    """
    rng = np.random.default_rng(seed)
    split_matches: Dict[str, List[str]] = {'train': [], 'val': [], 'test': []}
    for label in (0, 1):
        ms = sorted(m for m, v in match_labels.items() if v == label)
        if not ms:
            continue
        ms = [ms[i] for i in rng.permutation(len(ms))]
        n = len(ms)
        if n >= 3:
            n_va = max(1, int(round(n * val_ratio)))
            n_te = max(1, int(round(n * (1.0 - train_ratio - val_ratio))))
            n_tr = n - n_va - n_te
            if n_tr < 1:
                n_tr, n_va, n_te = n - 2, 1, 1
            split_matches['train'] += ms[:n_tr]
            split_matches['val'] += ms[n_tr:n_tr + n_va]
            split_matches['test'] += ms[n_tr + n_va:]
        elif n == 2:
            split_matches['train'].append(ms[0])
            split_matches['test'].append(ms[1])
        else:
            split_matches['train'].append(ms[0])

    def rows(split: str) -> pd.DataFrame:
        return df_meta[df_meta['match_id'].isin(split_matches[split])]

    test_rows = rows('test')
    test_players = set(test_rows['steamid'])
    val_all = rows('val')
    val_rows = val_all[~val_all['steamid'].isin(test_players)]
    blocked = test_players | set(val_rows['steamid'])
    train_all = rows('train')
    train_rows = train_all[~train_all['steamid'].isin(blocked)]

    dropped = (len(val_all) - len(val_rows)) + (len(train_all) - len(train_rows))
    logging.info(
        f"match_drop partition: train={len(train_rows)}, val={len(val_rows)}, test={len(test_rows)} windows; "
        f"dropped {dropped} windows of players shared with a higher-priority split."
    )

    splits = {'train': train_rows, 'val': val_rows, 'test': test_rows}
    for a_name, b_name in (('train', 'val'), ('train', 'test'), ('val', 'test')):
        a_df, b_df = splits[a_name], splits[b_name]
        assert set(a_df['match_id']).isdisjoint(b_df['match_id']), f"Data leakage: {a_name}/{b_name} share Match IDs!"
        assert set(a_df['steamid']).isdisjoint(b_df['steamid']), f"Data leakage: {a_name}/{b_name} share Player IDs!"
    if any(len(df) == 0 for df in splits.values()):
        raise ValueError(
            f"match_drop partition yielded an empty split (train={len(train_rows)}, val={len(val_rows)}, "
            f"test={len(test_rows)}). More independent matches are required."
        )
    return train_rows['fpath'].tolist(), val_rows['fpath'].tolist(), test_rows['fpath'].tolist()


class PlayerBalancedBatchSampler(Sampler):
    """
    P x K batch sampler for Supervised InfoNCE (Thesis Eq 17).

    Random batches of 32 windows from hundreds of players almost never contain two windows
    of the same player, so the contrastive loss has no positive pairs and returns 0. Each
    batch here holds `players_per_batch` players with up to `samples_per_player` windows each.
    Players with a single window still appear (as negatives for the others).

    Player IDs are parsed from filenames '<match>_p<steamid>_seg<k>.parquet', which is the
    same convention partition_dataset_files uses.
    """

    def __init__(self, file_paths: List[str], players_per_batch: int = 8, samples_per_player: int = 4, seed: int = 42):
        self.players_per_batch = max(1, players_per_batch)
        self.samples_per_player = max(1, samples_per_player)
        self.seed = seed
        self.epoch = 0
        by_player: Dict[str, List[int]] = {}
        for idx, f in enumerate(file_paths):
            bname = os.path.basename(f)
            pid = bname.split('_p')[1].split('_seg')[0] if '_p' in bname else bname
            by_player.setdefault(pid, []).append(idx)
        self.by_player = by_player
        self.num_files = len(file_paths)

    def _batches(self) -> List[List[int]]:
        rng = np.random.default_rng(self.seed + self.epoch)
        pools = {p: list(rng.permutation(ix)) for p, ix in self.by_player.items()}
        batches: List[List[int]] = []
        while pools:
            players = list(pools.keys())
            chosen = [players[i] for i in rng.permutation(len(players))[:self.players_per_batch]]
            batch: List[int] = []
            for p in chosen:
                take, pools[p] = pools[p][:self.samples_per_player], pools[p][self.samples_per_player:]
                batch.extend(int(i) for i in take)
                if not pools[p]:
                    del pools[p]
            batches.append(batch)
        return batches

    def __iter__(self) -> Iterator[List[int]]:
        batches = self._batches()
        self.epoch += 1
        return iter(batches)

    def __len__(self) -> int:
        return len(self._batches())


def partition_dataset_files(
    data_dir: str,
    train_ratio: float = 0.80,
    val_ratio: float = 0.10,
    test_ratio: float = 0.10,
    seed: int = 42,
    strategy: str = "auto"
) -> Tuple[List[str], List[str], List[str]]:
    """
    Partitions Parquet files by connected components of Match-ID and Player-ID
    to strictly prevent data leakage across splits.

    strategy:
      - "components": whole connected components are atomic units (original behaviour).
      - "match_drop": matches are split by class, then any window whose player already
        appears in a higher-priority split is dropped (priority test > val > train).
        Still zero-leakage (P and M pairwise disjoint), at the cost of discarding some
        windows, mostly from train.
      - "auto" (default): "components", unless the component graph is degenerate. That
        happens when one giant component holds most windows (common when the same high-ELO
        or pro players meet repeatedly) or when a class has many matches but fewer than 3
        components. In that case it switches to "match_drop" and logs why.
    Guarantees:
      P_train ∩ P_val = ∅, P_train ∩ P_test = ∅, P_val ∩ P_test = ∅
      M_train ∩ M_val = ∅, M_train ∩ M_test = ∅, M_val ∩ M_test = ∅

    Connected components are preserved intact as atomic units.
    Never silently returns an empty train, validation, or test partition.

    Class Representation Guarantees by Component Counts (clean=C, cheater=X):
    - C >= 3 and X >= 3:
      Train, Validation, and Test all contain BOTH clean and cheater components.
      Enables complete dual-class ROC-based operational threshold calibration.
    - C >= 3 and X == 2:
      Train and Test contain BOTH classes (1 cheater component each).
      Validation contains CLEAN components ONLY (permits zero-false-alarm threshold
      calibration, while validation TPR is unmeasured until held-out test evaluation).
    - C == 2 and X >= 3:
      Train and Test contain BOTH classes (1 clean component each).
      Validation contains CHEATER components ONLY (zero clean validation samples;
      FPR-based threshold calibration is unavailable on validation).
    - C == 2 and X == 2:
      Infeasible (4 components consumed by Train and Test, leaving Validation empty).
      Raises descriptive ValueError.
    - Single-class datasets (C == 0 or X == 0):
      Requires total components >= 3 to populate non-empty Train, Val, and Test.

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
    # A match is labeled as cheater if ANY of its extracted segments contain is_aimbot == 1.
    match_labels = {}
    for m, group in df_meta.groupby('match_id'):
        # Labels come from the Parquet 'is_aimbot' column only. Filename heuristics
        # ('cheat' in path) mislabel clean players stored under a cheaters/ folder.
        has_cheater = 0
        for sample_f in group['fpath']:
            if _read_file_label(sample_f) == 1:
                has_cheater = 1
                break
        match_labels[m] = has_cheater

    clean_clusters = [c for c in clusters if all(match_labels.get(m, 0) == 0 for m in c[0])]
    cheat_clusters = [c for c in clusters if any(match_labels.get(m, 0) == 1 for m in c[0])]
    C = len(clean_clusters)
    X = len(cheat_clusters)
    total_clusters = len(clusters)

    if total_clusters == 0:
        raise ValueError(f"No connected components found in {data_dir}. Cannot partition dataset.")

    match_file_counts = df_meta.groupby('match_id').size().to_dict()
    if strategy not in ("auto", "components", "match_drop"):
        raise ValueError(f"Unknown partition strategy '{strategy}'.")
    if strategy == "match_drop" or (strategy == "auto" and _components_are_degenerate(
            clusters, clean_clusters, cheat_clusters, match_labels, match_file_counts, val_ratio, test_ratio)):
        logging.info("Partition strategy: match_drop (connected-component graph is degenerate or match_drop requested).")
        return _partition_by_match_drop(df_meta, match_labels, train_ratio, val_ratio, seed)

    # Feasibility validation under strict zero-leakage constraints
    if C > 0 and X > 0:
        # Two-class dataset
        if C == 2 and X == 2:
            raise ValueError(
                f"Infeasible zero-leakage dataset partition: found {C} clean and {X} cheater connected components "
                f"(total={total_clusters}). Allocating both classes to Train (1 clean, 1 cheater) and held-out Test "
                f"(1 clean, 1 cheater) consumes all 4 components, leaving Validation empty. "
                f"To guarantee non-empty Train, Validation, and Test partitions with two-class representation in Train "
                f"and Test, at least 5 connected components are required (e.g., clean >= 3 and cheater >= 2, or "
                f"clean >= 2 and cheater >= 3). Cannot partition without leaving Validation empty or violating zero-leakage."
            )
        if (C < 2 and X < 3) or (X < 2 and C < 3):
            raise ValueError(
                f"Infeasible zero-leakage dataset partition: found {C} clean and {X} cheater connected components "
                f"(total={total_clusters}). Creating non-empty Train, Validation, and Test partitions under strict "
                f"zero-leakage constraints requires at least 4 components when one class has 1 component "
                f"(allocating 1 to Train and 3+ of the other class across Train, Val, Test), or at least 5 components "
                f"when both classes have at least 2 components. Found clean={C}, cheater={X}."
            )
    else:
        # Single-class dataset (pure clean or pure cheater)
        if total_clusters < 3:
            raise ValueError(
                f"Strict zero-leakage partitioning requires at least 3 connected components to form non-empty "
                f"Train, Validation, and Test partitions, but found only {total_clusters} component(s) "
                f"(clean={C}, cheater={X}). Under strict zero-leakage constraints, connected components cannot "
                f"be split across partitions. Additional independent matches are required."
            )

    def _cluster_size(cluster) -> int:
        return int(sum(match_file_counts.get(m, 0) for m in cluster[0]))

    def _split_clusters_three_way(
        cluster_list: List,
        tr_r: float,
        va_r: float,
        seed_val: int
    ) -> Tuple[List, List, List]:
        """
        Splits >= 3 clusters into non-empty Train, Val and Test, balancing by window count
        (not component count). The largest component always goes to Train so a giant
        component can never become the whole test set.
        """
        n = len(cluster_list)
        assert n >= 3, f"Requires at least 3 clusters, got {n}"
        rng = np.random.default_rng(seed_val)
        sizes = np.array([_cluster_size(c) for c in cluster_list], dtype=float)
        largest = int(np.argmax(sizes))
        others = [int(i) for i in rng.permutation(n) if i != largest]

        assignment = {0: [largest], 1: [others[0]], 2: [others[1]]}
        totals = np.array([sizes[largest], sizes[others[0]], sizes[others[1]]])
        targets = np.array([tr_r, va_r, max(1e-9, 1.0 - tr_r - va_r)]) * sizes.sum()
        for i in others[2:]:
            split = int(np.argmax((targets - totals) / targets))
            assignment[split].append(i)
            totals[split] += sizes[i]

        return tuple([cluster_list[i] for i in assignment[k]] for k in range(3))

    def _split_clusters_two_way(
        cluster_list: List,
        seed_val: int
    ) -> Tuple[List, List, List]:
        """Splits exactly 2 clusters into Train and Test (1 each), leaving Val empty for this class."""
        assert len(cluster_list) == 2, f"Requires exactly 2 clusters, got {len(cluster_list)}"
        rng = np.random.default_rng(seed_val)
        indices = np.arange(2)
        rng.shuffle(indices)
        return [cluster_list[indices[0]]], [], [cluster_list[indices[1]]]

    if C > 0 and X > 0:
        if C >= 3 and X >= 3:
            tr_clean, val_clean, test_clean = _split_clusters_three_way(clean_clusters, train_ratio, val_ratio, seed)
            tr_cheat, val_cheat, test_cheat = _split_clusters_three_way(cheat_clusters, train_ratio, val_ratio, seed + 1)
        elif C >= 3 and X == 2:
            tr_clean, val_clean, test_clean = _split_clusters_three_way(clean_clusters, train_ratio, val_ratio, seed)
            tr_cheat, val_cheat, test_cheat = _split_clusters_two_way(cheat_clusters, seed + 1)
            logging.info(
                "Cheater class has exactly 2 connected components: allocated 1 to Train and 1 to Test. "
                "Validation split contains clean samples only (FPR calibration available; validation TPR unmeasured until test evaluation)."
            )
        elif C == 2 and X >= 3:
            tr_clean, val_clean, test_clean = _split_clusters_two_way(clean_clusters, seed)
            tr_cheat, val_cheat, test_cheat = _split_clusters_three_way(cheat_clusters, train_ratio, val_ratio, seed + 1)
            logging.info(
                "Clean class has exactly 2 connected components: allocated 1 to Train and 1 to Test. "
                "Validation split contains cheater samples only (0 clean samples; FPR threshold calibration unavailable on validation)."
            )
        elif C >= 3 and X == 1:
            tr_clean, val_clean, test_clean = _split_clusters_three_way(clean_clusters, train_ratio, val_ratio, seed)
            tr_cheat, val_cheat, test_cheat = [cheat_clusters[0]], [], []
            logging.warning(
                "Cheater class has only 1 connected cluster; allocated to Train. "
                "Test partition will lack cheater samples. Held-out binary classification metrics require >= 2 cheater clusters."
            )
        elif C == 1 and X >= 3:
            tr_clean, val_clean, test_clean = [clean_clusters[0]], [], []
            tr_cheat, val_cheat, test_cheat = _split_clusters_three_way(cheat_clusters, train_ratio, val_ratio, seed + 1)
            logging.warning(
                "Clean class has only 1 connected cluster; allocated to Train. "
                "Test partition will lack clean samples."
            )
        else:
            raise ValueError(f"Infeasible cluster counts: clean={C}, cheater={X}.")

        train_clusters = tr_clean + tr_cheat
        val_clusters = val_clean + val_cheat
        test_clusters = test_clean + test_cheat
    else:
        # Single-class dataset
        train_clusters, val_clusters, test_clusters = _split_clusters_three_way(
            clusters, train_ratio, val_ratio, seed
        )

    train_matches = set().union(*[c[0] for c in train_clusters])
    val_matches = set().union(*[c[0] for c in val_clusters])
    test_matches = set().union(*[c[0] for c in test_clusters])

    train_players = set().union(*[c[1] for c in train_clusters])
    val_players = set().union(*[c[1] for c in val_clusters])
    test_players = set().union(*[c[1] for c in test_clusters])

    # Strict Zero-Data-Leakage Verification
    assert train_matches.isdisjoint(val_matches), "Data leakage: Train and Val share Match IDs!"
    assert train_matches.isdisjoint(test_matches), "Data leakage: Train and Test share Match IDs!"
    assert val_matches.isdisjoint(test_matches), "Data leakage: Val and Test share Match IDs!"
    assert train_players.isdisjoint(val_players), "Data leakage: Train and Val share Player IDs!"
    assert train_players.isdisjoint(test_players), "Data leakage: Train and Test share Player IDs!"
    assert val_players.isdisjoint(test_players), "Data leakage: Val and Test share Player IDs!"

    train_files = df_meta[df_meta['match_id'].isin(train_matches)]['fpath'].tolist()
    val_files = df_meta[df_meta['match_id'].isin(val_matches)]['fpath'].tolist()
    test_files = df_meta[df_meta['match_id'].isin(test_matches)]['fpath'].tolist()

    if len(train_files) == 0 or len(val_files) == 0 or len(test_files) == 0:
        raise ValueError(
            f"Zero-leakage partitioning yielded an empty split: "
            f"train={len(train_files)}, val={len(val_files)}, test={len(test_files)}. "
            f"Component counts: clean={C}, cheater={X}."
        )

    # Report actual split sizes and cluster allocations (approximate targets: 80/10/10)
    total_files = len(files)
    logging.info(
        f"Zero-leakage dataset partition summary (nominal targets: {train_ratio:.0%}/{val_ratio:.0%}/{test_ratio:.0%}):\n"
        f"  Train: {len(train_files)} files ({len(train_files)/total_files:.1%}), "
        f"{len(train_clusters)} clusters ({len(train_clusters)/total_clusters:.1%}), "
        f"{len(train_matches)} matches, {len(train_players)} players\n"
        f"  Val:   {len(val_files)} files ({len(val_files)/total_files:.1%}), "
        f"{len(val_clusters)} clusters ({len(val_clusters)/total_clusters:.1%}), "
        f"{len(val_matches)} matches, {len(val_players)} players\n"
        f"  Test:  {len(test_files)} files ({len(test_files)/total_files:.1%}), "
        f"{len(test_clusters)} clusters ({len(test_clusters)/total_clusters:.1%}), "
        f"{len(test_matches)} matches, {len(test_players)} players"
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
    scaler_load_path: Optional[str] = None,
    samples_per_player: int = 0,
    partition_strategy: str = "auto"
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Partitions dataset by connected components of Match-ID and Player-ID to strictly prevent data leakage.
    samples_per_player > 0 enables the P x K PlayerBalancedBatchSampler for the training loader.
    Guarantees:
      P_train ∩ P_test = ∅
      M_train ∩ M_test = ∅
    """
    train_files, val_files, test_files = partition_dataset_files(
        data_dir,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
        strategy=partition_strategy
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
    
    if samples_per_player > 0:
        # P x K batches so Supervised InfoNCE sees positive pairs (batch_size ~= P * K).
        sampler = PlayerBalancedBatchSampler(
            train_files,
            players_per_batch=max(1, batch_size // samples_per_player),
            samples_per_player=samples_per_player,
            seed=seed
        )
        train_loader = DataLoader(train_ds, batch_sampler=sampler, collate_fn=collate_trajectory_batch)
    else:
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_trajectory_batch)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_trajectory_batch)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_trajectory_batch)
    
    return train_loader, val_loader, test_loader

