"""
CS2CD Pilot Feasibility Study (outline-defense evidence; NOT the final evaluation).

Runs on a seeded CS2CD subset ingested by ingest_cs2cd.py and answers, at the player-match
session level with leakage-free grouped cross-validation:
  1. Do the candidate channels carry signal on real cheaters? (feature-group ablation)
  2. Do engineered channels beat raw view angles?
  3. Is mouse-input/view-angle consistency a useful additional signal?
  4. Real-data sanity: tremor band power vs white noise, target aim error by class.

Design notes
- Sessions = (match, player). CS2CD player IDs exist only within one match, so grouping folds
  by match is also player-disjoint (zero leakage).
- Negatives come only from no_cheater_present matches (see cs2cd_adapter for the label policy).
- Classifier: gradient-boosted trees on per-session means of the 6 window statistics per channel
  (the Table 7 tabular baseline). This is a fast feasibility probe, not ST-Trans.
- Small sample: report fold spread and do not extrapolate to the 0.01% FPR target.

Usage:
    python pilot_study.py --windows data/processed_parquet_cs2cd --raw data/raw_demos/cs2cd
"""

import argparse
import glob
import json
import os
import sys
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedGroupKFold

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))
from features.kinematics import MODEL_FEATURE_COLUMNS, compute_tremor_band_power  # noqa: E402
from data.batch_processor import pseudonymize_match_id, pseudonymize_steamid  # noqa: E402
from data.cs2cd_adapter import CS2CD_FOLDERS, is_valid_cs2cd_file  # noqa: E402

STAT_NAMES = ['mean', 'std', 'min', 'max', 'skew', 'kurt']
CHANNELS = MODEL_FEATURE_COLUMNS + ['yaw']
GROUPS: Dict[str, List[str]] = {
    'All 9 model channels': MODEL_FEATURE_COLUMNS,
    'Self-kinematics only (no aim error)': [c for c in MODEL_FEATURE_COLUMNS if not c.startswith('aim_error')],
    'Target-relative only (aim error + rate)': ['aim_error', 'aim_error_rate'],
    'Tremor band power only': ['tremor_power_8_12hz'],
    'Raw angles only (yaw, pitch)': ['yaw', 'pitch'],
}
INPUT_COLS = ['input_corr', 'input_unexplained_frac', 'input_resid_med']


def window_stats(windows_dir: str, cache: str) -> pd.DataFrame:
    """One row per ATW window: ids, label and 6 statistics per channel. Cached to Parquet."""
    if os.path.exists(cache):
        return pd.read_parquet(cache)
    rows = []
    for f in glob.glob(os.path.join(windows_dir, "*.parquet")):
        df = pd.read_parquet(f, columns=CHANNELS + ['match_id', 'steamid', 'is_aimbot'])
        x = df[CHANNELS].to_numpy(dtype=np.float64)
        stats = np.stack([x.mean(0), x.std(0), x.min(0), x.max(0),
                          np.nan_to_num(skew(x, axis=0)), np.nan_to_num(kurtosis(x, axis=0))])
        row = {f"{c}__{s}": stats[i, j] for i, s in enumerate(STAT_NAMES) for j, c in enumerate(CHANNELS)}
        row.update(match_id=df['match_id'].iat[0], steamid=int(df['steamid'].iat[0]), label=int(df['is_aimbot'].iat[0]),
                   tbp_median=float(np.median(df['tremor_power_8_12hz'])), aim_error_median=float(np.median(df['aim_error'])))
        rows.append(row)
    out = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    out.to_parquet(cache, index=False)
    return out


def input_consistency(raw_root: str) -> pd.DataFrame:
    """
    Per player-match agreement between raw mouse input (usercmd_mouse_dx) and yaw change.
    A legitimate player's rotation is explained by mouse input; angles written directly by
    software are not. Hardware mouse emulators (e.g. DMA + KMBox) would still look consistent.
    """
    rows = []
    for folder in CS2CD_FOLDERS:
        for pq in sorted(glob.glob(os.path.join(raw_root, folder, "*.parquet"))):
            stem = os.path.splitext(os.path.basename(pq))[0]
            match_name = f"cs2cd_{folder}_{stem}"
            if not is_valid_cs2cd_file(pq):
                print(f'[warn] skipping incomplete raw file {pq}')
                continue
            df = pd.read_parquet(pq, columns=['tick', 'steamid', 'yaw', 'usercmd_mouse_dx', 'is_alive'])
            for sid, g in df[df['is_alive'].astype(bool)].groupby('steamid'):
                g = g.sort_values('tick')
                dyaw = ((g['yaw'].diff() + 180.0) % 360.0) - 180.0
                dx = g['usercmd_mouse_dx']
                ok = (g['tick'].diff() == 1) & dyaw.notna() & dx.notna() & (dyaw.abs() < 30)
                moving = ok & (dx != 0)
                if moving.sum() < 50:
                    continue
                x, y = dx[moving].to_numpy(float), dyaw[moving].to_numpy(float)
                slope = float(np.sum(x * y) / (np.sum(x * x) + 1e-9))
                rows.append({
                    'match_id': pseudonymize_match_id(match_name),
                    'steamid': pseudonymize_steamid(f"{match_name}:{sid}"),
                    'input_corr': float(np.corrcoef(x, y)[0, 1]) if np.std(y) > 0 else 0.0,
                    # rotation > 0.1 deg in a tick with no mouse input at all
                    'input_unexplained_frac': float(((dyaw.abs() > 0.1) & (dx == 0) & ok).sum() / max(1, (ok & (dyaw.abs() > 0.1)).sum())),
                    'input_resid_med': float(np.median(np.abs(y - slope * x))),
                })
    return pd.DataFrame(rows)


def match_context(raw_root: str) -> pd.DataFrame:
    """Per pseudonymized match: map and Valve average rank from CS2CD JSON (for confound checks)."""
    rows = []
    for folder in CS2CD_FOLDERS:
        for js in sorted(glob.glob(os.path.join(raw_root, folder, "*.json"))):
            stem = os.path.splitext(os.path.basename(js))[0]
            if not is_valid_cs2cd_file(js):
                continue
            with open(js, encoding='utf-8') as f:
                info = (json.load(f).get('CSstats_info') or [{}])[0]
            rows.append({'match_id': pseudonymize_match_id(f"cs2cd_{folder}_{stem}"),
                         'map': str(info.get('map', '')), 'avg_rank': str(info.get('avg_rank', ''))})
    return pd.DataFrame(rows)


def cv_scores(X: np.ndarray, y: np.ndarray, groups: np.ndarray, seeds: List[int]) -> Dict[str, float]:
    """Repeated stratified grouped 5-fold CV; returns mean/std AUROC, AUPRC and pooled TPR at low FPR."""
    try:
        from xgboost import XGBClassifier
        make = lambda s: XGBClassifier(n_estimators=200, max_depth=3, learning_rate=0.05, subsample=0.8,
                                       colsample_bytree=0.8, random_state=s, eval_metric='logloss')
    except ImportError:
        from sklearn.ensemble import GradientBoostingClassifier
        make = lambda s: GradientBoostingClassifier(n_estimators=200, max_depth=3, random_state=s)
    aurocs, auprcs, tpr1, tpr5 = [], [], [], []
    for s in seeds:
        oof = np.zeros(len(y))
        for tr, te in StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=s).split(X, y, groups):
            m = make(s).fit(X[tr], y[tr])
            oof[te] = m.predict_proba(X[te])[:, 1]
        aurocs.append(roc_auc_score(y, oof))
        auprcs.append(average_precision_score(y, oof))
        fpr, tpr, _ = roc_curve(y, oof)
        tpr1.append(float(tpr[fpr <= 0.01].max()) if (fpr <= 0.01).any() else 0.0)
        tpr5.append(float(tpr[fpr <= 0.05].max()) if (fpr <= 0.05).any() else 0.0)
    return {'AUROC_mean': np.mean(aurocs), 'AUROC_std': np.std(aurocs), 'AUPRC_mean': np.mean(auprcs),
            'TPR_at_FPR1%': np.mean(tpr1), 'TPR_at_FPR5%': np.mean(tpr5)}


def oof_predictions(X: np.ndarray, y: np.ndarray, groups: np.ndarray, seed: int = 0) -> np.ndarray:
    """Out-of-fold scores from one stratified grouped 5-fold run (same model as cv_scores)."""
    try:
        from xgboost import XGBClassifier
        model = lambda: XGBClassifier(n_estimators=200, max_depth=3, learning_rate=0.05, subsample=0.8,
                                      colsample_bytree=0.8, random_state=seed, eval_metric='logloss')
    except ImportError:
        from sklearn.ensemble import GradientBoostingClassifier
        model = lambda: GradientBoostingClassifier(n_estimators=200, max_depth=3, random_state=seed)
    oof = np.zeros(len(y))
    for tr, te in StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed).split(X, y, groups):
        oof[te] = model().fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
    return oof


def paired_match_bootstrap(y: np.ndarray, groups: np.ndarray, p_a: np.ndarray, p_b: np.ndarray,
                           n_boot: int = 2000, seed: int = 0) -> Dict[str, float]:
    """
    95% percentile intervals for AUROC(a) and AUROC(a) - AUROC(b), resampling whole matches with
    replacement (sessions within a match are dependent, so matches are the resampling unit).
    """
    rng = np.random.default_rng(seed)
    uniq = np.unique(groups)
    idx_by_group = {g: np.flatnonzero(groups == g) for g in uniq}
    a_s, d_s = [], []
    for _ in range(n_boot):
        idx = np.concatenate([idx_by_group[g] for g in rng.choice(uniq, size=len(uniq), replace=True)])
        if len(np.unique(y[idx])) < 2:
            continue
        a = roc_auc_score(y[idx], p_a[idx])
        a_s.append(a)
        d_s.append(a - roc_auc_score(y[idx], p_b[idx]))
    return {'AUROC_a': roc_auc_score(y, p_a), 'AUROC_a_lo': np.percentile(a_s, 2.5), 'AUROC_a_hi': np.percentile(a_s, 97.5),
            'diff': roc_auc_score(y, p_a) - roc_auc_score(y, p_b),
            'diff_lo': np.percentile(d_s, 2.5), 'diff_hi': np.percentile(d_s, 97.5)}


def main() -> None:
    ap = argparse.ArgumentParser(description="CS2CD pilot feasibility study")
    ap.add_argument("--windows", default="data/processed_parquet_cs2cd")
    ap.add_argument("--raw", default="data/raw_demos/cs2cd")
    ap.add_argument("--out", default="reports/pilot_cs2cd")
    ap.add_argument("--seeds", type=int, default=5)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    w = window_stats(args.windows, os.path.join(args.out, "window_stats.parquet"))
    stat_cols = [c for c in w.columns if '__' in c]
    sessions = w.groupby(['match_id', 'steamid']).agg({**{c: 'mean' for c in stat_cols}, 'label': 'max'}).reset_index()
    sessions['n_windows'] = w.groupby(['match_id', 'steamid']).size().to_numpy()
    ic = input_consistency(args.raw)
    sessions = sessions.merge(ic, on=['match_id', 'steamid'], how='left')
    sessions = sessions.merge(match_context(args.raw), on='match_id', how='left')

    y = sessions['label'].to_numpy()
    groups = sessions['match_id'].to_numpy()
    seeds = list(range(args.seeds))
    results = {}
    for name, chans in GROUPS.items():
        cols = [c for c in stat_cols if c.split('__')[0] in chans]
        results[name] = cv_scores(sessions[cols].fillna(0).to_numpy(), y, groups, seeds)
    has_ic = sessions[INPUT_COLS].notna().all(axis=1).to_numpy()
    sub = sessions[has_ic]
    results['Mouse-input consistency only'] = cv_scores(sub[INPUT_COLS].to_numpy(), sub['label'].to_numpy(), sub['match_id'].to_numpy(), seeds)
    all_cols = [c for c in stat_cols if c.split('__')[0] in MODEL_FEATURE_COLUMNS] + INPUT_COLS
    results['All 9 channels + input consistency'] = cv_scores(sub[all_cols].fillna(0).to_numpy(), sub['label'].to_numpy(), sub['match_id'].to_numpy(), seeds)
    # Confound baselines: how far can match context alone (no aiming data) separate the classes?
    for name, col in (('Confound check: map only', 'map'), ('Confound check: avg rank only', 'avg_rank')):
        X = pd.get_dummies(sessions[col].fillna('?')).to_numpy(dtype=float)
        results[name] = cv_scores(X, y, groups, seeds)
    table = pd.DataFrame(results).T.round(3)

    # Paired match-level bootstrap for the comparisons the thesis argues about.
    def X_of(chans):
        return sessions[[c for c in stat_cols if c.split('__')[0] in chans]].fillna(0).to_numpy()
    p_full = oof_predictions(X_of(MODEL_FEATURE_COLUMNS), y, groups)
    comparisons = {
        'All 9 vs raw angles (yaw, pitch)': oof_predictions(X_of(['yaw', 'pitch']), y, groups),
        'All 9 vs self-kinematics only': oof_predictions(X_of(GROUPS['Self-kinematics only (no aim error)']), y, groups),
        'All 9 vs map only (confound)': oof_predictions(pd.get_dummies(sessions['map'].fillna('?')).to_numpy(dtype=float), y, groups),
    }
    boot = pd.DataFrame({k: paired_match_bootstrap(y, groups, p_full, v) for k, v in comparisons.items()}).T.round(3)
    boot.to_csv(os.path.join(args.out, "session_bootstrap.csv"))
    table.to_csv(os.path.join(args.out, "session_ablation.csv"))

    rng = np.random.default_rng(0)
    noise_tbp = float(np.median([np.median(compute_tremor_band_power(rng.normal(size=(256, 2)))) for _ in range(200)]))
    by_class = w.groupby('label').agg(windows=('label', 'size'), tbp_median=('tbp_median', 'median'),
                                      aim_error_median_deg=('aim_error_median', lambda v: float(np.degrees(np.median(v)))))
    ic_class = sessions[has_ic].groupby('label')[INPUT_COLS].median()
    summary = {
        'matches': int(sessions['match_id'].nunique()),
        'sessions': int(len(sessions)), 'positive_sessions': int(y.sum()), 'negative_sessions': int((y == 0).sum()),
        'windows': int(len(w)), 'white_noise_tbp': round(noise_tbp, 3),
        'by_class': by_class.round(4).to_dict(orient='index'),
        'input_consistency_median_by_class': ic_class.round(4).to_dict(orient='index'),
        'seeds': seeds,
        'labeled_cheaters_per_cheater_match_median': float(sessions[sessions['label'] == 1].groupby('match_id').size().median()),
        'maps_by_class': sessions.groupby('label')['map'].value_counts().unstack(fill_value=0).to_dict(orient='index'),
    }
    with open(os.path.join(args.out, "summary.json"), 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, default=float)
    print(json.dumps(summary, indent=2, default=float))
    print("\nSession-level grouped 5-fold CV (mean over seeds):\n" + table.to_string())
    print("\nPaired match-level bootstrap (95% intervals, seed-0 out-of-fold scores):\n" + boot.to_string())


if __name__ == "__main__":
    main()
