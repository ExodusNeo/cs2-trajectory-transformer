"""
Synthetic CS2 Trajectory Generator — PIPELINE VERIFICATION ONLY.

Scores on this data are NOT evidence for the thesis hypotheses: the generator encodes the
very assumptions under test (e.g. humans get tremor, scripts do not). Use it to check that
ingestion, training and evaluation run end to end, and to stress-test the model with a
"humanized" aimbot whose motion shape mimics the human model while its target lock does not.

Realism knobs:
- View angles are quantized to one mouse count (0.022 deg x sensitivity), like real replays.
- A target appears after a random delay and strafes; 'aim_error' is the angle from the
  crosshair to the target (pi before it appears), matching features.kinematics.compute_aim_error.

Output: syn_*.parquet files in data/synthetic_parquet/ (never the real processed_parquet store).
"""

import os
import sys
import glob
import logging
from typing import Optional, Tuple

import numpy as np
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))
from features.kinematics import compute_kinematic_features

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

M_YAW_DEG = 0.022  # CS2 degrees per mouse count at sensitivity 1.0
SYNTHETIC_PREFIX = "syn_"


def _quantize(angles_deg: np.ndarray, sensitivity: float) -> np.ndarray:
    """Rounds cumulative view angles to whole mouse counts (0.022 deg x sensitivity)."""
    step = M_YAW_DEG * sensitivity
    return np.round(angles_deg / step) * step


def _min_jerk(t_norm: np.ndarray) -> np.ndarray:
    """Thesis Eq (2): Flash & Hogan minimum-jerk position profile on [0, 1]."""
    t = np.clip(t_norm, 0.0, 1.0)
    return 10 * t ** 3 - 15 * t ** 4 + 6 * t ** 5


def _target_path(n_ticks: int, rng: np.random.Generator, tick_rate: float) -> Tuple[np.ndarray, np.ndarray, int]:
    """Target direction (yaw, pitch in degrees) that appears at t_appear and strafes."""
    t = np.arange(n_ticks) / tick_rate
    t_appear = int(rng.integers(10, 40))
    yaw0, pitch0 = rng.uniform(-45, 45), rng.uniform(-15, 15)
    strafe_rate = rng.uniform(-25, 25)  # deg/s angular strafe as seen by the shooter
    strafe = strafe_rate * np.clip(t - t_appear / tick_rate, 0.0, None)
    reversal = np.sin(2 * np.pi * rng.uniform(0.3, 0.8) * t) * rng.uniform(0, 4)  # A-D strafing
    return 180.0 + yaw0 + strafe + reversal, np.full(n_ticks, pitch0), t_appear


def _aim_error(yaw: np.ndarray, pitch: np.ndarray, tyaw: np.ndarray, tpitch: np.ndarray, t_appear: int) -> np.ndarray:
    """Great-circle angle (rad) between view and target direction; pi before the target appears."""
    def unit(y, p):
        y, p = np.radians(y), np.radians(p)
        return np.stack([np.cos(p) * np.cos(y), np.cos(p) * np.sin(y), np.sin(p)], axis=-1)
    dot = np.clip(np.sum(unit(yaw, pitch) * unit(tyaw, tpitch), axis=-1), -1.0, 1.0)
    err = np.arccos(dot)
    err[:t_appear] = np.pi
    return err


def _track(tyaw: np.ndarray, tpitch: np.ndarray, t_start: int, lag_ticks: float,
           acquire_ticks: int, endpoint_error_deg: float, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    """
    Minimum-jerk acquisition of the target starting at t_start, then first-order tracking
    with time constant lag_ticks (closed-form, no per-tick loop).
    """
    n = len(tyaw)
    idx = np.arange(n)
    end = min(n, t_start + acquire_ticks)
    goal_yaw = tyaw[min(end, n - 1)] + rng.normal(0, endpoint_error_deg)
    goal_pitch = tpitch[min(end, n - 1)] + rng.normal(0, endpoint_error_deg)
    s = _min_jerk((idx - t_start) / max(1, acquire_ticks))
    yaw = 180.0 + s * (goal_yaw - 180.0)
    pitch = s * goal_pitch
    if end < n and lag_ticks > 0:
        # Closed-form first-order follow: error to the moving target decays with exp(-k / lag).
        k = idx[end:] - end
        decay = np.exp(-k / lag_ticks)
        yaw[end:] = tyaw[end:] + (goal_yaw - tyaw[end]) * decay - np.gradient(tyaw)[end:] * lag_ticks * (1 - decay)
        pitch[end:] = tpitch[end:] + (goal_pitch - tpitch[end]) * decay
    elif end < n:
        yaw[end:] = tyaw[end:]
        pitch[end:] = tpitch[end:]
    return yaw, pitch


def generate_human_trajectory(
    n_ticks: int = 256,
    elo: float = 2000.0,
    tick_rate: float = 64.0,
    rng: Optional[np.random.Generator] = None
) -> pd.DataFrame:
    """Synthetic human: 150-300 ms reaction, minimum-jerk flick with endpoint error, laggy tracking."""
    rng = rng or np.random.default_rng()
    skill = np.clip(elo / 2500.0, 0.2, 1.2)
    tyaw, tpitch, t_appear = _target_path(n_ticks, rng, tick_rate)
    reaction = int(rng.uniform(0.15, 0.30) / skill ** 0.3 * tick_rate)
    yaw, pitch = _track(tyaw, tpitch, t_appear + reaction, lag_ticks=rng.uniform(5, 10) / skill,
                        acquire_ticks=int(rng.uniform(10, 20)), endpoint_error_deg=rng.uniform(0.5, 2.0) / skill, rng=rng)
    t = np.arange(n_ticks) / tick_rate
    tremor = rng.uniform(0.005, 0.03) * np.sin(2 * np.pi * rng.uniform(8.5, 11.5) * t + rng.uniform(0, np.pi))
    sens = rng.choice([0.8, 1.0, 1.25, 1.5, 2.0])
    yaw = _quantize(yaw + tremor + rng.normal(0, 0.02 / skill, n_ticks), sens)
    pitch = _quantize(pitch + tremor + rng.normal(0, 0.02 / skill, n_ticks), sens)
    return pd.DataFrame({'tick': np.arange(n_ticks), 'yaw': yaw, 'pitch': pitch,
                         'aim_error': _aim_error(yaw, pitch, tyaw, tpitch, t_appear)})


def generate_cheater_trajectory(
    n_ticks: int = 256,
    cheat_type: str = 'snap',
    tick_rate: float = 64.0,
    rng: Optional[np.random.Generator] = None
) -> pd.DataFrame:
    """
    Synthetic aim assistance:
    - 'snap': 1-tick acquisition, exact lock.
    - 'smooth': linear interpolation acquisition, exact lock.
    - 'humanized': human-like minimum-jerk flick + injected 8-12 Hz tremor + mouse quantization,
      but superhuman reaction (30-80 ms), zero endpoint error and near-zero tracking lag.
    """
    rng = rng or np.random.default_rng()
    tyaw, tpitch, t_appear = _target_path(n_ticks, rng, tick_rate)
    sens = rng.choice([0.8, 1.0, 1.25, 1.5, 2.0])
    if cheat_type == 'snap':
        yaw, pitch = _track(tyaw, tpitch, t_appear + 2, lag_ticks=0, acquire_ticks=1, endpoint_error_deg=0.0, rng=rng)
    elif cheat_type == 'smooth':
        start, dur = t_appear + 4, int(rng.integers(15, 30))
        s = np.clip((np.arange(n_ticks) - start) / dur, 0.0, 1.0)
        yaw = 180.0 + s * (tyaw - 180.0)
        pitch = s * tpitch
    elif cheat_type == 'humanized':
        reaction = int(rng.uniform(0.03, 0.08) * tick_rate)
        yaw, pitch = _track(tyaw, tpitch, t_appear + reaction, lag_ticks=rng.uniform(0.5, 1.5),
                            acquire_ticks=int(rng.uniform(10, 20)), endpoint_error_deg=0.0, rng=rng)
        t = np.arange(n_ticks) / tick_rate
        tremor = rng.uniform(0.005, 0.03) * np.sin(2 * np.pi * rng.uniform(8.5, 11.5) * t)
        yaw, pitch = yaw + tremor, pitch + tremor
    else:
        raise ValueError(f"Unknown cheat_type '{cheat_type}'")
    yaw, pitch = _quantize(yaw, sens), _quantize(pitch, sens)
    return pd.DataFrame({'tick': np.arange(n_ticks), 'yaw': yaw, 'pitch': pitch,
                         'aim_error': _aim_error(yaw, pitch, tyaw, tpitch, t_appear)})


def create_synthetic_dataset(
    output_dir: str = "data/synthetic_parquet",
    num_samples: int = 600,
    seed: int = 42
) -> int:
    """
    Writes syn_*.parquet windows for 10 clean and 10 cheater synthetic players (5 matches each class).
    Only previously generated syn_*.parquet files in output_dir are removed; nothing else is touched.
    """
    os.makedirs(output_dir, exist_ok=True)
    for old in glob.glob(os.path.join(output_dir, f"{SYNTHETIC_PREFIX}*.parquet")):
        os.remove(old)

    rng = np.random.default_rng(seed)
    elo_tiers = [700.0, 1150.0, 1600.0, 2200.0]
    clean_players = [(76561198000000010 + i, float(rng.choice(elo_tiers))) for i in range(10)]
    cheat_types = ['snap', 'smooth', 'humanized', 'humanized']
    cheater_players = [(76561198000000050 + i, cheat_types[i % len(cheat_types)], float(rng.choice(elo_tiers)))
                       for i in range(10)]
    per_player = num_samples // (len(clean_players) + len(cheater_players))
    total_saved = 0

    def save(raw_df: pd.DataFrame, match_id: str, p_id: int, seg: int, label: int, elo: float) -> None:
        feat_df = compute_kinematic_features(raw_df, tick_rate=64.0, extract_tremor=True)
        feat_df['match_id'] = match_id
        feat_df['steamid'] = p_id
        feat_df['segment_id'] = seg
        feat_df['is_aimbot'] = label
        feat_df['player_elo'] = elo  # cheaters share the clean ELO tiers so ELO cannot leak the label
        feat_df.to_parquet(os.path.join(output_dir, f"{match_id}_p{p_id}_seg{seg}.parquet"), index=False)

    for p_id, elo in clean_players:
        for seg in range(per_player):
            df = generate_human_trajectory(n_ticks=int(rng.integers(128, 256)), elo=elo, rng=rng)
            save(df, f"{SYNTHETIC_PREFIX}c{p_id % 5}", p_id, seg, 0, elo)
            total_saved += 1

    for p_id, cheat_type, elo in cheater_players:
        for seg in range(per_player):
            df = generate_cheater_trajectory(n_ticks=int(rng.integers(128, 256)), cheat_type=cheat_type, rng=rng)
            save(df, f"{SYNTHETIC_PREFIX}x{p_id % 5}", p_id, seg, 1, elo)
            total_saved += 1

    logging.info(f"Generated {total_saved} synthetic windows in {output_dir} (pipeline verification only).")
    return total_saved


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Generate synthetic pipeline-verification windows")
    ap.add_argument("--output_dir", type=str, default="data/synthetic_parquet")
    ap.add_argument("--num_samples", type=int, default=600)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    create_synthetic_dataset(a.output_dir, a.num_samples, a.seed)
