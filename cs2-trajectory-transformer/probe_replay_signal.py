"""
Replay Signal Probe — empirical check of what 64 Hz CS2 view angles can physically show.

For each demo, reports per player:
- p05_step_deg: 5th percentile of non-zero yaw steps; typically one mouse count
  (0.022 deg x sensitivity). Rarer, smaller steps exist (recoil / view punch).
- zero_motion_frac: share of live, contiguous ticks with no yaw change
- tbp_median: median 8-12 Hz relative band power of signed angular rates over moving windows

and a white-noise TBP reference. If real-player TBP sits at or below the noise reference,
the tremor channel carries little physiological signal on this data. Report that honestly.

Usage:
    python probe_replay_signal.py data/raw_demos/clean/*.dem
"""

import os
import sys
import argparse
from typing import Dict, List

import numpy as np
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))
from features.kinematics import compute_signed_angular_rates, compute_tremor_band_power


def probe_demo(demo_path: str, window: int = 256, windows_per_player: int = 80, seed: int = 0) -> pd.DataFrame:
    """Returns one row per player with quantization and tremor-band statistics."""
    from demoparser2 import DemoParser
    df = DemoParser(demo_path).parse_ticks(["pitch", "yaw", "is_alive"])
    df = df[df["is_alive"] == True].sort_values(["steamid", "tick"])
    rng = np.random.default_rng(seed)
    rows: List[Dict[str, float]] = []
    for sid, g in df.groupby("steamid"):
        contiguous = np.diff(g["tick"].to_numpy()) == 1
        dyaw = (((np.diff(g["yaw"].to_numpy()) + 180.0) % 360.0) - 180.0)[contiguous]
        nonzero = np.abs(dyaw[np.abs(dyaw) > 1e-7])
        rates = compute_signed_angular_rates(np.radians(g["pitch"].to_numpy()), np.radians(g["yaw"].to_numpy()))
        tbp = []
        if len(g) > window:
            for s in rng.integers(0, len(g) - window, windows_per_player):
                seg = rates[s:s + window]
                if np.mean(np.abs(seg).sum(axis=1) > 0) >= 0.3:  # skip mostly-static windows
                    tbp.append(np.median(compute_tremor_band_power(seg)))
        rows.append({
            "demo": os.path.basename(demo_path),
            "player": str(sid)[-4:],
            "p05_step_deg": float(np.percentile(nonzero, 5)) if len(nonzero) else np.nan,
            "zero_motion_frac": float(np.mean(np.abs(dyaw) < 1e-7)) if len(dyaw) else np.nan,
            "tbp_median": float(np.median(tbp)) if tbp else np.nan,
            "moving_windows": len(tbp),
        })
    return pd.DataFrame(rows)


def white_noise_tbp(n: int = 200, window: int = 256, seed: int = 0) -> float:
    """Median TBP of signed white-noise rates (about 5 of 30 in-band 1 Hz bins)."""
    rng = np.random.default_rng(seed)
    return float(np.median([np.median(compute_tremor_band_power(rng.normal(size=(window, 2)))) for _ in range(n)]))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Probe quantization and tremor-band signal in CS2 demos")
    ap.add_argument("demos", nargs="+", help=".dem files")
    args = ap.parse_args()
    results = pd.concat([probe_demo(d) for d in args.demos], ignore_index=True)
    print(results.round(4).to_string(index=False))
    print(f"\nreal TBP median across players: {results['tbp_median'].median():.3f}")
    print(f"white-noise TBP reference:      {white_noise_tbp():.3f}")
