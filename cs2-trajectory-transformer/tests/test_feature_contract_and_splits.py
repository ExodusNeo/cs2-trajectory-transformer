"""
Tests for the feature contract, signal-processing fixes, target-relative features,
P x K batching, giant-component splitting, ELO masking and synthetic-data safety.
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from features.kinematics import (
    MODEL_FEATURE_COLUMNS,
    compute_aim_error,
    compute_kinematic_features,
    compute_signed_angular_rates,
    compute_tremor_band_power,
)
from data.atw_filter import find_fov_encounters
from data.dataset import (
    FEATURE_COLUMNS,
    CS2TrajectoryDataset,
    PlayerBalancedBatchSampler,
    collate_trajectory_batch,
    partition_dataset_files,
)
from data.demo_downloader import is_cheating_ban_reason
from models.losses import masked_smooth_l1_loss


# ------------------------------------------------------------------ feature contract

def test_feature_contract_excludes_absolute_yaw_and_is_shared():
    assert 'yaw' not in MODEL_FEATURE_COLUMNS
    assert {'aim_error', 'aim_error_rate'} <= set(MODEL_FEATURE_COLUMNS)
    assert FEATURE_COLUMNS == MODEL_FEATURE_COLUMNS


def test_compute_kinematic_features_emits_every_model_channel():
    n = 128
    df = pd.DataFrame({'tick': np.arange(n), 'yaw': np.linspace(0, 30, n), 'pitch': np.zeros(n)})
    out = compute_kinematic_features(df, aim_error=np.linspace(0.5, 0.0, n))
    for col in MODEL_FEATURE_COLUMNS:
        assert col in out.columns
        assert np.isfinite(out[col]).all(), col


# ------------------------------------------------------------------ tremor band power

def _reference_tbp(sig: np.ndarray, w: int = 64, fs: float = 64.0) -> np.ndarray:
    """Original per-tick loop implementation, kept as the oracle for the vectorized version."""
    n = len(sig)
    freqs = np.fft.rfftfreq(w, d=1.0 / fs)
    t_mask = (freqs >= 8.0) & (freqs <= 12.0)
    a_mask = (freqs >= 1.0) & (freqs <= 30.0)
    out = np.zeros(n)
    for i in range(n):
        start = max(0, i - w // 2)
        sub = sig[start:min(n, start + w)]
        if len(sub) < w:
            sub = np.pad(sub, (0, w - len(sub)), mode='edge')
        p = np.abs(np.fft.rfft((sub - sub.mean()) * np.hanning(w))) ** 2
        tot = p[a_mask].sum()
        out[i] = 0.0 if tot < 1e-4 else p[t_mask].sum() / (tot + 1e-9)
    return out


def test_vectorized_tbp_matches_reference_loop():
    rng = np.random.default_rng(0)
    sig = rng.normal(size=200) + np.sin(2 * np.pi * 10 * np.arange(200) / 64.0)
    np.testing.assert_allclose(compute_tremor_band_power(sig), _reference_tbp(sig), rtol=1e-4, atol=1e-5)


def test_signed_rates_keep_tremor_in_band_while_speed_magnitude_doubles_it():
    """A 10 Hz yaw oscillation: |omega| rectifies it to 20 Hz (out of band); signed rates do not."""
    t = np.arange(256) / 64.0
    yaw = np.radians(0.5 * np.sin(2 * np.pi * 10.0 * t))
    pitch = np.zeros_like(yaw)
    rates = compute_signed_angular_rates(pitch, yaw)
    speed = np.abs(rates[:, 0])
    tbp_signed = np.median(compute_tremor_band_power(rates))
    tbp_speed = np.median(compute_tremor_band_power(speed))
    assert tbp_signed > 0.8
    assert tbp_speed < 0.3


# ------------------------------------------------------------------ target-relative features

def _player(n, yaw_deg=0.0, pitch_deg=0.0):
    return pd.DataFrame({'tick': np.arange(n), 'X': 0.0, 'Y': 0.0, 'Z': 0.0,
                         'pitch': pitch_deg, 'yaw': yaw_deg})


def test_aim_error_geometry_ahead_behind_and_absent():
    n = 5
    ahead = pd.DataFrame({'tick': np.arange(n), 'X': 1000.0, 'Y': 0.0, 'Z': 0.0, 'is_alive': True})
    behind = pd.DataFrame({'tick': np.arange(n), 'X': -1000.0, 'Y': 0.0, 'Z': 0.0, 'is_alive': True})
    assert np.allclose(compute_aim_error(_player(n), ahead), 0.0, atol=1e-6)
    assert np.allclose(compute_aim_error(_player(n), behind), np.pi, atol=1e-6)
    assert np.allclose(compute_aim_error(_player(n), None), np.pi)
    # Nearest of several enemies wins; dead enemies are ignored.
    both = pd.concat([behind, ahead.assign(is_alive=False)])
    assert np.allclose(compute_aim_error(_player(n), both), np.pi, atol=1e-6)


def test_aim_error_respects_cs2_pitch_sign():
    """Positive pitch looks down: an enemy below the player is acquired at positive pitch."""
    n = 3
    below = pd.DataFrame({'tick': np.arange(n), 'X': 1000.0, 'Y': 0.0, 'Z': -1000.0})
    assert np.allclose(compute_aim_error(_player(n, pitch_deg=45.0), below), 0.0, atol=1e-6)


def test_fov_encounters_ignore_dead_enemies_and_split_on_gaps():
    n = 10
    enemy = pd.DataFrame({'tick': np.arange(n), 'X': 1000.0, 'Y': 0.0, 'Z': 0.0,
                          'is_alive': [True] * 4 + [False] * 2 + [True] * 4})
    assert find_fov_encounters(_player(n), enemy) == [(0, 3), (6, 9)]


# ------------------------------------------------------------------ ban reasons

@pytest.mark.parametrize("reason,expected", [
    ("Cheating", True), ("cheat detected", True), ("Aimbot usage", True), ("wallhack", True),
    ("Smurfing", False), ("Ban evasion", False), ("toxic behaviour: false claim", False),
])
def test_cheating_ban_reason_uses_word_boundaries(reason, expected):
    assert is_cheating_ban_reason(reason) is expected


# ------------------------------------------------------------------ P x K sampler

def test_player_balanced_sampler_yields_positive_pairs():
    files = [f"m{i % 6}_p{pid}_seg{i}.parquet" for pid in range(12) for i in range(5)]
    sampler = PlayerBalancedBatchSampler(files, players_per_batch=4, samples_per_player=3, seed=1)
    seen = []
    for batch in sampler:
        pids = [files[i].split('_p')[1].split('_seg')[0] for i in batch]
        counts = pd.Series(pids).value_counts()
        assert (counts >= 2).any(), "every batch must contain at least one positive pair"
        seen.extend(batch)
    assert sorted(seen) == list(range(len(files))), "each window is used exactly once per epoch"


# ------------------------------------------------------------------ partitioning

def _write(tmp_path, match_id, player, label, elo=1500.0):
    n = 64
    df = pd.DataFrame({c: np.zeros(n) for c in FEATURE_COLUMNS})
    df['match_id'] = match_id
    df['steamid'] = player
    df['segment_id'] = 0
    df['is_aimbot'] = label
    df['player_elo'] = elo
    df.to_parquet(tmp_path / f"{match_id}_p{player}_seg0.parquet", index=False)


def _ids(files):
    m = {os.path.basename(f).split('_p')[0] for f in files}
    p = {os.path.basename(f).split('_p')[1].split('_seg')[0] for f in files}
    return m, p


def test_giant_component_falls_back_to_leakage_free_match_drop(tmp_path):
    # 20 clean matches chained through a shared "regular" player -> one giant clean component.
    for i in range(20):
        _write(tmp_path, f"clean{i}", 9000, 0)
        _write(tmp_path, f"clean{i}", 100 + i, 0)
        _write(tmp_path, f"clean{i}", 200 + i, 0)
    for i in range(6):
        _write(tmp_path, f"cheat{i}", 500 + i, 1)
        _write(tmp_path, f"cheat{i}", 600 + i, 0)

    tr, va, te = partition_dataset_files(str(tmp_path), seed=3)
    (mt, pt), (mv, pv), (me, pe) = _ids(tr), _ids(va), _ids(te)
    assert mt.isdisjoint(mv) and mt.isdisjoint(me) and mv.isdisjoint(me)
    assert pt.isdisjoint(pv) and pt.isdisjoint(pe) and pv.isdisjoint(pe)
    assert any(m.startswith('clean') for m in me) and any(m.startswith('cheat') for m in me)

    with pytest.raises(ValueError):
        partition_dataset_files(str(tmp_path), strategy="bogus")


def test_largest_component_never_lands_in_test(tmp_path):
    # One big clean component (players shared across 4 matches) plus small independent ones.
    for i in range(4):
        for p in range(10):
            _write(tmp_path, f"big{i}", 7000 + p, 0)
    for i in range(12):
        _write(tmp_path, f"small{i}", 100 + i, 0)
    for i in range(4):
        _write(tmp_path, f"cheat{i}", 300 + i, 1)
    tr, va, te = partition_dataset_files(str(tmp_path), strategy="components", seed=0)
    assert all(os.path.basename(f).startswith('big') is False for f in te + va)


# ------------------------------------------------------------------ ELO masking

def test_unknown_elo_is_masked_not_defaulted(tmp_path):
    _write(tmp_path, "m0", 1, 0, elo=float('nan'))
    _write(tmp_path, "m0", 2, 0, elo=1800.0)
    files = sorted(str(p) for p in tmp_path.glob("*.parquet"))
    batch = collate_trajectory_batch([CS2TrajectoryDataset(files)[i] for i in range(2)])
    assert batch['elo_mask'].flatten().tolist() == [False, True]
    pred = torch.tensor([[5.0], [0.9]], requires_grad=True)
    loss = masked_smooth_l1_loss(pred, batch['elo_labels'], batch['elo_mask'])
    assert loss.item() == pytest.approx(0.0, abs=1e-6)  # 0.9 == 1800 / 2000; the NaN row is ignored
    assert masked_smooth_l1_loss(pred, batch['elo_labels'], torch.zeros_like(batch['elo_mask'])).item() == 0.0


# ------------------------------------------------------------------ synthetic data safety

def test_synthetic_generator_only_touches_its_own_files(tmp_path):
    from generate_benchmark_dataset import create_synthetic_dataset
    keep = tmp_path / "match_real_p1_seg0.parquet"
    keep.write_bytes(b"real data")
    create_synthetic_dataset(str(tmp_path), num_samples=40, seed=0)
    create_synthetic_dataset(str(tmp_path), num_samples=40, seed=0)
    assert keep.read_bytes() == b"real data"
    assert len(list(tmp_path.glob("syn_*.parquet"))) == 40


# ------------------------------------------------------------------ halftime side swap

def test_enemies_are_resolved_per_tick_across_halftime_swap():
    """Teams swap sides at halftime; the player must never become their own 'enemy'."""
    from data.batch_processor import extract_player_feature_windows
    n = 400
    ticks = np.arange(n)
    first_half = ticks < n // 2
    player = pd.DataFrame({'tick': ticks, 'steamid': 1, 'team_num': np.where(first_half, 2, 3),
                           'X': 0.0, 'Y': 0.0, 'Z': 0.0, 'pitch': 0.0, 'yaw': 0.0, 'is_alive': True})
    # The only opponent stands directly behind the player for the whole match and swaps sides too.
    enemy = pd.DataFrame({'tick': ticks, 'steamid': 2, 'team_num': np.where(first_half, 3, 2),
                          'X': -1000.0, 'Y': 0.0, 'Z': 0.0, 'pitch': 0.0, 'yaw': 0.0, 'is_alive': True})
    windows = extract_player_feature_windows(pd.concat([player, enemy]), 1, event_ticks=list(range(0, n, 50)))
    assert windows, "event buffers should produce ATWs"
    aim = np.concatenate([w['aim_error'].to_numpy() for w in windows])
    assert np.allclose(aim, np.pi, atol=1e-6), "opponent is behind the player at every tick"
