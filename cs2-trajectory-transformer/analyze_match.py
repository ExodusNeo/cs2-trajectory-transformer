"""
End-to-End CS2 Match Analysis & Anti-Cheat / Smurf Auditing Tool.
Analyzes any real CS2 .dem / .dem.zst replay and produces a complete
player-by-player telemetry security report.

Usage:
    python analyze_match.py --demo path/to/match.dem
    python analyze_match.py --demo data/raw_demos/clean/1-1cfcda8f-0d0c-46ee-8863-f746235e48e7-1-1.dem
    python analyze_match.py --faceit_match_id 1-1cfcda8f-0d0c-46ee-8863-f746235e48e7
"""

import os
import sys
import argparse
import time
import numpy as np
import pandas as pd
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

from data.demo_parser import CS2DemoParser
from data.demo_downloader import CS2ReplayDownloader
from data.atw_filter import extract_active_tracking_windows
from features.kinematics import compute_kinematic_features
from models.st_transformer import STTrajectoryTransformer

FEATURE_COLS = [
    'yaw', 'pitch', 'angular_velocity', 'angular_accel',
    'angular_jerk', 'trajectory_curvature', 'curvature_entropy', 'tremor_power_8_12hz'
]


def load_st_transformer(checkpoint_path: str, device: torch.device):
    model = STTrajectoryTransformer(
        feature_dim=len(FEATURE_COLS),
        d_model=64,
        nhead=4,
        num_layers=4,
        embed_dim=32,
        dim_feedforward=256
    ).to(device)

    if os.path.exists(checkpoint_path):
        weights = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(weights)
    model.eval()
    return model


def analyze_demo(demo_path: str, model_path: str = "models/checkpoints/best_model.pt"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    downloader = CS2ReplayDownloader()

    # Automatically decompress if archive
    if demo_path.endswith('.zst') or demo_path.endswith('.gz') or demo_path.endswith('.bz2'):
        print(f"[*] Decompressing replay archive: {os.path.basename(demo_path)}...")
        extracted = downloader.decompress_archive(demo_path, os.path.dirname(demo_path))
        if extracted:
            demo_path = extracted[0]

    if not os.path.exists(demo_path):
        print(f"[!] Error: Demo file not found at {demo_path}")
        return

    print("=" * 80)
    print("      CS2 TRAJECTORY TRANSFORMER // FULL MATCH SECURITY AUDIT")
    print("=" * 80)
    print(f"  Target Replay:  {os.path.basename(demo_path)}")
    print(f"  Inference Unit: {device}")

    # 1. Parse Demo
    start_t = time.time()
    parser = CS2DemoParser(demo_path)
    header = parser.parse_header()
    print(f"  Map:            {header.get('map_name', 'Unknown')}")
    print(f"  Server / Patch: {header.get('server_name', 'FACEIT/Valve')} (Build {header.get('patch_version', 'N/A')})")

    print("\n[*] Ingesting high-frequency tick telemetry & combat events...")
    ticks_df = parser.parse_ticks()
    events_dict = parser.parse_events()
    fire_events = events_dict.get('weapon_fire', pd.DataFrame())

    parse_time = time.time() - start_t
    print(f"[OK] Ingested {len(ticks_df):,} telemetry ticks across all players in {parse_time:.2f}s.")

    # 2. Load Model
    model = load_st_transformer(model_path, device)

    # 3. Analyze each player
    players_data = []
    player_names = {}
    for steamid in ticks_df['steamid'].dropna().unique():
        p_ticks = ticks_df[ticks_df['steamid'] == steamid].sort_values('tick').reset_index(drop=True)
        if len(p_ticks) < 64:
            continue
        p_name = p_ticks['name'].dropna().iloc[0] if 'name' in p_ticks.columns and not p_ticks['name'].dropna().empty else f"Player_{steamid}"
        player_names[steamid] = p_name

        # Weapon fire ticks
        if not fire_events.empty and 'user_steamid' in fire_events.columns:
            p_fires = fire_events[fire_events['user_steamid'].astype(str) == str(steamid)]
            event_ticks = p_fires['tick'].tolist() if 'tick' in p_fires.columns else []
        else:
            event_ticks = []

        # Extract Active Tracking Windows
        atws = extract_active_tracking_windows(p_ticks, event_ticks=event_ticks, tick_buffer=64, min_window_len=64)

        if not atws:
            continue

        # Run ST-Trans on all engagement segments for this player
        aimbot_scores = []
        elo_predictions = []
        suspicious_segments = []

        for seg_idx, seg_df in enumerate(atws):
            feat_df = compute_kinematic_features(seg_df, tick_rate=128.0, extract_tremor=True)
            raw_array = feat_df[FEATURE_COLS].values.astype(np.float32)
            mean = np.mean(raw_array, axis=0, keepdims=True)
            std = np.std(raw_array, axis=0, keepdims=True) + 1e-6
            norm_array = (raw_array - mean) / std
            input_tensor = torch.tensor(norm_array, dtype=torch.float32).unsqueeze(0).to(device)

            with torch.no_grad():
                aim_prob, emb, elo_pred = model(input_tensor)

            prob_val = aim_prob.item() * 100.0
            aimbot_scores.append(prob_val)
            elo_predictions.append(elo_pred.item() * 2000.0)

            if prob_val >= 75.0:
                start_tick = seg_df['tick'].iloc[0]
                end_tick = seg_df['tick'].iloc[-1]
                suspicious_segments.append((start_tick, end_tick, prob_val))

        mean_aim = np.mean(aimbot_scores) if aimbot_scores else 0.0
        peak_aim = np.max(aimbot_scores) if aimbot_scores else 0.0
        est_elo = np.median(elo_predictions) if elo_predictions else 1500.0

        if peak_aim >= 85.0 or mean_aim >= 65.0:
            verdict = "[!] HIGH RISK (AIMBOT)"
        elif peak_aim >= 70.0 or mean_aim >= 50.0:
            verdict = "[?] SUSPICIOUS"
        else:
            verdict = "[OK] CLEAN HUMAN"

        players_data.append({
            'Player': p_name,
            'SteamID': steamid,
            'Engagements': len(atws),
            'Mean Aimbot %': f"{mean_aim:.1f}%",
            'Peak Aimbot %': f"{peak_aim:.1f}%",
            'Predicted ELO': f"{est_elo:.0f}",
            'Verdict': verdict,
            'Suspicious': suspicious_segments
        })

    # 4. Display Report Table
    print("\n" + "=" * 80)
    print(f"{'PLAYER':<18}{'ENGAGEMENTS':<14}{'MEAN AIM %':<13}{'PEAK AIM %':<13}{'PRED ELO':<11}{'VERDICT'}")
    print("=" * 80)
    for p in sorted(players_data, key=lambda x: float(x['Peak Aimbot %'].replace('%', '')), reverse=True):
        print(f"{p['Player'][:16]:<18}{p['Engagements']:<14}{p['Mean Aimbot %']:<13}{p['Peak Aimbot %']:<13}{p['Predicted ELO']:<11}{p['Verdict']}")

    print("=" * 80)

    # 5. Anomaly Details
    flagged = [p for p in players_data if p['Suspicious']]
    if flagged:
        print("\n[!] SUSPICIOUS ENGAGEMENT FLICK ANOMALIES DETECTED:")
        for p in flagged:
            print(f"  > {p['Player']} ({p['SteamID']}):")
            for st, et, score in p['Suspicious'][:3]:
                print(f"    - Ticks {st:,} -> {et:,} (Duration: {et-st} ticks): {score:.1f}% Aimbot Confidence")
    else:
        print("\n[✓] Zero high-risk aimbot anomalies detected across all engagement windows.")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Audit CS2 Match Demo with ST-Trans")
    parser.add_argument("--demo", type=str, default=None, help="Path to local .dem / .dem.zst file")
    parser.add_argument("--faceit_match_id", type=str, default=None, help="Faceit match ID to fetch and analyze")
    parser.add_argument("--model_path", type=str, default="models/checkpoints/best_model.pt", help="Checkpoint path")
    args = parser.parse_args()

    if args.faceit_match_id:
        downloader = CS2ReplayDownloader()
        api_key = os.environ.get("FACEIT_API_KEY")
        print(f"[*] Fetching match replay for Faceit Match ID: {args.faceit_match_id}")
        dems = downloader.fetch_faceit_match_demo(args.faceit_match_id, api_key=api_key)
        if dems:
            analyze_demo(dems[0], model_path=args.model_path)
    elif args.demo:
        analyze_demo(args.demo, model_path=args.model_path)
    else:
        # Default to our downloaded test demo if available
        default_demo = "data/raw_demos/clean/1-1cfcda8f-0d0c-46ee-8863-f746235e48e7-1-1.dem"
        if os.path.exists(default_demo):
            analyze_demo(default_demo, model_path=args.model_path)
        else:
            parser.print_help()


if __name__ == "__main__":
    main()
