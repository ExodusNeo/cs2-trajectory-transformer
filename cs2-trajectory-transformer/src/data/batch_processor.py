"""
Batch Demo Processing Pipeline for CS2 Replays.
Scans raw .dem files, extracts active combat tracking trajectories,
computes 8D micro-kinematic feature vectors, and saves structured Parquet datasets.
"""

import os
import glob
import logging
import hashlib
import numpy as np
import pandas as pd
from typing import List, Optional, Dict, Tuple, Union, Iterable
from concurrent.futures import ProcessPoolExecutor, as_completed

from .demo_parser import CS2DemoParser
from .atw_filter import extract_active_tracking_windows
from features.kinematics import MODEL_FEATURE_COLUMNS, compute_aim_error, compute_kinematic_features

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')


def pseudonymize_steamid(steamid: Union[str, int], salt: Optional[str] = None) -> int:
    """
    Thesis Reference: Section 3.3.1 — RA 10173 Cryptographic Pseudonymization.
    Transforms raw 64-bit SteamID to an irreversible 60-bit pseudorandom hash integer:
    h = SHA-256(SteamID || salt)
    using a managed secret salt (configured via CS2_PSEUDONYMIZATION_SALT environment variable or salt parameter).
    Obscures direct links to personal profiles and gamer handles, providing pseudonymous telemetry.
    Strictly enforces presence of secret salt to comply with RA 10173 data privacy governance.
    """
    effective_salt = salt or os.environ.get("CS2_PSEUDONYMIZATION_SALT")
    if not effective_salt:
        raise ValueError(
            "Cryptographic pseudonymization under RA 10173 requires a managed secret salt. "
            "Set the 'CS2_PSEUDONYMIZATION_SALT' environment variable or provide an explicit 'salt' parameter."
        )
    raw_str = f"{steamid}_{effective_salt}".encode('utf-8')
    h = hashlib.sha256(raw_str).hexdigest()
    # Map first 15 hex characters to positive 60-bit integer (fits PyTorch int64 and Parquet int64)
    return int(h[:15], 16)


def pseudonymize_match_id(match_id: str, salt: Optional[str] = None) -> str:
    """
    Cryptographically pseudonymizes external match identifiers (e.g. FACEIT match UUIDs)
    using a managed secret salt, preventing direct linking against public match scoreboards.
    Strictly enforces presence of secret salt to comply with RA 10173 data privacy governance.
    """
    effective_salt = salt or os.environ.get("CS2_PSEUDONYMIZATION_SALT")
    if not effective_salt:
        raise ValueError(
            "Cryptographic pseudonymization under RA 10173 requires a managed secret salt. "
            "Set the 'CS2_PSEUDONYMIZATION_SALT' environment variable or provide an explicit 'salt' parameter."
        )
    raw_str = f"{match_id}_{effective_salt}".encode('utf-8')
    h = hashlib.sha256(raw_str).hexdigest()
    return f"match_{h[:16]}"



FEATURE_COLUMNS = MODEL_FEATURE_COLUMNS

COMBAT_EVENT_NAMES = ['weapon_fire', 'player_hurt', 'player_death']
PLAYING_TEAMS = {2, 3}  # CS2 team_num: 2 = T, 3 = CT (0/1 are unassigned/spectators)


def collect_combat_event_ticks(events_dict: Dict[str, pd.DataFrame]) -> Dict[str, set]:
    """
    Thesis Reference: Chapter 3, Section 3.2.3 — Combat Temporal Buffers.
    Maps str(steamid) -> set of ticks where that player fired, dealt or took damage, or died/killed.
    """
    event_ticks_by_player: Dict[str, set] = {}
    for evt_name in COMBAT_EVENT_NAMES:
        evt_df = events_dict.get(evt_name, pd.DataFrame())
        if evt_df.empty or 'tick' not in evt_df.columns:
            continue
        for col in ['user_steamid', 'attacker_steamid']:
            if col in evt_df.columns:
                for s_id, grp in evt_df.groupby(col):
                    event_ticks_by_player.setdefault(str(s_id), set()).update(
                        grp['tick'].dropna().astype(int).tolist()
                    )
    return event_ticks_by_player


def extract_player_feature_windows(
    ticks_df: pd.DataFrame,
    steamid: Union[str, int],
    event_ticks: List[int],
    min_window_len: int = 64,
    tick_rate: float = 64.0,
    use_fov_filter: bool = True
) -> List[pd.DataFrame]:
    """
    Single preprocessing path shared by training ingestion and match analysis
    (prevents train/inference skew).

    1. Selects the player's live ticks and the living opposing team's ticks.
    2. Extracts Active Tracking Windows (FOV cone + combat event buffers).
    3. Computes per-tick target aim error over the player's full stream, then slices it per ATW.
    4. Computes the candidate kinematic channels per ATW.

    Returns a list of featured ATW DataFrames (columns include MODEL_FEATURE_COLUMNS).
    """
    p_df = ticks_df[ticks_df['steamid'] == steamid]
    if 'is_alive' in p_df.columns:
        p_df = p_df[p_df['is_alive'].astype(bool)]
    p_df = p_df.sort_values('tick').reset_index(drop=True)
    if len(p_df) < min_window_len:
        return []

    enemy_df = None
    if 'team_num' in p_df.columns:
        # Opponents are decided per tick: teams swap sides at halftime, so a team fixed from the
        # first tick would turn teammates (and the player themself, at distance 0) into "enemies".
        p_team_by_tick = p_df[['tick', 'team_num']].rename(columns={'team_num': 'p_team'})
        others = ticks_df[(ticks_df['steamid'] != steamid) & ticks_df['team_num'].isin(PLAYING_TEAMS)]
        others = others.merge(p_team_by_tick, on='tick')
        enemy_df = others[others['team_num'] != others['p_team']].drop(columns='p_team')

    atw_segments = extract_active_tracking_windows(
        player_df=p_df,
        enemy_df=enemy_df if use_fov_filter else None,
        event_ticks=sorted(event_ticks),
        fov_deg=30.0,
        tick_buffer=64,
        min_window_len=min_window_len,
        max_window_len=512
    )
    if not atw_segments:
        return []

    aim_error_by_tick = pd.Series(compute_aim_error(p_df, enemy_df), index=p_df['tick'].to_numpy())
    featured = []
    for seg_df in atw_segments:
        seg_aim_error = aim_error_by_tick.reindex(seg_df['tick'].to_numpy()).fillna(np.pi).to_numpy()
        featured.append(compute_kinematic_features(seg_df, tick_rate=tick_rate, extract_tremor=True, aim_error=seg_aim_error))
    return featured



def process_single_demo(
    demo_path: str, 
    output_dir: str, 
    is_cheater_demo: bool = False,
    banned_steamids: Optional[Iterable[Union[str, int]]] = None,
    player_elos: Optional[Dict[Union[str, int], float]] = None,
    default_elo: Optional[float] = None,
    min_window_len: int = 64,
    tick_rate: float = 64.0,
    use_fov_filter: bool = True,
    pseudonymize_matches: bool = True,
    salt: Optional[str] = None
) -> int:
    """
    Processes a single CS2 .dem replay:
    1. Parses player ticks and weapon events.
    2. Identifies all players and teams.
    3. Extracts Active Tracking Windows (30-deg FOV cones & combat event buffers) and computes
       the candidate channels via extract_player_feature_windows (shared with analyze_match.py).
    4. Labels ONLY accounts in the verified ban set (account-level weak label; see loop comment).
    6. Saves trajectory segments to Parquet.
    
    Returns the count of extracted ATW trajectory segments.
    """
    match_name = os.path.splitext(os.path.basename(demo_path))[0]
    os.makedirs(output_dir, exist_ok=True)
    
    banned_set = {str(b) for b in banned_steamids} if banned_steamids else set()
    
    try:
        parser = CS2DemoParser(demo_path)
        ticks_df = parser.parse_ticks()
        if ticks_df.empty:
            logging.warning(f"No tick data in {demo_path}")
            return 0
            
        events_dict = parser.parse_events()
        event_ticks_by_player = collect_combat_event_ticks(events_dict)
        
        # Load external banned steamid and ELO manifests if needed
        discovered_elos = {}
        manifest_paths = [
            os.path.join(os.path.dirname(demo_path), "banned_steamids.json"),
            os.path.join(os.path.dirname(demo_path), "..", "banned_steamids.json"),
            os.path.join("data", "banned_steamids.json"),
            os.path.join("data", "staging_manifest.json"),
            os.path.join("data", "raw_demos", "staging", "staging_manifest.json"),
            os.path.join("data", "raw_demos", "staging_manifest.json"),
            os.path.join(os.path.dirname(demo_path), "..", "staging", "staging_manifest.json"),
            os.path.join("data", "player_elos.json")
        ]
        for candidate_path in manifest_paths:
            if os.path.exists(candidate_path):
                try:
                    import json
                    with open(candidate_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    if isinstance(data, list):
                        if not banned_set:
                            banned_set.update(str(x) for x in data)
                    elif isinstance(data, dict):
                        for k, v in data.items():
                            if isinstance(v, (int, float)):
                                discovered_elos[str(k)] = float(v)
                            elif isinstance(v, dict) and 'players' in v:
                                for p in v.get('players', []):
                                    s_id = str(p.get('steam_id', ''))
                                    if s_id:
                                        if 'elo' in p:
                                            discovered_elos[s_id] = float(p['elo'])
                                        if v.get('status') == 'cheater_detected' and (p.get('nickname') == v.get('cheater') or p.get('player_id') == v.get('cheater')):
                                            banned_set.add(s_id)
                            elif isinstance(v, list) and not banned_set:
                                banned_set.update(str(x) for x in v)
                except Exception as e:
                    logging.warning(f"Failed to load manifest from {candidate_path}: {e}")

        # Combine discovered ELOs with explicitly passed dictionary
        combined_elos = {**discovered_elos, **(player_elos or {})}
        
        if 'team_num' in ticks_df.columns:
            steam_ids = ticks_df.loc[ticks_df['team_num'].isin(PLAYING_TEAMS), 'steamid'].dropna().unique()
        else:
            steam_ids = ticks_df['steamid'].dropna().unique()
        total_segments = 0
        
        for steamid in steam_ids:
            featured_segments = extract_player_feature_windows(
                ticks_df,
                steamid,
                event_ticks=list(event_ticks_by_player.get(str(steamid), [])),
                min_window_len=min_window_len,
                tick_rate=tick_rate,
                use_fov_filter=use_fov_filter
            )
            
            # Ground-truth labeling (weak supervision): is_aimbot = 1 means "this account received
            # a FACEIT cheating ban", NOT "aim assistance was active in this window". Bans do not
            # say which cheat was used (wallhack-only accounts are included) and humanized aimbots
            # act in only some engagements, so window labels are noisy; report player-match
            # session metrics as the primary result. Never label a whole lobby as cheaters.
            if banned_set:
                is_player_cheater = int(str(steamid) in banned_set)
            elif is_cheater_demo:
                # Fallback: the cheater's SteamID is embedded in the demo filename.
                is_player_cheater = int(str(steamid) in match_name)
            else:
                is_player_cheater = 0
                
            # Missing ELO stays NaN (masked out of the ELO loss) instead of a fake 1500 default.
            p_elo = combined_elos.get(str(steamid), default_elo)
            
            anon_steamid = pseudonymize_steamid(steamid, salt=salt)
            anon_match_id = pseudonymize_match_id(match_name, salt=salt) if pseudonymize_matches else match_name
            for seg_idx, featured_df in enumerate(featured_segments):
                # Metadata tags (RA 10173 Cryptographically Pseudonymized)
                featured_df['match_id'] = anon_match_id
                featured_df['steamid'] = anon_steamid
                featured_df['segment_id'] = seg_idx
                featured_df['is_aimbot'] = is_player_cheater
                featured_df['player_elo'] = float(p_elo) if p_elo is not None else np.nan
                
                # Export to Parquet
                out_filename = f"{anon_match_id}_p{anon_steamid}_seg{seg_idx}.parquet"
                featured_df.to_parquet(os.path.join(output_dir, out_filename), index=False)
                total_segments += 1

                
        logging.info(f"Processed {match_name}: {total_segments} ATW segments extracted.")
        return total_segments
        
    except Exception as e:
        logging.error(f"Error processing {demo_path}: {e}")
        return 0


def batch_process_demos(
    demo_dir: str, 
    output_dir: str, 
    is_cheater_dataset: bool = False,
    banned_steamids_map: Optional[Dict[str, List[Union[str, int]]]] = None,
    player_elos_map: Optional[Dict[str, Dict[Union[str, int], float]]] = None,
    tick_rate: float = 64.0,
    max_workers: int = 4,
    pseudonymize_matches: bool = True,
    salt: Optional[str] = None
) -> int:
    """Processes an entire directory of .dem files in parallel."""
    demo_files = glob.glob(os.path.join(demo_dir, "*.dem"))
    logging.info(f"Found {len(demo_files)} demos in {demo_dir}")
    
    total_extracted = 0
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        for demo in demo_files:
            m_name = os.path.splitext(os.path.basename(demo))[0]
            b_ids = banned_steamids_map.get(m_name) if banned_steamids_map else None
            p_elos = player_elos_map.get(m_name) if player_elos_map else None
            fut = executor.submit(
                process_single_demo, demo, output_dir, is_cheater_dataset, b_ids, p_elos, None, 64, tick_rate, True, pseudonymize_matches, salt
            )
            futures[fut] = demo
            
        for future in as_completed(futures):
            demo_path = futures[future]
            try:
                count = future.result()
                total_extracted += count
            except Exception as e:
                logging.error(f"Failed {demo_path}: {e}")
                
    logging.info(f"Batch completed: {total_extracted} total trajectory segments saved.")
    return total_extracted


def _count_parquet_labels(files: List[str], check_columns: bool = True) -> Tuple[int, int]:
    """Helper to count clean (0) and cheater (1) segments across a list of Parquet files."""
    clean_count = 0
    cheater_count = 0
    if check_columns and files:
        try:
            import pyarrow.parquet as pq
            for f in files:
                try:
                    tbl = pq.read_table(f, columns=['is_aimbot'])
                    label = tbl['is_aimbot'][0].as_py()
                    if label == 1:
                        cheater_count += 1
                    else:
                        clean_count += 1
                except Exception:
                    if 'cheater' in f.lower():
                        cheater_count += 1
                    else:
                        clean_count += 1
            return clean_count, cheater_count
        except ImportError:
            pass
    for f in files:
        if 'cheater' in f.lower():
            cheater_count += 1
        else:
            clean_count += 1
    return clean_count, cheater_count


def audit_clean_atw_quota(
    processed_dir: str, 
    target_clean: int = 30000,
    check_columns: bool = True,
    partition_test_split: bool = True,
    train_ratio: float = 0.80,
    val_ratio: float = 0.10,
    test_ratio: float = 0.10,
    seed: int = 42
) -> Dict[str, Union[int, float, bool]]:
    """
    Thesis Reference: Section 3.2.2 & Section 3.2.9 — Empirical Stopping Condition Audit.
    Inspects processed Parquet inventory to verify progress toward the empirical stopping
    condition of >= target_clean (default 30,000) clean ATW trajectory segments in the
    held-out test split.
    
    Parameters:
    -----------
    processed_dir: str
        Root directory containing processed ATW Parquet files (or 'clean' and 'cheaters' subdirs).
    target_clean: int
        Target count of verified clean ATWs (default 30,000 for Rule of Three FPR_95% <= 0.01%
        in the held-out test split).
    check_columns: bool
        If True, reads the 'is_aimbot' column to strictly distinguish clean (0) vs cheater (1) segments.
    partition_test_split: bool
        If True, applies the strict zero-leakage bipartite graph partitioner to extract the held-out
        test split files, auditing clean ATW quota directly against the test partition.
        If False, audits across the entire overall corpus directory.
        
    Returns:
    --------
    Dict containing:
      - 'test_clean_count': int, verified clean ATWs in held-out test partition
      - 'test_cheater_count': int, cheater ATWs in held-out test partition
      - 'test_total_count': int, total ATWs in held-out test partition
      - 'overall_clean_count': int, verified clean ATWs across entire corpus
      - 'overall_cheater_count': int, cheater ATWs across entire corpus
      - 'overall_total_count': int, total ATWs across entire corpus
      - 'clean_count': int, evaluated clean count (test_clean_count if partitioned, else overall)
      - 'cheater_count': int, evaluated cheater count
      - 'total_count': int, evaluated total count
      - 'target_clean': int, target quota
      - 'quota_met': bool, True if clean_count >= target_clean (strictly False if partition_test_split and test split unavailable)
      - 'progress_pct': float, clean_count / target_clean * 100.0
      - 'deficit': int, max(0, target_clean - clean_count)
      - 'is_test_split_audited': bool, True if quota is evaluated on held-out test partition
      - 'partition_error': Optional[str], error message if test split partitioning failed
    """
    if not os.path.exists(processed_dir):
        return {
            'test_clean_count': 0,
            'test_cheater_count': 0,
            'test_total_count': 0,
            'overall_clean_count': 0,
            'overall_cheater_count': 0,
            'overall_total_count': 0,
            'clean_count': 0,
            'cheater_count': 0,
            'total_count': 0,
            'target_clean': int(target_clean),
            'quota_met': False,
            'progress_pct': 0.0,
            'deficit': int(target_clean),
            'is_test_split_audited': False,
            'partition_error': f"Directory does not exist: {processed_dir}"
        }
        
    pattern = os.path.join(processed_dir, "**", "*.parquet")
    files = glob.glob(pattern, recursive=True)
    if not files:
        files = glob.glob(os.path.join(processed_dir, "*.parquet"))
    files = list(set(files))
    
    overall_clean, overall_cheater = _count_parquet_labels(files, check_columns=check_columns)
    overall_total = overall_clean + overall_cheater
    
    test_clean = 0
    test_cheater = 0
    is_test_audited = False
    partition_error = None
    
    if partition_test_split:
        if len(files) == 0:
            partition_error = "No parquet files found to partition."
        else:
            try:
                try:
                    from .dataset import partition_dataset_files
                except ImportError:
                    from data.dataset import partition_dataset_files
                    
                _, _, test_files = partition_dataset_files(
                    processed_dir,
                    train_ratio=train_ratio,
                    val_ratio=val_ratio,
                    test_ratio=test_ratio,
                    seed=seed
                )
                test_clean, test_cheater = _count_parquet_labels(test_files, check_columns=check_columns)
                is_test_audited = True
            except Exception as e:
                partition_error = str(e)
                logging.warning(
                    f"Dataset cannot yet be partitioned into disjoint test split ({e}); "
                    f"test split quota remains UNMET (failing closed)."
                )
                test_clean, test_cheater = 0, 0
                is_test_audited = False
                
    if partition_test_split:
        # Strictly enforce fail-closed evaluation: quota is evaluated on held-out test split.
        # If test split is unavailable, clean_count is 0 and quota_met is strictly False.
        eval_clean = test_clean if is_test_audited else 0
        eval_cheater = test_cheater if is_test_audited else 0
        eval_total = eval_clean + eval_cheater
        quota_met = bool(is_test_audited and (eval_clean >= target_clean))
        progress_pct = float(eval_clean / max(1, target_clean) * 100.0) if is_test_audited else 0.0
        deficit = int(max(0, target_clean - eval_clean))
    else:
        # User explicitly requested unpartitioned corpus-wide audit
        eval_clean = overall_clean
        eval_cheater = overall_cheater
        eval_total = overall_total
        quota_met = bool(eval_clean >= target_clean)
        progress_pct = float(eval_clean / max(1, target_clean) * 100.0)
        deficit = int(max(0, target_clean - eval_clean))
    
    return {
        'test_clean_count': test_clean,
        'test_cheater_count': test_cheater,
        'test_total_count': test_clean + test_cheater,
        'overall_clean_count': overall_clean,
        'overall_cheater_count': overall_cheater,
        'overall_total_count': overall_total,
        'clean_count': eval_clean,
        'cheater_count': eval_cheater,
        'total_count': eval_total,
        'target_clean': int(target_clean),
        'quota_met': quota_met,
        'progress_pct': progress_pct,
        'deficit': deficit,
        'is_test_split_audited': is_test_audited,
        'partition_error': partition_error
    }
