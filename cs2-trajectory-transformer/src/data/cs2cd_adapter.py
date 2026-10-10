"""
CS2CD (Counter-Strike 2 Cheat Detection, Loo & Luzkov, ITU; CC-BY-4.0, doi:10.57967/hf/5654) adapter.

Dataset layout (https://huggingface.co/datasets/CS2CD/CS2CD.Counter-Strike_2_Cheat_Detection):
    no_cheater_present/N.parquet + N.json    (478 matches)
    with_cheater_present/N.parquet + N.json  (317 matches)
- N.parquet: demoparser2 tick table, 10 rows per tick (one per player), 64 Hz.
- N.json: events keyed by name (weapon_fire, player_hurt, player_death, ...), the labels as
  "cheaters": [{"steamid": "Player_9"}], and "CSstats_info": [{map, server, avg_rank, match_making_type}].

Labeling policy (from the dataset card's own audit):
- Listed cheaters are positives.
- In with_cheater_present matches the "not cheater" label is only ~55.6% precise, so unlabeled
  players there are SKIPPED by default (not used as negatives).
- Negatives come from no_cheater_present matches (~97.2% of players clean per the card). This
  contamination bounds how low an FPR can be measured on CS2CD.

Source differences vs FACEIT .dem data: Valve Official Matchmaking with VAC bans and Valve ranks
(no FACEIT ELO, so player_elo is NaN). Player IDs are "Player_1..10" per match, so they are
namespaced by match before pseudonymization: CS2CD supports no cross-match player identity.
"""

import glob
import json
import logging
import os
import random
import time
import urllib.request
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd

from .batch_processor import (
    COMBAT_EVENT_NAMES,
    PLAYING_TEAMS,
    collect_combat_event_ticks,
    extract_player_feature_windows,
    pseudonymize_match_id,
    pseudonymize_steamid,
    write_feature_windows,
)

HF_REPO = "CS2CD/CS2CD.Counter-Strike_2_Cheat_Detection"
HF_API_TREE = f"https://huggingface.co/api/datasets/{HF_REPO}/tree/main/{{folder}}"
HF_RESOLVE = f"https://huggingface.co/datasets/{HF_REPO}/resolve/main/{{path}}"
CS2CD_FOLDERS = ("with_cheater_present", "no_cheater_present")
TICK_COLUMNS = ['tick', 'steamid', 'team_num', 'X', 'Y', 'Z', 'pitch', 'yaw', 'is_alive']


def load_cs2cd_match(parquet_path: str, json_path: str) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame], Set[str], Dict]:
    """
    Loads one CS2CD match into the same shapes the .dem pipeline uses.

    Returns (ticks_df with TICK_COLUMNS, events_dict of combat-event DataFrames,
    set of cheater steamid strings, CSstats_info dict).
    """
    ticks_df = pd.read_parquet(parquet_path, columns=TICK_COLUMNS)
    with open(json_path, encoding='utf-8') as f:
        meta = json.load(f)
    events = {name: pd.DataFrame(meta.get(name, [])) for name in COMBAT_EVENT_NAMES}
    cheaters = {str(c['steamid']) for c in meta.get('cheaters', []) if c.get('steamid')}
    info_list = meta.get('CSstats_info') or [{}]
    return ticks_df, events, cheaters, dict(info_list[0])


def process_cs2cd_match(
    parquet_path: str,
    json_path: str,
    output_dir: str,
    folder: str,
    salt: Optional[str] = None,
    include_unlabeled_in_cheater_matches: bool = False,
    min_window_len: int = 64,
    skip_existing: bool = True
) -> Dict[str, int]:
    """
    Extracts featured ATWs for every playing participant of one CS2CD match and writes Parquet
    windows with source='cs2cd'. Returns counts {'positive', 'negative', 'skipped_players'}.
    With skip_existing, a match whose windows already exist in output_dir is not reprocessed.
    """
    if folder not in CS2CD_FOLDERS:
        raise ValueError(f"folder must be one of {CS2CD_FOLDERS}, got '{folder}'")
    os.makedirs(output_dir, exist_ok=True)
    match_name = f"cs2cd_{folder}_{os.path.splitext(os.path.basename(parquet_path))[0]}"
    anon_match_id = pseudonymize_match_id(match_name, salt=salt)
    if skip_existing and glob.glob(os.path.join(output_dir, f"{anon_match_id}_p*.parquet")):
        return {'positive': 0, 'negative': 0, 'skipped_players': 0, 'already_processed': 1}
    ticks_df, events, cheaters, info = load_cs2cd_match(parquet_path, json_path)
    if folder == "no_cheater_present" and cheaters:
        logging.warning(f"{match_name}: no_cheater_present match lists cheaters {sorted(cheaters)}; labelling them positive.")

    event_ticks = collect_combat_event_ticks(events)
    extra = {'source': 'cs2cd', 'map': str(info.get('map', '')), 'avg_rank': str(info.get('avg_rank', ''))}
    counts = {'positive': 0, 'negative': 0, 'skipped_players': 0}

    players = ticks_df.loc[ticks_df['team_num'].isin(PLAYING_TEAMS), 'steamid'].dropna().unique()
    for steamid in players:
        sid = str(steamid)
        if sid in cheaters:
            label = 1
        elif folder == "no_cheater_present" or include_unlabeled_in_cheater_matches:
            label = 0
        else:
            counts['skipped_players'] += 1
            continue
        windows = extract_player_feature_windows(
            ticks_df, steamid, event_ticks=list(event_ticks.get(sid, [])), min_window_len=min_window_len
        )
        # "Player_k" repeats in every match: namespace by match before hashing.
        anon_steamid = pseudonymize_steamid(f"{match_name}:{sid}", salt=salt)
        n = write_feature_windows(windows, output_dir, anon_match_id, anon_steamid, label=label, elo=None, extra=extra)
        counts['positive' if label else 'negative'] += n
    return counts


def list_cs2cd_matches(folder: str, timeout: float = 30.0) -> List[str]:
    """Lists match numbers (as strings) that have both .parquet and .json in a CS2CD folder."""
    url = HF_API_TREE.format(folder=folder)
    paths: List[str] = []
    while url:
        req = urllib.request.Request(url, headers={'User-Agent': 'CS2TrajectoryTransformer/1.0 (Thesis Research)'})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            paths += [e['path'] for e in json.loads(resp.read().decode('utf-8'))]
            link = resp.headers.get('Link', '') or ''
        url = link.split(';')[0].strip('<> ') if 'rel="next"' in link else None
    stems = {os.path.splitext(os.path.basename(p))[0]: set() for p in paths}
    for p in paths:
        stem, ext = os.path.splitext(os.path.basename(p))
        stems[stem].add(ext)
    return sorted((s for s, exts in stems.items() if {'.parquet', '.json'} <= exts), key=lambda s: int(s) if s.isdigit() else s)


def is_valid_cs2cd_file(path: str) -> bool:
    """True if the file exists and is complete: Parquet has head+footer magic, JSON parses."""
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return False
    if path.endswith('.parquet'):
        with open(path, 'rb') as f:
            head = f.read(4)
            f.seek(-4, os.SEEK_END)
            return head == b'PAR1' and f.read(4) == b'PAR1'
    try:
        with open(path, encoding='utf-8') as f:
            json.load(f)
        return True
    except (ValueError, OSError):
        return False


def _download_verified(url: str, dst: str, retries: int = 3) -> None:
    """
    Streams url to dst via a .part file. A dropped connection can end the stream early without an
    exception, so the byte count is checked against Content-Length and the file is validated
    before it replaces dst. Raises after `retries` failed attempts.
    """
    tmp = dst + '.part'
    for attempt in range(1, retries + 1):
        req = urllib.request.Request(url, headers={'User-Agent': 'CS2TrajectoryTransformer/1.0 (Thesis Research)'})
        try:
            with urllib.request.urlopen(req, timeout=120) as resp, open(tmp, 'wb') as out:
                expected = int(resp.headers.get('Content-Length') or -1)
                written = 0
                while chunk := resp.read(1 << 20):
                    out.write(chunk)
                    written += len(chunk)
            if expected >= 0 and written != expected:
                raise IOError(f"truncated: {written} of {expected} bytes")
            os.replace(tmp, dst)
            if not is_valid_cs2cd_file(dst):
                os.remove(dst)
                raise IOError("downloaded file failed validation")
            return
        except OSError as e:
            logging.warning(f"download attempt {attempt}/{retries} failed for {url}: {e}")
            time.sleep(2.0 * attempt)
    raise IOError(f"could not download {url} after {retries} attempts")


def download_cs2cd_subset(dest_root: str, per_folder: int, seed: int = 0, pause_s: float = 0.5) -> Dict[str, List[str]]:
    """
    Downloads a seeded random subset of `per_folder` matches from each CS2CD folder into
    dest_root/<folder>/. Existing complete files are skipped (resumable). Returns chosen stems.
    """
    chosen: Dict[str, List[str]] = {}
    rng = random.Random(seed)
    for folder in CS2CD_FOLDERS:
        stems = list_cs2cd_matches(folder)
        pick = sorted(rng.sample(stems, min(per_folder, len(stems))), key=lambda s: int(s) if s.isdigit() else s)
        chosen[folder] = pick
        os.makedirs(os.path.join(dest_root, folder), exist_ok=True)
        for stem in pick:
            for ext in ('.json', '.parquet'):
                dst = os.path.join(dest_root, folder, stem + ext)
                if is_valid_cs2cd_file(dst):
                    continue
                _download_verified(HF_RESOLVE.format(path=f"{folder}/{stem}{ext}"), dst)
                time.sleep(pause_s)
            logging.info(f"downloaded {folder}/{stem}")
    return chosen


def _process_one(args: Tuple[str, str, str, str, Optional[str]]) -> Tuple[str, Dict[str, int]]:
    pq_path, js_path, out_dir, folder, salt = args
    try:
        return pq_path, process_cs2cd_match(pq_path, js_path, out_dir, folder, salt=salt)
    except Exception as e:  # one corrupt match must not stop the batch; it is reported, not hidden
        logging.error(f"CS2CD match failed {pq_path}: {e}")
        return pq_path, {'error': 1}


def process_cs2cd_directory(root: str, output_dir: str, salt: Optional[str] = None, max_workers: int = 4) -> Dict[str, int]:
    """Processes every downloaded match under root/<folder>/ in parallel. Call under __main__ on Windows."""
    jobs = []
    for folder in CS2CD_FOLDERS:
        fdir = os.path.join(root, folder)
        if not os.path.isdir(fdir):
            continue
        for name in sorted(os.listdir(fdir)):
            if name.endswith('.parquet'):
                js = os.path.join(fdir, name[:-len('.parquet')] + '.json')
                if not (is_valid_cs2cd_file(os.path.join(fdir, name)) and is_valid_cs2cd_file(js)):
                    logging.error(f"Skipping incomplete/corrupt CS2CD match {os.path.join(fdir, name)}; rerun the download to repair it.")
                    continue
                if os.path.exists(js):
                    jobs.append((os.path.join(fdir, name), js, output_dir, folder, salt))
    totals = {'matches': 0, 'positive': 0, 'negative': 0, 'skipped_players': 0, 'already_processed': 0, 'errors': 0}
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        for fut in as_completed([ex.submit(_process_one, j) for j in jobs]):
            _, c = fut.result()
            totals['matches'] += 1
            totals['errors'] += c.get('error', 0)
            for k in ('positive', 'negative', 'skipped_players', 'already_processed'):
                totals[k] += c.get(k, 0)
    logging.info(f"CS2CD processing totals: {totals}")
    return totals
