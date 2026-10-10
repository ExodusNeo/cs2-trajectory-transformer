"""Tests for the CS2CD Parquet + JSON adapter (labels, namespacing, skipping noisy negatives)."""

import json
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from data.cs2cd_adapter import load_cs2cd_match, process_cs2cd_match

SALT = "test-salt"


def _write_match(folder_path, stem, cheaters, n_ticks=300):
    """Two players per team; team 3 player stands 1000 units ahead of each team 2 player."""
    rows = []
    for tick in range(n_ticks):
        for i, (team, x) in enumerate([(2, 0.0), (2, 0.0), (3, 1000.0), (3, 1000.0)]):
            rows.append({'tick': tick, 'steamid': f'Player_{i + 1}', 'team_num': float(team), 'X': x, 'Y': 50.0 * i,
                         'Z': 0.0, 'pitch': 0.0, 'yaw': 0.0 if team == 2 else 180.0, 'is_alive': True,
                         'extra_col': 1.0})
    pd.DataFrame(rows).to_parquet(os.path.join(folder_path, f"{stem}.parquet"), index=False)
    fires = [{'tick': t, 'user_steamid': f'Player_{i + 1}', 'weapon': 'ak47'} for t in (100, 200) for i in range(4)]
    meta = {'weapon_fire': fires, 'player_hurt': [], 'player_death': [],
            'cheaters': [{'steamid': c} for c in cheaters],
            'CSstats_info': [{'map': 'de_test', 'server': 'x', 'avg_rank': 'Gold Nova I', 'match_making_type': 'Official Matchmaking'}]}
    with open(os.path.join(folder_path, f"{stem}.json"), 'w', encoding='utf-8') as f:
        json.dump(meta, f)


def _labels(out_dir):
    files = sorted(out_dir.glob("*.parquet"))
    frames = [pd.read_parquet(f, columns=['steamid', 'match_id', 'is_aimbot', 'player_elo', 'source']) for f in files]
    return pd.concat([f.iloc[:1] for f in frames]) if frames else pd.DataFrame()


def test_load_cs2cd_match_shapes(tmp_path):
    _write_match(tmp_path, "7", cheaters=["Player_3"])
    ticks, events, cheaters, info = load_cs2cd_match(str(tmp_path / "7.parquet"), str(tmp_path / "7.json"))
    assert list(ticks.columns) == ['tick', 'steamid', 'team_num', 'X', 'Y', 'Z', 'pitch', 'yaw', 'is_alive']
    assert cheaters == {"Player_3"} and info['map'] == 'de_test'
    assert len(events['weapon_fire']) == 8 and events['player_hurt'].empty


def test_cheater_match_skips_unlabeled_players_by_default(tmp_path):
    raw = tmp_path / "with_cheater_present"
    raw.mkdir()
    _write_match(raw, "1", cheaters=["Player_3"])
    out = tmp_path / "out"
    counts = process_cs2cd_match(str(raw / "1.parquet"), str(raw / "1.json"), str(out), "with_cheater_present", salt=SALT)
    lab = _labels(out)
    assert counts['skipped_players'] == 3 and counts['negative'] == 0 and counts['positive'] > 0
    assert set(lab['is_aimbot']) == {1}
    assert (lab['source'] == 'cs2cd').all() and lab['player_elo'].isna().all()


def test_clean_match_players_are_negatives_and_ids_are_namespaced(tmp_path):
    out = tmp_path / "out"
    for stem in ("1", "2"):
        d = tmp_path / "no_cheater_present"
        d.mkdir(exist_ok=True)
        _write_match(d, stem, cheaters=[])
        process_cs2cd_match(str(d / f"{stem}.parquet"), str(d / f"{stem}.json"), str(out), "no_cheater_present", salt=SALT)
    lab = _labels(out)
    assert set(lab['is_aimbot']) == {0}
    # "Player_1" exists in both matches but must map to two different pseudonymous players.
    per_match = lab.groupby('match_id')['steamid'].apply(set).tolist()
    assert len(per_match) == 2 and per_match[0].isdisjoint(per_match[1])


def test_invalid_folder_rejected(tmp_path):
    _write_match(tmp_path, "1", cheaters=[])
    with pytest.raises(ValueError):
        process_cs2cd_match(str(tmp_path / "1.parquet"), str(tmp_path / "1.json"), str(tmp_path / "o"), "bogus", salt=SALT)


def test_file_validation_detects_truncated_parquet_and_bad_json(tmp_path):
    from data.cs2cd_adapter import is_valid_cs2cd_file
    _write_match(tmp_path, "5", cheaters=[])
    good = tmp_path / "5.parquet"
    assert is_valid_cs2cd_file(str(good)) and is_valid_cs2cd_file(str(tmp_path / "5.json"))
    truncated = tmp_path / "6.parquet"
    truncated.write_bytes(good.read_bytes()[:-100])
    assert not is_valid_cs2cd_file(str(truncated))
    (tmp_path / "6.json").write_text('{"cheaters": [', encoding='utf-8')
    assert not is_valid_cs2cd_file(str(tmp_path / "6.json"))


def test_already_processed_match_is_skipped(tmp_path):
    raw = tmp_path / "no_cheater_present"
    raw.mkdir()
    _write_match(raw, "1", cheaters=[])
    out = tmp_path / "out"
    first = process_cs2cd_match(str(raw / "1.parquet"), str(raw / "1.json"), str(out), "no_cheater_present", salt=SALT)
    second = process_cs2cd_match(str(raw / "1.parquet"), str(raw / "1.json"), str(out), "no_cheater_present", salt=SALT)
    assert first['negative'] > 0 and second.get('already_processed') == 1
