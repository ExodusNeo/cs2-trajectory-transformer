"""
Unit Tests for CS2 Replay Downloader and Decompression Engine.
"""

import sys
import os
import gzip
import zipfile
import tempfile
import shutil
import pytest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from data.demo_downloader import CS2ReplayDownloader


@pytest.fixture
def temp_downloader():
    temp_dir = tempfile.mkdtemp()
    downloader = CS2ReplayDownloader(base_dir=temp_dir)
    yield downloader, temp_dir
    shutil.rmtree(temp_dir)


def test_gz_decompression(temp_downloader):
    downloader, temp_dir = temp_downloader
    dummy_dem_content = b"HL2DEMO_HEADER_TEST_BYTES_CS2"
    
    # Create fake .dem.gz
    gz_path = os.path.join(temp_dir, "test_match.dem.gz")
    with gzip.open(gz_path, 'wb') as f:
        f.write(dummy_dem_content)
        
    extracted = downloader.decompress_archive(gz_path, downloader.clean_dir)
    
    assert len(extracted) == 1
    assert os.path.exists(extracted[0])
    assert extracted[0].endswith(".dem")
    
    with open(extracted[0], 'rb') as f:
        assert f.read() == dummy_dem_content


def test_zip_decompression(temp_downloader):
    downloader, temp_dir = temp_downloader
    dummy_content = b"CS2_DEMO_ZIP_PAYLOAD"
    
    # Create fake .zip containing match1.dem and match2.dem
    zip_path = os.path.join(temp_dir, "tournament_pack.zip")
    with zipfile.ZipFile(zip_path, 'w') as zf:
        zf.writestr("match1.dem", dummy_content)
        zf.writestr("match2.dem", dummy_content)
        zf.writestr("readme.txt", "Some text file")
        
    extracted = downloader.decompress_archive(zip_path, downloader.clean_dir)
    
    # Should only extract the two .dem files, ignoring readme.txt
    assert len(extracted) == 2
    for dem in extracted:
        assert os.path.exists(dem)
        assert dem.endswith(".dem")


def test_inventory_listing(temp_downloader):
    downloader, _ = temp_downloader
    
    # Create fake clean and cheater files
    with open(os.path.join(downloader.clean_dir, "clean1.dem"), "wb") as f:
        f.write(b"demo1")
    with open(os.path.join(downloader.cheater_dir, "cheat1.dem"), "wb") as f:
        f.write(b"demo2")
        
    inv = downloader.list_downloaded_demos()
    assert len(inv['clean']) == 1
    assert len(inv['cheaters']) == 1
    assert inv['total'] == 2


def test_zip_slip_protection(temp_downloader):
    downloader, temp_dir = temp_downloader
    dummy_content = b"MALICIOUS_SLIP_CONTENT"
    
    # Create a zip containing a path traversal entry
    zip_path = os.path.join(temp_dir, "traversal_pack.zip")
    with zipfile.ZipFile(zip_path, 'w') as zf:
        zf.writestr("../../escape.dem", dummy_content)
        zf.writestr("subfolder/nested.dem", dummy_content)
        
    extracted = downloader.decompress_archive(zip_path, downloader.clean_dir)
    
    # Verify all extracted files remain strictly inside clean_dir
    assert len(extracted) == 2
    for dem in extracted:
        assert os.path.dirname(os.path.abspath(dem)) == os.path.abspath(downloader.clean_dir)
        assert os.path.exists(dem)


def test_download_url_https_validation(temp_downloader):
    downloader, _ = temp_downloader
    insecure_url = "http://insecure-cdn.example.com/match.dem.zst"
    result = downloader.download_url(insecure_url)
    assert result == []


def test_cheating_ban_verification(monkeypatch, temp_downloader):
    """Verify that is_banned_for_cheating identifies cheating bans while ignoring others."""
    downloader, _ = temp_downloader

    # Mock get_player_bans to return a cheating ban
    monkeypatch.setattr(downloader, 'get_player_bans', lambda pid, api_key=None: [
        {'reason': 'verbal abuse', 'starts_at': 1000},
        {'reason': 'Cheating / Aimbot Detected', 'starts_at': 2000}
    ])
    
    is_cheat, starts_at, reason = downloader.is_banned_for_cheating('test_player_1')
    assert is_cheat is True
    assert starts_at == 2000
    assert 'Cheating' in reason

    # Mock non-cheater player (e.g. only AFK or clean)
    monkeypatch.setattr(downloader, 'get_player_bans', lambda pid, api_key=None: [
        {'reason': 'leaving match', 'starts_at': 500}
    ])
    is_cheat, starts_at, reason = downloader.is_banned_for_cheating('test_player_2')
    assert is_cheat is False
    assert starts_at is None


def test_pre_ban_match_filtering(monkeypatch, temp_downloader):
    """Verify that pre-ban matches are selected based on match timestamps <= ban starts_at."""
    downloader, _ = temp_downloader
    ban_timestamp = 1700000000

    monkeypatch.setattr(downloader, 'is_banned_for_cheating', lambda pid, api_key=None: (True, ban_timestamp, 'cheating'))
    
    # Mock player profile resolution
    import json
    fake_player_body = json.dumps({'player_id': 'pid_123', 'nickname': 'cheat_user'}).encode('utf-8')
    
    # Mock history: match 1 is after ban (e.g. unbanned or edge), match 2 is right before ban
    fake_history_body = json.dumps({
        'items': [
            {'match_id': 'm_post_ban', 'started_at': ban_timestamp + 10000},
            {'match_id': 'm_pre_ban', 'started_at': ban_timestamp - 1500, 'finished_at': ban_timestamp - 500}
        ]
    }).encode('utf-8')
    
    from data import demo_downloader
    def mock_polite_request(req, max_retries=3, initial_delay=0.4):
        url = req.full_url
        if '/players?nickname=' in url:
            return fake_player_body
        elif '/history' in url:
            return fake_history_body
        return b""
        
    monkeypatch.setattr(demo_downloader, 'polite_request', mock_polite_request)
    
    downloaded_matches = []
    monkeypatch.setattr(downloader, 'fetch_faceit_match_demo', lambda mid, api_key=None, is_cheater=True: [f"{mid}.dem"])
    
    dems = downloader.fetch_banned_cheater_matches(['cheat_user'], matches_per_player=1, verify_ban=True)
    assert len(dems) == 1
    assert dems[0] == "m_pre_ban.dem"


def test_staging_manifest_and_audit(temp_downloader, monkeypatch):
    """Verify staging buffer manifest tracking and deferred ban promotion/graduation."""
    import time
    downloader, temp_dir = temp_downloader
    
    # 1. Create fake staged demo file
    fake_demo = os.path.join(downloader.staging_dir, "test_staged.dem")
    with open(fake_demo, "wb") as f:
        f.write(b"HL2DEMO_STAGED")
        
    manifest = {
        "match_cheat": {
            "match_id": "match_cheat",
            "staged_at": int(time.time()),
            "finished_at": int(time.time()),
            "demo_files": ["test_staged.dem"],
            "players": [{"player_id": "p_cheat", "nickname": "cheater1"}],
            "status": "pending_audit"
        },
        "match_clean": {
            "match_id": "match_clean",
            "staged_at": int(time.time() - 25 * 86400),  # 25 days old
            "finished_at": int(time.time() - 25 * 86400),
            "demo_files": ["test_clean.dem"],
            "players": [{"player_id": "p_clean", "nickname": "clean1"}],
            "status": "pending_audit"
        }
    }
    # Create fake clean staged demo
    with open(os.path.join(downloader.staging_dir, "test_clean.dem"), "wb") as f:
        f.write(b"HL2DEMO_CLEAN")
        
    downloader.save_staging_manifest(manifest)
    
    # Mock ban check: p_cheat is banned, p_clean is clean
    def mock_is_banned(pid, api_key=None):
        if pid == "p_cheat":
            return True, int(time.time()), "cheating"
        return False, None, None
        
    monkeypatch.setattr(downloader, 'is_banned_for_cheating', mock_is_banned)
    
    stats = downloader.audit_staging(graduation_days=21, auto_extract=False)
    
    assert stats['cheaters_detected'] == 1
    assert stats['clean_graduated'] == 1
    
    # Check that demo files were moved to appropriate folders
    assert os.path.exists(os.path.join(downloader.cheater_dir, "test_staged.dem"))
    assert os.path.exists(os.path.join(downloader.clean_dir, "test_clean.dem"))
    
    updated_manifest = downloader.load_staging_manifest()
    assert updated_manifest["match_cheat"]["status"] == "cheater_detected"
    assert updated_manifest["match_clean"]["status"] == "clean_graduated"



