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
