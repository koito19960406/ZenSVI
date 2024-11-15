import shutil
import pandas as pd
import pytest

from zensvi.download.ams import AMSDownloader


@pytest.fixture(autouse=True)
def cleanup_after_test(output_dir):
    """Fixture to clean up downloaded files after each test"""
    yield
    if output_dir.exists():
        shutil.rmtree(output_dir)


@pytest.fixture
def output_dir(base_output_dir, ensure_dir):
    output_dir = base_output_dir / "ams_svi"
    if output_dir.exists():
        shutil.rmtree(output_dir)
    ensure_dir(output_dir)
    return output_dir


@pytest.fixture
def sv_downloader(output_dir):
    return AMSDownloader(log_path=output_dir / "log.log")


def test_download_asv(output_dir, sv_downloader):
    sv_downloader.download_svi(output_dir, lat=52.356768, lon=4.907408, buffer=10)
    assert len(list(output_dir.iterdir())) > 0


def test_download_asv_metadata_only(output_dir, sv_downloader):
    sv_downloader.download_svi(output_dir, lat=52.356768, lon=4.907408, buffer=10, metadata_only=True)
    df = pd.read_csv(output_dir / "ams_pids.csv")
    assert df.shape[0] > 1


def test_csv_download_asv(output_dir, sv_downloader, input_dir):
    sv_downloader.download_svi(output_dir, input_csv_file=input_dir / "test_ams.csv", buffer=10)
    assert len(list(output_dir.iterdir())) > 0


def test_shp_download_asv(output_dir, sv_downloader, input_dir):
    file_list = ["point_ams.geojson", "line_ams.geojson", "polygon_ams.geojson"]
    for file in file_list:
        sv_downloader.download_svi(output_dir, input_shp_file=input_dir / file, buffer=10)
        assert len(list(output_dir.iterdir())) > 0


def test_place_name_download_asv(output_dir, sv_downloader):
    sv_downloader.download_svi(output_dir, input_place_name="Amsterdam Landlust", buffer=10)
    assert len(list(output_dir.iterdir())) > 0
