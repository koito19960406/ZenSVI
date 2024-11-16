import shutil

import pandas as pd
import pytest

from zensvi.download.ams import AMSDownloader

from .conftest import TimeoutException


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


def test_download_asv(output_dir, sv_downloader, timeout):
    try:
        with timeout(300):  # Let it run for 5 minutes
            sv_downloader.download_svi(output_dir, lat=52.356768, lon=4.907408, buffer=10)
    except TimeoutException:
        pass  # Allow timeout, we'll check results next

    assert len(list(output_dir.iterdir())) > 0, "No files downloaded within 5 minutes"


def test_download_asv_metadata_only(output_dir, sv_downloader, timeout):
    try:
        with timeout(300):  # Let it run for 5 minutes
            sv_downloader.download_svi(output_dir, lat=52.356768, lon=4.907408, buffer=10, metadata_only=True)
            # Try to check CSV first if within timeout
            df = pd.read_csv(output_dir / "ams_pids.csv")
            assert df.shape[0] > 1
    except TimeoutException:
        # Fall back to checking if any files exist
        assert len(list(output_dir.iterdir())) > 0, "No files downloaded within 5 minutes"


def test_csv_download_asv(output_dir, sv_downloader, input_dir, timeout):
    try:
        with timeout(300):  # Let it run for 5 minutes
            sv_downloader.download_svi(output_dir, input_csv_file=input_dir / "test_ams.csv", buffer=10)
    except TimeoutException:
        pass  # Allow timeout, we'll check results next

    assert len(list(output_dir.iterdir())) > 0, "No files downloaded within 5 minutes"


def test_shp_download_asv(output_dir, sv_downloader, input_dir, timeout):
    file_list = ["point_ams.geojson", "line_ams.geojson", "polygon_ams.geojson"]
    for file in file_list:
        try:
            with timeout(300):  # Let it run for 5 minutes
                sv_downloader.download_svi(output_dir, input_shp_file=input_dir / file, buffer=10)
        except TimeoutException:
            pass  # Allow timeout, we'll check results next

        assert len(list(output_dir.iterdir())) > 0, f"No files downloaded within 5 minutes for {file}"


# def test_place_name_download_asv(output_dir, sv_downloader, timeout):
#     try:
#         with timeout(30):  # Let it run for 5 minutes
#             sv_downloader.download_svi(output_dir, input_place_name="West Amsterdam, Netherlands", buffer=10)
#     except TimeoutException:
#         pass  # Allow timeout, we'll check results next

#     assert len(list(output_dir.iterdir())) > 0, "No files downloaded within 5 minutes"
