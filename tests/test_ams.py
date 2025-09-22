import shutil

import pandas as pd
import pytest

from zensvi.download.ams import AMSDownloader

from .conftest import TimeoutException


@pytest.fixture(scope="function")  # Explicitly set function scope
def output_dir(base_output_dir, ensure_dir):
    output_dir = base_output_dir / "ams_svi"
    if output_dir.exists():
        print(f"Cleaning up existing {output_dir} before test function")  # Optional: for debugging
        shutil.rmtree(output_dir)
    ensure_dir(output_dir)
    return output_dir


@pytest.fixture
def sv_downloader(output_dir):
    return AMSDownloader(log_path=output_dir / "log.log")


def test_download_asv(output_dir, sv_downloader, timeout_decorator, timeout_seconds):
    try:
        with timeout_decorator():
            sv_downloader.download_svi(output_dir, lat=52.356768, lon=4.907408, buffer=50)
    except TimeoutException:
        pass

    assert len(list(output_dir.iterdir())) > 0, f"No files downloaded within {timeout_seconds} seconds"


def test_download_asv_metadata_only(output_dir, sv_downloader, timeout_decorator, timeout_seconds):
    try:
        with timeout_decorator():
            sv_downloader.download_svi(output_dir, lat=52.356768, lon=4.907408, buffer=50, metadata_only=True)
            df = pd.read_csv(output_dir / "ams_pids.csv")
            assert df.shape[0] > 1
    except TimeoutException:
        assert len(list(output_dir.iterdir())) > 0, f"No files downloaded within {timeout_seconds} seconds"


def test_csv_download_asv(output_dir, sv_downloader, input_dir, timeout_decorator, timeout_seconds):
    try:
        with timeout_decorator():
            sv_downloader.download_svi(output_dir, input_csv_file=input_dir / "test_ams.csv", buffer=50)
    except TimeoutException:
        pass

    assert len(list(output_dir.iterdir())) > 0, f"No files downloaded within {timeout_seconds} seconds"


def test_shp_download_asv(output_dir, sv_downloader, input_dir, timeout_decorator, timeout_seconds):
    file_list = ["point_ams.geojson", "line_ams.geojson", "polygon_ams.geojson"]
    for file in file_list:
        try:
            with timeout_decorator():
                sv_downloader.download_svi(output_dir, input_shp_file=input_dir / file, buffer=50)
        except TimeoutException:
            pass

        assert len(list(output_dir.iterdir())) > 0, f"No files downloaded within {timeout_seconds} seconds for {file}"


# def test_place_name_download_asv(output_dir, sv_downloader, timeout_decorator):
#     try:
#         with timeout(30):  # Let it run for 5 minutes
#             sv_downloader.download_svi(output_dir, input_place_name="West Amsterdam, Netherlands", buffer=10)
#     except TimeoutException:
#         pass  # Allow timeout, we'll check results next

#     assert len(list(output_dir.iterdir())) > 0, "No files downloaded within 5 minutes"


def test_verbosity_levels(output_dir):
    """Test that verbosity levels work correctly in the AMSDownloader class."""

    # Test with verbosity=0 (no progress bars)
    silent_downloader = AMSDownloader(log_path=output_dir / "log_silent.log", verbosity=0)
    assert silent_downloader.verbosity == 0

    # Test with verbosity=1 (outer loops only - default)
    default_downloader = AMSDownloader(log_path=output_dir / "log_default.log")
    assert default_downloader.verbosity == 1

    # Test with verbosity=2 (all loops)
    verbose_downloader = AMSDownloader(log_path=output_dir / "log_verbose.log", verbosity=2)
    assert verbose_downloader.verbosity == 2

    # Test changing verbosity after initialization
    silent_downloader.verbosity = 2
    assert silent_downloader.verbosity == 2

    # Test setting verbosity in download_svi method
    default_downloader.download_svi(
        output_dir / "test_verbosity", lat=52.356768, lon=4.907408, metadata_only=True, verbosity=0
    )
    assert default_downloader.verbosity == 0


def test_async_download(output_dir, sv_downloader, timeout_decorator, timeout_seconds):
    """Test async download functionality"""
    output_dir = output_dir / "test_async"
    try:
        with timeout_decorator():
            sv_downloader.download_svi(
                output_dir,
                lat=52.356768,
                lon=4.907408,
                buffer=50,
                use_async=True,
                max_concurrency=5,
                metadata_only=False,
            )
            assert (output_dir / "ams_pids.csv").stat().st_size > 0
    except TimeoutException:
        assert len(list(output_dir.iterdir())) > 0, f"No files downloaded within {timeout_seconds} seconds"


def test_max_concurrency_property(output_dir):
    """Test max_concurrency property works correctly"""
    import os

    ams_downloader = AMSDownloader(log_path=output_dir / "log.log")

    # Test default value
    ams_downloader.max_concurrency = None
    assert ams_downloader.max_concurrency == min(100, os.cpu_count() * 4)

    # Test setting custom value
    ams_downloader.max_concurrency = 10
    assert ams_downloader.max_concurrency == 10
