import shutil

import geopandas as gp
import pandas as pd
import pytest

from zensvi.download.kartaview import download_functions as kv
from zensvi.download.kv import KVDownloader

from .conftest import TimeoutException


@pytest.fixture(scope="function")  # Explicitly set function scope
def output_dir(base_output_dir, ensure_dir):
    output_dir = base_output_dir / "kv_svi"
    if output_dir.exists():
        print(f"Cleaning up existing {output_dir} before test function")  # Optional: for debugging
        shutil.rmtree(output_dir)
    ensure_dir(output_dir)
    return output_dir


@pytest.fixture
def kv_input_files(input_dir):
    return {
        "multipolygon": input_dir / "test_multipolygon_sg.geojson",
        "polygon": input_dir / "test_polygon_sg.geojson",
    }


def test_interface(output_dir, kv_input_files, timeout_decorator, timeout_seconds):
    output_file = output_dir / "test_interface.csv"
    if output_file.exists():
        pytest.skip("Result exists")

    try:
        with timeout_decorator():
            gdf = gp.read_file(kv_input_files["polygon"])
            output_data = kv.get_points_in_shape(gdf)
            output_data.to_csv(output_file)
            assert len(output_data) > 0
    except TimeoutException:
        assert len(list(output_dir.iterdir())) > 0, f"No files downloaded within {timeout_seconds} seconds"


@pytest.mark.parametrize(
    "input_type,expected_files,verbosity",
    [
        ("coordinates", 1, 0),  # lat/lon input with no progress bars
        ("csv", 1, 1),  # CSV file input with outer progress bars
        ("polygon", 1, 2),  # Single polygon with all progress bars
        ("multipolygon", 1, 1),  # Multiple polygons with outer progress bars
        ("place_name", 1, 1),  # Place name input with outer progress bars
    ],
)
def test_downloader_input_types(
    output_dir, kv_input_files, input_dir, input_type, expected_files, verbosity, timeout_decorator, timeout_seconds
):
    """Test downloading with different input types"""
    output_dir = output_dir / f"test_{input_type}"
    kv_downloader = KVDownloader(log_path=output_dir / "log.log", max_workers=1, verbosity=verbosity)

    # Set up input parameters based on type
    kwargs = {}
    if input_type == "coordinates":
        kwargs = {"lat": 1.3140256, "lon": 103.7624098}
    elif input_type == "csv":
        test_csv = input_dir / "test_sg.csv"
        if not test_csv.exists():
            pd.DataFrame({"latitude": [1.3140256], "longitude": [103.7624098]}).to_csv(test_csv, index=False)
        kwargs = {"input_csv_file": str(test_csv)}
    elif input_type == "polygon":
        kwargs = {"input_shp_file": str(kv_input_files["polygon"])}
    elif input_type == "multipolygon":
        kwargs = {"input_shp_file": str(kv_input_files["multipolygon"])}
    else:  # place_name
        kwargs = {"input_place_name": "Yong Peng, Malaysia"}

    try:
        with timeout_decorator():
            kv_downloader.download_svi(output_dir, **kwargs)
    except TimeoutException:
        pass  # Allow timeout, we'll check results next

    # Check if files were downloaded within the timeout window
    files = list(output_dir.iterdir())
    assert len(files) >= expected_files, f"No files downloaded within {timeout_seconds} seconds for {input_type}"


def test_downloader_metadata_only(output_dir, kv_input_files, timeout_decorator, timeout_seconds):
    output_dir = output_dir / "test_metadata_only"
    kv_downloader = KVDownloader(log_path=output_dir / "log.log", max_workers=1)

    try:
        with timeout_decorator():
            kv_downloader.download_svi(output_dir, input_shp_file=kv_input_files["polygon"], metadata_only=True)
            assert (output_dir / "kv_pids.csv").stat().st_size > 0
    except TimeoutException:
        assert len(list(output_dir.iterdir())) > 0, f"No files downloaded within {timeout_seconds} seconds"


def test_downloader_with_buffer(output_dir, timeout_decorator, timeout_seconds):
    output_dir = output_dir / "test_buffer"
    kv_downloader = KVDownloader(log_path=output_dir / "log.log", max_workers=1)

    try:
        with timeout_decorator():
            kv_downloader.download_svi(output_dir, lat=1.3140256, lon=103.7624098, buffer=50)
    except TimeoutException:
        pass  # Allow timeout, we'll check results next

    assert len(list(output_dir.iterdir())) > 0, f"No files downloaded within {timeout_seconds} seconds"


def test_downloader_date_filter(output_dir, kv_input_files, timeout_decorator, timeout_seconds):
    output_dir = output_dir / "test_date_filter"
    kv_downloader = KVDownloader(log_path=output_dir / "log.log", max_workers=1)

    try:
        with timeout_decorator():
            kv_downloader.download_svi(
                output_dir,
                input_shp_file=kv_input_files["polygon"],
                start_date="2020-01-01",
                end_date="2023-12-31",
                metadata_only=True,
            )
            # Try to check CSV first if within timeout
            pids_file = output_dir / "kv_pids.csv"
            assert pids_file.exists()
            df = pd.read_csv(pids_file)
            dates = pd.to_datetime(df["shotDate"])
            assert all(dates >= "2020-01-01") and all(dates <= "2023-12-31")
    except TimeoutException:
        assert len(list(output_dir.iterdir())) > 0, f"No files downloaded within {timeout_seconds} seconds"


def test_downloader_batch_processing(output_dir, kv_input_files, timeout_decorator, timeout_seconds):
    output_dir = output_dir / "test_batch"
    kv_downloader = KVDownloader(log_path=output_dir / "log.log", max_workers=1)

    try:
        with timeout_decorator():
            kv_downloader.download_svi(
                output_dir, input_shp_file=kv_input_files["polygon"], batch_size=5, metadata_only=False
            )
    except TimeoutException:
        pass  # Allow timeout, we'll check results next

    assert len(list(output_dir.iterdir())) > 0, f"No files downloaded within {timeout_seconds} seconds"
    assert (output_dir / "kv_svi").exists(), f"SVI directory not created within {timeout_seconds} seconds"


def test_downloader_cropped_images(output_dir, kv_input_files, timeout_decorator, timeout_seconds):
    output_dir = output_dir / "test_cropped"
    kv_downloader = KVDownloader(log_path=output_dir / "log.log", max_workers=1)

    try:
        with timeout_decorator():
            kv_downloader.download_svi(
                output_dir, input_shp_file=kv_input_files["polygon"], cropped=True, metadata_only=False
            )
            # Try to check image dimensions if within timeout
            from PIL import Image

            svi_dir = output_dir / "kv_svi"
            if any(svi_dir.glob("**/*.png")):
                img_path = next(svi_dir.glob("**/*.png"))
                img = Image.open(img_path)
                assert img.size[1] <= img.size[0] / 2  # Height should be at most half the width
    except TimeoutException:
        assert len(list(output_dir.iterdir())) > 0, f"No files downloaded within {timeout_seconds} seconds"


def test_error_handling(output_dir, timeout_decorator, timeout_seconds):
    output_dir = output_dir / "test_errors"
    kv_downloader = KVDownloader(log_path=output_dir / "log.log")

    try:
        with timeout_decorator():
            # Test invalid date format
            with pytest.raises(ValueError):
                kv_downloader.download_svi(output_dir, lat=1.3140256, lon=103.7624098, start_date="invalid_date")

            # Test missing required parameters
            with pytest.raises(ValueError):
                kv_downloader.download_svi(output_dir)
    except TimeoutException:
        pass  # For error handling tests, timeout is acceptable


def test_verbosity_levels(output_dir):
    """Test that verbosity levels work correctly."""
    output_dir = output_dir / "test_verbosity"

    # Test verbosity=0
    kv_downloader_0 = KVDownloader(log_path=output_dir / "log.log", verbosity=0)
    assert kv_downloader_0.verbosity == 0

    # Test verbosity=1
    kv_downloader_1 = KVDownloader(log_path=output_dir / "log.log", verbosity=1)
    assert kv_downloader_1.verbosity == 1

    # Test verbosity=2
    kv_downloader_2 = KVDownloader(log_path=output_dir / "log.log", verbosity=2)
    assert kv_downloader_2.verbosity == 2

    # Test setting verbosity after initialization
    kv_downloader_1.verbosity = 2
    assert kv_downloader_1.verbosity == 2


def test_async_download(output_dir, kv_input_files, timeout_decorator, timeout_seconds):
    """Test async download functionality"""
    output_dir = output_dir / "test_async"
    kv_downloader = KVDownloader(log_path=output_dir / "log.log", max_workers=1, verbosity=0)

    try:
        with timeout_decorator():
            kv_downloader.download_svi(
                output_dir,
                input_shp_file=str(kv_input_files["polygon"]),
                use_async=True,
                max_concurrency=5,
                metadata_only=False,
            )
            assert (output_dir / "kv_pids.csv").stat().st_size > 0
    except TimeoutException:
        assert len(list(output_dir.iterdir())) > 0, f"No files downloaded within {timeout_seconds} seconds"


def test_max_concurrency_property(output_dir):
    """Test max_concurrency property works correctly"""
    import os

    kv_downloader = KVDownloader(log_path=output_dir / "log.log")

    # Test default value
    kv_downloader.max_concurrency = None
    assert kv_downloader.max_concurrency == min(100, os.cpu_count() * 4)

    # Test setting custom value
    kv_downloader.max_concurrency = 10
    assert kv_downloader.max_concurrency == 10
