import json
import os
import shutil

import pandas as pd
import pytest

from zensvi.download import MLYDownloader
from zensvi.download.mapillary import interface

from .conftest import TimeoutException


@pytest.fixture(scope="function")  # Explicitly set function scope
def output_dir(base_output_dir, ensure_dir):  # Changed from 'output' to 'output_dir'
    output_dir = base_output_dir / "mly_output"
    if output_dir.exists():
        shutil.rmtree(output_dir)
    ensure_dir(output_dir)
    return output_dir


@pytest.fixture
def mly_api_key():
    api_key = os.getenv("MLY_API_KEY")
    if not api_key:
        pytest.skip("MLY_API_KEY environment variable not set")
    interface.set_access_token(api_key)
    return api_key


@pytest.fixture
def mly_input_files(input_dir):
    return {
        "multipolygon": input_dir / "test_multipolygon_sg.geojson",
        "polygon": input_dir / "test_polygon_sg.geojson",
    }


def test_interface(output_dir, mly_input_files, mly_api_key, timeout_decorator, timeout_seconds):
    output_file = output_dir / "test_interface.json"
    if output_file.exists():
        pytest.skip("Result exists")

    try:
        with timeout_decorator():
            with open(mly_input_files["polygon"]) as f:
                geojson = json.load(f)
            output_data = interface.images_in_geojson(geojson)
            assert len(output_data.to_dict()) > 0
    except TimeoutException:
        assert len(list(output_dir.iterdir())) > 0, f"No files downloaded within {timeout_seconds} seconds"


@pytest.mark.parametrize(
    "input_type,expected_files,verbosity",
    [
        ("coordinates", 3, 0),  # Test with no progress bars
        ("csv", 3, 1),  # Test with outer progress bars only
        ("polygon", 3, 2),  # Test with all progress bars
        ("multipolygon", 3, 1),
        ("place_name", 3, 1),
    ],
)
def test_downloader_input_types(
    output_dir,
    mly_input_files,
    input_dir,
    mly_api_key,
    input_type,
    expected_files,
    verbosity,
    timeout_decorator,
    timeout_seconds,
):
    """Test downloading with different input types"""
    output_dir = output_dir / f"test_{input_type}"
    mly_downloader = MLYDownloader(mly_api_key, log_path=output_dir / "log.log", max_workers=1, verbosity=verbosity)

    # Set up input parameters based on type
    kwargs = {}
    if input_type == "coordinates":
        kwargs = {"lat": 11.827575599999989, "lon": 13.146558000000027, "buffer": 100}
    elif input_type == "csv":
        test_csv = input_dir / "test_sg.csv"
        if not test_csv.exists():
            pd.DataFrame({"latitude": [11.827575599999989], "longitude": [13.146558000000027]}).to_csv(
                test_csv, index=False
            )
        kwargs = {"input_csv_file": str(test_csv), "buffer": 100}
    elif input_type == "polygon":
        kwargs = {"input_shp_file": str(mly_input_files["polygon"])}
    elif input_type == "multipolygon":
        kwargs = {"input_shp_file": str(mly_input_files["multipolygon"])}
    else:  # place_name
        kwargs = {"input_place_name": "Maiduguri, Nigeria"}

    try:
        with timeout_decorator():
            mly_downloader.download_svi(output_dir, **kwargs)
    except TimeoutException:
        pass  # Allow timeout, we'll check results next

    # Check if files were downloaded within the 60-second window
    files = list((output_dir / "mly_svi/batch_1").iterdir())
    assert len(files) >= expected_files, f"No files downloaded within {timeout_seconds} seconds for {input_type}"


def test_downloader_metadata_only(output_dir, mly_input_files, mly_api_key, timeout_decorator, timeout_seconds):
    output_dir = output_dir / "test_metadata"
    if (output_dir / "mly_pids.csv").exists():
        pytest.skip("Result exists")

    try:
        with timeout_decorator():
            mly_downloader = MLYDownloader(mly_api_key, log_path=str(output_dir / "log.log"), max_workers=1)
            mly_downloader.download_svi(output_dir, input_shp_file=mly_input_files["polygon"], metadata_only=True)
            assert (output_dir / "mly_pids.csv").stat().st_size > 0
    except TimeoutException:
        assert len(list(output_dir.iterdir())) > 0, f"No files downloaded within {timeout_seconds} seconds"


def test_downloader_with_buffer(output_dir, mly_api_key, timeout_decorator, timeout_seconds):
    output_dir = output_dir / "test_buffer"
    try:
        with timeout_decorator():
            mly_downloader = MLYDownloader(mly_api_key, max_workers=1)
            mly_downloader.download_svi(output_dir, lat=11.827575599999989, lon=13.146558000000027, buffer=100)
    except TimeoutException:
        pass  # Allow timeout, we'll check the results next

    assert len(list(output_dir.iterdir())) > 0, f"No files downloaded within {timeout_seconds} seconds"


def test_downloader_kwargs(output_dir, mly_input_files, mly_api_key, timeout_decorator, timeout_seconds):
    output_dir = output_dir / "test_kwargs"
    if (output_dir / "mly_svi").exists():
        pytest.skip("Result exists")

    try:
        with timeout_decorator():
            mly_downloader = MLYDownloader(mly_api_key, log_path=str(output_dir / "log.log"), max_workers=1)
            kwarg = {
                "image_type": "flat",
                "min_captured_at": 1484549945000,
                "max_captured_at": 1642935417694,
                "organization_id": [1805883732926354],
                "compass_angle": (0, 180),
            }
            mly_downloader.download_svi(output_dir, input_shp_file=mly_input_files["polygon"], **kwarg)
    except TimeoutException:
        pass  # Allow timeout, we'll check the results next

    assert len(list(output_dir.iterdir())) > 0, f"No files downloaded within {timeout_seconds} seconds"


def test_error_handling(output_dir, mly_api_key, timeout_decorator, timeout_seconds):
    output_dir = output_dir / "test_errors"
    mly_downloader = MLYDownloader(mly_api_key, log_path=output_dir / "log.log")

    try:
        with timeout_decorator():
            # Test invalid date format
            with pytest.raises(ValueError):
                mly_downloader.download_svi(
                    output_dir, lat=11.827575599999989, lon=13.146558000000027, start_date="invalid_date"
                )

            # # Test missing required parameters
            # with pytest.raises(ValueError):
            #     mly_downloader.download_svi(output_dir)
    except TimeoutException:
        pass  # For error handling tests, timeout is acceptable


def test_verbosity_levels(output_dir, mly_api_key, timeout_decorator, timeout_seconds):
    """Test that verbosity levels work correctly"""
    output_dir = output_dir / "test_verbosity"

    # Test with verbosity=0 (should disable all progress bars)
    mly_downloader_no_verbosity = MLYDownloader(
        mly_api_key, log_path=output_dir / "log_no_verbosity.log", max_workers=1, verbosity=0
    )

    # Test with verbosity=1 (should show only outer loop)
    mly_downloader_low_verbosity = MLYDownloader(
        mly_api_key, log_path=output_dir / "log_low_verbosity.log", max_workers=1, verbosity=1
    )

    # Test with verbosity=2 (should show all loops)
    mly_downloader_high_verbosity = MLYDownloader(
        mly_api_key, log_path=output_dir / "log_high_verbosity.log", max_workers=1, verbosity=2
    )

    # Check that verbosity property works correctly
    assert mly_downloader_no_verbosity.verbosity == 0
    assert mly_downloader_low_verbosity.verbosity == 1
    assert mly_downloader_high_verbosity.verbosity == 2

    # Change verbosity and check it updates
    mly_downloader_no_verbosity.verbosity = 2
    assert mly_downloader_no_verbosity.verbosity == 2


def test_async_download(output_dir, mly_input_files, mly_api_key, timeout_decorator, timeout_seconds):
    """Test async download functionality"""
    output_dir = output_dir / "test_async"
    mly_downloader = MLYDownloader(mly_api_key, log_path=output_dir / "log.log", max_workers=1, verbosity=0)

    try:
        with timeout_decorator():
            mly_downloader.download_svi(
                output_dir,
                input_shp_file=str(mly_input_files["polygon"]),
                use_async=True,
                max_concurrency=5,
                metadata_only=False,
            )
            assert (output_dir / "mly_pids.csv").stat().st_size > 0
    except TimeoutException:
        assert len(list(output_dir.iterdir())) > 0, f"No files downloaded within {timeout_seconds} seconds"


def test_max_concurrency_property(output_dir, mly_api_key):
    """Test max_concurrency property works correctly"""
    mly_downloader = MLYDownloader(mly_api_key, log_path=output_dir / "log.log")

    # Test default value
    mly_downloader.max_concurrency = None
    assert mly_downloader.max_concurrency == min(100, os.cpu_count() * 4)

    # Test setting custom value
    mly_downloader.max_concurrency = 10
    assert mly_downloader.max_concurrency == 10
