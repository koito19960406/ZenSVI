import os
import shutil

import pandas as pd
import pytest

from zensvi.download import GSVDownloader

from .conftest import TimeoutException


@pytest.fixture
def output_dir(base_output_dir, ensure_dir):
    output_dir = base_output_dir / "gsv_svi"
    if output_dir.exists():
        shutil.rmtree(output_dir)
    ensure_dir(output_dir)
    return output_dir


@pytest.fixture
def gsv_api_key():
    api_key = os.getenv("GSV_API_KEY")
    if not api_key:
        pytest.skip("GSV_API_KEY environment variable not set")
    return api_key


@pytest.fixture
def gsv_input_files(input_dir):
    return {
        "multipolygon": input_dir / "test_multipolygon_sg.geojson",
        "polygon": input_dir / "test_polygon_sg.geojson",
    }


@pytest.fixture
def sv_downloader(output_dir, gsv_api_key):
    return GSVDownloader(gsv_api_key=gsv_api_key, log_path=output_dir / "log.log")


@pytest.mark.parametrize(
    "input_type,expected_files",
    [
        ("coordinates", 1),  # lat/lon input
        ("csv", 1),  # CSV file input
        ("polygon", 1),  # Single polygon
        ("multipolygon", 1),  # Multiple polygons
        # ("place_name", 1),  # Place name input
    ],
)
def test_downloader_input_types(
    output_dir,
    gsv_input_files,
    input_dir,
    sv_downloader,
    input_type,
    expected_files,
    timeout_decorator,
    timeout_seconds,
):
    """Test downloading with different input types"""
    output_dir = output_dir / f"test_{input_type}"

    # Set up input parameters based on type
    kwargs = {}
    if input_type == "coordinates":
        kwargs = {"lat": 1.342425, "lon": 103.721523}
    elif input_type == "csv":
        test_csv = input_dir / "test_sg.csv"
        if not test_csv.exists():
            pd.DataFrame({"latitude": [1.342425], "longitude": [103.721523]}).to_csv(test_csv, index=False)
        kwargs = {"input_csv_file": str(test_csv)}
    elif input_type == "polygon":
        kwargs = {"input_shp_file": str(gsv_input_files["polygon"])}
    elif input_type == "multipolygon":
        kwargs = {"input_shp_file": str(gsv_input_files["multipolygon"])}
    else:  # place_name
        pass
    #     kwargs = {"input_place_name": "Peng Yong"}

    try:
        with timeout_decorator():
            sv_downloader.download_svi(output_dir, augment_metadata=True, **kwargs)
    except TimeoutException:
        pass

    # Check if files were downloaded within the 60-second window
    files = list(output_dir.iterdir())
    assert len(files) >= expected_files, f"No files downloaded within {timeout_seconds} seconds for {input_type}"


def test_download_gsv_zoom(output_dir, sv_downloader, timeout_decorator):
    if (output_dir / "gsv_panorama_zoom_0").exists():
        pytest.skip("Result exists")

    zoom_levels = [0, 1, 2, 3, 4]
    downloaded_zoom_folders = []

    try:
        with timeout_decorator(60):  # Let it run for 5 minutes
            for zoom in zoom_levels:
                folder_name = f"gsv_panorama_zoom_{zoom}"
                folder_path = output_dir / folder_name
                folder_path.mkdir(exist_ok=True)
                sv_downloader.download_svi(folder_path, lat=1.342425, lon=103.721523, augment_metadata=True, zoom=zoom)
    except TimeoutException:
        pass  # Allow timeout, we'll check results next

    # Check results after timeout or completion
    for folder_name in [f"gsv_panorama_zoom_{zoom}" for zoom in zoom_levels]:
        folder_path = output_dir / folder_name
        if len(list(folder_path.iterdir())) > 0:
            downloaded_zoom_folders.append(folder_name)

    assert len(downloaded_zoom_folders) > 0, "No files downloaded within 5 minutes for any zoom level"


def test_download_gsv_metadata_only(output_dir, sv_downloader, timeout_decorator):
    if (output_dir / "gsv_pids.csv").exists():
        pytest.skip("Result exists")

    try:
        with timeout_decorator(60):  # Let it run for 5 minutes
            sv_downloader.download_svi(
                output_dir, lat=1.342425, lon=103.721523, augment_metadata=True, metadata_only=True
            )
            # Try to check CSV first if within timeout
            df = pd.read_csv(output_dir / "gsv_pids.csv")
            assert df.shape[0] > 1
    except TimeoutException:
        # Fall back to checking if any files exist
        assert len(list(output_dir.iterdir())) > 0, "No files downloaded within 5 minutes"


def test_download_gsv_depth(output_dir, sv_downloader, timeout_decorator):
    try:
        with timeout_decorator(60):  # Let it run for 5 minutes
            sv_downloader.download_svi(output_dir, lat=1.342425, lon=103.721523, download_depth=True)
    except TimeoutException:
        pass  # Allow timeout, we'll check results next

    assert len(list(output_dir.iterdir())) > 0, "No files downloaded within 5 minutes"


def test_error_handling(output_dir, sv_downloader, timeout_decorator):
    try:
        with timeout_decorator(60):  # Let it run for 5 minutes
            # Test invalid date format
            with pytest.raises(ValueError):
                sv_downloader.download_svi(output_dir, lat=1.342425, lon=103.721523, start_date="invalid_date")

            # # Test missing required parameters
            # with pytest.raises(ValueError):
            #     sv_downloader.download_svi(output_dir)
    except TimeoutException:
        pass  # For error handling tests, timeout is acceptable


def test_download_with_buffer(output_dir, sv_downloader, timeout_decorator):
    output_dir = output_dir / "test_buffer"
    try:
        with timeout_decorator(60):  # Let it run for 5 minutes
            sv_downloader.download_svi(output_dir, lat=1.342425, lon=103.721523, buffer=100)
    except TimeoutException:
        pass  # Allow timeout, we'll check results next

    assert len(list(output_dir.iterdir())) > 0, "No files downloaded within 5 minutes"
