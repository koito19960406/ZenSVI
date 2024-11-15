import geopandas as gp
import pandas as pd
import pytest

from zensvi.download.kartaview import download_functions as kv
from zensvi.download.kv import KVDownloader


@pytest.fixture
def output(base_output_dir, ensure_dir):
    output_dir = base_output_dir / "kv_output"
    ensure_dir(output_dir)
    return output_dir


@pytest.fixture
def kv_input_files(input_dir):
    return {
        "multipolygon": input_dir / "test_multipolygon_sg.geojson",
        "polygon": input_dir / "test_polygon_sg.geojson",
    }


def test_interface(output, kv_input_files):
    output_file = output / "test_interface.csv"
    if output_file.exists():
        pytest.skip("Result exists")

    gdf = gp.read_file(kv_input_files["polygon"])
    output_data = kv.get_points_in_shape(gdf)
    output_data.to_csv(output_file)
    assert len(output_data) > 0


@pytest.mark.parametrize(
    "input_type,expected_files",
    [
        ("coordinates", 1),  # lat/lon input
        ("csv", 1),  # CSV file input
        ("polygon", 1),  # Single polygon
        ("multipolygon", 1),  # Multiple polygons
        ("place_name", 1),  # Place name input
    ],
)
def test_downloader_input_types(output, kv_input_files, input_dir, input_type, expected_files):
    """Test downloading with different input types"""
    output_dir = output / f"test_{input_type}"
    kv_downloader = KVDownloader(log_path=output_dir / "log.log", max_workers=1)

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

    kv_downloader.download_svi(output_dir, **kwargs)
    assert len(list(output_dir.iterdir())) >= expected_files


def test_downloader_metadata_only(output, kv_input_files):
    output_dir = output / "test_metadata_only"
    kv_downloader = KVDownloader(log_path=output_dir / "log.log", max_workers=1)
    kv_downloader.download_svi(output_dir, input_shp_file=kv_input_files["polygon"], metadata_only=True)
    assert (output_dir / "kv_pids.csv").stat().st_size > 0


def test_downloader_with_buffer(output):
    output_dir = output / "test_buffer"
    kv_downloader = KVDownloader(log_path=output_dir / "log.log", max_workers=1)
    kv_downloader.download_svi(output_dir, lat=1.3140256, lon=103.7624098, buffer=50)
    assert len(list(output_dir.iterdir())) > 0


def test_downloader_date_filter(output, kv_input_files):
    output_dir = output / "test_date_filter"
    kv_downloader = KVDownloader(log_path=output_dir / "log.log", max_workers=1)
    kv_downloader.download_svi(
        output_dir,
        input_shp_file=kv_input_files["polygon"],
        start_date="2020-01-01",
        end_date="2023-12-31",
        metadata_only=True,
    )

    pids_file = output_dir / "kv_pids.csv"
    assert pids_file.exists()
    df = pd.read_csv(pids_file)
    dates = pd.to_datetime(df["shotDate"])
    assert all(dates >= "2020-01-01") and all(dates <= "2023-12-31")


def test_downloader_batch_processing(output, kv_input_files):
    output_dir = output / "test_batch"
    kv_downloader = KVDownloader(log_path=output_dir / "log.log", max_workers=1)
    kv_downloader.download_svi(output_dir, input_shp_file=kv_input_files["polygon"], batch_size=5, metadata_only=False)
    assert len(list(output_dir.iterdir())) > 0
    assert (output_dir / "kv_svi").exists()


def test_downloader_cropped_images(output, kv_input_files):
    output_dir = output / "test_cropped"
    kv_downloader = KVDownloader(log_path=output_dir / "log.log", max_workers=1)
    kv_downloader.download_svi(output_dir, input_shp_file=kv_input_files["polygon"], cropped=True, metadata_only=False)
    assert len(list(output_dir.iterdir())) > 0

    # Check if images are cropped (should be half height)
    from PIL import Image

    svi_dir = output_dir / "kv_svi"
    if any(svi_dir.glob("**/*.png")):
        img_path = next(svi_dir.glob("**/*.png"))
        img = Image.open(img_path)
        assert img.size[1] <= img.size[0] / 2  # Height should be at most half the width


def test_error_handling(output):
    output_dir = output / "test_errors"
    kv_downloader = KVDownloader(log_path=output_dir / "log.log")

    # Test invalid date format
    with pytest.raises(ValueError):
        kv_downloader.download_svi(output_dir, lat=1.3140256, lon=103.7624098, start_date="invalid_date")

    # Test missing required parameters
    with pytest.raises(ValueError):
        kv_downloader.download_svi(output_dir)
