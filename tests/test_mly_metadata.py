import shutil

import pytest

from zensvi.metadata import MLYMetadata


@pytest.fixture(scope="function")  # Explicitly set function scope
def output_dir(base_output_dir, ensure_dir):
    output_dir = base_output_dir / "metadata"
    if output_dir.exists():
        print(f"Cleaning up existing {output_dir} before test function")  # Optional: for debugging
        shutil.rmtree(output_dir)
    ensure_dir(output_dir)
    return output_dir


@pytest.fixture
def metadata(input_dir, output_dir):
    return MLYMetadata(
        str(input_dir / "metadata/mly_pids.csv"),
        log_path=output_dir / "log.log",
    )


def test_image_level_metadata(output_dir, metadata):
    df = metadata.compute_metadata(unit="image", path_output=output_dir / "image_metadata.csv")
    assert "relative_angle" in df.columns
    assert not df["relative_angle"].empty


def test_grid_level_metadata(output_dir, metadata):
    df = metadata.compute_metadata(
        unit="grid",
        grid_resolution=12,
        coverage_buffer=10,
        path_output=output_dir / "grid_metadata.geojson",
    )
    assert "coverage" in df.columns
    assert not df["coverage"].empty
    df.to_csv(output_dir / "grid_metadata.csv", index=False)


def test_street_level_metadata(output_dir, metadata):
    df = metadata.compute_metadata(
        unit="street",
        coverage_buffer=10,
        path_output=output_dir / "street_metadata.geojson",
    )
    assert "coverage" in df.columns
    assert not df["coverage"].empty
    df.to_csv(output_dir / "street_metadata.csv", index=False)


def test_image_level_partial_metadata(output_dir, metadata):
    indicator_list = "day daytime_nighttime relative_angle"
    df = metadata.compute_metadata(
        unit="image",
        indicator_list=indicator_list,
        path_output=output_dir / "image_metadata_partial.csv",
    )
    assert "relative_angle" in df.columns
    assert not df["relative_angle"].empty
    df.to_csv(output_dir / "image_metadata_partial.csv", index=False)


def test_grid_level_partial_metadata(output_dir, metadata):
    indicator_list = "coverage most_recent_date average_is_pano number_of_daytime"
    df = metadata.compute_metadata(
        unit="grid",
        grid_resolution=12,
        coverage_buffer=10,
        indicator_list=indicator_list,
        path_output=output_dir / "grid_metadata_partial.geojson",
    )
    assert "coverage" in df.columns
    assert not df["coverage"].empty
    df.to_csv(output_dir / "grid_metadata_partial.csv", index=False)


def test_street_level_partial_metadata(output_dir, metadata):
    indicator_list = "coverage most_recent_date average_is_pano number_of_daytime"
    df = metadata.compute_metadata(
        unit="street",
        coverage_buffer=10,
        indicator_list=indicator_list,
        path_output=output_dir / "street_metadata_partial.geojson",
    )
    assert "coverage" in df.columns
    assert not df["coverage"].empty
    df.to_csv(output_dir / "street_metadata_partial.csv", index=False)
