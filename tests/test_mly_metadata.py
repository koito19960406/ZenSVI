import shutil

import polars as pl
import pytest

from zensvi.metadata import MLYMetadata

pytestmark = pytest.mark.slow


@pytest.fixture(scope="function")  # Explicitly set function scope
def output_dir(base_output_dir, ensure_dir):
    output_dir = base_output_dir / "metadata"
    if output_dir.exists():
        print(f"Cleaning up existing {output_dir} before test function")  # Optional: for debugging
        shutil.rmtree(output_dir)
    ensure_dir(output_dir)
    return output_dir


@pytest.fixture(scope="module")
def _metadata_cached(input_dir):
    """Download OSM street network once per module (expensive network call)."""
    md = MLYMetadata(str(input_dir / "metadata/mly_pids.csv"))
    md._ensure_street_network()  # pre-warm so copies in the per-test fixture are valid
    return md


@pytest.fixture
def metadata(_metadata_cached):
    """Create a lightweight copy that reuses the cached street network and projected CRS.

    MLYMetadata.__init__ downloads the OSM street network (slow network call).
    We do that once in _metadata_cached (module scope), then create fresh instances
    that share the immutable network data but have their own mutable state.
    """
    md = object.__new__(MLYMetadata)
    # Shared immutable state from cached instance
    md._tf_instance = _metadata_cached._tf_instance
    md.path_input = _metadata_cached.path_input
    md.projected_crs = _metadata_cached.projected_crs
    # Per-test mutable state (fresh copies)
    md.logger = None
    md.df = pl.read_csv(md.path_input)
    md.metadata = None
    md.street_network = _metadata_cached.street_network.copy()
    # Rebuild method dicts bound to this new instance
    md.indicator_metadata_image = {
        k: getattr(md, v.__name__) for k, v in _metadata_cached.indicator_metadata_image.items()
    }
    md.indicator_metadata_grid_street = {
        k: getattr(md, v.__name__) for k, v in _metadata_cached.indicator_metadata_grid_street.items()
    }
    return md


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
