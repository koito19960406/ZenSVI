import shutil

import polars as pl
import pytest

from zensvi.metadata import KVMetadata

pytestmark = pytest.mark.slow


@pytest.fixture(scope="function")
def output_dir(base_output_dir, ensure_dir):
    output_dir = base_output_dir / "kv_metadata"
    if output_dir.exists():
        shutil.rmtree(output_dir)
    ensure_dir(output_dir)
    return output_dir


@pytest.fixture(scope="module")
def _metadata_cached(input_dir):
    """Download OSM street network once per module (expensive network call)."""
    md = KVMetadata(str(input_dir / "metadata/kv_pids.csv"))
    md._ensure_street_network()  # pre-warm so copies in the per-test fixture are valid
    return md


@pytest.fixture
def metadata(_metadata_cached):
    """Create a lightweight copy that reuses the cached street network and projected CRS.

    KVMetadata.__init__ downloads the OSM street network (slow network call). We do that
    once in _metadata_cached (module scope), then create fresh instances that share the
    immutable network data but have their own mutable state (re-applying KV normalization).
    """
    md = object.__new__(KVMetadata)
    md._tf_instance = _metadata_cached._tf_instance
    md.path_input = _metadata_cached.path_input
    md.projected_crs = _metadata_cached.projected_crs
    md.logger = None
    # Re-apply KartaView column normalization since __init__ is bypassed.
    md.df = md._normalize_columns(pl.read_csv(md.path_input))
    md.metadata = None
    md.street_network = _metadata_cached.street_network.copy()
    md._joined_daynight_cache = None
    md._seasons_grouped_cache = None
    md.indicator_metadata_image = {
        k: getattr(md, v.__name__) for k, v in _metadata_cached.indicator_metadata_image.items()
    }
    md.indicator_metadata_grid_street = {
        k: getattr(md, v.__name__) for k, v in _metadata_cached.indicator_metadata_grid_street.items()
    }
    return md


def test_kv_normalization():
    """KartaView columns are normalized to the BaseMetadata schema (no network)."""
    md = object.__new__(KVMetadata)
    df = md._normalize_columns(
        pl.DataFrame(
            {
                "id": [1],
                "lat": [1.3134],
                "lon": [103.793],
                "shotDate": ["2023-03-06 02:29:12.000"],
                "heading": [178.57],
                "sequenceId": [8326665],
                "userId": [44],
                "orgCode": ["CMNT"],
                "is_pano": [True],
            }
        )
    )
    for col in ["captured_at", "compass_angle", "sequence_id", "creator_id", "organization_id", "is_pano"]:
        assert col in df.columns
    # captured_at is ms since the Unix epoch (2023-03-06 02:29:12 UTC).
    assert df["captured_at"][0] == 1678069752000


def test_image_level_metadata(output_dir, metadata):
    df = metadata.compute_metadata(unit="image", path_output=output_dir / "image_metadata.csv")
    assert "relative_angle" in df.columns
    assert "year" in df.columns


def test_grid_level_metadata(output_dir, metadata):
    df = metadata.compute_metadata(
        unit="grid",
        grid_resolution=12,
        coverage_buffer=10,
        path_output=output_dir / "grid_metadata.geojson",
    )
    assert "coverage" in df.columns
    assert not df["coverage"].empty


def test_street_level_metadata(output_dir, metadata):
    df = metadata.compute_metadata(
        unit="street",
        coverage_buffer=10,
        path_output=output_dir / "street_metadata.geojson",
    )
    assert "coverage" in df.columns
    assert not df["coverage"].empty
