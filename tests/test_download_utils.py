"""Tests for zensvi.download.utils (helpers, imtool)."""

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from zensvi.download.utils.helpers import check_and_buffer, standardize_column_names
from zensvi.download.utils.imtool import ImageTool


class TestStandardizeColumnNames:
    """Tests for standardize_column_names function."""

    def test_renames_lat_lon(self):
        """Should rename 'lat'/'lon' to 'latitude'/'longitude'."""
        df = pd.DataFrame({"lat": [1.0], "lon": [2.0]})
        result = standardize_column_names(df)
        assert "latitude" in result.columns
        assert "longitude" in result.columns

    def test_renames_x_y(self):
        """Should rename 'x'/'y' to 'longitude'/'latitude'."""
        df = pd.DataFrame({"x": [1.0], "y": [2.0]})
        result = standardize_column_names(df)
        assert "longitude" in result.columns
        assert "latitude" in result.columns

    def test_renames_long_lt(self):
        """Should rename 'long'/'lt' variants."""
        df = pd.DataFrame({"long": [1.0], "lt": [2.0]})
        result = standardize_column_names(df)
        assert "longitude" in result.columns
        assert "latitude" in result.columns

    def test_renames_lng(self):
        """Should rename 'lng' to 'longitude'."""
        df = pd.DataFrame({"lng": [1.0], "latitude": [2.0]})
        result = standardize_column_names(df)
        assert "longitude" in result.columns

    def test_case_insensitive(self):
        """Should handle uppercase column names."""
        df = pd.DataFrame({"LAT": [1.0], "LON": [2.0]})
        result = standardize_column_names(df)
        assert "latitude" in result.columns
        assert "longitude" in result.columns

    def test_preserves_other_columns(self):
        """Should not touch unrelated columns."""
        df = pd.DataFrame({"lat": [1.0], "lon": [2.0], "name": ["a"]})
        result = standardize_column_names(df)
        assert "name" in result.columns

    def test_already_standard(self):
        """Should not change already-standard names."""
        df = pd.DataFrame({"latitude": [1.0], "longitude": [2.0]})
        result = standardize_column_names(df)
        assert "latitude" in result.columns
        assert "longitude" in result.columns


class TestImageTool:
    """Tests for ImageTool static methods."""

    def _make_image(self, width, height, color=(255, 0, 0)):
        """Create a solid-color test image."""
        img = Image.new("RGB", (width, height), color)
        return img

    def test_concat_horizontally(self):
        """Should concatenate two images side by side."""
        im1 = self._make_image(100, 50, (255, 0, 0))
        im2 = self._make_image(80, 50, (0, 255, 0))
        result = ImageTool.concat_horizontally(im1, im2)
        assert result.width == 180
        assert result.height == 50

    def test_concat_vertically(self):
        """Should concatenate two images top to bottom."""
        im1 = self._make_image(100, 50, (255, 0, 0))
        im2 = self._make_image(100, 30, (0, 255, 0))
        result = ImageTool.concat_vertically(im1, im2)
        assert result.width == 100
        assert result.height == 80

    def test_concat_horizontally_pixel_content(self):
        """Left side should contain pixels from im1, right from im2."""
        im1 = self._make_image(10, 10, (255, 0, 0))
        im2 = self._make_image(10, 10, (0, 255, 0))
        result = ImageTool.concat_horizontally(im1, im2)
        arr = np.array(result)
        # Left half should be red
        assert tuple(arr[5, 2, :]) == (255, 0, 0)
        # Right half should be green
        assert tuple(arr[5, 12, :]) == (0, 255, 0)

    def test_concat_vertically_pixel_content(self):
        """Top should contain pixels from im1, bottom from im2."""
        im1 = self._make_image(10, 10, (255, 0, 0))
        im2 = self._make_image(10, 10, (0, 0, 255))
        result = ImageTool.concat_vertically(im1, im2)
        arr = np.array(result)
        # Top should be red
        assert tuple(arr[2, 5, :]) == (255, 0, 0)
        # Bottom should be blue
        assert tuple(arr[12, 5, :]) == (0, 0, 255)


class TestFetchImageWithProxy:
    """Tests for ImageTool.fetch_image_with_proxy URL construction."""

    def test_uses_streetviewpixels_endpoint(self, monkeypatch):
        """Should request the current streetviewpixels tile endpoint with correct params."""
        from unittest.mock import MagicMock

        captured = {}

        def fake_get(url, **kwargs):
            captured["url"] = url
            return MagicMock(raw=MagicMock())

        sentinel = object()
        monkeypatch.setattr("zensvi.download.utils.imtool.requests.get", fake_get)
        monkeypatch.setattr("zensvi.download.utils.imtool.Image.open", lambda raw: sentinel)

        result = ImageTool.fetch_image_with_proxy(
            pano_id="PANO", zoom=2, x=1, y=0, ua={"User-Agent": "test"}, proxies=[{}]
        )

        assert result is sentinel
        url = captured["url"]
        assert url.startswith("https://streetviewpixels-pa.googleapis.com/v1/tile")
        assert "cb_client=maps_sv.tactile" in url
        assert "panoid=PANO" in url
        assert "x=1" in url
        assert "y=0" in url
        assert "zoom=2" in url
        # The deprecated cbk0 endpoint must not be used.
        assert "cbk0.google.com" not in url


class TestCreatePointGrid:
    """Tests for GeoProcessor.create_point_grid (grid sampling)."""

    def test_grid_over_small_polygon_is_bounded(self):
        """A ~40m polygon should yield a small, finite grid (regression: lon/lat were swapped)."""
        import geopandas as gpd
        from shapely.geometry import Point

        from zensvi.download.utils.geoprocess import GeoProcessor

        gdf = gpd.GeoDataFrame(geometry=[Point(103.7624, 1.3140)], crs="EPSG:4326")
        gdf_m = gdf.to_crs(gdf.estimate_utm_crs())
        gdf_m["geometry"] = gdf_m.buffer(40)
        poly = gdf_m.to_crs("EPSG:4326").geometry.iloc[0]

        gp = GeoProcessor(gdf, grid=True, grid_size=50)
        points, utm_crs = gp.create_point_grid(poly, grid_size=50)
        # An ~80m bbox at 50m spacing must produce only a handful of points, not millions.
        assert 0 < len(points) <= 25


class TestCheckAndBuffer:
    """Tests for check_and_buffer function."""

    def test_point_with_zero_buffer_raises(self):
        """Should raise ValueError for Point geometry with buffer=0."""
        import geopandas as gpd
        from shapely.geometry import Point

        gdf = gpd.GeoDataFrame(geometry=[Point(0, 0)], crs="EPSG:4326")
        with pytest.raises(ValueError, match="Buffer cannot be 0"):
            check_and_buffer(gdf, 0)

    def test_linestring_with_zero_buffer_raises(self):
        """Should raise ValueError for LineString geometry with buffer=0."""
        import geopandas as gpd
        from shapely.geometry import LineString

        gdf = gpd.GeoDataFrame(geometry=[LineString([(0, 0), (1, 1)])], crs="EPSG:4326")
        with pytest.raises(ValueError, match="Buffer cannot be 0"):
            check_and_buffer(gdf, 0)

    def test_point_with_buffer_returns_polygon(self):
        """Point with non-zero buffer should return polygon geometry."""
        import geopandas as gpd
        from shapely.geometry import Point

        gdf = gpd.GeoDataFrame(geometry=[Point(0, 0)], crs="EPSG:4326")
        result = check_and_buffer(gdf, 100)
        assert result.geom_type.iloc[0] == "Polygon"
        assert result.crs.to_epsg() == 4326
