"""Fast, no-network unit tests for the KartaView download_functions (proximity discovery)."""

from unittest.mock import patch

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import Polygon

from zensvi.download.kartaview import download_functions as kv
from zensvi.download.kv import KVDownloader


def _photo(pid, lat, lng, seq_id="100", user_id="7"):
    """Build a proximity-endpoint-shaped photo dict."""
    return {
        "id": pid,
        "lat": lat,
        "lng": lng,
        "shotDate": "2023-03-06 02:29:12.000",
        "fileurlProc": f"https://storage/{pid}.jpg",
        "heading": 178.57,
        "orgCode": "CMNT",
        "fieldOfView": 360,
        "projection": "SPHERE",
        "sequence": {"id": seq_id, "userId": user_id},
    }


class TestGetAllPages:
    """Pagination loops pages and stops on hasMoreData=False, capped at 150 items/page."""

    def test_paginates_until_no_more_data(self):
        pages = [
            {"data": [{"id": 1}, {"id": 2}], "hasMoreData": True},
            {"data": [{"id": 3}], "hasMoreData": False},
        ]
        requested_urls = []

        def fake_get_result(url):
            requested_urls.append(url)
            return pages[len(requested_urls) - 1]

        with patch.object(kv, "get_result_from_url", side_effect=fake_get_result):
            data = kv.get_all_pages("https://api/photo/?lat=1&lng=2")

        assert [d["id"] for d in data] == [1, 2, 3]
        assert len(requested_urls) == 2
        # Every request must respect the 150 cap and increment the page.
        assert "itemsPerPage=150&page=1" in requested_urls[0]
        assert "itemsPerPage=150&page=2" in requested_urls[1]

    def test_caps_items_per_page_at_150(self):
        with patch.object(kv, "get_result_from_url", return_value={"data": [], "hasMoreData": False}) as m:
            kv.get_all_pages("https://api/photo/?lat=1&lng=2", items_per_page=1000000)
        assert "itemsPerPage=150" in m.call_args.args[0]

    def test_stops_on_empty_result(self):
        with patch.object(kv, "get_result_from_url", return_value=None):
            assert kv.get_all_pages("https://api/photo/?lat=1&lng=2") == []


class TestGetPhotosNearPoint:
    """Proximity query shrinks radius on a timeout and gives up at the floor."""

    def test_shrinks_radius_on_timeout(self):
        calls = []

        def fake_pages(base_url, items_per_page=150):
            calls.append(base_url)
            if "radius=50" in base_url:
                raise ValueError("API Error: Query timeout. Narrow your filter")
            return [{"id": 1}]

        with patch.object(kv, "get_all_pages", side_effect=fake_pages):
            out = kv.get_photos_near_point(1.0, 2.0, radius=50)

        assert out == [{"id": 1}]
        assert "radius=50" in calls[0] and "radius=25" in calls[1]

    def test_gives_up_at_min_radius(self):
        with patch.object(kv, "get_all_pages", side_effect=ValueError("Query timeout")):
            assert kv.get_photos_near_point(1.0, 2.0, radius=50, min_radius=25) == []


class TestRateLimit:
    """Rate-limit responses are retriable; ordinary API errors are not."""

    def test_rate_limit_error_is_retriable(self):
        assert kv.is_retriable_exception(kv.RateLimitError("Too many requests")) is True

    def test_value_error_not_retriable(self):
        assert kv.is_retriable_exception(ValueError("API Error: bad request")) is False


class TestDateValidation:
    """KVDownloader validates date format up front, before any network fetch."""

    def test_invalid_start_date_raises_before_fetch(self):
        dl = KVDownloader()
        with pytest.raises(ValueError, match="start_date"):
            dl._get_raw_pids(start_date="not-a-date", end_date=None)

    def test_invalid_end_date_raises_before_fetch(self):
        dl = KVDownloader()
        with pytest.raises(ValueError, match="end_date"):
            dl._get_raw_pids(start_date=None, end_date="2023/01/01")


class TestFlattenPhotos:
    """Nested sequence is flattened, is_pano derived, lng renamed to lon."""

    def test_flatten(self):
        df = pd.DataFrame([_photo(1, 1.314, 103.762)])
        out = kv._flatten_photos(df)
        assert "sequence" not in out.columns
        assert out.loc[0, "sequenceId"] == "100"
        assert out.loc[0, "userId"] == "7"
        assert bool(out.loc[0, "is_pano"]) is True
        assert "lon" in out.columns and "lng" not in out.columns


class TestGetPointsInShape:
    """End-to-end (mocked): dedupe by id, clip to shape, preserve schema."""

    def _shape(self):
        poly = Polygon([(103.76, 1.313), (103.764, 1.313), (103.764, 1.316), (103.76, 1.316)])
        return gpd.GeoDataFrame(geometry=[poly], crs="EPSG:4326")

    def test_dedupe_clip_and_schema(self, monkeypatch):
        # Two sample points whose proximity results overlap (shared id=1) and include
        # one photo (id=99) outside the shape that must be clipped away.
        sample_points = pd.DataFrame({"longitude": [103.762, 103.7625], "latitude": [1.314, 1.3142]})

        class FakeGeoProcessor:
            def __init__(self, *a, **k):
                pass

            def get_lat_lon(self):
                return sample_points

        results = {
            (1.314, 103.762): [_photo(1, 1.314, 103.762), _photo(2, 1.3141, 103.7621)],
            (1.3142, 103.7625): [_photo(1, 1.314, 103.762), _photo(99, 1.20, 103.50)],
        }

        def fake_near(lat, lon, radius=50, min_radius=10):
            return results[(lat, lon)]

        monkeypatch.setattr(kv, "GeoProcessor", FakeGeoProcessor)
        monkeypatch.setattr(kv, "get_photos_near_point", fake_near)
        out = kv.get_points_in_shape(self._shape(), verbosity=0)

        ids = sorted(out["id"].tolist())
        assert ids == [1, 2]  # id=1 deduped, id=99 clipped out
        for col in ["id", "lon", "lat", "sequenceId", "shotDate", "fileurlProc"]:
            assert col in out.columns

    def test_returns_none_when_no_photos(self, monkeypatch):
        sample_points = pd.DataFrame({"longitude": [103.762], "latitude": [1.314]})

        class FakeGeoProcessor:
            def __init__(self, *a, **k):
                pass

            def get_lat_lon(self):
                return sample_points

        monkeypatch.setattr(kv, "GeoProcessor", FakeGeoProcessor)
        monkeypatch.setattr(kv, "get_photos_near_point", lambda *a, **k: [])
        assert kv.get_points_in_shape(self._shape(), verbosity=0) is None
