"""Fast, no-network tests for AMSDownloader pagination and grid default."""

import inspect
import json

from zensvi.download.ams import AMSDownloader


class _FakeResp:
    def __init__(self, payload):
        self.content = json.dumps(payload).encode()

    def raise_for_status(self):
        pass


def _downloader():
    dl = AMSDownloader()
    # Ensure proxy/user-agent pools are non-empty for random.choice (no network).
    dl._proxies = [{}]
    dl._user_agents = [{"user_agent": "test"}]
    return dl


def test_get_raw_pids_paginates(monkeypatch):
    """_get_raw_pids should follow _links.next and collect every page."""
    pages = [
        {
            "_embedded": {"panoramas": [{"pano_id": "a"}, {"pano_id": "b"}]},
            "_links": {"next": {"href": "https://api/x?page=2"}},
        },
        {"_embedded": {"panoramas": [{"pano_id": "c"}]}, "_links": {"next": None}},
    ]
    state = {"i": 0}

    def fake_get(url, **kwargs):
        resp = _FakeResp(pages[state["i"]])
        state["i"] += 1
        return resp

    monkeypatch.setattr("zensvi.download.ams.requests.get", fake_get)
    ids = _downloader()._get_raw_pids(52.0, 4.0, 50)
    assert ids == ["a", "b", "c"]


def test_get_raw_pids_stops_on_empty_page(monkeypatch):
    """A page with no panoramas ends pagination even if a next link is present."""
    monkeypatch.setattr(
        "zensvi.download.ams.requests.get",
        lambda url, **kw: _FakeResp({"_embedded": {"panoramas": []}, "_links": {"next": {"href": "x"}}}),
    )
    assert _downloader()._get_raw_pids(52.0, 4.0, 50) == []


def test_download_svi_defaults_to_grid():
    """AMS download should default to grid sampling."""
    sig = inspect.signature(AMSDownloader.download_svi)
    assert sig.parameters["grid"].default is True
    assert sig.parameters["grid_size"].default == 50
