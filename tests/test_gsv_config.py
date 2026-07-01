"""Fast, no-network tests for GSVDownloader configuration defaults."""

import warnings

from zensvi.download import GSVDownloader


def _make():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # GSVDownloader warns when no API key is given
        return GSVDownloader()


def test_default_sampling_is_grid():
    """GSV should default to grid sampling (avoids the slow OSM street-network lookup)."""
    dl = _make()
    assert dl._grid is True
    assert dl._grid_size == 50


def test_street_network_still_available():
    """grid=False must remain available for street-network sampling."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        dl = GSVDownloader(grid=False, distance=1)
    assert dl._grid is False
    assert dl._distance == 1
