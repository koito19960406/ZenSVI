"""Tests for the base downloader module."""

import pytest
from unittest.mock import Mock, patch
from zensvi.download.base import BaseDownloader


class ConcreteDownloader(BaseDownloader):
    """Concrete implementation of BaseDownloader for testing."""

    def __init__(self, log_path=None):
        """Initialize the concrete downloader."""
        super().__init__(log_path)

    def _filter_pids_date(self, pid_df, start_date, end_date):
        """Dummy implementation of abstract method."""
        return pid_df

    def download_svi(
        self,
        dir_output,
        lat=None,
        lon=None,
        input_csv_file="",
        input_shp_file="",
        input_place_name="",
        id_columns=None,
        buffer=0,
        update_pids=False,
        start_date=None,
        end_date=None,
        metadata_only=False,
    ):
        """Dummy implementation of abstract method."""
        pass


class TestBaseDownloader:
    """Test suite for BaseDownloader class."""

    def test_initialization(self):
        """Test that BaseDownloader can be initialized."""
        downloader = ConcreteDownloader()
        assert downloader is not None
        assert downloader.log_path is None

    def test_initialization_with_log_path(self):
        """Test initialization with log path."""
        log_path = "/tmp/test_log.txt"
        downloader = ConcreteDownloader(log_path=log_path)
        assert downloader.log_path == log_path

    def test_log_path_property(self):
        """Test log_path property getter and setter."""
        downloader = ConcreteDownloader()
        assert downloader.log_path is None

        new_log_path = "/tmp/new_log.txt"
        downloader.log_path = new_log_path
        assert downloader.log_path == new_log_path

    def test_get_user_agents(self):
        """Test _get_ua method returns a list of user agents."""
        downloader = ConcreteDownloader()
        user_agents = downloader._user_agents

        assert isinstance(user_agents, list)
        assert len(user_agents) > 0
        # Check that each item has a 'user_agent' key
        for ua in user_agents:
            assert isinstance(ua, dict)
            assert "user_agent" in ua
            assert isinstance(ua["user_agent"], str)
            assert len(ua["user_agent"]) > 0

    def test_get_proxies(self):
        """Test _get_proxies method returns a list of proxies."""
        downloader = ConcreteDownloader()
        proxies = downloader._proxies

        assert isinstance(proxies, list)
        # Proxies list may be empty (fallback behavior), so we just check it's a list
        # If not empty, check structure
        if len(proxies) > 0:
            for proxy in proxies:
                assert isinstance(proxy, dict)

    def test_user_agents_not_empty(self):
        """Test that user agents list is not empty."""
        downloader = ConcreteDownloader()
        assert len(downloader._user_agents) > 0

    def test_importlib_resources_usage(self):
        """Test that importlib.resources is being used (not pkg_resources)."""
        # This test ensures that the package can be imported without pkg_resources
        try:
            from zensvi.download.base import BaseDownloader

            assert True
        except ImportError as e:
            if "pkg_resources" in str(e):
                pytest.fail("pkg_resources dependency not removed")
            else:
                raise

    def test_user_agent_fallback(self):
        """Test that user agent fallback works if file is not found."""
        # This test verifies that the fallback mechanism works
        with patch("zensvi.download.base.files") as mock_files:
            # Simulate file not found
            mock_files.side_effect = Exception("File not found")

            downloader = ConcreteDownloader()
            user_agents = downloader._user_agents

            # Should have fallback user agent
            assert isinstance(user_agents, list)
            assert len(user_agents) > 0
            assert "user_agent" in user_agents[0]

    def test_proxy_fallback(self):
        """Test that proxy fallback works if file is not found."""
        # This test verifies that the fallback mechanism works
        with patch("zensvi.download.base.files") as mock_files:
            # Simulate file not found
            mock_files.side_effect = Exception("File not found")

            downloader = ConcreteDownloader()
            proxies = downloader._proxies

            # Should return empty list as fallback
            assert isinstance(proxies, list)
            assert len(proxies) == 0

    def test_log_write(self, tmp_path):
        """Test _log_write method."""
        log_file = tmp_path / "test_log.txt"
        downloader = ConcreteDownloader(log_path=str(log_file))

        pids = ["pid1", "pid2", "pid3"]
        downloader._log_write(pids)

        # Check that file was created and contains the PIDs
        assert log_file.exists()
        content = log_file.read_text()
        for pid in pids:
            assert pid in content

    def test_log_write_no_path(self):
        """Test _log_write when no log path is set."""
        downloader = ConcreteDownloader()
        # Should not raise an error
        downloader._log_write(["pid1", "pid2"])

    def test_check_already(self, tmp_path):
        """Test _check_already method."""
        # Create a temporary directory with some "downloaded" files
        panorama_dir = tmp_path / "panoramas"
        panorama_dir.mkdir()

        # Create some dummy files
        (panorama_dir / "pid1.jpg").touch()
        (panorama_dir / "pid2.jpg").touch()
        (panorama_dir / "pid3.jpg").touch()

        downloader = ConcreteDownloader()
        downloader.panorama_output = str(panorama_dir)

        # Test with some PIDs that are already downloaded and some that are not
        all_pids = ["pid1", "pid2", "pid3", "pid4", "pid5"]
        remaining = downloader._check_already(all_pids)

        # Should only return PIDs that are not already downloaded
        assert "pid4" in remaining
        assert "pid5" in remaining
        assert "pid1" not in remaining
        assert "pid2" not in remaining
        assert "pid3" not in remaining

    def test_read_pids(self, tmp_path):
        """Test _read_pids method."""
        # Create a temporary CSV file with PIDs
        csv_file = tmp_path / "pids.csv"
        csv_file.write_text("panoid,date\npid1,2020-01-01\npid2,2020-01-02\npid3,2020-01-03\n")

        downloader = ConcreteDownloader()
        pids = downloader._read_pids(str(csv_file), None, None)

        assert isinstance(pids, list)
        assert len(pids) == 3
        assert "pid1" in pids
        assert "pid2" in pids
        assert "pid3" in pids
