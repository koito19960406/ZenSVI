"""Tests for zensvi.utils.log module."""

import logging
from pathlib import Path

import pytest

from zensvi.utils.log import Logger, verbosity_tqdm


class TestLogger:
    """Tests for the Logger class."""

    def setup_method(self):
        """Clear the singleton registry before each test."""
        Logger._loggers.clear()

    def test_logger_creates_file(self, tmp_path):
        """Logger should create the log file and parent directories."""
        log_file = tmp_path / "subdir" / "test.log"
        logger = Logger(str(log_file))
        logger.log_info("test message")
        assert log_file.exists()

    def test_logger_singleton(self, tmp_path):
        """Same log path should return same Logger instance."""
        log_file = str(tmp_path / "test.log")
        logger1 = Logger(log_file)
        logger2 = Logger(log_file)
        assert logger1 is logger2

    def test_logger_different_paths(self, tmp_path):
        """Different log paths should return different Logger instances."""
        logger1 = Logger(str(tmp_path / "a.log"))
        logger2 = Logger(str(tmp_path / "b.log"))
        assert logger1 is not logger2

    def test_log_info(self, tmp_path):
        """log_info should write INFO-level message to file."""
        log_file = tmp_path / "test.log"
        logger = Logger(str(log_file))
        logger.log_info("hello world")
        content = log_file.read_text()
        assert "INFO" in content
        assert "hello world" in content

    def test_log_error(self, tmp_path):
        """log_error should write ERROR-level message to file."""
        log_file = tmp_path / "test.log"
        logger = Logger(str(log_file))
        logger.log_error("something broke")
        content = log_file.read_text()
        assert "ERROR" in content
        assert "something broke" in content

    def test_log_warning(self, tmp_path):
        """log_warning should write WARNING-level message to file."""
        log_file = tmp_path / "test.log"
        logger = Logger(str(log_file))
        logger.log_warning("caution")
        content = log_file.read_text()
        assert "WARNING" in content
        assert "caution" in content

    def test_log_args(self, tmp_path):
        """log_args should log function name and arguments."""
        log_file = tmp_path / "test.log"
        logger = Logger(str(log_file))
        logger.log_args("my_func", "pos_arg", key="value")
        content = log_file.read_text()
        assert "my_func" in content
        assert "pos_arg" in content
        assert "key='value'" in content

    def test_log_failed_tile(self, tmp_path):
        """log_failed_tile should log failed tile name."""
        log_file = tmp_path / "test.log"
        logger = Logger(str(log_file))
        logger.log_failed_tile("tile_42")
        content = log_file.read_text()
        assert "ERROR" in content
        assert "tile_42" in content

    def test_log_failed_pid(self, tmp_path):
        """log_failed_pid should log failed panorama ID."""
        log_file = tmp_path / "test.log"
        logger = Logger(str(log_file))
        logger.log_failed_pid("abc123")
        content = log_file.read_text()
        assert "ERROR" in content
        assert "abc123" in content

    def test_log_failed_pid_int(self, tmp_path):
        """log_failed_pid should accept integer IDs."""
        log_file = tmp_path / "test.log"
        logger = Logger(str(log_file))
        logger.log_failed_pid(42)
        content = log_file.read_text()
        assert "42" in content


class TestVerbosityTqdm:
    """Tests for the verbosity_tqdm function."""

    def test_verbosity_zero_returns_plain_iterable(self):
        """Verbosity 0 should return the iterable without wrapping."""
        data = [1, 2, 3]
        result = verbosity_tqdm(data, verbosity=0)
        assert result is data

    def test_verbosity_one_level_one_wraps(self):
        """Verbosity 1, level 1 should wrap in tqdm."""
        data = [1, 2, 3]
        result = verbosity_tqdm(data, verbosity=1, level=1)
        # tqdm wraps the iterable; it should not be the same object
        assert result is not data
        # But iterating should produce the same values
        assert list(result) == data

    def test_verbosity_one_level_two_disabled(self):
        """Verbosity 1, level 2 should not wrap (inner loop suppressed)."""
        data = [1, 2, 3]
        result = verbosity_tqdm(data, verbosity=1, level=2)
        assert result is data

    def test_verbosity_two_level_two_wraps(self):
        """Verbosity 2 should wrap even level 2 loops."""
        data = [1, 2, 3]
        result = verbosity_tqdm(data, verbosity=2, level=2)
        assert result is not data
        assert list(result) == data

    def test_disable_flag_overrides(self):
        """Explicit disable=True should bypass verbosity."""
        data = [1, 2, 3]
        result = verbosity_tqdm(data, verbosity=2, level=1, disable=True)
        assert result is data
