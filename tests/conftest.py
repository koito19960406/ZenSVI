import platform
import shutil
import signal
from pathlib import Path
from contextlib import contextmanager

import pytest
import torch


# Define timeout duration as a constant
DOWNLOAD_TIMEOUT_SECONDS = 60


class TimeoutException(Exception):
    pass


@contextmanager
def timeout(seconds=DOWNLOAD_TIMEOUT_SECONDS):
    """
    A context manager that forces timeout on processes and their child threads.
    This works even when the function being timed out uses ThreadPoolExecutor internally.

    Args:
        seconds (int): Maximum execution time in seconds. Defaults to DOWNLOAD_TIMEOUT_SECONDS

    Raises:
        TimeoutException: If execution time exceeds the specified seconds
    """

    def signal_handler(signum, frame):
        raise TimeoutException(f"Operation timed out after {seconds} seconds")

    # Register the signal handler and set the alarm
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)

    try:
        yield
    finally:
        # Disable the alarm
        signal.alarm(0)


@pytest.fixture(scope="session")
def timeout_decorator():
    """Fixture to provide timeout context manager"""
    return timeout


@pytest.fixture(scope="session")
def timeout_seconds():
    """Fixture to provide timeout duration"""
    return DOWNLOAD_TIMEOUT_SECONDS


@pytest.fixture(scope="session")
def input_dir():
    return Path("tests/data/input")


@pytest.fixture(scope="session")
def base_output_dir(tmp_path_factory):
    output_dir = tmp_path_factory.mktemp("output")
    return output_dir


@pytest.fixture
def ensure_dir():
    def _ensure_dir(directory):
        directory.mkdir(parents=True, exist_ok=True)

    return _ensure_dir


@pytest.fixture(params=["cpu", "cuda", "mps"])
def all_devices(request):
    device = request.param
    # Skip CUDA tests if not available
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA device not available")
    # Skip MPS tests if not on Mac
    if device == "mps" and platform.system() != "Darwin":
        pytest.skip("MPS device only available on Mac")
    return device


@pytest.fixture(autouse=True, scope="function")
def cleanup_after_test(output_dir):
    """Fixture to clean up downloaded files after each test function"""
    yield  # Wait for the test function to complete
    if output_dir.exists():
        print(f"Cleaning up {output_dir} after test function")  # Optional: for debugging
        shutil.rmtree(output_dir)
