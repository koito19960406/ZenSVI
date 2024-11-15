import pytest
import signal
from contextlib import contextmanager
from zensvi.download import GSDownloader


class TimeoutException(Exception):
    pass


@contextmanager
def time_limit(seconds):
    """Context manager to limit execution time of a block of code."""

    def signal_handler(signum, frame):
        raise TimeoutException("Download test timed out")

    # Register the signal handler
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)  # Set the alarm

    try:
        yield
    finally:
        signal.alarm(0)  # Disable the alarm


@pytest.fixture
def output_dir(base_output_dir, ensure_dir):
    output_dir = base_output_dir / "gs_download"
    ensure_dir(output_dir)
    return output_dir


@pytest.fixture
def gs_download():
    return GSDownloader()


def test_partial_download(output_dir, gs_download):
    """Test that download starts successfully by letting it run for 5 seconds."""
    try:
        with time_limit(5):  # Allow 5 seconds of downloading
            gs_download.download_all_data(str(output_dir))
    except TimeoutException:
        pass  # Expected to timeout

    # Check if any files were created
    files = list(output_dir.iterdir())
    assert len(files) > 0

    # Check if at least one file has content
    for file in files:
        if file.is_file():
            assert file.stat().st_size > 0, "Downloaded file should contain some data"


def test_download_structure(output_dir, gs_download):
    """Test that download creates correct folder structure in first few seconds."""
    try:
        with time_limit(3):
            gs_download.download_manual_labels(str(output_dir))
    except TimeoutException:
        pass

    # Verify the folder structure was created
    assert (output_dir / "manual_labels").exists()


def test_multiple_downloads(output_dir, gs_download, ensure_dir):
    """Test multiple download methods start successfully."""
    download_methods = [
        (gs_download.download_train, "train"),
        (gs_download.download_test, "test"),
        (gs_download.download_img_tar, "img"),
    ]

    for download_func, folder_name in download_methods:
        test_dir = output_dir / folder_name
        ensure_dir(test_dir)

        try:
            with time_limit(2):  # 2 seconds per download type
                download_func(str(test_dir))
        except TimeoutException:
            pass

        # Verify download started
        assert any(test_dir.iterdir()), f"Download should have started for {folder_name}"


def test_api_connection(gs_download):
    """Test that we can connect to HuggingFace and list files."""
    from huggingface_hub import HfApi

    api = HfApi()
    files = api.list_repo_files(gs_download.repo_id, repo_type=gs_download.repo_type)

    assert len(files) > 0, "Should be able to list repository files"
