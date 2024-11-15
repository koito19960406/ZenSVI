import platform
from pathlib import Path

import pytest
import torch


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
