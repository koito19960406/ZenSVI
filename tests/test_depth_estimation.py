import shutil
from pathlib import Path

import pytest

from zensvi.cv import DepthEstimator


@pytest.fixture(scope="function")  # Explicitly set function scope
def output_dir(base_output_dir, ensure_dir):
    output_dir = base_output_dir / "kv_svi"
    if output_dir.exists():
        print(f"Cleaning up existing {output_dir} before test function")  # Optional: for debugging
        shutil.rmtree(output_dir)
    ensure_dir(output_dir)
    return output_dir


def test_classify_directory(output_dir, input_dir, all_devices):
    classifier = DepthEstimator(device=all_devices)
    image_input = str(input_dir / "images")
    dir_image_output = str(Path(output_dir) / f"{all_devices}/directory/summary")
    classifier.estimate_depth(
        image_input,
        dir_image_output=dir_image_output,
        batch_size=1,
    )
    assert len(list(Path(dir_image_output).iterdir())) > 0


def test_classify_single_image(output_dir, input_dir, all_devices):
    classifier = DepthEstimator(device=all_devices)
    image_input = str(input_dir / "images/-3vfS0_iiYVZKh_LEVlHew.jpg")
    dir_image_output = str(Path(output_dir) / f"{all_devices}/single/summary")
    classifier.estimate_depth(
        image_input,
        dir_image_output=dir_image_output,
    )
    assert len(list(Path(dir_image_output).iterdir())) > 0


def test_classify_relative_depth(output_dir, input_dir, all_devices):
    classifier = DepthEstimator(device=all_devices, task="relative")
    image_input = str(input_dir / "images")
    dir_image_output = str(Path(output_dir) / f"{all_devices}/relative/summary")
    classifier.estimate_depth(
        image_input,
        dir_image_output=dir_image_output,
        batch_size=1,
    )
    assert len(list(Path(dir_image_output).iterdir())) > 0


def test_classify_absolute_depth(output_dir, input_dir, all_devices):
    classifier = DepthEstimator(device=all_devices, task="absolute")
    image_input = str(input_dir / "images")
    dir_image_output = str(Path(output_dir) / f"{all_devices}/absolute/summary")
    classifier.estimate_depth(
        image_input,
        dir_image_output=dir_image_output,
        batch_size=1,
    )
    assert len(list(Path(dir_image_output).iterdir())) > 0
