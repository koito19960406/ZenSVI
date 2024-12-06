import shutil
from pathlib import Path

import pytest

from zensvi.cv import ClassifierGlare


@pytest.fixture(scope="function")  # Explicitly set function scope
def output_dir(base_output_dir, ensure_dir):
    output_dir = base_output_dir / "glare"
    if output_dir.exists():
        print(f"Cleaning up existing {output_dir} before test function")  # Optional: for debugging
        shutil.rmtree(output_dir)
    ensure_dir(output_dir)
    return output_dir


def test_classify_directory(output_dir, input_dir, all_devices):
    classifier = ClassifierGlare(device=all_devices)
    image_input = str(input_dir / "images")
    dir_summary_output = str(Path(output_dir) / "directory/summary")
    classifier.classify(
        image_input,
        dir_summary_output=dir_summary_output,
        batch_size=3,
    )
    assert len(list(Path(dir_summary_output).iterdir())) > 0


def test_classify_single_image(output_dir, input_dir, all_devices):
    classifier = ClassifierGlare(device=all_devices)
    image_input = str(input_dir / "images/-3vfS0_iiYVZKh_LEVlHew.jpg")
    dir_summary_output = str(Path(output_dir) / "single/summary")
    classifier.classify(
        image_input,
        dir_summary_output=dir_summary_output,
    )
    assert len(list(Path(dir_summary_output).iterdir())) > 0


def test_classify_with_device(output_dir, input_dir, all_devices):
    classifier = ClassifierGlare(device=all_devices)
    image_input = str(input_dir / "images")
    dir_summary_output = str(Path(output_dir) / f"{all_devices}/summary")
    classifier.classify(
        image_input,
        dir_summary_output=dir_summary_output,
        batch_size=3,
    )
    assert len(list(Path(dir_summary_output).iterdir())) > 0
