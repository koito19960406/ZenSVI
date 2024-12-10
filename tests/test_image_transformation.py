import shutil
from pathlib import Path

import pytest

from zensvi.transform import ImageTransformer


@pytest.fixture(scope="function")  # Explicitly set function scope
def output_dir(base_output_dir, ensure_dir):
    output_dir = base_output_dir / "image_transformation"
    if output_dir.exists():
        print(f"Cleaning up existing {output_dir} before test function")  # Optional: for debugging
        shutil.rmtree(output_dir)
    ensure_dir(output_dir)
    return output_dir


def test_transform_images(output_dir, input_dir):
    dir_input = str(input_dir / "images")
    dir_output = str(output_dir)
    transformer = ImageTransformer(dir_input, dir_output, log_path=output_dir / "transformation.log")
    transformer.transform_images()

    expected_subdirs = [
        "equidistant_fisheye",
        "equisolid_fisheye",
        "orthographic_fisheye",
        "perspective",
        "stereographic_fisheye",
    ]

    for sub_dir in expected_subdirs:
        sub_path = Path(dir_output) / sub_dir
        assert sub_path.is_dir()
        assert len(list(sub_path.rglob("*"))) > 0


def test_transform_single_image(output_dir, input_dir):
    dir_input = str(input_dir / "images/-3vfS0_iiYVZKh_LEVlHew.jpg")
    dir_output = str(output_dir / "single_image")
    transformer = ImageTransformer(dir_input, dir_output, log_path=output_dir / "transformation.log")
    transformer.transform_images()

    expected_subdirs = [
        "equidistant_fisheye",
        "equisolid_fisheye",
        "orthographic_fisheye",
        "perspective",
        "stereographic_fisheye",
    ]

    for sub_dir in expected_subdirs:
        sub_path = Path(dir_output) / sub_dir
        assert sub_path.is_dir()
        assert len(list(sub_path.rglob("*"))) > 0


def test_upper_half(output_dir, input_dir):
    dir_input = str(input_dir / "images/-3vfS0_iiYVZKh_LEVlHew.jpg")
    dir_output = str(output_dir / "upper_half")
    transformer = ImageTransformer(dir_input, dir_output, log_path=output_dir / "transformation.log")
    transformer.transform_images(use_upper_half=True)

    expected_subdirs = [
        "equidistant_fisheye",
        "equisolid_fisheye",
        "orthographic_fisheye",
        "perspective",
        "stereographic_fisheye",
    ]

    for sub_dir in expected_subdirs:
        sub_path = Path(dir_output) / sub_dir
        assert sub_path.is_dir()
        assert len(list(sub_path.rglob("*"))) > 0
