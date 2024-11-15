from pathlib import Path

import pytest

from zensvi.transform import ImageTransformer


@pytest.fixture
def output(base_output_dir, ensure_dir):
    output_dir = base_output_dir / "transformation"
    ensure_dir(output_dir)
    return output_dir


def test_transform_images(output, input_dir):
    dir_input = str(input_dir / "images")
    dir_output = str(output)
    transformer = ImageTransformer(dir_input, dir_output, log_path=output / "transformation.log")
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


def test_transform_single_image(output, input_dir):
    dir_input = str(input_dir / "images/-3vfS0_iiYVZKh_LEVlHew.jpg")
    dir_output = str(output / "single_image")
    transformer = ImageTransformer(dir_input, dir_output, log_path=output / "transformation.log")
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


def test_upper_half(output, input_dir):
    dir_input = str(input_dir / "images/-3vfS0_iiYVZKh_LEVlHew.jpg")
    dir_output = str(output / "upper_half")
    transformer = ImageTransformer(dir_input, dir_output, log_path=output / "transformation.log")
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
