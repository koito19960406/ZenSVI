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


def test_verbosity_levels(output_dir, input_dir):
    """Test that verbosity levels work correctly in the ImageTransformer class."""
    dir_input = str(input_dir / "images/-3vfS0_iiYVZKh_LEVlHew.jpg")

    # Test default verbosity (should be 1)
    default_transformer = ImageTransformer(
        dir_input, str(output_dir / "default_verbosity"), log_path=output_dir / "default_verbosity.log"
    )
    assert default_transformer.verbosity == 1

    # Test with verbosity=0 (no progress bars)
    silent_transformer = ImageTransformer(
        dir_input, str(output_dir / "silent_verbosity"), log_path=output_dir / "silent_verbosity.log", verbosity=0
    )
    assert silent_transformer.verbosity == 0

    # Test with verbosity=2 (all progress bars)
    verbose_transformer = ImageTransformer(
        dir_input, str(output_dir / "verbose_verbosity"), log_path=output_dir / "verbose_verbosity.log", verbosity=2
    )
    assert verbose_transformer.verbosity == 2

    # Test the transform_images method respects verbosity parameter
    # First run with default verbosity
    default_transformer.transform_images(style_list="perspective equidistant_fisheye")

    # Run with verbosity set explicitly in method call (overriding instance verbosity)
    silent_transformer.transform_images(style_list="perspective equidistant_fisheye", verbosity=2)
    assert silent_transformer.verbosity == 0  # Should still be 0 after the call

    # Check that transformations were successful
    for transformer_dir in ["default_verbosity", "silent_verbosity", "verbose_verbosity"]:
        output_path = output_dir / transformer_dir
        if output_path.exists():  # Only check directories that were actually created
            for sub_dir in ["perspective", "equidistant_fisheye"]:
                sub_path = output_path / sub_dir
                if sub_path.exists():  # Some tests might not have created all subdirs
                    assert sub_path.is_dir()
                    assert len(list(sub_path.rglob("*"))) > 0
