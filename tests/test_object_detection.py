import shutil
from pathlib import Path

import pytest

from zensvi.cv import ObjectDetector


@pytest.fixture(scope="function")  # Explicitly set function scope
def output_dir(base_output_dir, ensure_dir):
    output_dir = base_output_dir / "object_detection"
    if output_dir.exists():
        print(f"Cleaning up existing {output_dir} before test function")
        shutil.rmtree(output_dir)
    ensure_dir(output_dir)
    return output_dir


def test_detect_objects_directory(output_dir, input_dir):
    detector = ObjectDetector(text_prompt="tree .", box_threshold=0.45, text_threshold=0.25)
    image_input = str(input_dir / "images")
    dir_image_output = str(output_dir / "images")
    summary_output = str(output_dir / "summary")
    detector.detect_objects(
        dir_input=image_input, dir_image_output=dir_image_output, dir_summary_output=summary_output, max_workers=1
    )
    # Check that at least one file (annotated image) is created in the  subdirectory
    assert len(list(Path(dir_image_output).glob("*.jpg"))) > 0
    # Check that summary files were created
    assert Path(summary_output).joinpath("detection_summary.json").exists()


def test_detect_objects_single_image(output_dir, input_dir):
    detector = ObjectDetector(text_prompt="tree .", box_threshold=0.45, text_threshold=0.25)
    # Specify a single image file inside the images directory.
    image_input = str(input_dir / "images" / "-3vfS0_iiYVZKh_LEVlHew.jpg")
    dir_image_output = str(output_dir / "images")
    summary_output = str(output_dir / "summary")
    detector.detect_objects(
        dir_input=image_input, dir_image_output=dir_image_output, dir_summary_output=summary_output, max_workers=1
    )
    # Check that the annotated image is created in the  subdirectory
    assert len(list(Path(dir_image_output).glob("*.jpg"))) > 0
    # Check that summary files were created
    assert Path(summary_output).joinpath("detection_summary.json").exists()


def test_detect_objects_image_output_only(output_dir, input_dir):
    detector = ObjectDetector(text_prompt="tree .", box_threshold=0.45, text_threshold=0.25)
    image_input = str(input_dir / "images")
    dir_image_output = str(output_dir / "images_only")
    detector.detect_objects(
        dir_input=image_input, dir_image_output=dir_image_output, dir_summary_output=None, max_workers=1
    )
    # Check that at least one file (annotated image) is created in the  subdirectory
    assert len(list(Path(dir_image_output).glob("*.jpg"))) > 0


def test_detect_objects_summary_output_only(output_dir, input_dir):
    detector = ObjectDetector(text_prompt="tree .", box_threshold=0.45, text_threshold=0.25)
    image_input = str(input_dir / "images")
    summary_output = str(output_dir / "summary_only")
    detector.detect_objects(
        dir_input=image_input, dir_image_output=None, dir_summary_output=summary_output, max_workers=1
    )
    # Check that summary files were created
    assert Path(summary_output).joinpath("detection_summary.json").exists()


def test_detect_objects_no_output_directories_error(output_dir, input_dir):
    detector = ObjectDetector(text_prompt="tree .", box_threshold=0.45, text_threshold=0.25)
    image_input = str(input_dir / "images")
    with pytest.raises(ValueError, match="At least one of dir_image_output or dir_summary_output must be provided"):
        detector.detect_objects(dir_input=image_input, dir_image_output=None, dir_summary_output=None, max_workers=1)


def test_detect_objects_single_image_summary_only(output_dir, input_dir):
    detector = ObjectDetector(text_prompt="tree .", box_threshold=0.45, text_threshold=0.25)
    # Specify a single image file inside the images directory
    image_input = str(input_dir / "images" / "-3vfS0_iiYVZKh_LEVlHew.jpg")
    summary_output = str(output_dir / "single_image_summary_only")
    detector.detect_objects(
        dir_input=image_input, dir_image_output=None, dir_summary_output=summary_output, max_workers=1
    )
    # Check that summary files were created
    assert Path(summary_output).joinpath("detection_summary.json").exists()


def test_detect_objects_grouped_summary(output_dir, input_dir):
    detector = ObjectDetector(text_prompt="tree .", box_threshold=0.45, text_threshold=0.25)
    image_input = str(input_dir / "images")
    summary_output = str(output_dir / "grouped_summary")
    detector.detect_objects(
        dir_input=image_input,
        dir_image_output=None,
        dir_summary_output=summary_output,
        max_workers=1,
        group_by_object=True,
    )
    # Check that both detailed and grouped summary files were created
    assert Path(summary_output).joinpath("detection_summary.json").exists()
    assert Path(summary_output).joinpath("detection_summary_grouped.json").exists()

    # Additional test for CSV format
    summary_output_csv = str(output_dir / "grouped_summary_csv")
    detector.detect_objects(
        dir_input=image_input,
        dir_image_output=None,
        dir_summary_output=summary_output_csv,
        max_workers=1,
        group_by_object=True,
        save_format="csv",
    )
    # Check that both detailed and grouped summary CSV files were created
    assert Path(summary_output_csv).joinpath("detection_summary.csv").exists()
    assert Path(summary_output_csv).joinpath("detection_summary_grouped.csv").exists()


def test_detect_objects_combined_with_grouping(output_dir, input_dir):
    detector = ObjectDetector(text_prompt="tree .", box_threshold=0.45, text_threshold=0.25)
    image_input = str(input_dir / "images")
    dir_image_output = str(output_dir / "combined_images")
    summary_output = str(output_dir / "combined_summary")
    detector.detect_objects(
        dir_input=image_input,
        dir_image_output=dir_image_output,
        dir_summary_output=summary_output,
        max_workers=1,
        group_by_object=True,
        save_format="json csv",
    )
    # Check that images are created
    assert len(list(Path(dir_image_output).glob("*.jpg"))) > 0

    # Check that all summary files were created
    assert Path(summary_output).joinpath("detection_summary.json").exists()
    assert Path(summary_output).joinpath("detection_summary_grouped.json").exists()
    assert Path(summary_output).joinpath("detection_summary.csv").exists()
    assert Path(summary_output).joinpath("detection_summary_grouped.csv").exists()
