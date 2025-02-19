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
    detector = ObjectDetector(
        text_prompt="tree .",
        box_threshold=0.45,
        text_threshold=0.25
    )
    image_input = str(input_dir / "images")
    dir_image_output = str(Path(output_dir) / "directory" / "summary")
    detector.detect_objects(
        dir_input=image_input,
        dir_output=dir_image_output,
        max_workers=1  # you can adjust this value as needed
    )
    # Check that at least one file (annotated image) is created in the output directory
    assert len(list(Path(dir_image_output).iterdir())) > 0

def test_detect_objects_single_image(output_dir, input_dir):
    detector = ObjectDetector(
        text_prompt="tree .",
        box_threshold=0.45,
        text_threshold=0.25
    )
    # Specify a single image file inside the images directory.
    image_input = str(input_dir / "images" / "single_image.jpg")
    dir_image_output = str(Path(output_dir) / "single" / "summary")
    detector.detect_objects(
        dir_input=image_input,
        dir_output=dir_image_output,
    )
    # Check that at least one file (annotated image) is created in the output directory
    assert len(list(Path(dir_image_output).iterdir())) > 0