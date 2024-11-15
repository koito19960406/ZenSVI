import pytest
from pathlib import Path
from zensvi.cv import ClassifierPlaces365


@pytest.fixture
def output(base_output_dir, ensure_dir):
    output_dir = base_output_dir / "classification/places365"
    ensure_dir(output_dir)
    return output_dir


def test_classify_directory(output, input_dir, all_devices):
    classifier = ClassifierPlaces365(device=all_devices)
    image_input = str(input_dir / "images")
    dir_image_output = str(output / f"{all_devices}/directory/image")
    dir_summary_output = str(output / f"{all_devices}/directory/summary")
    classifier.classify(
        image_input,
        dir_image_output=dir_image_output,
        dir_summary_output=dir_summary_output,
        csv_format="wide",
        batch_size=3,
    )
    assert len(list(Path(dir_image_output).iterdir())) > 0
    assert len(list(Path(dir_summary_output).iterdir())) > 0


def test_classify_single_image(output, input_dir, all_devices):
    classifier = ClassifierPlaces365(device=all_devices)
    image_input = str(input_dir / "images/-3vfS0_iiYVZKh_LEVlHew.jpg")
    dir_image_output = str(output / f"{all_devices}/single/image")
    dir_summary_output = str(output / f"{all_devices}/single/summary")
    classifier.classify(
        image_input,
        dir_image_output=dir_image_output,
        dir_summary_output=dir_summary_output,
        csv_format="wide",
    )
    assert len(list(Path(dir_image_output).iterdir())) > 0
    assert len(list(Path(dir_summary_output).iterdir())) > 0


def test_classify_to_long_format(output, input_dir, all_devices):
    classifier = ClassifierPlaces365(device=all_devices)
    image_input = str(input_dir / "images")
    dir_image_output = str(output / f"{all_devices}/long/image")
    dir_summary_output = str(output / f"{all_devices}/long/summary")
    classifier.classify(
        image_input,
        dir_image_output=dir_image_output,
        dir_summary_output=dir_summary_output,
        csv_format="long",
        batch_size=3,
    )
    assert len(list(Path(dir_image_output).iterdir())) > 0
    assert len(list(Path(dir_summary_output).iterdir())) > 0
