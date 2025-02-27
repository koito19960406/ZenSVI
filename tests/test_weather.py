import shutil
from pathlib import Path

import pytest

from zensvi.cv import ClassifierWeather


@pytest.fixture(scope="function")  # Explicitly set function scope
def output_dir(base_output_dir, ensure_dir):
    output_dir = base_output_dir / "weather"
    if output_dir.exists():
        print(f"Cleaning up existing {output_dir} before test function")  # Optional: for debugging
        shutil.rmtree(output_dir)
    ensure_dir(output_dir)
    return output_dir


def test_classify_directory(output_dir, input_dir, all_devices):
    classifier = ClassifierWeather(device=all_devices)
    image_input = str(input_dir / "images")
    dir_summary_output = str(output_dir / f"{all_devices}/directory/summary")
    classifier.classify(
        image_input,
        dir_summary_output=dir_summary_output,
        batch_size=3,
    )
    assert len(list(Path(dir_summary_output).iterdir())) > 0


def test_classify_single_image(output_dir, input_dir, all_devices):
    classifier = ClassifierWeather(device=all_devices)
    image_input = str(input_dir / "images/-3vfS0_iiYVZKh_LEVlHew.jpg")
    dir_summary_output = str(output_dir / f"{all_devices}/single/summary")
    classifier.classify(
        image_input,
        dir_summary_output=dir_summary_output,
    )
    assert len(list(Path(dir_summary_output).iterdir())) > 0


def test_verbosity_levels(output_dir, input_dir):
    """Test that verbosity levels work correctly in the ClassifierWeather class."""
    image_input = str(input_dir / "images/-3vfS0_iiYVZKh_LEVlHew.jpg")
    
    # Test with verbosity=0 (no progress bars)
    silent_classifier = ClassifierWeather(device="cpu", verbosity=0)
    assert silent_classifier.verbosity == 0
    
    # Test with default verbosity (1)
    default_classifier = ClassifierWeather(device="cpu")
    assert default_classifier.verbosity == 1
    
    # Test with verbosity=2 (all progress bars)
    verbose_classifier = ClassifierWeather(device="cpu", verbosity=2)
    assert verbose_classifier.verbosity == 2
    
    # Test the classify method with different verbosity levels
    # Run with default verbosity
    default_classifier.classify(
        image_input,
        dir_summary_output=str(output_dir / "default_verbosity/summary")
    )
    
    # Run with method-level verbosity override
    silent_classifier.classify(
        image_input,
        dir_summary_output=str(output_dir / "override_verbosity/summary"),
        verbosity=2  # Override instance verbosity
    )
    assert silent_classifier.verbosity == 0  # Make sure the instance verbosity remained unchanged
    
    # Check that results were produced
    assert len(list(Path(output_dir / "default_verbosity/summary").iterdir())) > 0
    assert len(list(Path(output_dir / "override_verbosity/summary").iterdir())) > 0
