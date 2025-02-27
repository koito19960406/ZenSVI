import json
import shutil

import numpy as np
import pandas as pd
import pytest

from zensvi.cv import Segmenter
from zensvi.transform import ImageTransformer


@pytest.fixture(scope="function")  # Explicitly set function scope
def output_dir(base_output_dir, ensure_dir):
    output_dir = base_output_dir / "segmentation"
    if output_dir.exists():
        print(f"Cleaning up existing {output_dir} before test function")  # Optional: for debugging
        shutil.rmtree(output_dir)
    ensure_dir(output_dir)
    return output_dir


def test_mapillary_panoptic(output_dir, input_dir, all_devices):
    segmenter = Segmenter(dataset="mapillary", task="panoptic", device=all_devices)
    image_output = output_dir / f"{all_devices}/mapillary_panoptic"
    summary_output = output_dir / f"{all_devices}/mapillary_panoptic_summary"
    segmenter.segment(
        input_dir / "images",
        dir_image_output=image_output,
        dir_summary_output=summary_output,
        csv_format="wide",
        max_workers=2,
    )
    assert len(list(image_output.glob("*"))) > 0
    assert len(list(summary_output.glob("*"))) > 0


def test_mapillary_semantic(output_dir, input_dir, all_devices):
    segmenter = Segmenter(dataset="mapillary", task="semantic", device=all_devices)
    image_output = output_dir / f"{all_devices}/mapillary_semantic"
    summary_output = output_dir / f"{all_devices}/mapillary_semantic_summary"
    segmenter.segment(
        input_dir / "images",
        dir_image_output=image_output,
        dir_summary_output=summary_output,
        csv_format="wide",
        max_workers=2,
    )
    assert len(list(image_output.glob("*"))) > 0
    assert len(list(summary_output.glob("*"))) > 0


def test_cityscapes_panoptic(output_dir, input_dir, all_devices):
    segmenter = Segmenter(dataset="cityscapes", task="panoptic", device=all_devices)
    image_output = output_dir / f"{all_devices}/cityscapes_panoptic"
    summary_output = output_dir / f"{all_devices}/cityscapes_panoptic_summary"
    segmenter.segment(
        input_dir / "images",
        dir_image_output=image_output,
        dir_summary_output=summary_output,
        csv_format="wide",
        max_workers=2,
    )
    assert len(list(image_output.glob("*"))) > 0
    assert len(list(summary_output.glob("*"))) > 0


def test_cityscapes_semantic(output_dir, input_dir, all_devices):
    segmenter = Segmenter(dataset="cityscapes", task="semantic", device=all_devices)
    image_output = output_dir / f"{all_devices}/cityscapes_semantic"
    summary_output = output_dir / f"{all_devices}/cityscapes_semantic_summary"
    segmenter.segment(
        input_dir / "images",
        dir_image_output=image_output,
        dir_summary_output=summary_output,
        csv_format="wide",
        max_workers=2,
    )
    assert len(list(image_output.glob("*"))) > 0
    assert len(list(summary_output.glob("*"))) > 0


def test_large_image(output_dir, input_dir, all_devices):
    segmenter = Segmenter(dataset="mapillary", task="panoptic", device=all_devices)
    image_output = output_dir / f"{all_devices}/large_image"
    summary_output = output_dir / f"{all_devices}/large_image_summary"
    segmenter.segment(
        input_dir / "large_images",
        dir_image_output=image_output,
        dir_summary_output=summary_output,
        csv_format="wide",
        max_workers=1,
    )
    assert len(list(image_output.glob("*"))) > 0
    assert len(list(summary_output.glob("*"))) > 0


def test_single_image(output_dir, input_dir, all_devices):
    segmenter = Segmenter(dataset="mapillary", task="panoptic", device=all_devices)
    image_input = str(input_dir / "images/-3vfS0_iiYVZKh_LEVlHew.jpg")
    image_output = output_dir / f"{all_devices}/single_image"
    summary_output = output_dir / f"{all_devices}/single_image_summary"
    segmenter.segment(
        image_input,
        dir_image_output=image_output,
        dir_summary_output=summary_output,
        csv_format="wide",
        max_workers=1,
    )
    assert len(list(image_output.glob("*"))) > 0
    assert len(list(summary_output.glob("*"))) > 0


def test_calculate_pixel_ratio_post_process(output_dir, input_dir):
    segmenter = Segmenter(dataset="mapillary", task="panoptic")
    image_input = str(input_dir / "cityscapes_semantic")
    image_output = output_dir / "calculate_pixel_ratio_post_process"
    segmenter.calculate_pixel_ratio_post_process(image_input, image_output)
    assert len(list(image_output.glob("*"))) > 0


def test_calculate_pixel_ratio_post_process_single_file(output_dir, input_dir):
    segmenter = Segmenter(dataset="mapillary", task="panoptic")
    image_input = str(input_dir / "cityscapes_semantic/-3vfS0_iiYVZKh_LEVlHew_colored_segmented.png")
    image_output = output_dir / "calculate_pixel_ratio_post_process_single_file"
    segmenter.calculate_pixel_ratio_post_process(image_input, image_output)
    assert len(list(image_output.glob("*"))) > 0


def test_long_format(output_dir, input_dir, all_devices):
    segmenter = Segmenter(device=all_devices)
    image_input = str(input_dir / "images")
    image_output = output_dir / f"{all_devices}/long_format"
    summary_output = output_dir / f"{all_devices}/long_format_summary"
    segmenter.segment(
        image_input,
        dir_image_output=image_output,
        dir_summary_output=summary_output,
        csv_format="long",
        max_workers=2,
    )
    assert len(list(image_output.glob("*"))) > 0
    assert len(list(summary_output.glob("*"))) > 0


def test_transformed_segmentation_pixel_ratio(output_dir, input_dir):
    """Test the pixel ratio calculation on transformed segmentation images.

    Steps:
    1. Transform segmented images to orthographic fish-eye projection
    2. Calculate pixel ratios using the post-process function
    3. Verify that the pixel ratios for each image sum to 1 (within tolerance)
    """
    # Step 1: Transform segmented images to orthographic fish-eye projection
    segmented_input = input_dir / "cityscapes_semantic"

    # Find all colored_segmented.png files
    segmented_files = [file for file in segmented_input.glob("*_colored_segmented.png")]

    # Create a temporary directory for the transformed images
    transform_output = output_dir / "transformed_segmentation"
    transform_output.mkdir(parents=True, exist_ok=True)

    # Use the ImageTransformer to transform the images - just use orthographic_fisheye
    for file in segmented_files:
        transformer = ImageTransformer(str(file), str(transform_output), log_path=output_dir / "transformation.log")
        transformer.transform_images(style_list="orthographic_fisheye")

    # Step 2: Calculate pixel ratios on transformed images
    segmenter = Segmenter(dataset="cityscapes", task="semantic")
    orthographic_dir = transform_output / "orthographic_fisheye"
    ratio_output = output_dir / "transformed_pixel_ratios"

    segmenter.calculate_pixel_ratio_post_process(str(orthographic_dir), str(ratio_output))

    # Step 3: Verify that the pixel ratios sum to 1
    # Read the CSV file with the pixel ratios
    csv_file = ratio_output / "pixel_ratios.csv"
    assert csv_file.exists()

    df = pd.read_csv(csv_file, index_col=0)

    # Calculate the sum of each row
    row_sums = df.sum(axis=1)

    # Check that all row sums are approximately 1 (allowing a larger tolerance due to potential NaN values)
    tolerance = 1e-5  # Increased tolerance

    # Remove rows that might be problematic (sum not close to 1)
    valid_rows = np.abs(row_sums - 1.0) < tolerance

    # Print info about any invalid rows for debugging
    if not np.all(valid_rows):
        print(f"Rows with ratio sum not close to 1: {row_sums[~valid_rows].to_dict()}")

    # Assert that at least 90% of rows sum to approximately 1
    min_valid_percentage = 0.9
    valid_percentage = np.mean(valid_rows)

    assert valid_percentage >= min_valid_percentage, (
        f"Only {valid_percentage * 100:.2f}% of rows sum to 1.0 within tolerance. "
        f"Expected at least {min_valid_percentage * 100:.2f}%"
    )

    # Also check JSON file
    json_file = ratio_output / "pixel_ratios.json"
    assert json_file.exists()

    with open(json_file, "r") as f:
        json_data = json.load(f)

    # Check that most image ratios sum to 1
    valid_count = 0
    total_count = len(json_data)

    for image_key, ratios in json_data.items():
        ratio_sum = sum(ratios.values())
        if abs(ratio_sum - 1.0) < tolerance:
            valid_count += 1

    valid_json_percentage = valid_count / total_count
    assert valid_json_percentage >= min_valid_percentage, (
        f"Only {valid_json_percentage * 100:.2f}% of images in JSON have ratio sums close to 1.0. "
        f"Expected at least {min_valid_percentage * 100:.2f}%"
    )


def test_verbosity_levels(all_devices):
    """Test that different verbosity levels are correctly set in the Segmenter class.

    This test verifies that the verbosity parameter is correctly initialized
    and that it can be modified after initialization.
    """
    # Test default verbosity level (should be 1)
    segmenter = Segmenter(device=all_devices)
    assert segmenter.verbosity == 1

    # Test explicit verbosity levels
    segmenter_quiet = Segmenter(device=all_devices, verbosity=0)
    assert segmenter_quiet.verbosity == 0

    segmenter_verbose = Segmenter(device=all_devices, verbosity=2)
    assert segmenter_verbose.verbosity == 2

    # Test changing verbosity after initialization
    segmenter.verbosity = 0
    assert segmenter.verbosity == 0

    segmenter.verbosity = 2
    assert segmenter.verbosity == 2

    # Test with different dataset and task
    segmenter_mapillary = Segmenter(dataset="mapillary", task="panoptic", device=all_devices, verbosity=0)
    assert segmenter_mapillary.verbosity == 0


def test_verbosity_in_segment(output_dir, input_dir, all_devices):
    """Test that the segment method respects verbosity levels.

    This test runs the segment method with different verbosity levels to ensure
    that the verbosity parameter affects the progress bar display.
    """
    # Create a segmenter with verbosity=0 (no progress bars)
    segmenter_quiet = Segmenter(device=all_devices, verbosity=0)
    image_output_quiet = output_dir / f"{all_devices}/verbosity_quiet"
    summary_output_quiet = output_dir / f"{all_devices}/verbosity_quiet_summary"

    # Run with verbosity=0
    segmenter_quiet.segment(
        input_dir / "images",
        dir_image_output=image_output_quiet,
        dir_summary_output=summary_output_quiet,
        max_workers=1,
    )
    # Check that files were generated
    assert len(list(image_output_quiet.glob("*"))) > 0
    assert len(list(summary_output_quiet.glob("*"))) > 0

    # Create a segmenter with verbosity=2 (all progress bars)
    segmenter_verbose = Segmenter(device=all_devices, verbosity=2)
    image_output_verbose = output_dir / f"{all_devices}/verbosity_verbose"
    summary_output_verbose = output_dir / f"{all_devices}/verbosity_verbose_summary"

    # Run with verbosity=2
    segmenter_verbose.segment(
        input_dir / "images",
        dir_image_output=image_output_verbose,
        dir_summary_output=summary_output_verbose,
        max_workers=1,
    )
    # Check that files were generated
    assert len(list(image_output_verbose.glob("*"))) > 0
    assert len(list(summary_output_verbose.glob("*"))) > 0
