import pytest
from zensvi.cv import Segmenter


@pytest.fixture
def output(base_output_dir, ensure_dir):
    output_dir = base_output_dir / "segmentation"
    ensure_dir(output_dir)
    return output_dir


def test_mapillary_panoptic(output, input_dir, all_devices):
    segmenter = Segmenter(dataset="mapillary", task="panoptic", device=all_devices)
    image_output = output / f"{all_devices}/mapillary_panoptic"
    summary_output = output / f"{all_devices}/mapillary_panoptic_summary"
    segmenter.segment(
        input_dir / "images",
        dir_image_output=image_output,
        dir_summary_output=summary_output,
        csv_format="wide",
        max_workers=2,
    )
    assert len(list(image_output.glob("*"))) > 0
    assert len(list(summary_output.glob("*"))) > 0


def test_mapillary_semantic(output, input_dir, all_devices):
    segmenter = Segmenter(dataset="mapillary", task="semantic", device=all_devices)
    image_output = output / f"{all_devices}/mapillary_semantic"
    summary_output = output / f"{all_devices}/mapillary_semantic_summary"
    segmenter.segment(
        input_dir / "images",
        dir_image_output=image_output,
        dir_summary_output=summary_output,
        csv_format="wide",
        max_workers=2,
    )
    assert len(list(image_output.glob("*"))) > 0
    assert len(list(summary_output.glob("*"))) > 0


def test_cityscapes_panoptic(output, input_dir, all_devices):
    segmenter = Segmenter(dataset="cityscapes", task="panoptic", device=all_devices)
    image_output = output / f"{all_devices}/cityscapes_panoptic"
    summary_output = output / f"{all_devices}/cityscapes_panoptic_summary"
    segmenter.segment(
        input_dir / "images",
        dir_image_output=image_output,
        dir_summary_output=summary_output,
        csv_format="wide",
        max_workers=2,
    )
    assert len(list(image_output.glob("*"))) > 0
    assert len(list(summary_output.glob("*"))) > 0


def test_cityscapes_semantic(output, input_dir, all_devices):
    segmenter = Segmenter(dataset="cityscapes", task="semantic", device=all_devices)
    image_output = output / f"{all_devices}/cityscapes_semantic"
    summary_output = output / f"{all_devices}/cityscapes_semantic_summary"
    segmenter.segment(
        input_dir / "images",
        dir_image_output=image_output,
        dir_summary_output=summary_output,
        csv_format="wide",
        max_workers=2,
    )
    assert len(list(image_output.glob("*"))) > 0
    assert len(list(summary_output.glob("*"))) > 0


def test_large_image(output, input_dir, all_devices):
    segmenter = Segmenter(dataset="mapillary", task="panoptic", device=all_devices)
    image_output = output / f"{all_devices}/large_image"
    summary_output = output / f"{all_devices}/large_image_summary"
    segmenter.segment(
        input_dir / "large_images",
        dir_image_output=image_output,
        dir_summary_output=summary_output,
        csv_format="wide",
        max_workers=1,
    )
    assert len(list(image_output.glob("*"))) > 0
    assert len(list(summary_output.glob("*"))) > 0


def test_single_image(output, input_dir, all_devices):
    segmenter = Segmenter(dataset="mapillary", task="panoptic", device=all_devices)
    image_input = str(input_dir / "images/-3vfS0_iiYVZKh_LEVlHew.jpg")
    image_output = output / f"{all_devices}/single_image"
    summary_output = output / f"{all_devices}/single_image_summary"
    segmenter.segment(
        image_input,
        dir_image_output=image_output,
        dir_summary_output=summary_output,
        csv_format="wide",
        max_workers=1,
    )
    assert len(list(image_output.glob("*"))) > 0
    assert len(list(summary_output.glob("*"))) > 0


def test_calculate_pixel_ratio_post_process(output, input_dir):
    segmenter = Segmenter(dataset="mapillary", task="panoptic")
    image_input = str(input_dir / "cityscapes_semantic")
    image_output = output / "calculate_pixel_ratio_post_process"
    segmenter.calculate_pixel_ratio_post_process(image_input, image_output)
    assert len(list(image_output.glob("*"))) > 0


def test_calculate_pixel_ratio_post_process_single_file(output, input_dir):
    segmenter = Segmenter(dataset="mapillary", task="panoptic")
    image_input = str(input_dir / "cityscapes_semantic/-3vfS0_iiYVZKh_LEVlHew_colored_segmented.png")
    image_output = output / "calculate_pixel_ratio_post_process_single_file"
    segmenter.calculate_pixel_ratio_post_process(image_input, image_output)
    assert len(list(image_output.glob("*"))) > 0


def test_long_format(output, input_dir, all_devices):
    segmenter = Segmenter(device=all_devices)
    image_input = str(input_dir / "images")
    image_output = output / f"{all_devices}/long_format"
    summary_output = output / f"{all_devices}/long_format_summary"
    segmenter.segment(
        image_input,
        dir_image_output=image_output,
        dir_summary_output=summary_output,
        csv_format="long",
        max_workers=2,
    )
    assert len(list(image_output.glob("*"))) > 0
    assert len(list(summary_output.glob("*"))) > 0
