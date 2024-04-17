import json
import unittest
import os
from pathlib import Path
import shutil

from zensvi.cv import Segmenter


class TestSegmentation(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.image_input = Path("tests/data/input/images")
        self.output = Path("tests/data/output/segmentation")
        Path(self.output).mkdir(parents=True, exist_ok=True)
        pass

    def tearDown(self):
        # remove output directory
        shutil.rmtree(self.output, ignore_errors=True)

    def test_mapillary_panoptic(self):
        segmenter = Segmenter(dataset="mapillary", task="panoptic")
        image_output = self.output / "mapillary_panoptic"
        summary_output = self.output / "mapillary_panoptic_summary"
        segmenter.segment(
            self.image_input,
            dir_image_output=image_output,
            dir_summary_output=summary_output,
            csv_format="wide",
            max_workers=4,
        )
        # assert True if files in image_output and summary_output are not empty
        self.assertTrue(
            len(list(image_output.glob("*"))) > 0
            and len(list(summary_output.glob("*"))) > 0
        )

    def test_mapillary_semantic(self):
        segmenter = Segmenter(dataset="mapillary", task="semantic")
        image_output = self.output / "mapillary_semantic"
        summary_output = self.output / "mapillary_semantic_summary"
        segmenter.segment(
            self.image_input,
            dir_image_output=image_output,
            dir_summary_output=summary_output,
            csv_format="wide",
            max_workers=4,
        )
        # assert True if files in image_output and summary_output are not empty
        self.assertTrue(
            len(list(image_output.glob("*"))) > 0
            and len(list(summary_output.glob("*"))) > 0
        )

    def test_cityscapes_panoptic(self):
        segmenter = Segmenter(dataset="cityscapes", task="panoptic")
        image_output = self.output / "cityscapes_panoptic"
        summary_output = self.output / "cityscapes_panoptic_summary"
        segmenter.segment(
            self.image_input,
            dir_image_output=image_output,
            dir_summary_output=summary_output,
            csv_format="wide",
            max_workers=4,
        )
        # assert True if files in image_output and summary_output are not empty
        self.assertTrue(
            len(list(image_output.glob("*"))) > 0
            and len(list(summary_output.glob("*"))) > 0
        )

    def test_cityscapes_semantic(self):
        segmenter = Segmenter(dataset="cityscapes", task="semantic")
        image_output = self.output / "cityscapes_semantic"
        summary_output = self.output / "cityscapes_semantic_summary"
        segmenter.segment(
            self.image_input,
            dir_image_output=image_output,
            dir_summary_output=summary_output,
            csv_format="wide",
            max_workers=4,
        )
        # assert True if files in image_output and summary_output are not empty
        self.assertTrue(
            len(list(image_output.glob("*"))) > 0
            and len(list(summary_output.glob("*"))) > 0
        )

    def test_large_image(self):
        segmenter = Segmenter(dataset="mapillary", task="panoptic")
        image_output = self.output / "large_image"
        summary_output = self.output / "large_image_summary"
        segmenter.segment(
            self.image_input,
            dir_image_output=image_output,
            dir_summary_output=summary_output,
            csv_format="wide",
            max_workers=1,
        )
        # assert True if files in image_output and summary_output are not empty
        self.assertTrue(
            len(list(image_output.glob("*"))) > 0
            and len(list(summary_output.glob("*"))) > 0
        )

    def test_single_image(self):
        segmenter = Segmenter(dataset="mapillary", task="panoptic")
        image_input = str(self.image_input / "-3vfS0_iiYVZKh_LEVlHew.jpg")
        image_output = self.output / "single_image"
        summary_output = self.output / "single_image_summary"
        segmenter.segment(
            image_input,
            dir_image_output=image_output,
            dir_summary_output=summary_output,
            csv_format="wide",
            max_workers=1,
        )
        # assert True if files in image_output and summary_output are not empty
        self.assertTrue(
            len(list(image_output.glob("*"))) > 0
            and len(list(summary_output.glob("*"))) > 0
        )

    def test_mps_device(self):
        device = "mps"
        segmenter = Segmenter(dataset="mapillary", task="panoptic", device=device)
        image_output = self.output / "mps_device"
        summary_output = self.output / "mps_device_summary"
        segmenter.segment(
            self.image_input,
            dir_image_output=image_output,
            dir_summary_output=summary_output,
            csv_format="wide",
            max_workers=4,
        )
        # assert True if files in image_output and summary_output are not empty
        self.assertTrue(
            len(list(image_output.glob("*"))) > 0
            and len(list(summary_output.glob("*"))) > 0
        )

    def test_calculate_pixel_ratio_post_process(self):
        segmenter = Segmenter(dataset="mapillary", task="panoptic")
        image_input = "tests/data/input/cityscapes_semantic"
        image_output = self.output / "calculate_pixel_ratio_post_process"
        segmenter.calculate_pixel_ratio_post_process(image_input, image_output)
        # assert True if files in image_output are not empty
        self.assertTrue(len(list(image_output.glob("*"))) > 0)

    def test_calculate_pixel_ratio_post_process_single_file(self):
        segmenter = Segmenter(dataset="mapillary", task="panoptic")
        image_input = "tests/data/input/cityscapes_semantic/-3vfS0_iiYVZKh_LEVlHew_colored_segmented.png"
        image_output = self.output / "calculate_pixel_ratio_post_process_single_file"
        segmenter.calculate_pixel_ratio_post_process(image_input, image_output)
        # assert True if files in image_output are not empty
        self.assertTrue(len(list(image_output.glob("*"))) > 0)


if __name__ == "__main__":
    unittest.main()
