import os
import unittest
from pathlib import Path

from test_base import TestBase

from zensvi.cv import ClassifierPlaces365


class TestClassifierPlaces365(TestBase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.output = cls.base_output_dir / "classification/places365"
        cls.ensure_dir(cls.output)

    def test_classify_directory(self):
        classifier = ClassifierPlaces365()
        image_input = str(self.input_dir / "images")
        dir_image_output = str(self.output / "directory/image")
        dir_summary_output = str(self.output / "directory/summary")
        classifier.classify(
            image_input,
            dir_image_output=dir_image_output,
            dir_summary_output=dir_summary_output,
            csv_format="wide",
            batch_size=3,
        )
        self.assertTrue(os.listdir(dir_image_output) and os.listdir(dir_summary_output))

    def test_classify_single_image(self):
        classifier = ClassifierPlaces365()
        image_input = "tests/data/input/images/-3vfS0_iiYVZKh_LEVlHew.jpg"
        dir_image_output = str(Path(self.output) / "single/image")
        dir_summary_output = str(Path(self.output) / "single/summary")
        classifier.classify(
            image_input,
            dir_image_output=dir_image_output,
            dir_summary_output=dir_summary_output,
            csv_format="wide",
        )
        # assert True if files in dir_image_output and dir_summary_output are not empty
        self.assertTrue(os.listdir(dir_image_output) and os.listdir(dir_summary_output))

    def test_classify_with_mps_device(self):
        device = "mps"
        classifier = ClassifierPlaces365(device=device)
        image_input = "tests/data/input/images"
        dir_image_output = str(Path(self.output) / "mps/image")
        dir_summary_output = str(Path(self.output) / "mps/summary")
        classifier.classify(
            image_input,
            dir_image_output=dir_image_output,
            dir_summary_output=dir_summary_output,
            csv_format="wide",
            batch_size=3,
        )
        # assert True if files in dir_image_output and dir_summary_output are not empty
        self.assertTrue(os.listdir(dir_image_output) and os.listdir(dir_summary_output))

    def test_classify_to_long_format(self):
        device = "mps"
        classifier = ClassifierPlaces365(device=device)
        image_input = "tests/data/input/images"
        dir_image_output = str(Path(self.output) / "long/image")
        dir_summary_output = str(Path(self.output) / "long/summary")
        classifier.classify(
            image_input,
            dir_image_output=dir_image_output,
            dir_summary_output=dir_summary_output,
            csv_format="long",
            batch_size=3,
        )
        # assert True if files in dir_image_output and dir_summary_output are not empty
        self.assertTrue(os.listdir(dir_image_output) and os.listdir(dir_summary_output))


if __name__ == "__main__":
    unittest.main()
