import unittest
import os
import shutil
from pathlib import Path

from zensvi.cv import DepthEstimator
from test_base import TestBase

class TestDepthEstimator(TestBase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.output = cls.base_output_dir / "depth_estimation"
        cls.ensure_dir(cls.output)

    def test_classify_directory(self):
        classifier = DepthEstimator()
        image_input = "tests/data/input/images"
        dir_image_output = str(Path(self.output) / "directory/summary")
        classifier.estimate_depth(
            image_input,
            dir_image_output=dir_image_output,
            batch_size=3,
        )
        # assert True if files in dir_image_output and dir_image_output are not empty
        self.assertTrue(os.listdir(dir_image_output))

    def test_classify_single_image(self):
        classifier = DepthEstimator()
        image_input = "tests/data/input/images/-3vfS0_iiYVZKh_LEVlHew.jpg"
        dir_image_output = str(Path(self.output) / "single/summary")
        classifier.estimate_depth(
            image_input,
            dir_image_output=dir_image_output,
        )
        # assert True if files in dir_image_output and dir_image_output are not empty
        self.assertTrue(os.listdir(dir_image_output))

    def test_classify_with_mps_device(self):
        device = "mps"
        classifier = DepthEstimator(device=device, task="relative")
        image_input = "tests/data/input/images"
        dir_image_output = str(Path(self.output) / "mps/summary")
        classifier.estimate_depth(
            image_input,
            dir_image_output=dir_image_output,
            batch_size=3,
        )
        # assert True if files in dir_image_output and dir_image_output are not empty
        self.assertTrue(os.listdir(dir_image_output))

    def test_classify_absolute_depth(self):
        classifier = DepthEstimator(task="absolute")
        image_input = "tests/data/input/images"
        dir_image_output = str(Path(self.output) / "absolute/summary")
        classifier.estimate_depth(
            image_input,
            dir_image_output=dir_image_output,
            batch_size=3,
        )
        # assert True if files in dir_image_output and dir_image_output are not empty
        self.assertTrue(os.listdir(dir_image_output))

    def test_classify_absolute_depth_mps(self):
        device = "mps"
        classifier = DepthEstimator(device=device, task="absolute")
        image_input = "tests/data/input/images"
        dir_image_output = str(Path(self.output) / "absolute_mps/summary")
        classifier.estimate_depth(
            image_input,
            dir_image_output=dir_image_output,
            batch_size=3,
        )
        # assert True if files in dir_image_output and dir_image_output are not empty
        self.assertTrue(os.listdir(dir_image_output))


if __name__ == "__main__":
    unittest.main()
