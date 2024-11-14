#!/usr/bin/env python3

import os
import unittest
import os
from pathlib import Path
from zensvi.cv import ClassifierPerception
from zensvi.cv import ClassifierPerceptionViT
from test_base import TestBase

from zensvi.cv import ClassifierPerception


class TestClassifierPerception(TestBase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.output = cls.base_output_dir / "classification/perception"
        cls.ensure_dir(cls.output)
    # def tearDown(self):
    #     # remove output directory
    #     shutil.rmtree(self.output, ignore_errors=True)

    def test_classify_directory(self):
        classifier = ClassifierPerception(perception_study="more boring")
        image_input = "tests/data/input/images"
        dir_summary_output = str(Path(self.output) / "directory/summary")
        classifier.classify(
            image_input,
            dir_summary_output=dir_summary_output,
            batch_size=3,
        )
        self.assertTrue(os.listdir(dir_summary_output))

    def test_classify_single_image(self):
        classifier = ClassifierPerception(perception_study="more boring")
        image_input = "tests/data/input/images/test1.jpg"
        dir_summary_output = str(Path(self.output) / "single/summary")
        classifier.classify(
            image_input,
            dir_summary_output=dir_summary_output,
        )
        self.assertTrue(os.listdir(dir_summary_output))

    def test_classify_with_mps_device(self):
        device = "mps"
        classifier = ClassifierPerception(
            perception_study="more boring", device=device)
        image_input = "tests/data/input/images"
        dir_summary_output = str(Path(self.output) / "mps/summary")
        classifier.classify(
            image_input,
            dir_summary_output=dir_summary_output,
            batch_size=3,
        )
        self.assertTrue(os.listdir(dir_summary_output))

    def test_classify_directory_vit(self):
        classifier = ClassifierPerceptionViT(perception_study = 'more boring')
        image_input = "tests/data/input/images"
        dir_summary_output = str(Path(self.output) / "directory/summary")
        classifier.classify(
            image_input,
            dir_summary_output=dir_summary_output,
            batch_size=3,
        )
        # assert True if files in dir_image_output and dir_summary_output are not empty
        self.assertTrue(os.listdir(dir_summary_output))


if __name__ == "__main__":
    unittest.main()
