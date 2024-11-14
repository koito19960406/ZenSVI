import os
import unittest

from test_base import TestBase

from zensvi.cv import ClassifierReflection


class TestClassifierReflection(TestBase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.output = cls.base_output_dir / "classification/reflection"
        cls.ensure_dir(cls.output)

    def test_classify_directory(self):
        classifier = ClassifierReflection()
        image_input = str(self.input_dir / "images")
        dir_summary_output = str(self.output / "directory/summary")
        classifier.classify(
            image_input,
            dir_summary_output=dir_summary_output,
            batch_size=3,
        )
        self.assertTrue(os.listdir(dir_summary_output))

    def test_classify_single_image(self):
        classifier = ClassifierReflection()
        image_input = str(self.input_dir / "images/-3vfS0_iiYVZKh_LEVlHew.jpg")
        dir_summary_output = str(self.output / "single/summary")
        classifier.classify(
            image_input,
            dir_summary_output=dir_summary_output,
        )
        self.assertTrue(os.listdir(dir_summary_output))

    def test_classify_with_mps_device(self):
        device = "mps"
        classifier = ClassifierReflection(device=device)
        image_input = str(self.input_dir / "images")
        dir_summary_output = str(self.output / "mps/summary")
        classifier.classify(
            image_input,
            dir_summary_output=dir_summary_output,
            batch_size=3,
        )
        self.assertTrue(os.listdir(dir_summary_output))


if __name__ == "__main__":
    unittest.main()
