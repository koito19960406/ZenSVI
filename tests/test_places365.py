import unittest

from zensvi.cv import ClassifierPlaces365


class TestClassifierPlaces365(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        pass

    def test_classify_single_image(self):
        classifier = ClassifierPlaces365()
        image_input = "tests/data/input/images"
        dir_image_output = "tests/data/output/classification/places365/image"
        dir_summary_output = "tests/data/output/classification/places365/summary"
        classifier.classify(
            image_input,
            dir_image_output=dir_image_output,
            dir_summary_output=dir_summary_output,
            csv_format="long",
        )


if __name__ == "__main__":
    unittest.main()
