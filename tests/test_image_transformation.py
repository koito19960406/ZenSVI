from zensvi.transform import ImageTransformer
import unittest
from pathlib import Path
import shutil


class TestImageTransformer(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.output = Path("tests/data/output/transformation")
        self.output.mkdir(parents=True, exist_ok=True)
        pass

    # def tearDown(self):
    #     # remove output directory
    #     shutil.rmtree(self.output, ignore_errors=True)

    # skip for now
    # @unittest.skip("skip for now")
    def test_transform_images(self):
        dir_input = "tests/data/input/images"
        dir_output = str(self.output)
        transformer = ImageTransformer(dir_input, dir_output)
        transformer.transform_images()
        # assert True if all the subdirectories are created in the output directory and files are not empty
        assert all(
            [
                Path(dir_output, sub_dir).is_dir()
                and len(list(Path(dir_output, sub_dir).rglob("*"))) > 0
                for sub_dir in [
                    "equidistant_fisheye",
                    "equisolid_fisheye",
                    "orthographic_fisheye",
                    "perspective",
                    "stereographic_fisheye",
                ]
            ]
        )

    def test_transform_single_image(self):
        dir_input = "tests/data/input/images/-3vfS0_iiYVZKh_LEVlHew.jpg"
        dir_output = str(self.output / "single_image")
        transformer = ImageTransformer(dir_input, dir_output)
        transformer.transform_images()
        # assert True if all the subdirectories are created in the output directory and files are not empty
        assert all(
            [
                Path(dir_output, sub_dir).is_dir()
                and len(list(Path(dir_output, sub_dir).rglob("*"))) > 0
                for sub_dir in [
                    "equidistant_fisheye",
                    "equisolid_fisheye",
                    "orthographic_fisheye",
                    "perspective",
                    "stereographic_fisheye",
                ]
            ]
        )

if __name__ == "__main__":
    unittest.main()
