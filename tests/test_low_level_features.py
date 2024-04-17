import json
import unittest
import os
from pathlib import Path
import shutil
import pandas as pd

from zensvi.cv import get_low_level_features


class TestLowLevel(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.output_dir = Path("tests/data/output/low_level")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        pass

    def tearDown(self):
        # recursively remove the output directory
        shutil.rmtree(self.output_dir)

    def test_low_level(self):
        image_input = "tests/data/input/images"
        image_output = self.output_dir / "images"
        summary_output = self.output_dir / "summary"
        get_low_level_features(
            image_input,
            dir_image_output=image_output,
            dir_summary_output=summary_output,
            save_format="json csv",
            csv_format="wide",
        )
        files_in_image_output = [f for f in image_output.rglob("*") if f.is_file()]
        df = pd.read_csv(summary_output / "low_level_features.csv")
        self.assertTrue(len(files_in_image_output) > 0 and len(df) > 0)


if __name__ == "__main__":
    unittest.main()
