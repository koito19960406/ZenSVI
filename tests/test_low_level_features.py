import json
import unittest
import os
from pathlib import Path
import pandas as pd
from test_base import TestBase

from zensvi.cv import get_low_level_features


class TestLowLevel(TestBase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.output = cls.base_output_dir / "low_level"
        cls.ensure_dir(cls.output)

    def test_low_level(self):
        image_input = str(self.input_dir / "images")
        image_output = self.output / "images"
        summary_output = self.output / "summary"
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
