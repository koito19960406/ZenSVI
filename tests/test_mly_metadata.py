from zensvi.metadata import MLYMetadata
import unittest
from pathlib import Path
import shutil


class TestMLYMetadata(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # create output directory
        cls.output_dir = Path("tests/data/output/metadata")
        cls.output_dir.mkdir(parents=True, exist_ok=True)

    # def tearDown(self):
    #     # remove output directory
    #     shutil.rmtree(self.output_dir, ignore_errors=True)

    def test_image_level_metadata(self):
        self.metadata = MLYMetadata("tests/data/input/metadata/mly_pids.csv", log_path=self.output_dir / "log.log")
        # test image-level metadata
        df = self.metadata.compute_metadata(unit="image", path_output=self.output_dir / "image_metadata.csv")
        print(df.head())
        # assert True if df has coverage column and it is not empty
        self.assertTrue("relative_angle" in df.columns)
        self.assertFalse(df["relative_angle"].empty)

    def test_grid_level_metadata(self):
        self.metadata = MLYMetadata("tests/data/input/metadata/mly_pids.csv", log_path=self.output_dir / "log.log")
        # test grid-level metadata
        df = self.metadata.compute_metadata(
            unit="grid",
            grid_resolution=12,
            coverage_buffer=10,
            path_output=self.output_dir / "grid_metadata.geojson",
        )
        print(df.head())
        # assert True if df has coverage column and it is not empty
        self.assertTrue("coverage" in df.columns)
        self.assertFalse(df["coverage"].empty)
        # save df as csv
        df.to_csv(
            self.output_dir / "grid_metadata.csv", index=False
        )

    def test_street_level_metadata(self):
        self.metadata = MLYMetadata("tests/data/input/metadata/mly_pids.csv", log_path=self.output_dir / "log.log")
        # test street-level metadata
        df = self.metadata.compute_metadata(
            unit="street",
            coverage_buffer=10,
            path_output=self.output_dir / "street_metadata.geojson",
        )
        print(df.head())
        # assert True if df has coverage column and it is not empty
        self.assertTrue("coverage" in df.columns)
        self.assertFalse(df["coverage"].empty)
        # save df as csv
        df.to_csv(
            self.output_dir / "street_metadata.csv", index=False
        )
    
    def test_image_level_partial_metadata(self):
        self.metadata = MLYMetadata("tests/data/input/metadata/mly_pids.csv", log_path=self.output_dir / "log.log")
        indicator_list = "day daytime_nighttime relative_angle"
        df = self.metadata.compute_metadata(
            unit="image",
            indicator_list=indicator_list,
            path_output=self.output_dir / "image_metadata_partial.csv",
        )
        print(df.head())
        # assert True if df has coverage column and it is not empty
        self.assertTrue("relative_angle" in df.columns)
        self.assertFalse(df["relative_angle"].empty)
        # save df as csv
        df.to_csv(self.output_dir / "image_metadata_partial.csv", index=False)
        
    def test_grid_level_partial_metadata(self):
        self.metadata = MLYMetadata("tests/data/input/metadata/mly_pids.csv", log_path=self.output_dir / "log.log")
        indicator_list = "coverage most_recent_date average_is_pano number_of_daytime"
        df = self.metadata.compute_metadata(
            unit="grid",
            grid_resolution=12,
            coverage_buffer=10,
            indicator_list=indicator_list,
            path_output=self.output_dir / "grid_metadata_partial.geojson",
        )
        print(df.head())
        # assert True if df has coverage column and it is not empty
        self.assertTrue("coverage" in df.columns)
        self.assertFalse(df["coverage"].empty)
        # save df as csv
        df.to_csv(
            self.output_dir / "grid_metadata_partial.csv", index=False
        )
        
    def test_street_level_partial_metadata(self):
        self.metadata = MLYMetadata("tests/data/input/metadata/mly_pids.csv", log_path=self.output_dir / "log.log")
        indicator_list = "coverage most_recent_date average_is_pano number_of_daytime"
        df = self.metadata.compute_metadata(
            unit="street",
            coverage_buffer=10,
            indicator_list=indicator_list,
            path_output=self.output_dir / "street_metadata_partial.geojson",
        )
        print(df.head())
        # assert True if df has coverage column and it is not empty
        self.assertTrue("coverage" in df.columns)
        self.assertFalse(df["coverage"].empty)
        # save df as csv
        df.to_csv(
            self.output_dir / "street_metadata_partial.csv", index=False
        )

    # # test with mly_pids_large.csv
    # def test_image_level_metadata_large(self):
    #     self.metadata = MLYMetadata("tests/data/input/metadata/mly_pids_large.csv")
    #     # test image-level metadata
    #     df = self.metadata.compute_metadata(unit="image", path_output=self.output_dir / "image_metadata_large.csv")
    #     print(df.head())
    #     # assert True if df has coverage column and it is not empty
    #     self.assertTrue("relative_angle" in df.columns)
    #     self.assertFalse(df["relative_angle"].empty)

    # def test_grid_level_metadata_large(self):
    #     self.metadata = MLYMetadata("tests/data/input/metadata/mly_pids_large.csv")
    #     # test grid-level metadata
    #     df = self.metadata.compute_metadata(
    #         unit="grid",
    #         grid_resolution=12,
    #         coverage_buffer=10,
    #         path_output=self.output_dir / "grid_metadata_large.geojson",
    #     )
    #     print(df.head())
    #     # assert True if df has coverage column and it is not empty
    #     self.assertTrue("coverage" in df.columns)
    #     self.assertFalse(df["coverage"].empty)
    #     # save df as csv
    #     df.to_csv(
    #         self.output_dir / "grid_metadata_large.csv", index=False
    #     )
        
    # def test_street_level_metadata_large(self):
    #     self.metadata = MLYMetadata("tests/data/input/metadata/mly_pids_large.csv")
    #     # test street-level metadata
    #     df = self.metadata.compute_metadata(
    #         unit="street",
    #         coverage_buffer=10,
    #         path_output=self.output_dir / "street_metadata_large.geojson",
    #     )
    #     print(df.head())
    #     # assert True if df has coverage column and it is not empty
    #     self.assertTrue("coverage" in df.columns)
    #     self.assertFalse(df["coverage"].empty)
    #     # save df as csv
    #     df.to_csv(
    #         self.output_dir / "street_metadata_large.csv", index=False
    #     )


if __name__ == "__main__":
    unittest.main()
