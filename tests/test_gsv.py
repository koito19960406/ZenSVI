import math
import os
import shutil
import unittest
from pathlib import Path

import pandas as pd

from zensvi.download import GSVDownloader


class TestStreetViewDownloader(unittest.TestCase):
    def setUp(self):
        Path("tests/data/output/gsv_svi").mkdir(parents=True, exist_ok=True)
        self.output_dir = "tests/data/output/gsv_svi"
        self.gsv_api_key = os.getenv("GSV_API_KEY")
        self.sv_downloader = GSVDownloader(gsv_api_key=self.gsv_api_key, log_path="tests/data/output/gsv_svi/log.log")

    def tearDown(self):
        # recursively remove the output directory
        shutil.rmtree(self.output_dir)

    def test_download_gsv(self):
        # Skip test if the output file already exists
        if os.path.exists(os.path.join(self.output_dir, "gsv_panorama")):
            self.skipTest("Result exists")
        self.sv_downloader.download_svi(self.output_dir, lat=1.342425, lon=103.721523, augment_metadata=True)
        # assert True if there are files in the output directory
        self.assertTrue(os.listdir(self.output_dir))

    def test_download_gsv_zoom(self):
        # skip test if the output file already exists
        if os.path.exists(os.path.join(self.output_dir, "gsv_panorama_zoom_0")):
            self.skipTest("Result exists")
        # Define the range of zoom levels to test
        zoom_levels = [0, 1, 2, 3, 4]  # example zoom levels

        # List to store information about the downloads for comparison
        downloaded_zoom_folders = []

        for zoom in zoom_levels:
            # Create a folder name based on the zoom level
            folder_name = f"gsv_panorama_zoom_{zoom}"
            folder_path = os.path.join(self.output_dir, folder_name)

            # Create the folder if it doesn't exist
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)

            # Assuming download_svi method can accept a directory path as the output
            self.sv_downloader.download_svi(
                folder_path,
                lat=1.342425,
                lon=103.721523,
                augment_metadata=True,
                zoom=zoom,
            )

            # Check if any file was downloaded in the folder
            if os.listdir(folder_path):
                downloaded_zoom_folders.append(folder_name)
            else:
                print(f"No images were downloaded for zoom level {zoom}")

        # Assert that at least one folder has images
        self.assertTrue(len(downloaded_zoom_folders) > 0, "No images were downloaded in any folder")

        # Print downloaded folder names for review
        print("Downloaded Zoom Level Folders:")
        for folder in downloaded_zoom_folders:
            print(folder)

    def test_download_gsv_metadata_only(self):
        # Skip test if the output file already exists
        if os.path.exists(os.path.join(self.output_dir, "gsv_pids.csv")):
            self.skipTest("Result exists")
        self.sv_downloader.download_svi(
            self.output_dir,
            lat=1.342425,
            lon=103.721523,
            augment_metadata=True,
            metadata_only=True,
        )
        # assert True if there are more than 1 rows in os.path.join(self.output_dir, "gsv_pids.csv")
        df = pd.read_csv(os.path.join(self.output_dir, "gsv_pids.csv"))
        self.assertTrue(df.shape[0] > 1)

    def test_download_gsv_depth(self):
        # Skip test if the output file already exists
        # if os.path.exists(os.path.join(self.output_dir, "gsv_depth")):
        #     self.skipTest("Result exists")
        self.sv_downloader.download_svi(self.output_dir, lat=1.342425, lon=103.721523, download_depth=True)
        # assert True if there are files in the output directory
        self.assertTrue(os.listdir(self.output_dir))


if __name__ == "__main__":
    unittest.main()
