from zensvi.download.ams import AMSDownloader
import unittest
import os 
from pathlib import Path
import shutil
import math
import pandas as pd


class TestAMSDownloader(unittest.TestCase):
    def setUp(self):
        Path("tests/data/output/ams_svi").mkdir(parents=True, exist_ok=True)
        self.output_dir = "tests/data/output/ams_svi"
        self.ams_api_key = None
        self.sv_downloader = AMSDownloader(ams_api_key=self.ams_api_key, log_path="tests/data/output/ams_svi/log.log")
     
    # def tearDown(self):
    #     # remove output directory
    #     shutil.rmtree(self.mly_output, ignore_errors=True)

    def test_download_asv(self):
        # Skip test if the output file already exists
        if os.path.exists(os.path.join(self.output_dir, "asv_panorama")):
            self.skipTest("Result exists")
        self.sv_downloader.download_svi(self.output_dir,lat=52.356768, lon=4.907408, buffer = 10)
        # assert True if there are files in the output directory
        self.assertTrue(os.listdir(self.output_dir))

    def test_download_asv_metadata_only(self):
        # Skip test if the output file already exists
        if os.path.exists(os.path.join(self.output_dir, "asv_pids.csv")):
            self.skipTest("Result exists")
        self.sv_downloader.download_svi(self.output_dir,lat=52.356768, lon=4.907408, buffer = 10, metadata_only=True)
        # assert True if there are more than 1 rows in os.path.join(self.output_dir, "asv_pids.csv")
        df = pd.read_csv(os.path.join(self.output_dir, "asv_pids.csv"))
        self.assertTrue(df.shape[0] > 1)

if __name__ == '__main__':
    unittest.main()
