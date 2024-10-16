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
        self.sv_downloader = AMSDownloader(log_path="tests/data/output/ams_svi/log.log")
     
    def tearDown(self):
        # remove output directory
        shutil.rmtree(self.output_dir, ignore_errors=True)

    def test_download_asv(self):
        self.sv_downloader.download_svi(self.output_dir,lat=52.356768, lon=4.907408, buffer = 10)
        # assert True if there are files in the output directory
        self.assertTrue(os.listdir(self.output_dir))

    def test_download_asv_metadata_only(self):
        self.sv_downloader.download_svi(self.output_dir,lat=52.356768, lon=4.907408, buffer = 10, metadata_only=True)
        # assert True if there are more than 1 rows in os.path.join(self.output_dir, "asm_pids.csv")
        df = pd.read_csv(os.path.join(self.output_dir, "ams_pids.csv"))
        self.assertTrue(df.shape[0] > 1)

    def test_csv_download_asv(self):
        self.sv_downloader.download_svi(self.output_dir, input_csv_file="tests/data/input/test.csv", buffer = 10)
        # assert True if there are files in the output directory
        self.assertTrue(os.listdir(self.output_dir))
    
    def test_shp_download_asv(self):
        file_list = ["tests/data/input/point.geojson", "tests/data/input/line.geojson", "tests/data/input/polygon.geojson"]
        for file in file_list:
            self.sv_downloader.download_svi(self.output_dir, input_shp_file=file, buffer = 10)
            # assert True if there are files in the output directory
            self.assertTrue(os.listdir(self.output_dir))
        
    def test_place_name_download_asv(self):
        self.sv_downloader.download_svi(self.output_dir, input_place_name="Amsterdam Landlust", buffer = 10)
        # assert True if there are files in the output directory
        self.assertTrue(os.listdir(self.output_dir))

if __name__ == '__main__':
    unittest.main()

