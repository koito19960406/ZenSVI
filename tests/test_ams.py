from zensvi.download.ams import AMSDownloader
import unittest
import os 
from pathlib import Path
import pandas as pd
from test_base import TestBase


class TestAMSDownloader(TestBase):
    def setUp(self):
        super().setUp()
        self.output_dir = self.base_output_dir / "ams_svi"
        self.ensure_dir(self.output_dir)
        self.sv_downloader = AMSDownloader(log_path=self.output_dir / "log.log")
     
    def test_download_asv(self):
        self.sv_downloader.download_svi(self.output_dir, lat=52.356768, lon=4.907408, buffer=10)
        self.assertTrue(os.listdir(self.output_dir))

    def test_download_asv_metadata_only(self):
        self.sv_downloader.download_svi(self.output_dir, lat=52.356768, lon=4.907408, buffer=10, metadata_only=True)
        df = pd.read_csv(self.output_dir / "ams_pids.csv")
        self.assertTrue(df.shape[0] > 1)

    def test_csv_download_asv(self):
        self.sv_downloader.download_svi(self.output_dir, input_csv_file=self.input_dir / "test.csv", buffer=10)
        self.assertTrue(os.listdir(self.output_dir))
    
    def test_shp_download_asv(self):
        file_list = ["point.geojson", "line.geojson", "polygon.geojson"]
        for file in file_list:
            self.sv_downloader.download_svi(self.output_dir, input_shp_file=self.input_dir / file, buffer=10)
            self.assertTrue(os.listdir(self.output_dir))
        
    def test_place_name_download_asv(self):
        self.sv_downloader.download_svi(self.output_dir, input_place_name="Amsterdam Landlust", buffer=10)
        self.assertTrue(os.listdir(self.output_dir))


if __name__ == '__main__':
    unittest.main()
