from zensvi.download import GSVDownloader
import unittest
import os 
from pathlib import Path
import shutil

class TestStreetViewDownloader(unittest.TestCase):
    def setUp(self):
        self.output_dir = "tests/data/output"
        self.gsv_api_key = os.getenv('GSV_API_KEY')
        self.sv_downloader = GSVDownloader(gsv_api_key=self.gsv_api_key)

    def tearDown(self):
        pass

    def test_download_gsv(self):
        self.sv_downloader.download_svi(self.output_dir,lat=1.342425, lon=103.721523, augment_metadata=True)
        self.assertTrue(os.path.exists(os.path.join(self.output_dir, "gsv_panorama")))
    
if __name__ == '__main__':
    unittest.main()

