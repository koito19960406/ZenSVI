import json
import unittest
import os 

from zensvi.download.mapillary import interface
from zensvi.download import MLYDownloader

class TestMapillary(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mly_api_key = os.getenv('MLY_API_KEY')
        interface.set_access_token(cls.mly_api_key)
        cls.mly_output = 'tests/data/output/mly_output/'
        os.makedirs(cls.mly_output, exist_ok=True)
        cls.mly_input = 'tests/data/input/test.geojson'
        cls.mly_output = 'tests/data/output/mly_output/output.geojson'
        cls.mly_svi_output = 'tests/data/output/mly_svi'
        pass
    
    def test_interface(self):
        # Skip test if the output file already exists
        if os.path.exists(self.mly_output):
            self.skipTest("Result exists")
        # read geojson as dict
        with open(self.mly_input) as f:
            geojson = json.load(f)
        output = interface.images_in_geojson(geojson)
        # save output as geojson
        with open(self.mly_output, 'w') as f:
            json.dump(output.to_dict(), f)

    def test_downloader(self):
        # Skip test if the output file already exists
        if os.path.exists(os.path.join(self.mly_svi_output, "mly_svi")):
            self.skipTest("Result exists")
        # download images
        mly_downloader = MLYDownloader(self.mly_api_key)
        mly_downloader.download_svi(self.mly_svi_output, input_shp_file=self.mly_input)    
        
    def test_downloader_metadata_only(self):
        # Skip test if the output file already exists
        if os.path.exists(os.path.join(self.mly_svi_output, "mly_pids.csv")):
            self.skipTest("Result exists")
        # download metadata only
        mly_downloader = MLYDownloader(self.mly_api_key, log_path="tests/data/output/mly_svi/log.log", max_workers=200)
        mly_downloader.download_svi(self.mly_svi_output, input_place_name="Singapore", metadata_only=True)
        # mly_downloader.download_svi(self.mly_svi_output, input_shp_file=self.mly_input, metadata_only=True, use_cache=False)
        
if __name__ == '__main__':
    unittest.main()