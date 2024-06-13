import json
import unittest
import os
import shutil
from pathlib import Path

from zensvi.download.mapillary import interface
from zensvi.download import MLYDownloader


class TestMapillary(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.mly_api_key = os.getenv("MLY_API_KEY")
        interface.set_access_token(self.mly_api_key)
        self.mly_output = "tests/data/output/mly_output/"
        os.makedirs(self.mly_output, exist_ok=True)
        self.mly_input_multipolygon = "tests/data/input/test_multipolygon.geojson"
        self.mly_input_polygon = "tests/data/input/test_polygon.geojson"
        self.mly_output_json = Path(self.mly_output) / "mly_output.json"
        self.mly_svi_output = Path(self.mly_output) / "mly_svi"
        self.mly_svi_output_multipolygon = Path(self.mly_output) / "mly_svi_multipolygon"
        self.mly_svi_output_polygon = Path(self.mly_output) / "mly_svi_polygon"
        pass
    
    @classmethod   
    def tearDown(self):
        # remove output directory
        shutil.rmtree(self.mly_output, ignore_errors=True)

    # def test_interface(self):
    #     # Skip test if the output file already exists
    #     if os.path.exists(self.mly_output_json):
    #         self.skipTest("Result exists")
    #     # read geojson as dict
    #     with open(self.mly_input_polygon) as f:
    #         geojson = json.load(f)
    #     output = interface.images_in_geojson(geojson)
    #     # assert True if output is not empty
    #     self.assertTrue(len(output.to_dict()) > 0)

    # def test_downloader(self):
    #     # Skip test if the output file already exists
    #     if os.path.exists(os.path.join(self.mly_svi_output, "mly_svi")):
    #         self.skipTest("Result exists")
    #     # download images
    #     mly_downloader = MLYDownloader(self.mly_api_key)
    #     mly_downloader.download_svi(self.mly_svi_output, input_shp_file=self.mly_input_polygon)

    #     # assert True if there are files in the output directory
    #     self.assertTrue(len(os.listdir(self.mly_svi_output)) > 0)

    # def test_downloader_metadata_only(self):
    #     # Skip test if the output file already exists
    #     if os.path.exists(os.path.join(self.mly_svi_output, "mly_pids.csv")):
    #         self.skipTest("Result exists")
    #     # download metadata only
    #     mly_downloader = MLYDownloader(
    #         self.mly_api_key,
    #         log_path="tests/data/output/mly_svi/log.log",
    #         max_workers=200,
    #     )
    #     # mly_downloader.download_svi(self.mly_svi_output, input_place_name="Singapore", metadata_only=True)
    #     mly_downloader.download_svi(
    #         self.mly_svi_output, input_shp_file=self.mly_input_polygon, metadata_only=True
    #     )
    #     # assert True if mly_pids.csv is not empty
    #     self.assertTrue(
    #         os.path.getsize(os.path.join(self.mly_svi_output, "mly_pids.csv")) > 0
    #     )

    # # test multipolygon
    # def test_downloader_multipolygon(self):
    #     # # Skip test if the output file already exists
    #     # if os.path.exists(os.path.join(self.mly_svi_output_multipolygon, "mly_svi")):
    #     #     self.skipTest("Result exists")
    #     # download images
    #     mly_downloader = MLYDownloader(self.mly_api_key, max_workers=200)
    #     mly_downloader.download_svi(
    #         self.mly_svi_output_multipolygon, input_shp_file=self.mly_input_multipolygon
    #     )
    #     # assert True if there are files in the output directory
    #     self.assertTrue(len(os.listdir(self.mly_svi_output_multipolygon)) > 0)

    # # test polygon
    # def test_downloader_polygon(self):
    #     # # Skip test if the output file already exists
    #     # if os.path.exists(os.path.join(self.mly_svi_output_polygon, "mly_svi")):
    #     #     self.skipTest("Result exists")
    #     # download images
    #     mly_downloader = MLYDownloader(self.mly_api_key, max_workers=200)
    #     mly_downloader.download_svi(
    #         self.mly_svi_output_polygon, input_shp_file=self.mly_input_polygon
    #     )
    #     # assert True if there are files in the output directory
    #     self.assertTrue(len(os.listdir(self.mly_svi_output_polygon)) > 0)


    # test with kwargs for mly
    def test_downloader_kwargs(self):
        # Skip test if the output file already exists
        if os.path.exists(os.path.join(self.mly_svi_output, "mly_svi")):
            self.skipTest("Result exists")
        # download images
        mly_downloader = MLYDownloader(self.mly_api_key)
        kwarg = {
            "image_type": "flat",  # The tile image_type to be obtained, either as 'flat', 'pano' (panoramic), or 'all'.
            "min_captured_at": 1484549945000,  # The min date. Format from 'YYYY', to 'YYYY-MM-DDTHH:MM:SS'
            "max_captured_at": 1642935417694,  # The max date. Format from 'YYYY', to 'YYYY-MM-DDTHH:MM:SS'
            "organization_id": [1805883732926354],  # The organization id, ID of the organization this image (or sets of images) belong to. It can be absent. Thus, default is -1 (None)
            "compass_angle": (0,180)
        }
        mly_downloader.download_svi(self.mly_svi_output, input_shp_file=self.mly_input_polygon, **kwarg)
        # assert True if there are files in the output directory
        self.assertTrue(len(os.listdir(self.mly_svi_output)) > 0)

if __name__ == "__main__":
    unittest.main()
