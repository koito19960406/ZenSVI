import json
import unittest
import os
import shutil
from pathlib import Path
import pandas as pd
from PIL import Image
import numpy as np
from scipy.spatial.transform import Rotation as R
import cv2
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
        self.mly_svi_output_buffer = Path(self.mly_output) / "mly_svi_buffer"
        pass
    
    # @classmethod   
    # def tearDown(self):
    #     # remove output directory
    #     shutil.rmtree(self.mly_output, ignore_errors=True)

    def test_interface(self):
        # Skip test if the output file already exists
        if os.path.exists(self.mly_output_json):
            self.skipTest("Result exists")
        # read geojson as dict
        with open(self.mly_input_polygon) as f:
            geojson = json.load(f)
        output = interface.images_in_geojson(geojson)
        # assert True if output is not empty
        self.assertTrue(len(output.to_dict()) > 0)

    def test_downloader(self):
        # Skip test if the output file already exists
        if os.path.exists(os.path.join(self.mly_svi_output, "mly_svi")):
            self.skipTest("Result exists")
        # download images
        mly_downloader = MLYDownloader(self.mly_api_key)
        mly_downloader.download_svi(self.mly_svi_output, input_shp_file=self.mly_input_polygon)

        # assert True if there are files in the output directory
        self.assertTrue(len(os.listdir(self.mly_svi_output)) > 0)

    def test_downloader_metadata_only(self):
        # Skip test if the output file already exists
        if os.path.exists(os.path.join(self.mly_svi_output, "mly_pids.csv")):
            self.skipTest("Result exists")
        # download metadata only
        mly_downloader = MLYDownloader(
            self.mly_api_key,
            log_path="tests/data/output/mly_svi/log.log",
            max_workers=200,
        )
        # mly_downloader.download_svi(self.mly_svi_output, input_place_name="Singapore", metadata_only=True)
        mly_downloader.download_svi(
            self.mly_svi_output, input_shp_file=self.mly_input_polygon, metadata_only=True
        )
        # assert True if mly_pids.csv is not empty
        self.assertTrue(
            os.path.getsize(os.path.join(self.mly_svi_output, "mly_pids.csv")) > 0
        )

    # test multipolygon
    def test_downloader_multipolygon(self):
        # # Skip test if the output file already exists
        # if os.path.exists(os.path.join(self.mly_svi_output_multipolygon, "mly_svi")):
        #     self.skipTest("Result exists")
        # download images
        mly_downloader = MLYDownloader(self.mly_api_key, max_workers=200)
        mly_downloader.download_svi(
            self.mly_svi_output_multipolygon, input_shp_file=self.mly_input_multipolygon
        )
        # assert True if there are files in the output directory
        self.assertTrue(len(os.listdir(self.mly_svi_output_multipolygon)) > 0)

    # test polygon
    def test_downloader_polygon(self):
        # # Skip test if the output file already exists
        # if os.path.exists(os.path.join(self.mly_svi_output_polygon, "mly_svi")):
        #     self.skipTest("Result exists")
        # download images
        mly_downloader = MLYDownloader(self.mly_api_key, max_workers=200)
        mly_downloader.download_svi(
            self.mly_svi_output_polygon, input_shp_file=self.mly_input_polygon
        )
        # assert True if there are files in the output directory
        self.assertTrue(len(os.listdir(self.mly_svi_output_polygon)) > 0)
        
    def test_downloader_single_field(self):
        # Set up a new output directory for this test
        single_field_output = Path(self.mly_output) / "mly_svi_single_field"
        
        # Download images with only the 'captured_at' field
        mly_downloader = MLYDownloader(self.mly_api_key, max_workers=300)
        mly_downloader.download_svi(
            single_field_output,
            input_shp_file=self.mly_input_polygon,
            additional_fields=["all"]
        )
        
        # Check if the output directory is created and contains files
        self.assertTrue(single_field_output.exists())
        self.assertTrue(len(list(single_field_output.glob('**/*'))) > 0)
        
        # Check if the mly_pids.csv file exists and contains all the expected columns
        pids_file = single_field_output / "pids_urls.csv"
        self.assertTrue(pids_file.exists())
        
        df = pd.read_csv(pids_file)
        
        # Check for all fields from Entities.get_image_fields()
        expected_columns = [
            'altitude', 'atomic_scale', 'camera_parameters', 'camera_type',
            'captured_at', 'compass_angle', 'computed_altitude',
            'computed_compass_angle', 'computed_geometry', 'computed_rotation',
            'exif_orientation', 'geometry', 'height', 'url', 'merge_cc', 'mesh',
            'quality_score', 'sequence', 'sfm_cluster', 'width'
        ]
        
        for column in expected_columns:
            self.assertIn(column, df.columns)
        
        # Optionally, you can check the content of some image files to ensure they're valid
        image_files = list(single_field_output.glob('**/*.jpg'))
        if image_files:
            with Image.open(image_files[0]) as img:
                self.assertTrue(img.format in ('JPEG', 'PNG'))

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

    # test with buffer
    def test_downloader_with_buffer(self):
        # Skip test if the output file already exists
        if os.path.exists(os.path.join(self.mly_svi_output_buffer, "mly_svi")):
            self.skipTest("Result exists")
        # download images
        mly_downloader = MLYDownloader(self.mly_api_key, max_workers=200)
        mly_downloader.download_svi(
            self.mly_svi_output_buffer, lat = 52.50, lon = 13.42, buffer=1000
        )
        # assert True if there are files in the output directory
        self.assertTrue(len(os.listdir(self.mly_svi_output_buffer)) > 0)

    # test input_place_name
    def test_downloader_place_name(self):
        # Skip test if the output file already exists
        if os.path.exists(os.path.join(self.mly_svi_output, "mly_svi")):
            self.skipTest("Result exists")
        # download images
        mly_downloader = MLYDownloader(self.mly_api_key, max_workers=300)
        mly_downloader.download_svi(self.mly_svi_output, input_place_name="Tbilisi")
        # assert True if there are files in the output directory
        self.assertTrue(len(os.listdir(self.mly_svi_output)) > 0)
        

if __name__ == "__main__":
    unittest.main()
