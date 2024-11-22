import json
import os
import unittest

from test_base import TestBase

from zensvi.download import MLYDownloader
from zensvi.download.mapillary import interface


class TestMapillary(TestBase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.mly_api_key = os.getenv("MLY_API_KEY")
        interface.set_access_token(cls.mly_api_key)
        cls.output = cls.base_output_dir / "mly_output"
        cls.ensure_dir(cls.output)
        cls.mly_input_multipolygon = cls.input_dir / "test_multipolygon.geojson"
        cls.mly_input_polygon = cls.input_dir / "test_polygon.geojson"
        cls.mly_output_test_interface = cls.output / "test_interface.json"
        cls.mly_output_test_downloader = cls.output / "test_downloader"
        cls.mly_output_test_metadata = cls.output / "test_metadata"
        cls.mly_output_test_multipolygon = cls.output / "test_multipolygon"
        cls.mly_output_test_polygon = cls.output / "test_polygon"
        cls.mly_output_test_kwargs = cls.output / "test_kwargs"
        cls.mly_output_test_buffer = cls.output / "test_buffer"
        cls.mly_output_test_placename = cls.output / "test_placename"

    def test_interface(self):
        # Skip test if the output file already exists
        if os.path.exists(self.mly_output_test_interface):
            self.skipTest("Result exists")
        # read geojson as dict
        with open(self.mly_input_polygon) as f:
            geojson = json.load(f)
        output = interface.images_in_geojson(geojson)
        # assert True if output is not empty
        self.assertTrue(len(output.to_dict()) > 0)

    def test_downloader(self):
        # Skip test if the output file already exists
        if os.path.exists(os.path.join(self.mly_output_test_downloader, "mly_svi")):
            self.skipTest("Result exists")
        # download images
        mly_downloader = MLYDownloader(self.mly_api_key)
        mly_downloader.download_svi(self.mly_output_test_downloader, input_shp_file=self.mly_input_polygon)

        # assert True if there are files in the output directory
        self.assertTrue(len(os.listdir(self.mly_output_test_downloader)) > 0)

    def test_downloader_metadata_only(self):
        # Skip test if the output file already exists
        if os.path.exists(os.path.join(self.mly_output_test_metadata, "mly_pids.csv")):
            self.skipTest("Result exists")
        # download metadata only
        mly_downloader = MLYDownloader(
            self.mly_api_key,
            log_path=str(self.mly_output_test_metadata / "log.log"),
            max_workers=200,
        )
        # mly_downloader.download_svi(self.mly_output_test_metadata, input_place_name="Singapore", metadata_only=True)
        mly_downloader.download_svi(
            self.mly_output_test_metadata, input_shp_file=self.mly_input_polygon, metadata_only=True
        )
        # assert True if mly_pids.csv is not empty
        self.assertTrue(os.path.getsize(os.path.join(self.mly_output_test_metadata, "mly_pids.csv")) > 0)

    # test multipolygon
    def test_downloader_multipolygon(self):
        # # Skip test if the output file already exists
        # if os.path.exists(os.path.join(self.mly_output_test_multipolygon, "mly_svi")):
        #     self.skipTest("Result exists")
        # download images
        mly_downloader = MLYDownloader(self.mly_api_key, max_workers=200)
        mly_downloader.download_svi(self.mly_output_test_multipolygon, input_shp_file=self.mly_input_multipolygon)
        # assert True if there are files in the output directory
        self.assertTrue(len(os.listdir(self.mly_output_test_multipolygon)) > 0)

    # test polygon
    def test_downloader_polygon(self):
        # # Skip test if the output file already exists
        # if os.path.exists(os.path.join(self.mly_output_test_polygon, "mly_svi")):
        #     self.skipTest("Result exists")
        # download images
        mly_downloader = MLYDownloader(self.mly_api_key, max_workers=200)
        mly_downloader.download_svi(self.mly_output_test_polygon, input_shp_file=self.mly_input_polygon)
        # assert True if there are files in the output directory
        self.assertTrue(len(os.listdir(self.mly_output_test_polygon)) > 0)

    def test_downloader_kwargs(self):
        # Skip test if the output file already exists
        if os.path.exists(os.path.join(self.mly_output_test_kwargs, "mly_svi")):
            self.skipTest("Result exists")
        # download images
        mly_downloader = MLYDownloader(
            self.mly_api_key, log_path=str(self.mly_output_test_kwargs / "log.log"), max_workers=200
        )
        kwarg = {
            "image_type": "flat",  # The tile image_type to be obtained, either as 'flat', 'pano' (panoramic), or 'all'.
            "min_captured_at": 1484549945000,  # The min date. Format from 'YYYY', to 'YYYY-MM-DDTHH:MM:SS'
            "max_captured_at": 1642935417694,  # The max date. Format from 'YYYY', to 'YYYY-MM-DDTHH:MM:SS'
            "organization_id": [
                1805883732926354
            ],  # The organization id, ID of the organization this image (or sets of images) belong to. It can be absent. Thus, default is -1 (None)
            "compass_angle": (0, 180),
        }
        mly_downloader.download_svi(self.mly_output_test_kwargs, input_shp_file=self.mly_input_polygon, **kwarg)
        # assert True if there are files in the output directory
        self.assertTrue(len(os.listdir(self.mly_output_test_kwargs)) > 0)

    # test with buffer
    def test_downloader_with_buffer(self):
        # Skip test if the output file already exists
        if os.path.exists(os.path.join(self.mly_output_test_buffer, "mly_svi")):
            self.skipTest("Result exists")
        # download images
        mly_downloader = MLYDownloader(self.mly_api_key, max_workers=200)
        mly_downloader.download_svi(self.mly_output_test_buffer, lat=52.50, lon=13.42, buffer=1000)
        # assert True if there are files in the output directory
        self.assertTrue(len(os.listdir(self.mly_output_test_buffer)) > 0)

    # test input_place_name
    def test_downloader_place_name(self):
        # Skip test if the output file already exists
        if os.path.exists(os.path.join(self.mly_output_test_placename, "mly_svi")):
            self.skipTest("Result exists")
        # download images
        mly_downloader = MLYDownloader(self.mly_api_key, max_workers=300)
        mly_downloader.download_svi(self.mly_output_test_placename, input_place_name="Raffles Place, Singapore")
        # assert True if there are files in the output directory
        self.assertTrue(len(os.listdir(self.mly_output_test_placename)) > 0)


if __name__ == "__main__":
    unittest.main()
