import os
import shutil
import unittest

import geopandas as gp
from test_base import TestBase

import zensvi.download.kartaview.download_functions as kv
from zensvi.download.kv import KVDownloader


class TestKartaView(TestBase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.output = cls.base_output_dir / "kv_output"
        cls.ensure_dir(cls.output)
        cls.kv_input_multipolygon = cls.input_dir / "test_multipolygon.geojson"
        cls.kv_input_polygon = cls.input_dir / "test_polygon.geojson"
        cls.kv_output_test_interface = cls.output / "test_interface.csv"
        cls.kv_output_test_downloader = cls.output / "test_downloader"
        cls.kv_output_test_polygon_metaonly = cls.output / "test_polygon_metaonly"
        cls.kv_output_test_multipolygon = cls.output / "test_multipolygon"
        cls.kv_output_test_polygon = cls.output / "test_polygon"

    @classmethod
    def tearDown(self):
        # remove output directory
        shutil.rmtree(self.kv_output, ignore_errors=True)

    def test_interface(self):
        print()
        print("--------------------------------")
        print("Testing interface...")
        # Skip test if the output file already exists
        if os.path.exists(self.kv_output_test_interface):
            print("Result exits, skipping")
            self.skipTest("Result exists")
        # read geojson as dict
        gdf = gp.read_file(self.kv_input_polygon)
        output = kv.get_points_in_shape(gdf)
        output.to_csv(self.kv_output_test_interface)
        # assert True if output is not empty
        self.assertTrue(len(output) > 0)

    def test_downloader(self):
        print()
        print("--------------------------------")
        print("Testing downloader...")
        # Skip test if the output file already exists
        if os.path.exists(os.path.join(self.kv_output_test_downloader, "kv_pids.csv")):
            print("Result exits, skipping")
            self.skipTest("Result exists")
        # download images
        kv_downloader = KVDownloader(log_path=os.path.join(self.kv_output_test_downloader, "log.log"), max_workers=2000)
        kv_downloader.download_svi(self.kv_output_test_downloader, input_shp_file=self.kv_input_polygon)

        # assert True if there are files in the output directory (other than log.log)
        self.assertTrue(len(os.listdir(self.kv_output_test_downloader)) > 1)

    def test_downloader_metadata_only(self):
        print()
        print("--------------------------------")
        print("Testing polygon input (metadata only)...")
        # Skip test if the output file already exists
        if os.path.exists(os.path.join(self.kv_output_test_polygon_metaonly, "kv_pids.csv")):
            print("Result exits, skipping")
            self.skipTest("Result exists")
        # download metadata only
        kv_downloader = KVDownloader(log_path=os.path.join(self.kv_output_test_polygon_metaonly, "log.log"), max_workers=2000)
        # kv_downloader.download_svi(self.kv_output_test_polygon_metaonly, input_place_name="Singapore", metadata_only=True)
        kv_downloader.download_svi(
            self.kv_output_test_polygon_metaonly,
            input_shp_file=self.kv_input_polygon,
            metadata_only=True,
        )
        # assert True if kv_pids.csv is not empty
        self.assertTrue(os.path.getsize(os.path.join(self.kv_output_test_polygon_metaonly, "kv_pids.csv")) > 0)

    # test multipolygon
    def test_downloader_multipolygon(self):
        print()
        print("--------------------------------")
        print("Testing multipolygon input...")
        # Skip test if the output file already exists
        # if os.path.exists(os.path.join(self.kv_output_test_multipolygon, "kv_pids.csv")):
        #     print('Result exits, skipping')
        #     self.skipTest("Result exists")
        # download images
        kv_downloader = KVDownloader(log_path=os.path.join(self.kv_output_test_multipolygon, "log.log"), max_workers=2000)
        kv_downloader.download_svi(self.kv_output_test_multipolygon, input_shp_file=self.kv_input_multipolygon)
        # assert True if there are files in the output directory (other than log.log)
        self.assertTrue(len(os.listdir(self.kv_output_test_multipolygon)) > 1)

    # test polygon
    def test_downloader_polygon(self):
        print()
        print("--------------------------------")
        print("Testing polygon input...")
        # Skip test if the output file already exists
        # if os.path.exists(os.path.join(self.kv_output_test_polygon, "kv_pids.csv")):
        #     print('Result exits, skipping')
        #     self.skipTest("Result exists")
        # download images
        kv_downloader = KVDownloader(log_path=os.path.join(self.kv_output_test_polygon, "log.log"), max_workers=2000)
        kv_downloader.download_svi(self.kv_output_test_polygon, input_shp_file=self.kv_input_polygon)
        # assert True if there are files in the output directory (other than log.log)
        self.assertTrue(len(os.listdir(self.kv_output_test_polygon)) > 1)


if __name__ == "__main__":
    unittest.main()
