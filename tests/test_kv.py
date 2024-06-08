import unittest
import os
import shutil
from pathlib import Path
import geopandas as gp

import zensvi.download.kartaview.download_functions as kv
from zensvi.download.kv import KVDownloader


class TestKartaView(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.kv_output = "tests/data/output/kv_output/"
        os.makedirs(self.kv_output, exist_ok=True)
        self.kv_input_multipolygon = "tests/data/input/test_multipolygon.geojson"
        self.kv_input_polygon = "tests/data/input/test_polygon.geojson"
        self.kv_output_csv = Path(self.kv_output) / "kv_output.csv"
        self.kv_svi_output = Path(self.kv_output) / "kv_svi"
        self.kv_svi_output_multipolygon = Path(self.kv_output) / "kv_svi_multipolygon"
        self.kv_svi_output_polygon = Path(self.kv_output) / "kv_svi_polygon"
        pass
    
    @classmethod   
    def tearDown(self):
        # remove output directory
        shutil.rmtree(self.kv_output, ignore_errors=True)

    def test_interface(self):
        # Skip test if the output file already exists
        if os.path.exists(self.kv_output_csv):
            self.skipTest("Result exists")
        # read geojson as dict
        gdf = gp.read_file(self.kv_input_polygon)
        output = kv.get_sequences_in_shape(gdf)
        output.to_csv(self.kv_output_csv)
        # assert True if output is not empty
        self.assertTrue(len(output.to_dict()) > 0)

    def test_downloader(self):
        # Skip test if the output file already exists
        if os.path.exists(os.path.join(self.kv_svi_output, "kv_svi")):
            self.skipTest("Result exists")
        # download images
        kv_downloader = KVDownloader()
        kv_downloader.download_svi(self.kv_svi_output, input_shp_file=self.kv_input_polygon)

        # assert True if there are files in the output directory
        self.assertTrue(len(os.listdir(self.kv_svi_output)) > 0)

    def test_downloader_metadata_only(self):
        # Skip test if the output file already exists
        if os.path.exists(os.path.join(self.kv_svi_output, "kv_pids.csv")):
            self.skipTest("Result exists")
        # download metadata only
        kv_downloader = KVDownloader(
            log_path="tests/data/output/kv_svi/log.log",
            max_workers=200,
        )
        # kv_downloader.download_svi(self.kv_svi_output, input_place_name="Singapore", metadata_only=True)
        kv_downloader.download_svi(
            self.kv_svi_output, input_shp_file=self.kv_input_polygon, metadata_only=True
        )
        # assert True if kv_pids.csv is not empty
        self.assertTrue(
            os.path.getsize(os.path.join(self.kv_svi_output, "kv_pids.csv")) > 0
        )

    # test multipolygon
    def test_downloader_multipolygon(self):
        # # Skip test if the output file already exists
        # if os.path.exists(os.path.join(self.kv_svi_output_multipolygon, "kv_svi")):
        #     self.skipTest("Result exists")
        # download images
        kv_downloader = KVDownloader(max_workers=200)
        kv_downloader.download_svi(
            self.kv_svi_output_multipolygon, input_shp_file=self.kv_input_multipolygon
        )
        # assert True if there are files in the output directory
        self.assertTrue(len(os.listdir(self.kv_svi_output_multipolygon)) > 0)

    # test polygon
    def test_downloader_polygon(self):
        # # Skip test if the output file already exists
        # if os.path.exists(os.path.join(self.kv_svi_output_polygon, "kv_svi")):
        #     self.skipTest("Result exists")
        # download images
        kv_downloader = KVDownloader(max_workers=200)
        kv_downloader.download_svi(
            self.kv_svi_output_polygon, input_shp_file=self.kv_input_polygon
        )
        # assert True if there are files in the output directory
        self.assertTrue(len(os.listdir(self.kv_svi_output_polygon)) > 0)


if __name__ == "__main__":
    unittest.main()
