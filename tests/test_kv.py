import unittest
import os
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
        self.kv_output_test_interface = Path(self.kv_output) / "test_interface.csv"
        self.kv_output_test_downloader = Path(self.kv_output) / "test_downloader"
        self.kv_output_test_polygon_metaonly = Path(self.kv_output) / "test_polygon_metaonly"
        self.kv_output_test_multipolygon = Path(self.kv_output) / "test_multipolygon"
        self.kv_output_test_polygon = Path(self.kv_output) / "test_polygon"
        pass
    
    @classmethod   
    # def tearDown(self):
    #     # remove output directory
    #     shutil.rmtree(self.kv_output, ignore_errors=True)

    def test_interface(self):
        print()
        print('--------------------------------')
        print('Testing interface...')
        # Skip test if the output file already exists
        if os.path.exists(self.kv_output_test_interface):
            print('Result exits, skipping')
            self.skipTest("Result exists")
        # read geojson as dict
        gdf = gp.read_file(self.kv_input_polygon)
        output = kv.get_points_in_shape(gdf)
        output.to_csv(self.kv_output_test_interface)
        # assert True if output is not empty
        self.assertTrue(len(output) > 0)

    def test_downloader(self):
        print()
        print('--------------------------------')
        print('Testing downloader...')
        # Skip test if the output file already exists
        if os.path.exists(os.path.join(self.kv_output_test_downloader, "kv_pids.csv")):
            print('Result exits, skipping')
            self.skipTest("Result exists")
        # download images
        kv_downloader = KVDownloader(
            log_path=os.path.join(self.kv_output_test_downloader, "log.log")
        )
        kv_downloader.download_svi(self.kv_output_test_downloader, input_shp_file=self.kv_input_polygon)

        # assert True if there are files in the output directory (other than log.log)
        self.assertTrue(len(os.listdir(self.kv_output_test_downloader)) > 1)

    def test_downloader_metadata_only(self):
        print()
        print('--------------------------------')
        print('Testing polygon input (metadata only)...')
        # Skip test if the output file already exists
        if os.path.exists(os.path.join(self.kv_output_test_polygon_metaonly, "kv_pids.csv")):
            print('Result exits, skipping')
            self.skipTest("Result exists")
        # download metadata only
        kv_downloader = KVDownloader(
            log_path=os.path.join(self.kv_output_test_polygon_metaonly, "log.log")
        )
        # kv_downloader.download_svi(self.kv_output_test_polygon_metaonly, input_place_name="Singapore", metadata_only=True)
        kv_downloader.download_svi(
            self.kv_output_test_polygon_metaonly, input_shp_file=self.kv_input_polygon, metadata_only=True
        )
        # assert True if kv_pids.csv is not empty
        self.assertTrue(
            os.path.getsize(os.path.join(self.kv_output_test_polygon_metaonly, "kv_pids.csv")) > 0
        )

    # test multipolygon
    def test_downloader_multipolygon(self):
        print()
        print('--------------------------------')
        print('Testing multipolygon input...')
        # Skip test if the output file already exists
        if os.path.exists(os.path.join(self.kv_output_test_multipolygon, "kv_pids.csv")):
            print('Result exits, skipping')
            self.skipTest("Result exists")
        # download images
        kv_downloader = KVDownloader(
            log_path=os.path.join(self.kv_output_test_multipolygon, "log.log")
        )
        kv_downloader.download_svi(
            self.kv_output_test_multipolygon, input_shp_file=self.kv_input_multipolygon
        )
        # assert True if there are files in the output directory (other than log.log)
        self.assertTrue(len(os.listdir(self.kv_output_test_multipolygon)) > 1)

    # test polygon
    def test_downloader_polygon(self):
        print()
        print('--------------------------------')
        print('Testing polygon input...')
        # Skip test if the output file already exists
        if os.path.exists(os.path.join(self.kv_output_test_polygon, "kv_pids.csv")):
            print('Result exits, skipping')
            self.skipTest("Result exists")
        # download images
        kv_downloader = KVDownloader(
            log_path=os.path.join(self.kv_output_test_polygon, "log.log")
        )
        kv_downloader.download_svi(
            self.kv_output_test_polygon, input_shp_file=self.kv_input_polygon
        )
        # assert True if there are files in the output directory (other than log.log)
        self.assertTrue(len(os.listdir(self.kv_output_test_polygon)) > 1)


if __name__ == "__main__":
    unittest.main()
