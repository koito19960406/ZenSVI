import os
import unittest
from pathlib import Path
import shutil
import unittest
import numpy as np
from unittest.mock import MagicMock
import time
import dotenv

from zensvi.download import StreetViewDownloader
from zensvi.cv import Segmenter, ImageDataset, create_cityscapes_label_colormap
from zensvi.transform import xyz2lonlat, lonlat2XY, ImageTransformer


class TestStreetViewDownloader(unittest.TestCase):

    def setUp(self):
        self.temp_dir = "temp_test_dir"
        Path(self.temp_dir).mkdir(parents=True, exist_ok=True)
        self.gsv_api_key = "YOUR_GSV_API_KEY"
        self.sv_downloader = StreetViewDownloader(self.temp_dir, gsv_api_key=self.gsv_api_key)

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_download_gsv(self):
        self.sv_downloader.download_gsv(lat=1.342425, lng=103.721523, augment_metadata=True)
        panorama_output = os.path.join(self.temp_dir, "panorama")
        self.assertTrue(os.path.exists(panorama_output))
        self.assertGreater(len(os.listdir(panorama_output)), 0)


class TestImageTransformerMethods(unittest.TestCase):
    def test_xyz2lonlat(self):
        xyz = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        expected = np.array([[np.pi / 2, 0.0], [0.0, np.pi / 2], [0.0, 0.0]])
        result = xyz2lonlat(xyz)
        np.testing.assert_almost_equal(result, expected, decimal=6)

    def test_lonlat2XY(self):
        lonlat = np.array([[0.0, 0.0], [0.0, np.pi / 4], [np.pi / 2, 0.0]])
        shape = (512, 1024, 3)
        expected = np.array([[511.5, 255.5], [511.5, 127.75], [767.25, 255.5]])
        result = lonlat2XY(lonlat, shape)
        np.testing.assert_almost_equal(result, expected, decimal=6)


    def test_ImageTransformer(self):
        dir_input = "test_input"
        dir_output = "test_output"
        transformer = ImageTransformer(dir_input, dir_output)

        self.assertIsInstance(transformer.dir_input, Path)
        self.assertIsInstance(transformer.dir_output, Path)
        self.assertEqual(transformer.dir_input, Path(dir_input))
        self.assertEqual(transformer.dir_output, Path(dir_output))

        with self.assertRaises(TypeError):
            transformer.dir_input = 42
        with self.assertRaises(TypeError):
            transformer.dir_output = 42


if __name__ == "__main__":
    # unittest.main()
    dotenv.load_dotenv(dotenv.find_dotenv())
    gsv_api_key = os.getenv('GSV_API_KEY')
    # # Code block 1
    # start_time = time.time()

    # downloader = StreetViewDownloaderAsync(gsv_api_key = gsv_api_key,
    #                                     grid = True, grid_size = 100)
    # downloader.download_gsv_async("/Users/koichiito/Desktop/test_async", 
    #                             input_shp_file = "/Users/koichiito/Downloads/Delft-subset/Delft-subset.shp",
    #                             augment_metadata=True) 

    # end_time = time.time()
    # print(f"Block 1 execution time: {end_time - start_time} seconds")

    # Code block 2
    start_time = time.time()

    downloader = StreetViewDownloader(gsv_api_key = gsv_api_key,
                                    distance=20,
                                    grid = False, grid_size = 20)
    downloader.download_gsv("tests/data/output/", 
                            # input_csv_file = "tests/data/input/count_station.csv",
                            input_shp_file = "tests/data/input/locations_point.shp",
                            # id_columns = "count_point_id",
                            buffer = 100,
                            augment_metadata=True) 
    segmenter = Segmenter()
    segmenter.segment("tests/data/output/panorama", 
                    # dir_image_output = "tests/data/output/segmentation", 
                    dir_pixel_ratio_output = "tests/data/output/",
                    batch_size=1)
    end_time = time.time()
    print(f"Block 2 execution time: {end_time - start_time} seconds")