import os
import unittest
from pathlib import Path
import shutil
import unittest
import numpy as np
from unittest.mock import MagicMock
import time
import dotenv

from zensvi.download import GSVDownloader, MLYDownloader
from zensvi.cv import Segmenter, ImageDataset, create_cityscapes_label_colormap
from zensvi.transform import xyz2lonlat, lonlat2XY, ImageTransformer

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
    mly_api_key = os.getenv('MLY_API_KEY')
    org_id = os.getenv('ORGANIZATION_ID')
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

    downloader = GSVDownloader(gsv_api_key = gsv_api_key,
                                    distance=20,
                                    grid = False, grid_size = 20)
    downloader.download_svi("tests/data/output",
                            # orchard road in singapore
                            lat = 1.286926, lon = 103.77931,
                            # input_csv_file = "tests/data/input/count_station_clean.csv",
                            # input_shp_file = "tests/data/input/c1_4326.shp",
                            # input_place_name="Bronkhorst, Netherlands",
                            # id_columns = "count_point_id",
                            # buffer = 50,
                            # network_type = "walk",
                            augment_metadata=False#,
                            # start_date="2021-01-01",
                            # end_date="2021-01-01"
                            ) 
    # downloader = MLYDownloader(mly_api_key=mly_api_key, max_workers = 4)
    # downloader.download_svi(dir_output = "tests/data/output", 
    #                         lat = 1.286926, lon = 103.77931#,
    #                         # input_csv_file="tests/data/input/Walking_count_sites 3.csv",
    #                         # input_shp_file="tests/data/input/pasir_panjang_rd_filtered.geojson",
    #                         # organization_id = [5498631900212193],
    #                         # input_place_name="Bronkhorst, Netherlands",
    #                         # network_type = "all_private",
    #                         # radius=100
    #                         # cropped=True,
    #                         # start_date="2023-01-01",
    #                         # end_date="2023-01-08"
    #                         )
    segmenter = Segmenter(dataset = "cityscapes", task="semantic")
    segmenter.segment(dir_input = "tests/data/output/gsv_panorama/batch_1",
                    dir_image_output = "tests/data/output/segmented", 
                    dir_segmentation_summary_output= "tests/data/output",
                    max_workers=1)
    
    transformer = ImageTransformer(dir_input = "tests/data/output/gsv_panorama/batch_1",
                                    dir_output = "tests/data/output/transformed")
    transformer.transform_images(style_list=["perspective", "equidistant_fisheye"])
    
    end_time = time.time()
    print(f"Block 2 execution time: {end_time - start_time} seconds")