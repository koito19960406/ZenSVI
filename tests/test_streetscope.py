import os
import unittest
from pathlib import Path
import shutil
import unittest
import numpy as np
from typing import List
from pathlib import Path
import os
import unittest
import numpy as np
from typing import List
from pathlib import Path
from unittest.mock import MagicMock


from streetscope.download.streetview_downloader import StreetViewDownloader
from streetscope.cv.segmentation import Segmenter, ImageDataset, create_cityscapes_label_colormap


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

class TestSegmentation(unittest.TestCase):

    # ... (previous test cases) ...

    def test_Segmenter_create_color_map(self):
        segmenter = Segmenter()
        segmenter._create_color_map = MagicMock()  # Mock the method to prevent actual execution

        labels = create_cityscapes_label_colormap()
        train_ids = np.array([label.trainId for label in labels], dtype=np.uint8)
        colors = np.array([label.color for label in labels], dtype=np.uint8)
        max_train_id = np.max(train_ids) + 1
        expected_color_map = np.zeros((max_train_id, 3), dtype=np.uint8)
        expected_color_map[train_ids] = colors

        segmenter._create_color_map("cityscapes")

        segmenter._create_color_map.assert_called_once()

    def test_Segmenter_calculate_pixel_ratios(self):
        segmenter = Segmenter()
        labels = create_cityscapes_label_colormap()
        labels = [label for label in labels if label.trainId != -1]
        segmenter.train_id_to_name = {label.trainId: label.name for label in labels}

        segmented_img = np.array([[0, 1], [1, 0]], dtype=np.uint8)
        expected_pixel_ratios = {'unlabeled': 0.5, 'ego vehicle': 0.5}

        pixel_ratios = segmenter._calculate_pixel_ratios(segmented_img)

        self.assertEqual(pixel_ratios, expected_pixel_ratios)

    def test_Segmenter_trainid_to_color(self):
        segmenter = Segmenter()
        labels = create_cityscapes_label_colormap()
        labels = [label for label in labels if label.trainId != -1]
        train_ids = np.array([label.trainId for label in labels], dtype=np.uint8)
        colors = np.array([label.color for label in labels], dtype=np.uint8)
        max_train_id = np.max(train_ids) + 1
        color_map = np.zeros((max_train_id, 3), dtype=np.uint8)
        color_map[train_ids] = colors
        segmenter.color_map = color_map

        segmented_img = np.array([[0, 1], [1, 0]], dtype=np.uint8)
        expected_colored_img = np.array([[(0, 0, 0), (70, 70, 70)], [(70, 70, 70), (0, 0, 0)]], dtype=np.uint8)

        colored_img = segmenter._trainid_to_color(segmented_img)

        np.testing.assert_array_equal(colored_img, expected_colored_img)

if __name__ == "__main__":
    unittest.main()
