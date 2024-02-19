import json
import unittest
import os 

from zensvi.cv import Segmenter

class TestSegmentation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        pass
        
    def test_mapillary_panoptic(self):
        segmenter = Segmenter(dataset="mapillary", task="panoptic")
        image_input = "tests/data/input/images"
        image_output = "tests/data/output/segmentation/mapillary_panoptic"
        summary_output = "tests/data/output/segmentation/mapillary_panoptic_summary"
        segmenter.segment(image_input, dir_image_output = image_output, dir_segmentation_summary_output = summary_output, csv_format = "wide", max_workers = 4)
    
    def test_mapillary_semantic(self):
        segmenter = Segmenter(dataset="mapillary", task="semantic")
        image_input = "tests/data/input/images"
        image_output = "tests/data/output/segmentation/mapillary_semantic"
        summary_output = "tests/data/output/segmentation/mapillary_semantic_summary"
        segmenter.segment(image_input, dir_image_output = image_output, dir_segmentation_summary_output = summary_output, csv_format = "wide", max_workers = 4)
    
    def test_cityscapes_panoptic(self):
        segmenter = Segmenter(dataset="cityscapes", task="panoptic")
        image_input = "tests/data/input/images"
        image_output = "tests/data/output/segmentation/cityscapes_panoptic"
        summary_output = "tests/data/output/segmentation/cityscapes_panoptic_summary"
        segmenter.segment(image_input, dir_image_output = image_output, dir_segmentation_summary_output = summary_output, csv_format = "wide", max_workers = 4)
    
    def test_cityscapes_semantic(self):
        segmenter = Segmenter(dataset="cityscapes", task="semantic")
        image_input = "tests/data/input/images"
        image_output = "tests/data/output/segmentation/cityscapes_semantic"
        summary_output = "tests/data/output/segmentation/cityscapes_semantic_summary"
        segmenter.segment(image_input, dir_image_output = image_output, dir_segmentation_summary_output = summary_output, csv_format = "wide", max_workers = 4)

    def test_large_image(self):
        segmenter = Segmenter(dataset="mapillary", task="panoptic")
        image_input = "tests/data/input/large_images"
        image_output = "tests/data/output/segmentation/large_image"
        summary_output = "tests/data/output/segmentation/large_image_summary"
        segmenter.segment(image_input, dir_image_output = image_output, dir_segmentation_summary_output = summary_output, csv_format = "wide", max_workers = 1)
    
    def test_single_image(self):
        segmenter = Segmenter(dataset="mapillary", task="panoptic")
        image_input = "tests/data/input/images/-3vfS0_iiYVZKh_LEVlHew.jpg"
        image_output = "tests/data/output/segmentation/single_image"
        summary_output = "tests/data/output/segmentation/single_image_summary"
        segmenter.segment(image_input, dir_image_output = image_output, dir_segmentation_summary_output = summary_output, csv_format = "wide", max_workers = 1)
    
    def test_calculate_pixel_ratio_post_process(self):
        segmenter = Segmenter(dataset="mapillary", task="panoptic")
        image_input = "tests/data/input/cityscapes_semantic"
        image_output = "tests/data/output/segmentation/calculate_pixel_ratio_post_process"
        segmenter.calculate_pixel_ratio_post_process(image_input, image_output)
    
    def test_calculate_pixel_ratio_post_process_single_file(self):
        segmenter = Segmenter(dataset="mapillary", task="panoptic")
        image_input = "tests/data/input/cityscapes_semantic/-3vfS0_iiYVZKh_LEVlHew_colored_segmented.png"
        image_output = "tests/data/output/segmentation/calculate_pixel_ratio_post_process_single_file"
        segmenter.calculate_pixel_ratio_post_process(image_input, image_output)
        
if __name__ == "__main__":
    unittest.main()