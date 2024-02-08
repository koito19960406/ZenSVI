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
        segmenter.segment(image_input, dir_image_output = image_output, dir_segmentation_summary_output = summary_output, max_workers = 4)
    
    def test_mapillary_semantic(self):
        segmenter = Segmenter(dataset="mapillary", task="semantic")
        image_input = "tests/data/input/images"
        image_output = "tests/data/output/segmentation/mapillary_semantic"
        summary_output = "tests/data/output/segmentation/mapillary_semantic_summary"
        segmenter.segment(image_input, dir_image_output = image_output, dir_segmentation_summary_output = summary_output, max_workers = 4)
    
    def test_cityscapes_panoptic(self):
        segmenter = Segmenter(dataset="cityscapes", task="panoptic")
        image_input = "tests/data/input/images"
        image_output = "tests/data/output/segmentation/cityscapes_panoptic"
        summary_output = "tests/data/output/segmentation/cityscapes_panoptic_summary"
        segmenter.segment(image_input, dir_image_output = image_output, dir_segmentation_summary_output = summary_output, max_workers = 4)
    
    def test_cityscapes_semantic(self):
        segmenter = Segmenter(dataset="cityscapes", task="semantic")
        image_input = "tests/data/input/images"
        image_output = "tests/data/output/segmentation/cityscapes_semantic"
        summary_output = "tests/data/output/segmentation/cityscapes_semantic_summary"
        segmenter.segment(image_input, dir_image_output = image_output, dir_segmentation_summary_output = summary_output, max_workers = 4)

    def test_large_image(self):
        segmenter = Segmenter(dataset="mapillary", task="panoptic")
        image_input = "tests/data/input/large_images"
        image_output = "tests/data/output/segmentation/large_image"
        summary_output = "tests/data/output/segmentation/large_image_summary"
        segmenter.segment(image_input, dir_image_output = image_output, dir_segmentation_summary_output = summary_output, max_workers = 1)
        
if __name__ == "__main__":
    unittest.main()