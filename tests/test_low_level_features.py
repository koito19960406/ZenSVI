import json
import unittest
import os 

from zensvi.cv import get_low_level_features

class TestLowLevel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        pass
        
    def test_edge_detection(self):
        image_input = "tests/data/input/images"
        image_output = "tests/data/output/low_level/edge"
        summary_output = "tests/data/output/low_level/edge_summary"
        get_low_level_features(image_input, dir_image_output = image_output, dir_summary_output = summary_output, save_format = ['json', 'csv'], csv_format = "wide")
    

if __name__ == '__main__':
    unittest.main()