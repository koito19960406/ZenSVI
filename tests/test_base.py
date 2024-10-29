# import os
# import shutil
import unittest
from pathlib import Path


class TestBase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.input_dir = Path("tests/data/input")
        cls.base_output_dir = Path("tests/data/output")
        cls.ensure_dir(cls.base_output_dir)

    # @classmethod
    # def tearDownClass(cls):
    #     if cls.base_output_dir.exists():
    #         shutil.rmtree(cls.base_output_dir)

    @staticmethod
    def ensure_dir(directory):
        directory.mkdir(parents=True, exist_ok=True)

    def setUp(self):
        self.output_dir = self.base_output_dir
        self.ensure_dir(self.output_dir)

    # def tearDown(self):
    #     if self.output_dir.exists():
    #         shutil.rmtree(self.output_dir)
