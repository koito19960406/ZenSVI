import json
import unittest
import os
from pathlib import Path
import shutil

from zensvi.cv import Embeddings


class TestEmbeddings(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.image_input = Path("tests/data/input/images")
        self.output = Path("tests/data/output/embeddings")
        Path(self.output).mkdir(parents=True, exist_ok=True)
        pass

    def tearDown(self):
        # remove output directory
        shutil.rmtree(self.output, ignore_errors=True)

    def test_mapillary_panoptic(self):
        embedding = Embeddings(model_name="alexnet", cuda=True)
        image_output = self.output / "images"
        emb_output = self.output / "embeddings"
        
        embs = embedding.generate_embedding(
            self.image_input,
            image_output,
            batch_size=1000,
        )
        
        # assert True if files in image_output and summary_output are not empty
        self.assertTrue(
            embs
        )



if __name__ == "__main__":
    unittest.main()
