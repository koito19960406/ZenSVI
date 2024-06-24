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

    def test_embeddings(self):
        embedding = Embeddings(model_name="alexnet", cuda=True)
        image_output = self.output / "images"
        emb_output = self.output / "embeddings"
        
        embs = embedding.generate_embedding(
           str(self.image_input),
            image_output,
            batch_size=1000,
        )
        
        # assert True if files in image_output and summary_output are not empty
        self.assertTrue(
            embs
        )

    def test_search_similar_images(self):
        embedding = Embeddings(model_name="alexnet", cuda=True)
        image_output = self.output / "images"
        emb_output = self.output / "embeddings"
        
        embs = embedding.generate_embedding(
            str(self.image_input),
            image_output,
            batch_size=1000,
        )
        
        # assert True if files in image_output and summary_output are not empty
        self.assertTrue(
            embs
        )

        # base file name of the first file in the image_input directory
        image_path = list(self.image_input.glob("*"))[0]
        image_base_name = image_path.stem
        
        similar_images = embedding.search_similar_images(
            image_base_name,
            image_output,
            5,
        )
        # assert True if files in image_output and summary_output are not empty
        self.assertTrue(
            similar_images
        )


if __name__ == "__main__":
    unittest.main()
