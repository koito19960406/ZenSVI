import json
import unittest
import os
from pathlib import Path
import shutil
from collections import namedtuple

from zensvi.cv import Embeddings
from test_base import TestBase


class TestEmbeddings(TestBase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.image_input = cls.input_dir / "images"
        cls.output = cls.base_output_dir / "embeddings"
        cls.ensure_dir(cls.output)
        pass


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

    # test all models
    def test_all_models_cpu(self):
        _Model = namedtuple('Model', ['name', 'layer', 'layer_output_size'])
        models_dict = {
            'resnet-18': _Model('resnet18', 'avgpool', 512),
            'alexnet': _Model('alexnet', 'classifier', 4096),
            'vgg': _Model('vgg11', 'classifier', 4096),
            'densenet': _Model('densenet', 'classifier', 1024),
            'efficientnet_b0': _Model('efficientnet_b0', 'avgpool', 1280),
            'efficientnet_b1': _Model('efficientnet_b1', 'avgpool', 1280),
            'efficientnet_b2': _Model('efficientnet_b2', 'avgpool', 1408),
            'efficientnet_b3': _Model('efficientnet_b3', 'avgpool', 1536),
            'efficientnet_b4': _Model('efficientnet_b4', 'avgpool', 1792),
            'efficientnet_b5': _Model('efficientnet_b5', 'avgpool', 2048),
            'efficientnet_b6': _Model('efficientnet_b6', 'avgpool', 2304),
            'efficientnet_b7': _Model('efficientnet_b7', 'avgpool', 2560),
        }
        for model_name in models_dict.keys():
            print(f"Testing model: {model_name}")
            embedding = Embeddings(model_name=model_name, cuda=False)
            image_output = self.output / model_name
            emb_output = self.output / "embeddings"
            
            embs = embedding.generate_embedding(
                str(self.image_input),
                image_output,
                batch_size=10,
            )
            
            # assert True if files in image_output and summary_output are not empty
            self.assertTrue(
                embs
            )

if __name__ == "__main__":
    unittest.main()
