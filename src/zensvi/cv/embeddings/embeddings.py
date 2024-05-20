import os
import tqdm
import torch
import numpy as np
import pandas as pd
from PIL import Image
import torch.nn as nn
from shutil import copyfile
from typing import List, Union
import matplotlib.pyplot as plt
from collections import namedtuple
from sklearn.cluster import KMeans
import torchvision.models as models
from torch.autograd import Variable
from img2vec_pytorch import Img2Vec
from sklearn.decomposition import PCA
import torchvision.transforms as transforms


_Model = namedtuple('Model', ['name', 'layer', 'layer_output_size'])

models_dict = {
    'resnet-18': _Model('resnet18', 'avgpool', 512),
    'alexnet': _Model('alexnet', 'classifier', 4096),
    'vgg-11': _Model('vgg11', 'classifier', 4096),
    'densenet': _Model('densenet', 'classifier', 1024),
    'efficientnet_b0': _Model('efficientnet_b0', '_avg_pooling', 1280),
    'efficientnet_b1': _Model('efficientnet_b1', '_avg_pooling', 1280),
    'efficientnet_b2': _Model('efficientnet_b2', '_avg_pooling', 1408),
    'efficientnet_b3': _Model('efficientnet_b3', '_avg_pooling', 1536),
    'efficientnet_b4': _Model('efficientnet_b4', '_avg_pooling', 1792),
    'efficientnet_b5': _Model('efficientnet_b5', '_avg_pooling', 2048),
    'efficientnet_b6': _Model('efficientnet_b6', '_avg_pooling', 2304),
    'efficientnet_b7': _Model('efficientnet_b7', '_avg_pooling', 2560),
}

# Embedding vector class for embedding vector operations. 
class EmbeddingVector:
    def __init__(self, vector):
        """
        :param vector: embedding vector is a multidimensional numpy array 
        """
        self.vector = vector

    def __add__(self, other):
        return EmbeddingVector(self.vector + other.vector)
    
    def __sub__(self, other):
        return EmbeddingVector(self.vector - other.vector)
    
    def distance_to(self, other):
        return np.linalg.norm(self.vector - other.vector)
    
    def get_dimension(self):
        return self.vector.shape[0]
    
    def __str__(self):
        return str(self.vector)
    
    def cosine_similarity(self, other):
        return np.dot(self.vector, other.vector) / (np.linalg.norm(self.vector) * np.linalg.norm(other.vector))



# create a class for extracting embeddings
class Embeddings:
    def __init__(self,
                 model_name: str ='resnet-18',
                 cuda: bool =False,
                 tensor: bool = True
                 ):
        """
        :param model_name: name of the model to be used for extracting embeddings (default: 'resnet-18') 
            Other available models: 'alexnet', 'vgg-11', 'densenet', 'efficientnet_b0', 'efficientnet_b1', 
            'efficientnet_b2', 'efficientnet_b3', 'efficientnet_b4', 'efficientnet_b5', 'efficientnet_b6', 'efficientnet_b7'
        :param cuda: whether to use cuda or not
        """
        self.model_name = model_name
        self.layer = models_dict[model_name].layer
        self.layer_output_size = models_dict[model_name].layer_output_size
        self.model, self.extraction_layer = self.get_model_and_layer()
        self.model.eval()
        self.cuda = cuda
        self.tensor = tensor

    def load_image(self, image_path):
        """
        :param image_path: path to the image
        :return: image
        """
        img = Image.open(image_path)
        img = img.resize((224, 224))
        return img

    def get_model_and_layer(self):
        """
        :return: model and layer
        """
        model = models.__dict__[models_dict[self.model_name].name](pretrained=True)
        layer = getattr(model, self.layer)
        return model, layer
    

    def get_image_embedding(self, 
                            image_path: Union[List[str], str], 
                            tensor: bool = None, 
                            cuda: bool = None):
        """
        :param image_path: path to the image
        :return: image embedding
        """
        if not tensor:
            tensor = self.tensor
        if not cuda:
            cuda = self.cuda
            
        img2vec = Img2Vec(cuda=cuda)

        img = self.load_image(image_path)
        return img2vec.get_vec(img)
        
    

    def generate_embedding(self, 
                           dir_images:List[str],
                           dir_embeddings_output: str,
                           batch_size: int = 100):
        """
        :param dir_images: directory containing the images to extract embeddings from
        :param dir_embeddings_output: directory to save the embeddings
        :param batch_size: batch size for extracting embeddings (default: 100)
        """

        for i in tqdm.tqdm(range(len(dir_images)//batch_size)):
            images = [self.load_image(image_path) for image_path in dir_images[i*batch_size:(i+1)*batch_size]]
            img2vec = Img2Vec(cuda=self.cuda)
            fvectors = img2vec.get_vec(images)
            np.save(dir_embeddings_output + f'embeddings_{i}.npy', fvectors)
            print(f'Batch {i+1} done!')
    
    def cosine_similarity(self, emb1, emb2):
        """
        :param emb1: embedding 1
        :param emb2: embedding 2
        :return: cosine similarity between the two embeddings
        """
        # make sure that emb1 and emb2 are tensors of the same shape:
        emb1 = torch.tensor(emb1)
        emb2 = torch.tensor(emb2)

        cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        cos_sim = cos(emb1.reshape(1, -1),
              emb2.reshape(1, -1))[0]

        print('\nCosine similarity: {0}\n'.format(cos_sim))
        return cos_sim

    def cluster(self, 
                input_path,
                vec_length: int = 512,
                k_value: int = 2,
                ):
        """
        :param dir_embeddings_output: directory containing the embeddings
        :param dir_summary_output: directory to save the summary of the clustering
        :param batch_size: batch size for clustering (default: 100)
        """
        files = os.listdir(input_path)
        img2vec = Img2Vec(self.model, cuda=self.cuda)
        samples = len(files)
        vec_mat = np.zeros((samples, vec_length))
        sample_indices = np.random.choice(range(0, len(files)), size=samples, replace=False)

        print('Reading images...')
        for index, i in enumerate(sample_indices):
            file = files[i]
            filename = os.fsdecode(file)
            img = Image.open(os.path.join(input_path, filename)).convert('RGB')
            vec = img2vec.get_vec(img)
            vec_mat[index, :] = vec

        print('Applying PCA...')
        reduced_data = PCA(n_components=2).fit_transform(vec_mat)
        kmeans = KMeans(init='k-means++', n_clusters=k_value, n_init=10)
        kmeans.fit(reduced_data)

        # Create a folder for each cluster (0, 1, 2, ..)
        for i in set(kmeans.labels_):
            try:
                os.mkdir('./' + str(i))
            except FileExistsError:
                continue

        print('Predicting...')
        preds = kmeans.predict(reduced_data)

        print('Copying images...')
        for index, i in enumerate(sample_indices):
            file = files[i]
            filename = os.fsdecode(file)
            copyfile(input_path + '/' + filename, './' + str(preds[index]) + '/' + filename)

        print('Done!')



if __name__ == '__main__':
    emb = Embeddings()



