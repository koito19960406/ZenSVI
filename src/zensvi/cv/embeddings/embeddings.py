# Description: This file contains the code to extract embeddings from a given image using a pre-trained model.
# Author : Mahmoud A. (arch.mahmoud.ouf111@gmail.com)
# This module is based on img2vec_pytorch

import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.autograd import Variable
from typing import List
from PIL import Image
from img2vec_pytorch import Img2Vec
import numpy as np
from collections import namedtuple
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import tqdm
import pandas as pd

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
                 cuda: bool =False
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

    def load_image(self, image_path):
        """
        :param image_path: path to the image
        :return: image
        """
        img = Image.open(image_path)
        img = img.resize((224, 224))
        img = transforms.ToTensor()(img)
        img = img.unsqueeze(0)
        print(img.shape)
        return img

    def get_model_and_layer(self):
        """
        :return: model and layer
        """
        model = models.__dict__[models_dict[self.model_name].name](pretrained=True)
        layer = getattr(model, self.layer)
        return model, layer
    

    def get_image_embedding(self, image_path):
        """
        :param image_path: path to the image
        :return: image embedding
        """
        img = self.load_image(image_path)
        if self.cuda:
            img = Variable(img).cuda()
        else:
            img = Variable(img)
        result = self.model(img)
        return result.detach().numpy()
    

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
        cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        cos_sim = cos(emb1.unsqueeze(0),
                    emb2.unsqueeze(0))
        print('\nCosine similarity: {0}\n'.format(cos_sim))
        return cos_sim

    def cluster(self, 
                dir_embeddings_output: List[str],
                dir_summary_output: str,
                num_clusters: int =100,
                batch_size: int =100):
        """
        :param dir_embeddings_output: directory containing the embeddings
        :param dir_summary_output: directory to save the summary of the clustering
        :param batch_size: batch size for clustering (default: 100)
        """
        embeddings = []
        for i in tqdm.tqdm(range(len(dir_embeddings_output))):
            embeddings.append(np.load(dir_embeddings_output + f'embeddings_{i}.npy'))
        embeddings = np.concatenate(embeddings, axis=0)
        kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(embeddings)
        np.save(dir_summary_output + 'labels.npy', kmeans.labels_)
        np.save(dir_summary_output + 'centers.npy', kmeans.cluster_centers_)
        print('Clustering done!')


    def search_similar(self, 
                        image_path : str,
                        dir_images: List[str], 
                        dir_embeddings_output : str, 
                        dir_summary_output: str, 
                        num_similar: int =5
                        ):
        """
        :param image_path: path to the image to search for similar images
        :param dir_images: directory containing the images
        :param dir_embeddings_output: directory containing the embeddings
        :param dir_summary_output: directory containing the summary of the clustering
        :param num_similar: number of similar images to return (default: 5)
        :return: similar images
        """
        # img = self.load_image(image_path)
        # img2vec = Img2Vec(cuda=self.cuda)
        # fvector = img2vec.get_vec([img])
        # labels = np.load(dir_summary_output + 'labels.npy')
        # centers = np.load(dir_summary_output + 'centers.npy')
        # kmeans = KMeans(n_clusters=centers.shape[0], init=centers, n_init=1)
        # kmeans.fit(fvector)
        # similar_images = []
        # for i in range(len(labels)):
        #     if labels[i] == kmeans.labels_[0]:
        #         similar_images.append(dir_images[i])
        # return similar_images[:num_similar]

if __name__ == '__main__':
    emb = Embeddings()



