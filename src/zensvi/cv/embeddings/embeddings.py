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
from torchvision import datasets
from collections import namedtuple
from sklearn.cluster import KMeans
import torchvision.models as models
from torch.autograd import Variable
from img2vec_pytorch import Img2Vec
from sklearn.decomposition import PCA
import torchvision.transforms as transforms
from concurrent.futures import ThreadPoolExecutor, as_completed
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToPILImage
import pyarrow.parquet as pq
import pyarrow as pa
import faiss




_Model = namedtuple('Model', ['name', 'layer', 'layer_output_size'])

models_dict = {
    'resnet-18': _Model('resnet18', 'avgpool', 512),
    'alexnet': _Model('alexnet', 'classifier', 4096),
    'vgg-11': _Model('vgg11', 'classifier', 4096),
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


class ImageDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        img = Image.open(image_path)
        image = img.resize((224, 224))
        if self.transform:
            image = self.transform(image)
        return str(image_path), image
    
    def collate_fn(self, data):
        image_paths, images = zip(*data)
        # Stack images to create a batch        
        images = torch.stack(images)
        return list(image_paths), images



# create a class for extracting embeddings
class Embeddings:
    def __init__(self,
                 model_name: str ='resnet-18',
                 cuda: bool =False,
                 tensor: bool = True, 
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
                           images_path: Union[List[str], str],
                           dir_embeddings_output: str,
                           batch_size: int = 100, 
                           maxWorkers: int = 8):
        
        if isinstance(images_path, str):
            image_paths = [os.path.join(images_path, image) for image in os.listdir(images_path)]
        else:
            image_paths = images_path

        if not os.path.exists(dir_embeddings_output):
            os.makedirs(dir_embeddings_output)

        batch_size = min(batch_size, len(image_paths))
        
        labels = [0] * len(image_paths)
        n_batches = (len(image_paths) + batch_size - 1) // batch_size
        print("Total number of images: ", len(image_paths))
        print("Number of batches: ", n_batches)

        img2vec = Img2Vec(cuda=self.cuda,model=self.model_name)

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((224, 224)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        dataset = ImageDataset(image_paths, transform=transform)  # Apply transformations if needed
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=dataset.collate_fn)
        to_pil = ToPILImage()


        def process_image(image):
            pil_image = to_pil(image)
            return pil_image

        with ThreadPoolExecutor(max_workers=maxWorkers, thread_name_prefix="emb") as executor:
            for i, (image_paths, images) in tqdm.tqdm(enumerate(dataloader), total=n_batches, desc='Progress', ncols=100, ):
                pil_images = list(executor.map(process_image, images))
                vec = img2vec.get_vec(pil_images, tensor=self.tensor)
                if isinstance(vec, torch.Tensor):
                    vec = vec.cpu().numpy()
                df = pd.DataFrame(vec)
                df.insert(0, 'file_key', [os.path.basename(image_path).split('.')[0] for image_path in image_paths])
                df.to_parquet(os.path.join(dir_embeddings_output, f'batch_{i}.parquet'), index=False)
    
    
    def search_similar_images(self, image_key:str, embeddings_dir: str, number_of_items: int = 10):
        embeddings_df = pq.read_table(embeddings_dir).to_pandas()
        embeddings_np_array = np.stack(embeddings_df[embeddings_df.columns[1:]].to_numpy())
        embeddings_layer_size = self.layer_output_size
        index = faiss.IndexFlatIP(embeddings_layer_size)
        index.add(embeddings_np_array)
        id_to_name = {k:v for k,v in enumerate(list(embeddings_df["file_key"]))}
        name_to_id = {v:k for k,v in id_to_name.items()}


        emb_df = embeddings_np_array[name_to_id[image_key]]
        D, I = index.search(np.expand_dims(emb_df, 0), number_of_items)     # actual search
        results = list(zip(D[0], [id_to_name[x] for x in I[0]]))
        results = [(i[0],i[1], i[1]+".png") for i in results]
        
        
        return results

        
        
    def get_all_models(self):
        return models_dict
                


if __name__ == '__main__':
    emb = Embeddings()



