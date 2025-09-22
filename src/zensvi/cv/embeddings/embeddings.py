import os
from collections import namedtuple
from concurrent.futures import ThreadPoolExecutor
from typing import List, Union

import faiss
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from img2vec_pytorch import Img2Vec
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import ToPILImage

from zensvi.utils.log import verbosity_tqdm

_Model = namedtuple("Model", ["name", "layer", "layer_output_size"])

_models_dict = {
    "resnet18": _Model("resnet18", "avgpool", 512),
    "alexnet": _Model("alexnet", "classifier", 4096),
    "vgg": _Model("vgg11", "classifier", 4096),
    "densenet": _Model("densenet121", "classifier", 1024),
    "efficientnet_b0": _Model("efficientnet_b0", "avgpool", 1280),
    "efficientnet_b1": _Model("efficientnet_b1", "avgpool", 1280),
    "efficientnet_b2": _Model("efficientnet_b2", "avgpool", 1408),
    "efficientnet_b3": _Model("efficientnet_b3", "avgpool", 1536),
    "efficientnet_b4": _Model("efficientnet_b4", "avgpool", 1792),
    "efficientnet_b5": _Model("efficientnet_b5", "avgpool", 2048),
    "efficientnet_b6": _Model("efficientnet_b6", "avgpool", 2304),
    "efficientnet_b7": _Model("efficientnet_b7", "avgpool", 2560),
}


class ImageDataset(Dataset):
    """A PyTorch Dataset for loading images.

    Args:
        image_paths: List of paths to image files.
        transform: Optional transform to be applied on the images.
    """

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
        """Custom collate function for batching data.

        Args:
            data: List of tuples containing (image_path, image).

        Returns:
            tuple: Contains:
                - list of image paths
                - tensor of stacked images
        """
        image_paths, images = zip(*data)
        # Stack images to create a batch
        images = torch.stack(images)
        return list(image_paths), images


# create a class for extracting embeddings
class Embeddings:
    """A class for extracting image embeddings using pre-trained models."""

    def __init__(
        self,
        model_name: str = "resnet18",
        cuda: bool = False,
        tensor: bool = True,
        verbosity: int = 1,
    ):
        """Initialize the Embeddings class for extracting image embeddings.

        This class uses pre-trained models from the Img2Vec package
        (https://github.com/christiansafka/img2vec) to extract feature vectors from images.
        These embeddings can be used for various downstream tasks such as image similarity,
        clustering, or as input to other machine learning models.

        The available models include popular architectures like ResNet, AlexNet, VGG,
        DenseNet, and various EfficientNet variants. Each model is configured to extract
        features from a specific layer, providing embeddings of different sizes
        depending on the chosen model.

        Args:
            model_name: Name of the model to be used for extracting embeddings.
                Default is 'resnet18'. Other options include 'alexnet', 'vgg',
                'densenet', and 'efficientnet_b0' through 'efficientnet_b7'.
            cuda: Whether to use CUDA for GPU acceleration. Default is False.
            tensor: Whether to return the embedding as a PyTorch tensor.
                If False, returns a numpy array. Default is True.
            verbosity: Verbosity level for tqdm progress bar. Default is 1.
        """
        self.model_name = model_name
        self.layer = _models_dict[model_name].layer
        self.layer_output_size = _models_dict[model_name].layer_output_size
        self.model, self.extraction_layer = self.get_model_and_layer()
        self.model.eval()
        self.cuda = cuda
        self.tensor = tensor
        self.verbosity = verbosity

    def load_image(self, image_path):
        """Load and preprocess an image from a file path.

        Args:
            image_path: Path to the image file.

        Returns:
            PIL.Image: Loaded and resized image.
        """
        img = Image.open(image_path)
        img = img.resize((224, 224))
        return img

    def get_model_and_layer(self):
        """Get the pre-trained model and extraction layer.

        Returns:
            tuple: Contains:
                - torch.nn.Module: The pre-trained model
                - torch.nn.Module: The extraction layer
        """
        model = models.__dict__[_models_dict[self.model_name].name](pretrained=True)
        layer = getattr(model, self.layer)
        return model, layer

    def get_image_embedding(self, image_path: Union[List[str], str], tensor: bool = None, cuda: bool = None):
        """Extract embedding for a single image.

        Args:
            image_path: Path to the image file.
            tensor: Whether to return the embedding as a PyTorch tensor.
                If None, uses the instance default.
            cuda: Whether to use CUDA for computation.
                If None, uses the instance default.

        Returns:
            Union[torch.Tensor, numpy.ndarray]: The image embedding.
        """
        if not tensor:
            tensor = self.tensor
        if not cuda:
            cuda = self.cuda

        img2vec = Img2Vec(cuda=cuda)

        img = self.load_image(image_path)
        return img2vec.get_vec(img)

    def generate_embedding(
        self,
        images_path: Union[List[str], str],
        dir_embeddings_output: str,
        batch_size: int = 100,
        maxWorkers: int = 8,
        verbosity: int = None,
    ):
        """Generate and save embeddings for multiple images.

        Args:
            images_path: Either a directory path containing images or a list of image paths.
            dir_embeddings_output: Directory where embeddings will be saved as parquet files.
            batch_size: Number of images to process in each batch. Default is 100.
            maxWorkers: Maximum number of worker threads for image processing. Default is 8.
            verbosity: Level of verbosity for progress bars.
                      0 = no progress bars, 1 = outer loops only, 2 = all loops.
                      If None, uses the instance's verbosity level.

        Returns:
            bool: True if embeddings were successfully generated and saved.
        """
        # Use instance verbosity if not specified
        if verbosity is None:
            verbosity = self.verbosity

        if isinstance(images_path, str):
            valid_extensions = [
                ".jpg",
                ".jpeg",
                ".png",
                ".tif",
                ".tiff",
                ".bmp",
                ".dib",
                ".pbm",
                ".pgm",
                ".ppm",
                ".sr",
                ".ras",
                ".exr",
                ".jp2",
            ]
            image_paths = [
                os.path.join(images_path, image)
                for image in os.listdir(images_path)
                if os.path.splitext(image)[1].lower() in valid_extensions
            ]
        else:
            image_paths = images_path

        if not os.path.exists(dir_embeddings_output):
            os.makedirs(dir_embeddings_output)

        batch_size = min(batch_size, len(image_paths))

        n_batches = (len(image_paths) + batch_size - 1) // batch_size
        print("Total number of images: ", len(image_paths))
        print("Number of batches: ", n_batches)

        img2vec = Img2Vec(cuda=self.cuda, model=self.model_name)

        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize((224, 224)),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

        dataset = ImageDataset(image_paths, transform=transform)  # Apply transformations if needed
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=dataset.collate_fn)
        to_pil = ToPILImage()

        def process_image(image):
            """Convert a tensor image to PIL Image.

            Args:
                image: Input tensor image.

            Returns:
                PIL.Image: Converted image.
            """
            pil_image = to_pil(image)
            return pil_image

        with ThreadPoolExecutor(max_workers=maxWorkers, thread_name_prefix="emb") as executor:
            for i, (image_paths, images) in verbosity_tqdm(
                enumerate(dataloader), total=n_batches, desc="Generating embeddings", verbosity=verbosity, level=1
            ):
                pil_images = list(executor.map(process_image, images))
                vec = img2vec.get_vec(pil_images, tensor=self.tensor)
                if isinstance(vec, torch.Tensor):
                    vec = vec.cpu().numpy()
                if vec.ndim > 2:
                    vec = vec.reshape(vec.shape[0], -1)
                df = pd.DataFrame(vec)
                df.insert(
                    0,
                    "filename_key",
                    [os.path.basename(image_path).split(".")[0] for image_path in image_paths],
                )
                # convert all the column names to string
                df.columns = [str(col) for col in df.columns]
                df.to_parquet(
                    os.path.join(dir_embeddings_output, f"batch_{i}.parquet"),
                    index=False,
                )
        return True

    def search_similar_images(self, image_key: str, embeddings_dir: str, number_of_items: int = 10):
        """Search for similar images using embeddings.

        Args:
            image_key: Key of the query image.
            embeddings_dir: Directory containing the embeddings parquet file.
            number_of_items: Number of similar images to return. Default is 10.

        Returns:
            list: List of tuples containing (similarity_score, image_key, image_filename).
        """
        embeddings_df = pq.read_table(embeddings_dir).to_pandas()
        embeddings_np_array = np.stack(embeddings_df[embeddings_df.columns[1:]].to_numpy())
        embeddings_layer_size = self.layer_output_size
        index = faiss.IndexFlatIP(embeddings_layer_size)
        index.add(embeddings_np_array)
        id_to_name = {k: v for k, v in enumerate(list(embeddings_df["filename_key"]))}
        name_to_id = {v: k for k, v in id_to_name.items()}

        emb_df = embeddings_np_array[name_to_id[image_key]]
        D, Index = index.search(np.expand_dims(emb_df, 0), number_of_items)  # actual search
        results = list(zip(D[0], [id_to_name[x] for x in Index[0]]))
        results = [(i[0], i[1], i[1] + ".png") for i in results]

        return results

    def get_all_models(self):
        """Get all the available models.

        Returns:
            dict: Dictionary of available models with their configurations.
        """
        return _models_dict
