import glob
import json
import shutil
from collections import defaultdict, namedtuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from math import ceil
from pathlib import Path
from typing import List, Tuple, Union

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from tqdm.contrib.concurrent import thread_map
from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation

# a label and all meta information
_Label = namedtuple(
    "_Label",
    [
        "name",  # The identifier of this label, e.g. 'car', 'person', ... .
        # We use them to uniquely name a class
        "id",  # An integer ID that is associated with this label.
        # The IDs are used to represent the label in ground truth images
        # An ID of -1 means that this label does not have an ID and thus
        # is ignored when creating ground truth images (e.g. license plate).
        # Do not modify these IDs, since exactly these IDs are expected by the
        # evaluation server.
        "trainId",  # Feel free to modify these IDs as suitable for your method. Then create
        # ground truth images with train IDs, using the tools provided in the
        # 'preparation' folder. However, make sure to validate or submit results
        # to our evaluation server using the regular IDs above!
        # For trainIds, multiple labels might have the same ID. Then, these labels
        # are mapped to the same class in the ground truth images. For the inverse
        # mapping, we use the label that is defined first in the list below.
        # For example, mapping all void-type classes to the same ID in training,
        # might make sense for some approaches.
        # Max value is 255!
        "category",  # The name of the category that this label belongs to
        "categoryId",  # The ID of this category. Used to create ground truth images
        # on category level.
        "hasInstances",  # Whether this label distinguishes between single instances or not
        "ignoreInEval",  # Whether pixels having this class as ground truth label are ignored
        # during evaluations or not
        "color",  # The color of this label
    ],
)


def _create_cityscapes_label_colormap():
    """Creates a label colormap used in CITYSCAPES segmentation benchmark.

    Args:

    Returns:
      : A colormap for visualizing segmentation results.

    """
    labels = [
        #       name                     id    trainId   category            catId     hasInstances   ignoreInEval   color
        _Label("unlabeled", 0, 255, "void", 0, False, True, (0, 0, 0)),
        _Label("ego vehicle", 1, 255, "void", 0, False, True, (0, 0, 0)),
        _Label("rectification border", 2, 255, "void", 0, False, True, (0, 0, 0)),
        _Label("out of roi", 3, 255, "void", 0, False, True, (0, 0, 0)),
        _Label("static", 4, 255, "void", 0, False, True, (0, 0, 0)),
        _Label("dynamic", 5, 255, "void", 0, False, True, (111, 74, 0)),
        _Label("ground", 6, 255, "void", 0, False, True, (81, 0, 81)),
        _Label("road", 7, 0, "flat", 1, False, False, (128, 64, 128)),
        _Label("sidewalk", 8, 1, "flat", 1, False, False, (244, 35, 232)),
        _Label("parking", 9, 255, "flat", 1, False, True, (250, 170, 160)),
        _Label("rail track", 10, 255, "flat", 1, False, True, (230, 150, 140)),
        _Label("building", 11, 2, "construction", 2, False, False, (70, 70, 70)),
        _Label("wall", 12, 3, "construction", 2, False, False, (102, 102, 156)),
        _Label("fence", 13, 4, "construction", 2, False, False, (190, 153, 153)),
        _Label("guard rail", 14, 255, "construction", 2, False, True, (180, 165, 180)),
        _Label("bridge", 15, 255, "construction", 2, False, True, (150, 100, 100)),
        _Label("tunnel", 16, 255, "construction", 2, False, True, (150, 120, 90)),
        _Label("pole", 17, 5, "object", 3, False, False, (153, 153, 153)),
        _Label("polegroup", 18, 255, "object", 3, False, True, (153, 153, 153)),
        _Label("traffic light", 19, 6, "object", 3, False, False, (250, 170, 30)),
        _Label("traffic sign", 20, 7, "object", 3, False, False, (220, 220, 0)),
        _Label("vegetation", 21, 8, "nature", 4, False, False, (107, 142, 35)),
        _Label("terrain", 22, 9, "nature", 4, False, False, (152, 251, 152)),
        _Label("sky", 23, 10, "sky", 5, False, False, (70, 130, 180)),
        _Label("person", 24, 11, "human", 6, True, False, (220, 20, 60)),
        _Label("rider", 25, 12, "human", 6, True, False, (255, 0, 0)),
        _Label("car", 26, 13, "vehicle", 7, True, False, (0, 0, 142)),
        _Label("truck", 27, 14, "vehicle", 7, True, False, (0, 0, 70)),
        _Label("bus", 28, 15, "vehicle", 7, True, False, (0, 60, 100)),
        _Label("caravan", 29, 255, "vehicle", 7, True, True, (0, 0, 90)),
        _Label("trailer", 30, 255, "vehicle", 7, True, True, (0, 0, 110)),
        _Label("train", 31, 16, "vehicle", 7, True, False, (0, 80, 100)),
        _Label("motorcycle", 32, 17, "vehicle", 7, True, False, (0, 0, 230)),
        _Label("bicycle", 33, 18, "vehicle", 7, True, False, (119, 11, 32)),
        _Label("license plate", -1, -1, "vehicle", 7, False, True, (0, 0, 142)),
    ]
    return labels


def _create_mapillary_vistas_label_colormap():
    """Creates a label colormap used in Mapillary Vistas segmentation benchmark.

    Args:

    Returns:
      : A list of labels for visualizing segmentation results.

    """
    labels = [
        _Label("Bird", 0, 0, "animal", 0, True, False, (165, 42, 42)),
        _Label("Ground Animal", 1, 1, "animal", 0, True, False, (0, 192, 0)),
        _Label("Curb", 2, 2, "construction", 1, False, False, (196, 196, 196)),
        _Label("Fence", 3, 3, "construction", 1, False, False, (190, 153, 153)),
        _Label("Guard Rail", 4, 4, "construction", 1, False, False, (180, 165, 180)),
        _Label("Barrier", 5, 5, "construction", 1, False, False, (102, 102, 156)),
        _Label("Wall", 6, 6, "construction", 1, False, False, (102, 102, 156)),
        _Label("Bike Lane", 7, 7, "flat", 2, False, False, (128, 64, 255)),
        _Label("Crosswalk - Plain", 8, 8, "flat", 2, False, False, (140, 140, 200)),
        _Label("Curb Cut", 9, 9, "flat", 2, False, False, (170, 170, 170)),
        _Label("Parking", 10, 10, "flat", 2, False, False, (250, 170, 160)),
        _Label("Pedestrian Area", 11, 11, "flat", 2, False, False, (96, 96, 96)),
        _Label("Rail Track", 12, 12, "flat", 2, False, False, (230, 150, 140)),
        _Label("Road", 13, 13, "flat", 2, False, False, (128, 64, 128)),
        _Label("Service Lane", 14, 14, "flat", 2, False, False, (110, 110, 110)),
        _Label("Sidewalk", 15, 15, "flat", 2, False, False, (244, 35, 232)),
        _Label("Bridge", 16, 16, "construction", 1, False, False, (150, 100, 100)),
        _Label("Building", 17, 17, "construction", 1, False, False, (70, 70, 70)),
        _Label("Tunnel", 18, 18, "construction", 1, False, False, (150, 120, 90)),
        _Label("Person", 19, 19, "human", 3, True, False, (220, 20, 60)),
        _Label("Bicyclist", 20, 20, "human", 3, True, False, (255, 0, 0)),
        _Label("Motorcyclist", 21, 21, "human", 3, True, False, (255, 0, 0)),
        _Label("Other Rider", 22, 22, "human", 3, True, False, (255, 0, 0)),
        _Label(
            "Lane Marking - Crosswalk",
            23,
            23,
            "marking",
            4,
            False,
            True,
            (200, 128, 128),
        ),
        _Label("Lane Marking - General", 24, 24, "marking", 4, True, False, (255, 255, 255)),
        _Label("Mountain", 25, 25, "nature", 5, False, False, (64, 170, 64)),
        _Label("Sand", 26, 26, "nature", 5, False, False, (230, 160, 50)),
        _Label("Sky", 27, 27, "sky", 6, False, False, (70, 130, 180)),
        _Label("Snow", 28, 28, "nature", 5, False, False, (190, 255, 255)),
        _Label("Terrain", 29, 29, "nature", 5, False, False, (152, 251, 152)),
        _Label("Vegetation", 30, 30, "nature", 5, False, False, (107, 142, 35)),
        _Label("Water", 31, 31, "water", 7, False, False, (0, 170, 30)),
        _Label("Banner", 32, 32, "object", 8, False, False, (255, 220, 0)),
        _Label("Bench", 33, 33, "object", 8, False, False, (255, 0, 0)),
        _Label("Bike Rack", 34, 34, "object", 8, False, False, (255, 0, 0)),
        _Label("Billboard", 35, 35, "object", 8, False, False, (255, 0, 0)),
        _Label("Catch Basin", 36, 36, "object", 8, False, False, (255, 0, 0)),
        _Label("CCTV Camera", 37, 37, "object", 8, False, False, (255, 0, 0)),
        _Label("Fire Hydrant", 38, 38, "object", 8, False, False, (255, 0, 0)),
        _Label("Junction Box", 39, 39, "object", 8, False, False, (255, 0, 0)),
        _Label("Mailbox", 40, 40, "object", 8, False, False, (255, 0, 0)),
        _Label("Manhole", 41, 41, "object", 8, False, False, (255, 0, 0)),
        _Label("Phone Booth", 42, 42, "object", 8, False, False, (255, 0, 0)),
        _Label("Pothole", 43, 43, "object", 8, False, False, (255, 0, 0)),
        _Label("Street Light", 44, 44, "object", 8, False, False, (255, 0, 0)),
        _Label("Pole", 45, 45, "object", 8, False, False, (255, 0, 0)),
        _Label("Traffic Sign Frame", 46, 46, "object", 8, False, False, (255, 0, 0)),
        _Label("Utility Pole", 47, 47, "object", 8, False, False, (255, 0, 0)),
        _Label("Traffic Light", 48, 48, "object", 8, False, False, (255, 0, 0)),
        _Label("Traffic Sign (Back)", 49, 49, "object", 8, False, False, (255, 0, 0)),
        _Label("Traffic Sign (Front)", 50, 50, "object", 8, False, False, (255, 0, 0)),
        _Label("Trash Can", 51, 51, "object", 8, False, False, (255, 0, 0)),
        _Label("Bicycle", 52, 52, "vehicle", 9, True, False, (119, 11, 32)),
        _Label("Boat", 53, 53, "vehicle", 9, False, False, (0, 0, 142)),
        _Label("Bus", 54, 54, "vehicle", 9, True, False, (0, 60, 100)),
        _Label("Car", 55, 55, "vehicle", 9, True, False, (0, 0, 142)),
        _Label("Caravan", 56, 56, "vehicle", 9, True, False, (0, 0, 90)),
        _Label("Motorcycle", 57, 57, "vehicle", 9, True, False, (0, 0, 230)),
        _Label("On Rails", 58, 58, "vehicle", 9, False, False, (0, 80, 100)),
        _Label("Other Vehicle", 59, 59, "vehicle", 9, True, False, (128, 64, 64)),
        _Label("Trailer", 60, 60, "vehicle", 9, True, False, (0, 0, 110)),
        _Label("Truck", 61, 61, "vehicle", 9, True, False, (0, 0, 70)),
        _Label("Wheeled Slow", 62, 62, "vehicle", 9, False, False, (0, 0, 192)),
        _Label("Car Mount", 63, 63, "vehicle", 9, True, False, (32, 32, 32)),
        _Label("Ego Vehicle", 64, 64, "vehicle", 9, True, False, (120, 10, 10)),
    ]
    return labels


def _get_resized_dimensions(width: int, height: int, max_size: int = 2048) -> Tuple[int, int]:
    """Calculate the new dimensions of an image to maintain aspect ratio.

    If both dimensions are less than or equal to max_size, the original dimensions are returned.

    Args:
        width (int): The original width of the image.
        height (int): The original height of the image.
        max_size (int, optional): The maximum size for either dimension. Defaults to 2048.

    Returns:
        Tuple[int, int]: The new dimensions (width, height) of the image.
    """
    if max(width, height) > max_size:
        scaling_factor = max_size / max(width, height)
        new_width = int(width * scaling_factor)
        new_height = int(height * scaling_factor)
        return new_width, new_height
    else:
        # Return original dimensions if resizing is not necessary
        return width, height


class ImageDataset(Dataset):
    """A dataset class for loading and processing images.

    This class handles the loading of images from specified file paths,
    resizing them to a maximum size while maintaining the aspect ratio,
    and converting them to RGB format if required.

    Args:
        image_files (List[Path]): A list of paths to the image files.
        max_size (int, optional): The maximum size for resizing the images. Defaults to 2048.
        rgb (bool, optional): If True, images will be converted to RGB format. Defaults to True.
    """

    def __init__(self, image_files: List[Path], max_size: int = 2048, rgb: bool = True) -> None:
        """Initializes the ImageDataset with the paths to images, maximum size for resizing,
        and color mode.

        Args:
            image_files (List[Path]): A list of paths to the image files.
            max_size (int, optional): The maximum size for resizing the images. Defaults to 2048.
            rgb (bool, optional): If True, images will be converted to RGB format. Defaults to True.
        """
        self.image_files = [
            image_file
            for image_file in image_files
            if image_file.suffix.lower() in [".jpg", ".jpeg", ".png"] and not image_file.name.startswith(".")
        ]
        self.max_size = max_size
        self.rgb = rgb

    def __len__(self) -> int:
        """Returns the number of images in the dataset.

        Returns:
            int: The number of images in the dataset.
        """
        return len(self.image_files)

    def __getitem__(self, idx: int) -> Tuple[str, cv2.Mat, Tuple[int, int]]:
        """Retrieves an image and its metadata from the dataset.

        Args:
            idx (int): The index of the image to retrieve.

        Returns:
            Tuple[str, cv2.Mat, Tuple[int, int]]: A tuple containing the image file path,
            the image data, and the dimensions of the image (height, width).

        Raises:
            ValueError: If the image cannot be read.
        """
        image_file = self.image_files[idx]
        img = cv2.imread(str(image_file))

        if img is None:
            raise ValueError(f"Unable to read image at {image_file}")

        original_height, original_width = img.shape[:2]
        new_width, new_height = _get_resized_dimensions(original_width, original_height, self.max_size)

        # Resize image if necessary
        if (original_width, original_height) != (new_width, new_height):
            img = cv2.resize(img, (new_width, new_height))

        if self.rgb:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        return str(image_file), img, (new_height, new_width)

    def collate_fn(
        self, data: List[Tuple[str, cv2.Mat, Tuple[int, int]]]
    ) -> Tuple[List[str], List[cv2.Mat], List[Tuple[int, int]]]:
        """Custom collate function for the dataset.

        Args:
            data (List[Tuple[str, cv2.Mat, Tuple[int, int]]]): A list of tuples containing
            image file path, image data, and original image dimensions.

        Returns:
            Tuple[List[str], List[cv2.Mat], List[Tuple[int, int]]]: A tuple containing lists
            of image file paths, image data, and original image dimensions.
        """
        image_files, images, original_img_shape = zip(*data)
        return list(image_files), list(images), list(original_img_shape)


class Segmenter:
    """A class for performing semantic and panoptic segmentation on images.

    The models used are from the Mask2Former (https://huggingface.co/docs/transformers/model_doc/mask2former).

    Attributes:
        device (str): The device to run the model on (e.g., "cuda" or "cpu").
        dataset (str): The name of the dataset (e.g., "cityscapes" or "mapillary").
        task (str): The type of segmentation task (e.g., "semantic" or "panoptic").
        model_name (str): The name of the pre-trained model corresponding to the dataset and task.
        model: The segmentation model.
        processor: The image processor for the model.
        color_map: A mapping of class IDs to colors.
        label_map: A mapping of class IDs to labels.
        id_to_name_map: A mapping of label IDs to label names.

    Args:
        dataset (str): The name of the dataset (default is "cityscapes").
        task (str): The type of task (default is "semantic").
        device (str, optional): The device to run the model on (e.g., "cuda" or "cpu"). If None, the default device will be used.

    Returns:
        None
    """

    def __init__(self, dataset="cityscapes", task="semantic", device=None):
        """Initializes the Segmenter with a model and dataset.

        Args:
            dataset (str): The name of the dataset (default is "cityscapes").
            task (str): The type of task (default is "semantic").
            device (str, optional): The device to run the model on (e.g., "cuda" or "cpu"). If None, the default device will be used.

        Returns:
            None
        """
        self.device = self._get_device(device)
        self.dataset = dataset
        self.task = task
        self.model_name = self._get_model_name(self.dataset, self.task)
        self.model, self.processor = self._get_model_processor(self.model_name)
        self.color_map = self._create_color_map(dataset)
        self.label_map = self._create_label_map(dataset)
        self.id_to_name_map = self._create_id_to_name_map(dataset)

    def _get_model_name(self, dataset: str, task: str) -> str:
        """Get the model name based on the dataset and task.

        Args:
            dataset (str): The name of the dataset (e.g., "cityscapes", "mapillary").
            task (str): The type of task (e.g., "semantic", "panoptic").

        Returns:
            str: The name of the pre-trained model corresponding to the dataset and task.

        Raises:
            ValueError: If the dataset is unknown.

        """
        if dataset == "cityscapes":
            if task == "semantic":
                return "facebook/mask2former-swin-tiny-cityscapes-semantic"
            elif task == "panoptic":
                return "facebook/mask2former-swin-tiny-cityscapes-panoptic"
        elif dataset == "mapillary":
            if task == "semantic":
                return "facebook/mask2former-swin-large-mapillary-vistas-semantic"
            elif task == "panoptic":
                return "facebook/mask2former-swin-large-mapillary-vistas-panoptic"
        else:
            raise ValueError(f"Unknown dataset: {dataset}")

    def _get_model_processor(self, model_name):
        """Get the model and processor for the given model name.

        Args:
          model_name(str): The name of the pre-trained model.

        Returns:
          Tuple: The model and processor.

        """
        # Add other models in the future
        if "mask2former" in model_name:
            processor = AutoImageProcessor.from_pretrained(model_name)
            model = Mask2FormerForUniversalSegmentation.from_pretrained(model_name).to(self.device)
        return model, processor

    def _create_color_map(self, dataset):
        """Create a color map based on the given dataset.

        Args:
          dataset(str): The name of the dataset.

        Returns:
          np.ndarray: A color map.

        """
        if dataset == "cityscapes":
            labels = _create_cityscapes_label_colormap()
        elif dataset == "mapillary":
            labels = _create_mapillary_vistas_label_colormap()
        else:
            raise ValueError(f"Unknown dataset: {dataset}")

        labels = [label for label in labels if label.trainId != -1]
        train_ids = np.array([label.trainId for label in labels], dtype=np.uint8)
        colors = np.array([label.color for label in labels], dtype=np.uint8)
        max_train_id = np.max(train_ids) + 1
        color_map = np.zeros((max_train_id, 3), dtype=np.uint8)
        color_map[train_ids] = colors

        # Add a train_id_to_name dictionary as an attribute
        self.train_id_to_name = {label.trainId: label.name for label in labels}

        return color_map

    def _create_label_map(self, dataset):
        """Create a label map based on the given dataset.

        Args:
          dataset(str): The name of the dataset.

        Returns:
          Dict[Tuple, _Label]: A dictionary mapping colors to labels.

        """
        if dataset == "cityscapes":
            labels = _create_cityscapes_label_colormap()
        elif dataset == "mapillary":
            labels = _create_mapillary_vistas_label_colormap()
        else:
            raise ValueError(f"Unknown dataset: {dataset}")

        color_to_label = {}
        for label in labels:
            color = label.color
            color_to_label[color] = label

        return color_to_label

    def _create_id_to_name_map(self, dataset):
        """Create a mapping from train IDs to label names based on the dataset.

        Args:
            dataset (str): The name of the dataset (e.g., "cityscapes" or "mapillary").

        Returns:
            dict: A dictionary mapping train IDs to label names.
        """
        if dataset == "cityscapes":
            labels = _create_cityscapes_label_colormap()
        elif dataset == "mapillary":
            labels = _create_mapillary_vistas_label_colormap()
        return {label.trainId: label.name for label in labels}

    def _get_device(self, device) -> torch.device:
        """Get the appropriate device for running the model.

        Args:
            device (str or None): The device to use (e.g., "cpu", "cuda", "mps"). If None, the function will select the best available device.

        Returns:
            torch.device: The device to use for running the model.

        Raises:
            ValueError: If the provided device is not recognized.
        """
        if device is not None:
            if device not in ["cpu", "cuda", "mps"]:
                raise ValueError(f"Unknown device: {device}")
            else:
                print(f"Using {device.upper()}")
                return torch.device(device)
        if torch.cuda.is_available():
            print("Using GPU")
            return torch.device("cuda")
        else:
            print("Using CPU")
            return torch.device("cpu")

    def _calculate_pixel_ratios(self, segmented_img):
        """Calculate pixel ratios for each class in the segmented image.

        Args:
            segmented_img (numpy.ndarray): Segmented image.

        Returns:
            dict: A dictionary with class names as keys and pixel ratios as values.
        """
        unique, counts = np.unique(segmented_img, return_counts=True)
        total_pixels = np.sum(counts)
        pixel_ratios = {
            self.train_id_to_name[train_id]: count / total_pixels for train_id, count in zip(unique, counts)
        }

        return pixel_ratios

    def _save_as_csv(self, input_dict: dict, dir_output: Path, value_name: str, csv_format: str) -> None:
        """Save pixel ratios as a CSV file.

        This function takes a dictionary of pixel ratios and saves it to a CSV file in either long or wide format.

        Args:
            input_dict (dict): A dictionary containing pixel ratios for each image and label.
            dir_output (Path): The directory where the CSV file will be saved.
            value_name (str): The name of the value to be saved in the CSV.
            csv_format (str): The format of the CSV file, either 'long' or 'wide'.

        Returns:
            None: This function does not return any value but saves the CSV file to the specified directory.
        """
        if csv_format == "long":
            df_list = [
                pd.DataFrame(
                    {
                        "filename_key": [filename_key],
                        "label_name": [key],
                        value_name: [value] if value is not None else [0],
                    }
                )
                for filename_key, inner_dict in input_dict.items()
                for key, value in inner_dict.items()
            ]

            pixel_ratios_df = pd.concat(df_list, ignore_index=True)

        elif csv_format == "wide":
            pixel_ratios_df = pd.DataFrame(input_dict).transpose().fillna(0)
            pixel_ratios_df.index.names = ["filename_key"]

        pixel_ratios_df.to_csv(dir_output / Path(value_name + ".csv"))

    def _panoptic_segmentation(self, images, original_img_shape):
        """Perform panoptic segmentation on the given images.

        Args:
          images(list): List of input images.
          original_img_shape(tuple): Original image shape.

        Returns:
          list: List of panoptic segmentation outputs.

        """
        inputs = self.processor(images=images, return_tensors="pt").to(self.model.device)
        outputs = self.model(**inputs)
        return self.processor.post_process_panoptic_segmentation(
            outputs, target_sizes=original_img_shape, label_ids_to_fuse=set([])
        )

    def _semantic_segmentation(self, images, original_img_shape):
        """Perform semantic segmentation on the given images.

        Args:
          images(list): List of input images.
          original_img_shape(tuple): Original image shape.

        Returns:
          tuple: Tuple containing list of semantic segmentation outputs and list of pixel ratios.

        """
        inputs = self.processor(images=images, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        segmentations = self.processor.post_process_semantic_segmentation(outputs, target_sizes=original_img_shape)
        return segmentations

    def _trainid_to_color(self, segmented_img):
        """Convert segmented image with train IDs to a colored image.

        Args:
          segmented_img(numpy.ndarray): Segmented image with train IDs.

        Returns:
          numpy.ndarray: Colored segmented image.

        """
        colored_img = self.color_map[segmented_img]
        return colored_img

    def _save_panoptic_segmentation_image(
        self, image_file: str, img: np.ndarray, dir_output: Path, output: dict
    ) -> None:
        """Save the panoptic segmentation image as a blended image with the original input image.

        Args:
            image_file (str): The input image file path.
            img (np.ndarray): The input image in the format of a NumPy array.
            dir_output (Path): The output directory path to save the blended image.
            output (dict): The output dictionary containing the segmentation data.

        Returns:
            None: This function does not return any value but saves the blended image and segmented image to the specified directory.
        """
        colored_segmented_img = self._trainid_to_color(output["label_segmentation"].cpu().numpy())
        alpha = 0.5
        blend_img = cv2.addWeighted(img, alpha, colored_segmented_img, 1 - alpha, 0)

        # Calculate the scale factor for text size
        height, width, _ = img.shape
        scale_factor = np.sqrt(height * width) / 1000  # Example scale, adjust as needed

        # Add annotations for each segment
        for segment_info in output["segments_info"]:
            segment_id = segment_info["id"]
            label_id = segment_info["label_id"]
            score = segment_info["score"]

            # Use the label name instead of the label_id
            label_name = self.id_to_name_map.get(label_id)

            # Find the center of the segment for the label placement
            y, x = np.where(output["segmentation"].cpu().numpy() == segment_id)
            center_x, center_y = np.mean(x), np.mean(y)

            # Add the annotation with dynamic font size
            font_scale = 1 * scale_factor  # Adjust base font size (1 here) as needed
            thickness = 1 * scale_factor  # Adjust base thickness (1 here) as needed
            cv2.putText(
                blend_img,
                f"{label_name}-{score:.2f}",
                (int(center_x), int(center_y)),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (255, 255, 255),
                ceil(thickness),
                cv2.LINE_AA,
            )

        output_file = dir_output / Path(image_file).name

        # Save images based on specified options
        if "segmented_image" in self.save_image_options:
            cv2.imwrite(
                str(output_file.with_name(output_file.stem + "_colored_segmented.png")),
                cv2.cvtColor(colored_segmented_img, cv2.COLOR_RGB2BGR),
            )
        if "blend_image" in self.save_image_options:
            cv2.imwrite(
                str(output_file.with_name(output_file.stem + "_blend.png")),
                cv2.cvtColor(blend_img, cv2.COLOR_RGB2BGR),
            )

    def _save_semantic_segmentation_image(self, image_file, img, dir_output, output):
        """Saves the semantic segmentation image as a colored segmented image and/or a
        blended image with the original input image.

        Args:
            image_file (str): The input image file path.
            img (np.array): The input image in the format of a NumPy array.
            dir_output (Path): The output directory path to save the colored segmented and/or blended image.
            output (Tensor): The output tensor containing the semantic segmentation data.

        Returns:
            None
        """
        colored_segmented_img = self._trainid_to_color(output.cpu().numpy())
        alpha = 0.5
        blend_img = cv2.addWeighted(img, alpha, colored_segmented_img, 1 - alpha, 0)

        output_file = dir_output / Path(image_file).name

        # Save images based on specified options
        if "segmented_image" in self.save_image_options:
            cv2.imwrite(
                str(output_file.with_name(output_file.stem + "_colored_segmented.png")),
                cv2.cvtColor(colored_segmented_img, cv2.COLOR_RGB2BGR),
            )
        if "blend_image" in self.save_image_options:
            cv2.imwrite(
                str(output_file.with_name(output_file.stem + "_blend.png")),
                cv2.cvtColor(blend_img, cv2.COLOR_RGB2BGR),
            )

    def _panoptic_count_labels(self, output):
        """Count the occurrences of each label in the panoptic segmentation output.

        Args:
            output (dict): The output dictionary containing segmentation information.
                It should have a key "segments_info" which is a list of dictionaries,
                each containing a "label_id".

        Returns:
            dict: A dictionary where keys are label names and values are the counts
            of each label in the segmentation output.
        """
        label_counts = {}

        # Loop through each segment in the image
        segments_info_list = output["segments_info"]
        for segments_info in segments_info_list:
            # Convert label_id to label_name
            label_name = self.id_to_name_map.get(segments_info["label_id"])
            # Increment the count for the name in label_counts
            if label_name in label_counts:
                label_counts[label_name] += 1
            else:
                label_counts[label_name] = 1
        return label_counts

    def _panoptic_segment_to_label(self, output):
        """This function converts the output of post_process_panoptic_segmentation
        function from segment_id to label_id.

        Args:
          output: The output dictionary from the
        post_process_panoptic_segmentation function

        Returns:
          : segmentation with label_ids instead of segment_ids

        """
        # Extract the segmentation and segments_info from the output
        segmentation = output["segmentation"]
        segments_info = output["segments_info"]

        # Create a mapping from segment_id to label_id
        id_map = {segment["id"]: segment["label_id"] for segment in segments_info}

        # Use the map to convert the segmentation tensor from segment_ids to label_ids
        new_segmentation = segmentation.clone()

        for seg_id, label_id in id_map.items():
            new_segmentation[segmentation == seg_id] = label_id

        return new_segmentation

    def _process_images(
        self,
        task,
        image_files,
        images,
        dir_output,
        pixel_ratio_dict,
        original_img_shape,
        panoptic_dict=None,
    ):
        """Process the input images for segmentation and save the output images.

        Args:
          task(str): The segmentation task to perform, either "panoptic" or "semantic".
          image_files(List[str]): The list of file paths of the input images.
          images(List[ndarray]): The list of input images in the form of numpy arrays.
          dir_output(Path): The output directory where the segmented images will be saved.
          pixel_ratio_dict(defaultdict): A dictionary to store the pixel ratios of the segmented images.
          original_img_shape(List[Tuple[int): The original shapes of the input images.
          panoptic_dict: (Default value = None)

        Returns:
          : None

        """
        outputs = None
        if task == "panoptic":
            outputs = self._panoptic_segmentation(images, original_img_shape)
            if outputs is not None:
                for image_file, img, output in zip(image_files, images, outputs):
                    # create a new segmentation with label_ids instead of segment_ids
                    output["label_segmentation"] = self._panoptic_segment_to_label(output)
                    if (len(self.save_image_options) > 0) & (dir_output is not None):
                        self._save_panoptic_segmentation_image(image_file, img, dir_output, output)
                    pixel_ratio = self._calculate_pixel_ratios(output["label_segmentation"].cpu().numpy())
                    label_counts = self._panoptic_count_labels(output)
                    image_file_key = Path(image_file).stem
                    pixel_ratio_dict[image_file_key] = pixel_ratio
                    panoptic_dict[image_file_key] = label_counts

        elif task == "semantic":
            segmentations = self._semantic_segmentation(images, original_img_shape)
            if segmentations is not None:
                for image_file, img, segmentation in zip(image_files, images, segmentations):
                    if (len(self.save_image_options) > 0) & (dir_output is not None):
                        self._save_semantic_segmentation_image(image_file, img, dir_output, segmentation)
                    pixel_ratio = self._calculate_pixel_ratios(segmentation.cpu().numpy())
                    image_file_key = Path(image_file).stem
                    pixel_ratio_dict[image_file_key] = pixel_ratio

    # Modify the segment method inside the Segmenter class
    def segment(
        self,
        dir_input: Union[str, Path],
        dir_image_output: Union[str, Path, None] = None,
        dir_summary_output: Union[str, Path, None] = None,
        batch_size=1,
        save_image_options="segmented_image blend_image",
        save_format="json csv",
        csv_format="long",  # "long" or "wide"
        max_workers: Union[int, None] = None,
    ):
        """Processes a batch of images for segmentation, saves the segmented images and
        summary statistics.

        This method handles the processing of images for segmentation, managing input/output directories,
        saving options, and parallel processing settings. The method requires specifying an input directory
        or a path to a single image and supports optional saving of output images and segmentation summaries.

        Args:
          dir_input: Union
          dir_image_output: Union
          are: saved
          dir_summary_output: Union
          segmentation: summary files are saved
          batch_size: int (Default value = 1)
          save_image_options: str (Default value = "segmented_image blend_image")
          segmented_image: and
          save_format: str (Default value = "json csv")
          Defaults: to
          csv_format: str (Default value = "long")
          Defaults: to
          max_workers: Union
          Defaults: to None
          dir_input: Union[str:
          Path]:
          dir_image_output: Union[str:
          Path:
          None]: (Default value = None)
          dir_summary_output: Union[str:
          # "long" or "wide"max_workers: Union[int:
          dir_input: Union[str:
          dir_image_output: Union[str:
          dir_summary_output: Union[str:
          # "long" or "wide"max_workers: Union[int:
          dir_input: Union[str:
          dir_image_output: Union[str:
          dir_summary_output: Union[str:
          # "long" or "wide"max_workers: Union[int:

        Returns:
          None: The method does not return any value but saves the processed results to specified directories.

        Raises:
          ValueError: If neither
          ValueError: If

        """
        # make sure that at least one of dir_image_output and dir_summary_output is not None
        if (dir_image_output is None) & (dir_summary_output is None):
            raise ValueError("At least one of dir_image_output and dir_summary_output must not be None.")

        # skip if there's pixel_ratio.json and/or pixel_ratios.csv in dir_summary_output, depending on save_format
        if dir_summary_output is not None:
            if "json" in save_format and "csv" in save_format:
                if (Path(dir_summary_output) / "pixel_ratios.json").exists() and (
                    Path(dir_summary_output) / "pixel_ratios.csv"
                ).exists():
                    print("Segmentation summary already exists. Skipping segmentation.")
                    return
            elif "json" in save_format:
                if (Path(dir_summary_output) / "pixel_ratios.json").exists():
                    print("Segmentation summary already exists. Skipping segmentation.")
                    return
            elif "csv" in save_format:
                if (Path(dir_summary_output) / "pixel_ratios.csv").exists():
                    print("Segmentation summary already exists. Skipping segmentation.")
                    return
        # save_image_options as a property of the class
        self.save_image_options = save_image_options

        # make directory
        dir_input = Path(dir_input)

        # initialize completed_image_files
        completed_image_files = set()
        if dir_image_output is not None:
            dir_image_output = Path(dir_image_output)
            dir_image_output.mkdir(parents=True, exist_ok=True)
            # get a list of .png files and _blend.png files in the output directory and get the file names as a set
            completed_image_files.update(
                [
                    str(Path(f).stem).replace("_blend", "").replace("_colored_segmented", "")
                    for f in dir_image_output.glob("*.png")
                ]
            )

        if dir_summary_output is not None:
            dir_summary_output = Path(dir_summary_output)
            dir_summary_output.mkdir(parents=True, exist_ok=True)
            # Create a new directory called "pixel_ratio_checkpoints"
            dir_cache_segmentation_summary = dir_summary_output / "pixel_ratio_checkpoints"
            dir_cache_segmentation_summary.mkdir(parents=True, exist_ok=True)

            # Load all the checkpoint json files
            checkpoints = glob.glob(str(dir_cache_segmentation_summary / "*.json"))
            checkpoint_start_index = len(checkpoints)

            if checkpoint_start_index > 0:
                for checkpoint in checkpoints:
                    with open(checkpoint, "r") as f:
                        checkpoint_dict = json.load(f)
                        completed_image_files.update(checkpoint_dict.keys())

            # also check pixel_ratios.json in dir_cache_segmentation_summary
            if (dir_cache_segmentation_summary / "pixel_ratios.json").exists():
                with open(dir_cache_segmentation_summary / "pixel_ratios.json", "r") as f:
                    pixel_ratio_dict = json.load(f)
                    completed_image_files.update(pixel_ratio_dict.keys())

        # Get the list of all image files and filter the ones that are not completed yet
        # Handle both single file and directory inputs
        if dir_input.is_file():
            # Process as a single file
            image_file_list = [dir_input]
        elif dir_input.is_dir():
            # Process all suitable files in the directory
            image_extensions = [
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
            # Get the list of all image files in the directory that are not completed yet
            image_file_list = [
                f
                for f in Path(dir_input).iterdir()
                if f.suffix in image_extensions and f.stem not in completed_image_files
            ]
        else:
            raise ValueError("dir_input must be either a file or a directory.")

        # skip if there are no image files to process
        if len(image_file_list) == 0:
            print("No image files to process. Skipping segmentation.")
            return

        outer_batch_size = 1000  # Number of inner batches in one outer batch
        num_outer_batches = (len(image_file_list) + outer_batch_size * batch_size - 1) // (
            outer_batch_size * batch_size
        )

        for i in tqdm(
            range(num_outer_batches),
            desc=f"Processing outer batches of size {min(outer_batch_size * batch_size, len(image_file_list))}",
        ):
            # Get the image files for the current outer batch
            outer_batch_image_file_list = image_file_list[
                i * outer_batch_size * batch_size : (i + 1) * outer_batch_size * batch_size
            ]

            dataset = ImageDataset(outer_batch_image_file_list)
            dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=dataset.collate_fn)

            # set up pixel_ratio_dict for the current outer batch
            pixel_ratio_dict = defaultdict(dict)  # reset pixel_ratio_dict for each outer batch
            panoptic_dict = defaultdict(dict)  # reset panoptic_dict for each outer batch
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = []

                for batch in dataloader:
                    image_files, images, original_img_shape = batch
                    if self.task == "panoptic":
                        future = executor.submit(
                            self._process_images,
                            self.task,
                            image_files,
                            images,
                            dir_image_output,
                            pixel_ratio_dict,
                            original_img_shape,
                            panoptic_dict,
                        )
                    elif self.task == "semantic":
                        future = executor.submit(
                            self._process_images,
                            self.task,
                            image_files,
                            images,
                            dir_image_output,
                            pixel_ratio_dict,
                            original_img_shape,
                        )
                    futures.append(future)

                for completed_future in tqdm(
                    as_completed(futures),
                    total=len(futures),
                    desc=f"Processing outer batch #{i+1}",
                ):
                    completed_future.result()

                if dir_summary_output is not None:
                    # Save checkpoint for each outer batch
                    with open(
                        f"{dir_cache_segmentation_summary}/checkpoint_batch_{checkpoint_start_index+i+1}_pixel_ratio.json",
                        "w",
                    ) as f:
                        json.dump(pixel_ratio_dict, f)

                    if self.task == "panoptic":
                        with open(
                            f"{dir_cache_segmentation_summary}/checkpoint_batch_{checkpoint_start_index+i+1}_panoptic.json",
                            "w",
                        ) as f:
                            json.dump(panoptic_dict, f)
        if dir_summary_output is not None:
            # Merge all checkpoints into a single pixel_ratio_dict
            pixel_ratio_dict = defaultdict(dict)
            for checkpoint in glob.glob(str(dir_cache_segmentation_summary / "*_pixel_ratio.json")):
                with open(checkpoint, "r") as f:
                    checkpoint_dict = json.load(f)
                    for key, value in checkpoint_dict.items():
                        pixel_ratio_dict[key] = value

            # Merge all checkpoints into a single panoptic_dict
            if self.task == "panoptic":
                panoptic_dict = defaultdict(dict)
                for checkpoint in glob.glob(str(dir_cache_segmentation_summary / "*_panoptic.json")):
                    with open(checkpoint, "r") as f:
                        checkpoint_dict = json.load(f)
                        for key, value in checkpoint_dict.items():
                            panoptic_dict[key] = value

            # Merge existing pixel_ratios.json with the new pixel_ratio_dict
            if (dir_summary_output / "pixel_ratios.json").exists():
                with open(dir_summary_output / "pixel_ratios.json", "r") as f:
                    existing_pixel_ratio_dict = json.load(f)
                    for key, value in existing_pixel_ratio_dict.items():
                        pixel_ratio_dict[key] = value

            # Merge existing label_counts.json with the new panoptic_dict
            if self.task == "panoptic":
                if (dir_summary_output / "label_counts.json").exists():
                    with open(dir_summary_output / "label_counts.json", "r") as f:
                        existing_panoptic_dict = json.load(f)
                        for key, value in existing_panoptic_dict.items():
                            panoptic_dict[key] = value

            # Save pixel_ratio_dict as a JSON or CSV file
            if "json" in save_format:
                with open(dir_summary_output / "pixel_ratios.json", "w") as f:
                    json.dump(pixel_ratio_dict, f)
                if self.task == "panoptic":
                    with open(dir_summary_output / "label_counts.json", "w") as f:
                        json.dump(panoptic_dict, f)
            if "csv" in save_format:
                self._save_as_csv(pixel_ratio_dict, dir_summary_output, "pixel_ratios", csv_format)
                if self.task == "panoptic":
                    self._save_as_csv(panoptic_dict, dir_summary_output, "label_counts", csv_format)

            # Delete the "pixel_ratio_checkpoints" directory
            shutil.rmtree(dir_cache_segmentation_summary, ignore_errors=True)

    def calculate_pixel_ratio_post_process(self, dir_input, dir_output, save_format="json csv"):
        """Calculates the pixel ratio of different classes present in the segmented
        images and saves the results in either JSON or CSV format.

        Args:
          dir_input: A string or Path object representing the input directory containing the segmented images.
          dir_output: A string or Path object representing the output directory where the pixel ratio results will be saved.
          save_format: A list containing the file formats in which the results will be saved. The allowed file formats are "json" and "csv". The default value is "json csv".

        Returns:
          : None

        """

        def calculate_label_ratios(image, label_map):
            """Calculates the pixel ratio of different classes present in a single
            image.

            Args:
              image: A numpy array representing an image.
              label_map: A dictionary containing the label names and their respective RGB colors.

            Returns:
              : A dictionary containing the pixel ratio of different classes in the given image.

            """
            label_ratios = {}
            total_pixels = image.shape[0] * image.shape[1]

            for color, label in label_map.items():
                color_pixels = np.count_nonzero(np.all(image == color, axis=-1))
                label_ratios[label.name] = color_pixels / total_pixels

            return label_ratios

        def process_image_file(image_file, label_map):
            """Calculates the pixel ratio of different classes in a single segmented
            image file.

            Args:
              image_file: A Path object representing the segmented image file.
              label_map: A dictionary containing the label names and their respective RGB colors.

            Returns:
              : A tuple containing the image file key and the pixel ratio of different classes in the given image.

            """
            image_file_key = str(Path(image_file).stem).replace("_colored_segmented", "")
            image = cv2.imread(str(image_file))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            label_ratios = calculate_label_ratios(image, label_map)
            return image_file_key, label_ratios

        def results_to_dataframe(results):
            """Converts the results obtained from processing each image file into a
            Pandas DataFrame.

            Args:
              results: A list of tuples, where each tuple contains the image file key and the pixel ratio of different classes in the corresponding image.

            Returns:
              : A Pandas DataFrame containing the pixel ratios of different classes in each image file.

            """
            pixel_ratio_dict = {}

            for image_file_key, label_ratios in results:
                pixel_ratio_dict[str(image_file_key)] = label_ratios

            pixel_ratios_df = pd.DataFrame(pixel_ratio_dict).transpose()
            pixel_ratios_df.fillna(0, inplace=True)
            pixel_ratios_df.index.names = ["filename_key"]

            return pixel_ratios_df

        def results_to_nested_dict(results):
            """Converts the results obtained from processing each image file into a
            nested dictionary.

            Args:
              results: A list of tuples, where each tuple contains the image file key and the pixel ratio of different classes in the corresponding image.

            Returns:
              : A nested dictionary containing the pixel ratios of different classes in each image file.

            """
            data = {}

            for image_file_key, label_ratios in results:
                image_file_key = str(image_file_key)
                data[image_file_key] = label_ratios

            return data

        # create dir_output
        dir_output = Path(dir_output)
        dir_output.mkdir(parents=True, exist_ok=True)

        # get files
        if isinstance(dir_input, str):
            dir_input = Path(dir_input)

        # Set image file extensions
        image_extensions = [".jpg", ".png"]

        if dir_input.is_file():
            image_files = [dir_input]
        elif dir_input.is_dir():
            image_files = [
                file
                for file in dir_input.rglob("*")
                if file.suffix.lower() in image_extensions and "_colored_segmented" in file.stem
            ]
        else:
            raise ValueError("dir_input must be either a file or a directory.")

        results = thread_map(process_image_file, image_files, [self.label_map] * len(image_files))

        if "json" in save_format:
            json_output_file = Path(dir_output) / "pixel_ratios.json"
            nested_dict = results_to_nested_dict(results)
            with open(json_output_file, "w") as f:
                json.dump(nested_dict, f, indent=2)

        if "csv" in save_format:
            csv_output_file = Path(dir_output) / "pixel_ratios.csv"
            df = results_to_dataframe(results)
            df.to_csv(csv_output_file)
