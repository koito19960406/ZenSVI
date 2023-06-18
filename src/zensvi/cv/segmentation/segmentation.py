import cv2
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Union, List
from torch.utils.data import Dataset, DataLoader
import torch
from torch import nn
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import namedtuple
from transformers import OneFormerProcessor, OneFormerForUniversalSegmentation
from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation
import os
from tqdm.auto import tqdm
import json
from collections import defaultdict
from tqdm.contrib.concurrent import thread_map
import glob
import shutil

# a label and all meta information
Label = namedtuple( 'Label' , [

            'name'        , # The identifier of this label, e.g. 'car', 'person', ... .
                            # We use them to uniquely name a class

            'id'          , # An integer ID that is associated with this label.
                            # The IDs are used to represent the label in ground truth images
                            # An ID of -1 means that this label does not have an ID and thus
                            # is ignored when creating ground truth images (e.g. license plate).
                            # Do not modify these IDs, since exactly these IDs are expected by the
                            # evaluation server.

            'trainId'     , # Feel free to modify these IDs as suitable for your method. Then create
                            # ground truth images with train IDs, using the tools provided in the
                            # 'preparation' folder. However, make sure to validate or submit results
                            # to our evaluation server using the regular IDs above!
                            # For trainIds, multiple labels might have the same ID. Then, these labels
                            # are mapped to the same class in the ground truth images. For the inverse
                            # mapping, we use the label that is defined first in the list below.
                            # For example, mapping all void-type classes to the same ID in training,
                            # might make sense for some approaches.
                            # Max value is 255!

            'category'    , # The name of the category that this label belongs to

            'categoryId'  , # The ID of this category. Used to create ground truth images
                            # on category level.

            'hasInstances', # Whether this label distinguishes between single instances or not

            'ignoreInEval', # Whether pixels having this class as ground truth label are ignored
                            # during evaluations or not

            'color'       , # The color of this label
            ] )
        
def create_cityscapes_label_colormap():
        """Creates a label colormap used in CITYSCAPES segmentation benchmark.
        Returns:
            A colormap for visualizing segmentation results.
        """
        
        labels = [
        #       name                     id    trainId   category            catId     hasInstances   ignoreInEval   color
        Label(  'unlabeled'            ,  0 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
        Label(  'ego vehicle'          ,  1 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
        Label(  'rectification border' ,  2 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
        Label(  'out of roi'           ,  3 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
        Label(  'static'               ,  4 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
        Label(  'dynamic'              ,  5 ,      255 , 'void'            , 0       , False        , True         , (111, 74,  0) ),
        Label(  'ground'               ,  6 ,      255 , 'void'            , 0       , False        , True         , ( 81,  0, 81) ),
        Label(  'road'                 ,  7 ,        0 , 'flat'            , 1       , False        , False        , (128, 64,128) ),
        Label(  'sidewalk'             ,  8 ,        1 , 'flat'            , 1       , False        , False        , (244, 35,232) ),
        Label(  'parking'              ,  9 ,      255 , 'flat'            , 1       , False        , True         , (250,170,160) ),
        Label(  'rail track'           , 10 ,      255 , 'flat'            , 1       , False        , True         , (230,150,140) ),
        Label(  'building'             , 11 ,        2 , 'construction'    , 2       , False        , False        , ( 70, 70, 70) ),
        Label(  'wall'                 , 12 ,        3 , 'construction'    , 2       , False        , False        , (102,102,156) ),
        Label(  'fence'                , 13 ,        4 , 'construction'    , 2       , False        , False        , (190,153,153) ),
        Label(  'guard rail'           , 14 ,      255 , 'construction'    , 2       , False        , True         , (180,165,180) ),
        Label(  'bridge'               , 15 ,      255 , 'construction'    , 2       , False        , True         , (150,100,100) ),
        Label(  'tunnel'               , 16 ,      255 , 'construction'    , 2       , False        , True         , (150,120, 90) ),
        Label(  'pole'                 , 17 ,        5 , 'object'          , 3       , False        , False        , (153,153,153) ),
        Label(  'polegroup'            , 18 ,      255 , 'object'          , 3       , False        , True         , (153,153,153) ),
        Label(  'traffic light'        , 19 ,        6 , 'object'          , 3       , False        , False        , (250,170, 30) ),
        Label(  'traffic sign'         , 20 ,        7 , 'object'          , 3       , False        , False        , (220,220,  0) ),
        Label(  'vegetation'           , 21 ,        8 , 'nature'          , 4       , False        , False        , (107,142, 35) ),
        Label(  'terrain'              , 22 ,        9 , 'nature'          , 4       , False        , False        , (152,251,152) ),
        Label(  'sky'                  , 23 ,       10 , 'sky'             , 5       , False        , False        , ( 70,130,180) ),
        Label(  'person'               , 24 ,       11 , 'human'           , 6       , True         , False        , (220, 20, 60) ),
        Label(  'rider'                , 25 ,       12 , 'human'           , 6       , True         , False        , (255,  0,  0) ),
        Label(  'car'                  , 26 ,       13 , 'vehicle'         , 7       , True         , False        , (  0,  0,142) ),
        Label(  'truck'                , 27 ,       14 , 'vehicle'         , 7       , True         , False        , (  0,  0, 70) ),
        Label(  'bus'                  , 28 ,       15 , 'vehicle'         , 7       , True         , False        , (  0, 60,100) ),
        Label(  'caravan'              , 29 ,      255 , 'vehicle'         , 7       , True         , True         , (  0,  0, 90) ),
        Label(  'trailer'              , 30 ,      255 , 'vehicle'         , 7       , True         , True         , (  0,  0,110) ),
        Label(  'train'                , 31 ,       16 , 'vehicle'         , 7       , True         , False        , (  0, 80,100) ),
        Label(  'motorcycle'           , 32 ,       17 , 'vehicle'         , 7       , True         , False        , (  0,  0,230) ),
        Label(  'bicycle'              , 33 ,       18 , 'vehicle'         , 7       , True         , False        , (119, 11, 32) ),
        Label(  'license plate'        , -1 ,       -1 , 'vehicle'         , 7       , False        , True         , (  0,  0,142) ),
        ]
        return labels

def create_mapillary_vistas_label_colormap():
    """Creates a label colormap used in Mapillary Vistas segmentation benchmark.
    Returns:
        A list of labels for visualizing segmentation results.
    """

    labels = [
        Label('Bird',                   0,      0, 'animal',             0,      True,         False,        (165, 42, 42)),
        Label('Ground Animal',          1,      1, 'animal',             0,      True,         False,        (0, 192, 0)),
        Label('Curb',                   2,      2, 'construction',       1,      False,        False,        (196, 196, 196)),
        Label('Fence',                  3,      3, 'construction',       1,      False,        False,        (190, 153, 153)),
        Label('Guard Rail',             4,      4, 'construction',       1,      False,        False,        (180, 165, 180)),
        Label('Barrier',                5,      5, 'construction',       1,      False,        False,        (102, 102, 156)),
        Label('Wall',                   6,      6, 'construction',       1,      False,        False,        (102, 102, 156)),
        Label('Bike Lane',              7,      7, 'flat',               2,      False,        False,        (128, 64, 255)),
        Label('Crosswalk - Plain',      8,      8, 'flat',               2,      False,        False,        (140, 140, 200)),
        Label('Curb Cut',               9,      9, 'flat',               2,      False,        False,        (170, 170, 170)),
        Label('Parking',               10,     10, 'flat',               2,      False,        False,        (250, 170, 160)),
        Label('Pedestrian Area',       11,     11, 'flat',               2,      False,        False,        (96, 96, 96)),
        Label('Rail Track',            12,     12, 'flat',               2,      False,        False,        (230, 150, 140)),
        Label('Road',                  13,     13, 'flat',               2,      False,        False,        (128, 64, 128)),
        Label('Service Lane',          14,     14, 'flat',               2,      False,        False,        (110, 110, 110)),
        Label('Sidewalk',              15,     15, 'flat',               2,      False,        False,        (244, 35, 232)),
        Label('Bridge',                16,     16, 'construction',       1,      False,        False,        (150, 100, 100)),
        Label('Building',              17,     17, 'construction',       1,      False,        False,        (70, 70, 70)),
        Label('Tunnel',                18,     18, 'construction',       1,      False,        False,        (150, 120, 90)),
        Label('Person',                19,     19, 'human',              3,      True,         False,        (220, 20, 60)),
        Label('Bicyclist',             20,     20, 'human',              3,      True,         False,        (255, 0, 0)),
        Label('Motorcyclist',          21,     21, 'human',              3,      True,         False,        (255, 0, 0)),
        Label('Other Rider',           22,     22, 'human',              3,      True,         False,        (255, 0, 0)),
        Label('Lane Marking - Crosswalk',23,  23, 'marking',            4,      False,        True,         (200, 128, 128)),
        Label('Lane Marking - General',24,     24, 'marking',            4,      True,         False,        (255, 255, 255)),
        Label('Mountain',              25,     25, 'nature',             5,      False,        False,        (64, 170, 64)),
        Label('Sand',                  26,     26, 'nature',             5,      False,        False,        (230, 160, 50)),
        Label('Sky',                   27,     27, 'sky',                6,      False,        False,        (70, 130, 180)),
        Label('Snow',                  28,     28, 'nature',             5,      False,        False,        (190, 255, 255)),
        Label('Terrain',               29,     29, 'nature',             5,      False,        False,        (152, 251, 152)),
        Label('Vegetation',            30,     30, 'nature',             5,      False,        False,        (107, 142, 35)),
        Label('Water',                 31,     31, 'water',              7,      False,        False,        (0, 170, 30)),
        Label('Banner',                32,     32, 'object',             8,      False,        False,        (255, 220, 0)),
        Label('Bench',                 33,     33, 'object',             8,      False,        False,        (255, 0, 0)),
        Label('Bike Rack',             34,     34, 'object',             8,      False,        False,        (255, 0, 0)),
        Label('Billboard',             35,     35, 'object',             8,      False,        False,        (255, 0, 0)),
        Label('Catch Basin',           36,     36, 'object',             8,      False,        False,        (255, 0, 0)),
        Label('CCTV Camera',           37,     37, 'object',             8,      False,        False,        (255, 0, 0)),
        Label('Fire Hydrant',          38,     38, 'object',             8,      False,        False,        (255, 0, 0)),
        Label('Junction Box',          39,     39, 'object',             8,      False,        False,        (255, 0, 0)),
        Label('Mailbox',               40,     40, 'object',             8,      False,        False,        (255, 0, 0)),
        Label('Manhole',               41,     41, 'object',             8,      False,        False,        (255, 0, 0)),
        Label('Phone Booth',           42,     42, 'object',             8,      False,        False,        (255, 0, 0)),
        Label('Pothole',               43,     43, 'object',             8,      False,        False,        (255, 0, 0)),
        Label('Street Light',          44,     44, 'object',             8,      False,        False,        (255, 0, 0)),
        Label('Pole',                  45,     45, 'object',             8,      False,        False,        (255, 0, 0)),
        Label('Traffic Sign Frame',    46,     46, 'object',             8,      False,        False,        (255, 0, 0)),
        Label('Utility Pole',          47,     47, 'object',             8,      False,        False,        (255, 0, 0)),
        Label('Traffic Light',         48,     48, 'object',             8,      False,        False,        (255, 0, 0)),
        Label('Traffic Sign (Back)',   49,     49, 'object',             8,      False,        False,        (255, 0, 0)),
        Label('Traffic Sign (Front)',  50,     50, 'object',             8,      False,        False,        (255, 0, 0)),
        Label('Trash Can',             51,     51, 'object',             8,      False,        False,        (255, 0, 0)),
        Label('Bicycle',               52,     52, 'vehicle',            9,      True,         False,        (119, 11, 32)),
        Label('Boat',                  53,     53, 'vehicle',            9,      False,        False,        (0, 0, 142)),
        Label('Bus',                   54,     54, 'vehicle',            9,      True,         False,        (0, 60, 100)),
        Label('Car',                   55,     55, 'vehicle',            9,      True,         False,        (0, 0, 142)),
        Label('Caravan',               56,     56, 'vehicle',            9,      True,         False,        (0, 0, 90)),
        Label('Motorcycle',            57,     57, 'vehicle',            9,      True,         False,        (0, 0, 230)),
        Label('On Rails',              58,     58, 'vehicle',            9,      False,        False,        (0, 80, 100)),
        Label('Other Vehicle',         59,     59, 'vehicle',            9,      True,         False,        (128, 64, 64)),
        Label('Trailer',               60,     60, 'vehicle',            9,      True,         False,        (0, 0, 110)),
        Label('Truck',                 61,     61, 'vehicle',            9,      True,         False,        (0, 0, 70)),
        Label('Wheeled Slow',          62,     62, 'vehicle',            9,      False,        False,        (0, 0, 192)),
        Label('Car Mount',             63,     63, 'vehicle',            9,      True,         False,        (32, 32, 32)),
        Label('Ego Vehicle',           64,     64, 'vehicle',            9,      True,         False,        (120, 10, 10))
    ]
    return labels

def get_resized_dimensions(image: cv2.Mat, max_size: int = 2048) -> Tuple[int, int]:
    """
    Get the resized dimensions of an image based on a maximum size.

    Args:
        image (cv2.Mat): Input image in OpenCV format.
        max_size (int, optional): Maximum size for the larger side of the image. Defaults to 2048.

    Returns:
        Tuple[int, int]: Tuple containing new height and width.
    """
    height, width = image.shape[:2]

    # Determine the larger side
    larger_side = max(height, width)

    # Check if the larger side exceeds the maximum size
    if larger_side > max_size:
        # Calculate the scaling factor
        scaling_factor = max_size / larger_side

        # Calculate the new dimensions
        new_width = int(width * scaling_factor)
        new_height = int(height * scaling_factor)

        return (new_height, new_width)

    return (height, width)

class ImageDataset(Dataset):
    """
    Custom Dataset class for images.
    """

    def __init__(self, image_files: List[Path], rgb: bool = True) -> None:
        """
        Initialize the ImageDataset.

        Args:
            image_files (List[Path]): List of image files.
            rgb (bool, optional): Flag to convert images to RGB. Defaults to True.
        """
        self.image_files = image_files
        self.rgb = rgb

    def __len__(self) -> int:
        """
        Get the length of the dataset.

        Returns:
            int: Number of image files in the dataset.
        """
        return len(self.image_files)

    def __getitem__(self, idx: int) -> Tuple[str, cv2.Mat, Tuple[int, int]]:
        """
        Get an item from the dataset.

        Args:
            idx (int): Index of the image.

        Returns:
            Tuple[str, cv2.Mat, Tuple[int, int]]: Tuple containing image file path, image data, and original image dimensions.
        """
        image_file = self.image_files[idx]
        img = cv2.imread(str(image_file))
        original_img_shape = get_resized_dimensions(img)
        if self.rgb:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return str(image_file), img, original_img_shape

    def collate_fn(self, data: List[Tuple[str, cv2.Mat, Tuple[int, int]]]) -> Tuple[List[str], List[cv2.Mat], List[Tuple[int, int]]]:
        """
        Custom collate function for the dataset.

        Args:
            data (List[Tuple[str, cv2.Mat, Tuple[int, int]]]): List of tuples containing image file path, image data, and original image dimensions.

        Returns:
            Tuple[List[str], List[cv2.Mat], List[Tuple[int, int]]]: Tuple containing lists of image file paths, image data, and original image dimensions.
        """
        image_files, images, original_img_shape = zip(*data)
        return list(image_files), list(images), list(original_img_shape)

class Segmenter:
    def __init__(self, dataset="cityscapes", task = "semantic"):
        """
        Initialize the Segmenter with a model and dataset.

        Args:
            model_name (str): The name of the pre-trained model.
            dataset (str): The name of the dataset.
        """
        self.device = self._get_device()
        self.dataset = dataset
        self.task = task
        self.model_name = self._get_model_name(self.dataset, self.task)
        self.model, self.processor = self._get_model_processor(self.model_name)
        self.color_map = self._create_color_map(dataset)
        self.label_map = self._create_label_map(dataset)
        self.id_to_name_map = self._create_id_to_name_map(dataset)
    
    def _get_model_name(self, dataset, task):
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
        """
        Get the model and processor for the given model name.

        Args:
            model_name (str): The name of the pre-trained model.

        Returns:
            Tuple: The model and processor.
        """
        # Add other models in the future
        if "mask2former" in model_name:
            processor = AutoImageProcessor.from_pretrained(model_name)
            model = Mask2FormerForUniversalSegmentation.from_pretrained(model_name).to(self.device)
        return model, processor

    def _create_color_map(self, dataset):
        """
        Create a color map based on the given dataset.

        Args:
            dataset (str): The name of the dataset.

        Returns:
            np.ndarray: A color map.
        """
        if dataset == "cityscapes":
            labels = create_cityscapes_label_colormap()
        elif dataset == "mapillary":
            labels = create_mapillary_vistas_label_colormap()
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
        """
        Create a label map based on the given dataset.

        Args:
            dataset (str): The name of the dataset.

        Returns:
            Dict[Tuple, Label]: A dictionary mapping colors to labels.
        """
        if dataset == "cityscapes":
            labels = create_cityscapes_label_colormap()
        elif dataset == "mapillary":
            labels = create_mapillary_vistas_label_colormap()
        else:
            raise ValueError(f"Unknown dataset: {dataset}")

        color_to_label = {}
        for label in labels:
            color = label.color
            color_to_label[color] = label

        return color_to_label

    def _create_id_to_name_map(self, dataset):
        if dataset == "cityscapes":
            labels = create_cityscapes_label_colormap()
        elif dataset == "mapillary":
            labels = create_mapillary_vistas_label_colormap()
        return {label.trainId: label.name for label in labels}
    
    def _get_device(self) -> torch.device:
        """
        Get the appropriate device for running the model.

        Returns:
            torch.device: The device to use for running the model.
        """
        if torch.cuda.is_available():
            print("Using GPU")
            return torch.device("cuda")
        else:
            print("Using CPU")
            return torch.device("cpu")

    def _calculate_pixel_ratios(self, segmented_img):
        """
        Calculate pixel ratios for each class in the segmented image.

        Args:
            segmented_img (numpy.ndarray): Segmented image.

        Returns:
            dict: A dictionary with class names as keys and pixel ratios as values.
        """
        unique, counts = np.unique(segmented_img, return_counts=True)
        total_pixels = np.sum(counts)
        pixel_ratios = {self.train_id_to_name[train_id]: count / total_pixels for train_id, count in zip(unique, counts)}

        return pixel_ratios

    def _save_as_csv(self, input_dict, dir_output, value_name, csv_format):
        if csv_format == "long":
            df_list = [pd.DataFrame({'filename_key': [filename_key], 'label_name': [key], value_name: [value] if value is not None else [0]}) 
                    for filename_key, inner_dict in input_dict.items() 
                    for key, value in inner_dict.items()]

            pixel_ratios_df = pd.concat(df_list, ignore_index=True)

        elif csv_format == "wide":
            pixel_ratios_df = pd.DataFrame(input_dict).transpose().fillna(0)
            pixel_ratios_df.index.names = ["filename_key"]

        pixel_ratios_df.to_csv(dir_output / Path(value_name + ".csv"))


    def _panoptic_segmentation(self, images, original_img_shape):
        """
        Perform panoptic segmentation on the given images.

        Args:
            images (list): List of input images.
            original_img_shape (tuple): Original image shape.

        Returns:
            list: List of panoptic segmentation outputs.
        """
        inputs = self.processor(images=images, task_inputs=["panoptic"], return_tensors="pt").to(self.model.device)
        outputs = self.model(**inputs)
        return self.processor.post_process_panoptic_segmentation(outputs, target_sizes=original_img_shape, label_ids_to_fuse = set([]))

    def _semantic_segmentation(self, images, original_img_shape):
        """
        Perform semantic segmentation on the given images.

        Args:
            images (list): List of input images.
            original_img_shape (tuple): Original image shape.

        Returns:
            tuple: Tuple containing list of semantic segmentation outputs and list of pixel ratios.
        """
        inputs = self.processor(images=images, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        segmentations = self.processor.post_process_semantic_segmentation(outputs, target_sizes=original_img_shape)
        return segmentations

    def _trainid_to_color(self, segmented_img):
        """
        Convert segmented image with train IDs to a colored image.

        Args:
            segmented_img (numpy.ndarray): Segmented image with train IDs.

        Returns:
            numpy.ndarray: Colored segmented image.
        """
        colored_img = self.color_map[segmented_img]
        return colored_img

    def _save_panoptic_segmentation_image(self, image_file, img, dir_output, output):
        """
        Save the panoptic segmentation image as a blended image with the original input image.

        Args:
            image_file (str): The input image file path.
            img (np.array): The input image in the format of a NumPy array.
            dir_output (Path): The output directory path to save the blended image.
            output (dict): The output dictionary containing the segmentation data.
        """
        colored_segmented_img = self._trainid_to_color(output['label_segmentation'].cpu().numpy())
        alpha = 0.5
        blend_img = cv2.addWeighted(img, alpha, colored_segmented_img, 1 - alpha, 0)

        # Add annotations for each segment
        for segment_info in output['segments_info']:
            segment_id = segment_info['id']
            label_id = segment_info['label_id']
            score = segment_info['score']
            
            # Use the label name instead of the label_id
            label_name = self.id_to_name_map.get(label_id)
            
            # Find the center of the segment for the label placement
            y, x = np.where(output['segmentation'].cpu().numpy() == segment_id)
            center_x, center_y = np.mean(x), np.mean(y)
            
            # Add the annotation
            text = f"{label_name}-{score:.2f}"
            cv2.putText(blend_img, text, (int(center_x), int(center_y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        output_file = dir_output / Path(image_file).name
        
        # Save images based on specified options
        if "segmented_image" in self.save_image_options:
            cv2.imwrite(str(output_file.with_name(output_file.stem + "_colored_segmented.png")), cv2.cvtColor(colored_segmented_img, cv2.COLOR_RGB2BGR))
        if "blend_image" in self.save_image_options:
            cv2.imwrite(str(output_file.with_name(output_file.stem + "_blend.png")), cv2.cvtColor(blend_img, cv2.COLOR_RGB2BGR))
        
    def _save_semantic_segmentation_image(self, image_file, img, dir_output, output):
        """
        Save the semantic segmentation image as a colored segmented image and/or a blended image with the original input image.

        Args:
            image_file (str): The input image file path.
            img (np.array): The input image in the format of a NumPy array.
            dir_output (Path): The output directory path to save the colored segmented and/or blended image.
            output (Tensor): The output tensor containing the semantic segmentation data.
        """
        colored_segmented_img = self._trainid_to_color(output.cpu().numpy())
        alpha = 0.5
        blend_img = cv2.addWeighted(img, alpha, colored_segmented_img, 1 - alpha, 0)

        output_file = dir_output / Path(image_file).name
        
        # Save images based on specified options
        if "segmented_image" in self.save_image_options:
            cv2.imwrite(str(output_file.with_name(output_file.stem + "_colored_segmented.png")), cv2.cvtColor(colored_segmented_img, cv2.COLOR_RGB2BGR))
        if "blend_image" in self.save_image_options:
            cv2.imwrite(str(output_file.with_name(output_file.stem + "_blend.png")), cv2.cvtColor(blend_img, cv2.COLOR_RGB2BGR))

    def _panoptic_count_labels(self, output):
        label_counts = {}

        # Loop through each segment in the image
        segments_info_list = output['segments_info']
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
        """
        This function converts the output of post_process_panoptic_segmentation function
        from segment_id to label_id.
        :param output: The output dictionary from the post_process_panoptic_segmentation function
        :return: segmentation with label_ids instead of segment_ids
        """
        # Extract the segmentation and segments_info from the output
        segmentation = output['segmentation']
        segments_info = output['segments_info']

        # Create a mapping from segment_id to label_id
        id_map = {segment['id']: segment['label_id'] for segment in segments_info}

        # Use the map to convert the segmentation tensor from segment_ids to label_ids
        new_segmentation = segmentation.clone()

        for seg_id, label_id in id_map.items():
            new_segmentation[segmentation == seg_id] = label_id

        return new_segmentation

    def _process_images(self, task, image_files, images, dir_output, pixel_ratio_dict, original_img_shape, panoptic_dict=None):
        """
        Process the input images for segmentation and save the output images.

        Args:
            task (str): The segmentation task to perform, either "panoptic" or "semantic".
            image_files (List[str]): The list of file paths of the input images.
            images (List[ndarray]): The list of input images in the form of numpy arrays.
            dir_output (Path): The output directory where the segmented images will be saved.
            pixel_ratio_dict (defaultdict): A dictionary to store the pixel ratios of the segmented images.
            original_img_shape (List[Tuple[int, int]]): The original shapes of the input images.

        Returns:
            None
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
    def segment(self, dir_input: Union[str, Path], 
                dir_image_output: Union[str, Path, None] = None, 
                dir_segmentation_summary_output: Union[str, Path, None] = None, 
                batch_size=1, 
                save_image_options = ["segmented_image", "blend_image"], 
                pixel_ratio_save_format = ["json", "csv"],
                csv_format = "long", # "long" or "wide"
                max_workers: Union[int, None] = None):
        # make sure that at least one of dir_image_output and dir_segmentation_summary_output is not None
        if (dir_image_output is None) & (dir_segmentation_summary_output is None):
            raise ValueError("At least one of dir_image_output and dir_segmentation_summary_output must not be None.")
        
        # save_image_options as a property of the class
        self.save_image_options = save_image_options
        
        # make directory
        dir_input = Path(dir_input)
        if dir_image_output is not None:
            dir_image_output = Path(dir_image_output)
            dir_image_output.mkdir(parents=True, exist_ok=True)
        if dir_segmentation_summary_output is not None:
            dir_segmentation_summary_output = Path(dir_segmentation_summary_output)
            dir_segmentation_summary_output.mkdir(parents=True, exist_ok=True)
            # Create a new directory called "pixel_ratio_checkpoints"
            dir_cache_segmentation_summary = dir_segmentation_summary_output / 'pixel_ratio_checkpoints'
            dir_cache_segmentation_summary.mkdir(parents=True, exist_ok=True)

            # Load all the checkpoint json files
            checkpoints = glob.glob(str(dir_cache_segmentation_summary / '*.json'))
            checkpoint_start_index = len(checkpoints)

            completed_image_files = set()  # completed_image_files will store the keys in the pixel_ratio_dict
            if checkpoint_start_index > 0:
                for checkpoint in checkpoints:
                    with open(checkpoint, 'r') as f:
                        checkpoint_dict = json.load(f)
                        completed_image_files.update(checkpoint_dict.keys())

        # Get the list of all image files and filter the ones that are not completed yet
        # List of possible image file extensions
        image_extensions = [".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".dib", ".pbm", ".pgm", ".ppm", ".sr", ".ras", ".exr", ".jp2"]

        # Get the list of all image files in the directory that are not completed yet
        image_file_list = [str(f) for f in Path(dir_input).iterdir() if f.suffix in image_extensions and f.stem not in completed_image_files]

        outer_batch_size = 1000  # Number of inner batches in one outer batch
        num_outer_batches = (len(image_file_list) + outer_batch_size * batch_size - 1) // (outer_batch_size * batch_size)

        for i in tqdm(range(num_outer_batches), desc=f"Processing outer batches of size {min(outer_batch_size * batch_size, len(image_file_list))}"):
            # Get the image files for the current outer batch
            outer_batch_image_file_list = image_file_list[i * outer_batch_size * batch_size : (i+1) * outer_batch_size * batch_size]

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
                        future = executor.submit(self._process_images, self.task, image_files, images, dir_image_output, pixel_ratio_dict, original_img_shape, panoptic_dict)
                    elif self.task == "semantic":
                        future = executor.submit(self._process_images, self.task, image_files, images, dir_image_output, pixel_ratio_dict, original_img_shape)
                    futures.append(future)

                for completed_future in tqdm(as_completed(futures), total=len(futures), desc=f"Processing outer batch #{i+1}"):
                    completed_future.result()

                # Save checkpoint for each outer batch
                with open(f'{dir_cache_segmentation_summary}/checkpoint_batch_{checkpoint_start_index+i+1}_pixel_ratio.json', 'w') as f:
                    json.dump(pixel_ratio_dict, f)
                    
                if self.task == "panoptic":
                    with open(f'{dir_cache_segmentation_summary}/checkpoint_batch_{checkpoint_start_index+i+1}_panoptic.json', 'w') as f:
                        json.dump(panoptic_dict, f)

        # Merge all checkpoints into a single pixel_ratio_dict
        pixel_ratio_dict = defaultdict(dict)
        for checkpoint in glob.glob(str(dir_cache_segmentation_summary / '*_pixel_ratio.json')):
            with open(checkpoint, 'r') as f:
                checkpoint_dict = json.load(f)
                for key, value in checkpoint_dict.items():
                    pixel_ratio_dict[key] = value
        
        # Merge all checkpoints into a single panoptic_dict
        if self.task == "panoptic":
            panoptic_dict = defaultdict(dict)
            for checkpoint in glob.glob(str(dir_cache_segmentation_summary / '*_panoptic.json')):
                with open(checkpoint, 'r') as f:
                    checkpoint_dict = json.load(f)
                    for key, value in checkpoint_dict.items():
                        panoptic_dict[key] = value

        # Save pixel_ratio_dict as a JSON or CSV file
        if "json" in pixel_ratio_save_format:
            with open(dir_segmentation_summary_output / "pixel_ratios.json", "w") as f:
                json.dump(pixel_ratio_dict, f)
            if self.task == "panoptic":
                with open(dir_segmentation_summary_output / "label_counts.json", "w") as f:
                    json.dump(panoptic_dict, f)
        if "csv" in pixel_ratio_save_format:
            self._save_as_csv(pixel_ratio_dict, dir_segmentation_summary_output, "pixel_ratios", csv_format)
            if self.task == "panoptic":
                self._save_as_csv(panoptic_dict, dir_segmentation_summary_output, "label_counts", csv_format)
                
            
        # Delete the "pixel_ratio_checkpoints" directory
        if dir_segmentation_summary_output is not None:
            shutil.rmtree(dir_cache_segmentation_summary)

            
    def calculate_pixel_ratio_post_process(self, dir_input, dir_output, pixel_ratio_save_format = ["json", "csv"]):
        """
        Calculates the pixel ratio of different classes present in the segmented images and saves the results in either JSON or CSV format.

        Args:
            dir_input: A string or Path object representing the input directory containing the segmented images.
            dir_output: A string or Path object representing the output directory where the pixel ratio results will be saved.
            pixel_ratio_save_format: A list containing the file formats in which the results will be saved. The allowed file formats are "json" and "csv". The default value is ["json", "csv"].

        Returns:
            None
        """
        def calculate_label_ratios(image, label_map):
            """
            Calculates the pixel ratio of different classes present in a single image.

            Args:
                image: A numpy array representing an image.
                label_map: A dictionary containing the label names and their respective RGB colors.

            Returns:
            A dictionary containing the pixel ratio of different classes in the given image.
            """
            label_ratios = {}
            total_pixels = image.shape[0] * image.shape[1]

            for color, label in label_map.items():
                color_pixels = np.count_nonzero(np.all(image == color, axis=-1))
                label_ratios[label.name] = color_pixels / total_pixels

            return label_ratios

        def process_image_file(image_file, label_map):
            """
            Calculates the pixel ratio of different classes in a single segmented image file.

            Args:
                image_file: A Path object representing the segmented image file.
                label_map: A dictionary containing the label names and their respective RGB colors.

            Returns:
                A tuple containing the image file key and the pixel ratio of different classes in the given image.
            """
            image_file_key = str(Path(image_file).stem).replace("_colored_segmented", "")
            image = cv2.imread(str(image_file))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            label_ratios = calculate_label_ratios(image, label_map)
            return image_file_key, label_ratios

        def results_to_dataframe(results):
            """
            Converts the results obtained from processing each image file into a Pandas DataFrame.

            Args:
                results: A list of tuples, where each tuple contains the image file key and the pixel ratio of different classes in the corresponding image.

            Returns:
                A Pandas DataFrame containing the pixel ratios of different classes in each image file.
            """
            pixel_ratio_dict = {}

            for image_file_key, label_ratios in results:
                pixel_ratio_dict[str(image_file_key)] = label_ratios

            pixel_ratios_df = pd.DataFrame(pixel_ratio_dict).transpose()
            pixel_ratios_df.fillna(0, inplace=True)
            pixel_ratios_df.index.names = ["filename_key"]

            return pixel_ratios_df
        
        def results_to_nested_dict(results):
            """
            Converts the results obtained from processing each image file into a nested dictionary.

            Args:
                results: A list of tuples, where each tuple contains the image file key and the pixel ratio of different classes in the corresponding image.

            Returns:
                A nested dictionary containing the pixel ratios of different classes in each image file.
            """
            data = {}

            for image_file_key, label_ratios in results:
                image_file_key = str(image_file_key)
                data[image_file_key] = label_ratios

            return data

        # get files
        if isinstance(dir_input, str):
            dir_input = Path(dir_input)

        # Set image file extensions
        image_extensions = ['.jpg', '.png']

        image_files = [file for file in dir_input.rglob('*') if file.suffix.lower() in image_extensions and '_colored_segmented' in file.stem]

        results = thread_map(process_image_file, image_files, [self.label_map] * len(image_files))

        if "json" in pixel_ratio_save_format:
            json_output_file = Path(dir_output) / 'pixel_ratio.json'
            nested_dict = results_to_nested_dict(results)
            with open(json_output_file, 'w') as f:
                json.dump(nested_dict, f, indent=2) 
                
        if "csv" in pixel_ratio_save_format:
            csv_output_file = Path(dir_output) / 'pixel_ratio.csv'
            df = results_to_dataframe(results)
            df.to_csv(csv_output_file)