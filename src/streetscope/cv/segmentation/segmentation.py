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

def create_cityscapes_label_colormap():
        """Creates a label colormap used in CITYSCAPES segmentation benchmark.
        Returns:
            A colormap for visualizing segmentation results.
        """
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
    Label = namedtuple('Label', [
        'name',
        'id',
        'trainId',
        'category',
        'categoryId',
        'hasInstances',
        'ignoreInEval',
        'color',
    ])

    labels = [
        #       name                id  trainId  category            catId  hasInstances  ignoreInEval  color
        Label('Bird',               0,      0, 'animal',             0,      True,         False,        (165, 42, 42)),
        Label('Ground Animal',      1,      1, 'animal',             0,      True,         False,        (0, 192, 0)),
        Label('Curb',               2,      2, 'construction',       1,      False,        False,        (196, 196, 196)),
        Label('Fence',              3,      3, 'construction',       1,      False,        False,        (190, 153, 153)),
        Label('Guard Rail',         4,      4, 'construction',       1,      False,        False,        (180, 165, 180)),
        Label('Barrier',            5,      5, 'construction',       1,      False,        False,        (102, 102, 156)),
        Label('Wall',               6,      6, 'construction',       1,      False,        False,        (102, 102, 156)),
        Label('Bike Lane',          7,      7, 'flat',               2,      False,        False,        (128, 64, 255)),
        Label('Crosswalk - Plain',  8,      8, 'flat',               2,      False,        False,        (140, 140, 200)),
        Label('Curb Cut',           9,      9, 'flat',               2,      False,        False,        (170, 170, 170)),
        Label('Parking',           10,     10, 'flat',               2,      False,        False,        (250, 170, 160)),
        Label('Pedestrian Area',   11,     11, 'flat',               2,      False,        False,        (96, 96, 96)),
        Label('Rail Track',        12,     12, 'flat',               2,      False,        False,        (230, 150, 140)),
        Label('Road',              13,     13, 'flat',               2,      False,        False,        (128, 64, 128)),
        Label('Service Lane',      14,     14, 'flat',               2,      False,        False,        (110, 110, 110)),
        Label('Sidewalk',          15,     15, 'flat',               2,      False,        False,        (244, 35, 232)),
        Label('Bridge',            16,     16, 'construction',       1,      False,        False,        (150, 100, 100)),
        Label('Building',           17,     17, 'construction',       1,      False,        False,        (70, 70, 70)),
        Label('Tunnel',             18,     18, 'construction',       1,      False,        False,        (150, 120, 90)),
        Label('Person',             19,     19, 'human',              3,      True,         False,        (220, 20, 60)),
        Label('Bicyclist',          20,     20, 'human',              3,      True,         False,        (255, 0, 0)),
        Label('Motorcyclist',       21,     21, 'human',              3,      True,         False,        (255, 0, 0)),
        Label('Other Rider',        22,     22, 'human',              3,      True,         False,        (255, 0, 0)),
        Label('Lane Marking - Crosswalk', 23, 23, 'marking',         4,      False,        False,        (200, 128, 128)),
        Label('Lane Marking - General', 24, 24, 'marking',           4,      False,        False,        (255, 255, 255)),
        Label('Mountain',           25,     25, 'nature',             5,      False,        False,        (64, 170, 64)),
        Label('Sand',               26,     26, 'nature',             5,      False,        False,        (230, 160, 50)),
        Label('Sky',                27,     27, 'nature',             5,      False,        False,        (70, 130, 180)),
        Label('Snow',               28,     28, 'nature',             5,      False,        False,        (190, 255, 255)),
        Label('Terrain',            29,     29, 'nature',             5,      False,        False,        (152, 251, 152)),
        Label('Vegetation',         30,     30, 'nature',             5,      False,        False,        (107, 142, 35)),
        Label('Water',              31,     31, 'nature',             5,      False,        False,        (0, 170, 30)),
        Label('Banner',             32,     32, 'object',             6,      False,        False,        (255, 255, 128)),
        Label('Bench',              33,     33, 'object',             6,      False,        False,        (255, 255, 128)),
        Label('Bike Rack',          34,     34, 'object',             6,      False,        False,        (255, 255, 128)),
        Label('Billboard',          35,     35, 'object',             6,      False,        False,        (255, 255, 128)),
        Label('Catch Basin',        36,     36, 'object',             6,      False,        False,        (255, 255, 128)),
        Label('CCTV Camera',        37,     37, 'object',             6,      False,        False,        (255, 255, 128)),
        Label('Fire Hydrant',       38,     38, 'object',             6,      False,        False,        (255, 255, 128)),
        Label('Junction Box',       39,     39, 'object',             6,      False,        False,        (255, 255, 128)),
        Label('Mailbox',            40,     40, 'object',             6,      False,        False,        (255, 255, 128)),
        Label('Manhole',            41,     41, 'object',             6,      False,        False,        (255, 255, 128)),
        Label('Phone Booth',        42,     42, 'object',             6,      False,        False,        (255, 255, 128)),
        Label('Pothole',            43,     43, 'object',             6,      False,        False,        (255, 255, 128)),
        Label('Street Light',       44,     44, 'object',             6,      False,        False,        (255, 255, 128)),
        Label('Traffic Cone',       45,     45, 'object',             6,      False,        False,        (255, 255, 128)),
        Label('Traffic Device',     46,     46, 'object',             6,      False,        False,        (255, 255, 128)),
        Label('Traffic Light',      47,     47, 'object',             6,      False,        False,        (255, 255, 128)),
        Label('Traffic Sign',       48,     48, 'object',             6,      False,        False,        (255, 255, 128)),
        Label('Traffic Sign Frame', 49,     49, 'object',             6,      False,        False,        (255, 255, 128)),
        Label('Trash Can',          50,     50, 'object',             6,      False,        False,        (255, 255, 128)),
        Label('Bicycle',            51,     51, 'vehicle',            7,      True,         False,        (119, 11, 32)),
        Label('Boat',               52,     52, 'vehicle',            7,      True,         False,        (0, 0, 142)),
        Label('Bus',                53,     53, 'vehicle',            7,      True,         False,        (0, 60, 100)),
        Label('Car',                54,     54, 'vehicle',            7,      True,         False,        (0, 0, 142)),
        Label('Caravan',            55,     55, 'vehicle',            7,      True,         False,        (0, 0, 90)),
        Label('Motorcycle',         56,     56, 'vehicle',            7,      True,         False,        (0, 0, 230)),
        Label('On Rails',           57,     57, 'vehicle',            7,      True,         False,        (0, 80, 100)),
        Label('Other Vehicle',      58,     58, 'vehicle',            7,      True,         False,        (128, 64, 128)),
        Label('Trailer',            59,     59, 'vehicle',            7,      True,         False,        (0, 0, 110)),
        Label('Truck',              60,     60, 'vehicle',            7,      True,         False,        (0, 0, 70)),
        Label('Wheeled Slow',       61,     61, 'vehicle',            7,      True,         False,        (0, 0, 192)),
        Label('Car Mount',          62,     62, 'vehicle',            7,      False,        False,        (32, 32, 32)),
        Label('Ego Vehicle',        63,     63, 'vehicle',            7,      False,        False,        (120, 10, 10))
    ]

    return labels

def get_resized_dimensions(image, max_size=2048):
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
    def __init__(self, dir_input: Union[str, Path], rgb = True) -> None:
        self.dir_input = Path(dir_input)
        self.image_files = list(self.dir_input.glob("*.jpg"))
        self.rgb = rgb

    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(self, idx: int) -> Tuple[str, cv2.Mat]:
        image_file = self.image_files[idx]
        img = cv2.imread(str(image_file))
        original_img_shape = get_resized_dimensions(img)
        if self.rgb:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # img = cv2.resize(img, (1024, 1024))
        return str(image_file), img, original_img_shape

    def collate_fn(self, data: List[Tuple[str, cv2.Mat]]) -> Tuple[List[str], torch.Tensor]:
        image_files, images, original_img_shape = zip(*data)
        # images = torch.stack([torch.from_numpy(img) for img in images])
        return list(image_files), list(images), list(original_img_shape)

class Segmenter:
    def __init__(self, model_name="facebook/mask2former-swin-tiny-cityscapes-semantic", dataset="cityscapes"):
        self.device = self._get_device()
        self.model, self.processor = self._get_model_processor(model_name)
        self.color_map = self._create_color_map(dataset)

    def _get_model_processor(self, model_name):
        #TODO add other models
        if "mask2former" in model_name:
            processor = AutoImageProcessor.from_pretrained(model_name)
            model = Mask2FormerForUniversalSegmentation.from_pretrained(model_name).to(self.device)
        return model, processor
    
    # Modify the _create_color_map method inside the Segmenter class
    def _create_color_map(self, dataset):
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

    def _get_device(self) -> torch.device:
        if torch.cuda.is_available():
            return torch.device("cuda")
        # elif torch.backends.mps.is_available():
        #     return torch.device("mps")
        else:
            return torch.device("cpu")

    def _calculate_pixel_ratios(self, segmented_img):
        unique, counts = np.unique(segmented_img, return_counts=True)
        total_pixels = np.sum(counts)
        pixel_ratios = {self.train_id_to_name[train_id]: count / total_pixels for train_id, count in zip(unique, counts)}

        return pixel_ratios
    
    # Add this method inside the Segmenter class
    def _save_pixel_ratios_as_csv(self, pixel_ratio_dict, dir_output):
        pixel_ratios_df = pd.DataFrame(pixel_ratio_dict).transpose()
        pixel_ratios_df.fillna(0, inplace=True)
        pixel_ratios_df.to_csv(dir_output / "pixel_ratios.csv")

        
    def _panoptic_segmentation(self, images, original_img_shape):
        inputs = self.processor(images=images, task_inputs=["panoptic"], return_tensors="pt").to(self.model.device)
        outputs = self.model(**inputs)
        return self.processor.post_process_panoptic_segmentation(outputs, target_sizes=original_img_shape)

    def _semantic_segmentation(self, images, original_img_shape):
        inputs = self.processor(images=images, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        # you can pass them to processor for postprocessing
        segmentations = self.processor.post_process_semantic_segmentation(outputs, target_sizes=original_img_shape)

        # Calculate pixel ratios
        pixel_ratios = [self._calculate_pixel_ratios(segmentation.cpu().numpy()) for segmentation in segmentations]

        return segmentations, pixel_ratios

    def _trainid_to_color(self, segmented_img):
        colored_img = self.color_map[segmented_img]
        return colored_img

    def _save_panoptic_segmentation_image(self, image_file, img, dir_output, output):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # img = img.transpose(1, 2, 0)  # Change img shape to (H, W, C)
        segmented_img = output["segmentation"].cpu().numpy()
        colored_segmented_img = self._trainid_to_color(segmented_img)
        # colored_segmented_img = colored_segmented_img.transpose(1, 2, 0)  # Change colored_segmented_img shape to (H, W, C)
        
        alpha = 0.5
        blend_img = cv2.addWeighted(img, alpha, colored_segmented_img, 1 - alpha, 0)

        output_file = dir_output / Path(image_file).name
        cv2.imwrite(str(output_file), cv2.cvtColor(blend_img, cv2.COLOR_RGB2BGR))
        
    def _save_semantic_segmentation_image(self, image_file, img, dir_output, output):
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        colored_segmented_img = self._trainid_to_color(output.cpu().numpy())
        # img = cv2.resize(img, (colored_segmented_img.shape[1], colored_segmented_img.shape[0]))
        alpha = 0.5
        blend_img = cv2.addWeighted(img, alpha, colored_segmented_img, 1 - alpha, 0)

        output_file = dir_output / Path(image_file).name
        # save images one by one
        if "segmented_image" in self.save_image_options:
            # save colored segmented image as "XXX_colored_segmented.jpg"
            cv2.imwrite(str(output_file.with_name(output_file.stem + "_colored_segmented" + output_file.suffix)), cv2.cvtColor(colored_segmented_img, cv2.COLOR_RGB2BGR))
        if "blend_image" in self.save_image_options:
            # save blended image as "XXX_blend.jpg"
            cv2.imwrite(str(output_file.with_name(output_file.stem + "_blend" + output_file.suffix)), cv2.cvtColor(blend_img, cv2.COLOR_RGB2BGR))

    def _save_segmentation_image(self, task, image_file, img, dir_output, output):
        if task == "panoptic":
            self._save_panoptic_segmentation_image(image_file, img, dir_output, output)
        elif task == "semantic":
            self._save_semantic_segmentation_image(image_file, img, dir_output, output)

    def _process_images(self, task, image_files, images, dir_output, pixel_ratio_dict, original_img_shape):
        outputs = None
        pixel_ratios = None
        if task == "panoptic":
            outputs = self._panoptic_segmentation(images, original_img_shape)
        elif task == "semantic":
            outputs, pixel_ratios = self._semantic_segmentation(images, original_img_shape)

        if outputs is not None:
            for image_file, img, output, pixel_ratio in zip(image_files, images, outputs, pixel_ratios):
                if len(self.save_image_options) > 0:
                    self._save_segmentation_image(task, image_file, img, dir_output, output)
                image_file_key = Path(image_file).stem
                pixel_ratio_dict[image_file_key] = pixel_ratio

    # Modify the segment method inside the Segmenter class
    def segment(self, dir_input: Union[str, Path], dir_output: Union[str, Path], task="semantic", batch_size=1, num_workers=0, save_image_options = ["segmented_image", "blend_image"], save_format="json"):
        # save_image_options as a property of the class
        self.save_image_options = save_image_options
        
        # make directory
        dir_input = Path(dir_input)
        dir_output = Path(dir_output)
        dir_output.mkdir(parents=True, exist_ok=True)

        dataset = ImageDataset(dir_input)
        dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=dataset.collate_fn, num_workers=num_workers)

        pixel_ratio_dict = defaultdict(dict)

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = []

            for batch in dataloader:
                image_files, images, original_img_shape = batch
                future = executor.submit(self._process_images, task, image_files, images, dir_output, pixel_ratio_dict, original_img_shape)
                futures.append(future)

            for completed_future in tqdm(as_completed(futures), total=len(futures), desc="Processing tasks"):
                completed_future.result()

        # Save pixel_ratio_dict as a JSON or CSV file
        if save_format == "json":
            with open(dir_output / "pixel_ratios.json", "w") as f:
                json.dump(pixel_ratio_dict, f)
        elif save_format == "csv":
            self._save_pixel_ratios_as_csv(pixel_ratio_dict, dir_output)
    
if __name__ == "__main__":
    segmentation = Segmenter()
    segmentation.segment("/Users/koichiito/Desktop/test2/panorama", "/Users/koichiito/Desktop/test2/panorama_segmented", batch_size=5, num_workers=5, save_image_options = ["segmented_image", "blend_image"], save_format="json")
    # segmentation = Segmenter(model_name="facebook/mask2former-swin-large-mapillary-vistas-semantic", dataset="mapillary")
    # segmentation.segment("/Users/koichiito/Desktop/test2/panorama", "/Users/koichiito/Desktop/test2/panorama_segmented", batch_size=5, num_workers=5, save_image_options = ["segmented_image", "blend_image"], save_format="csv")

