from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
from torchvision import transforms as trn
from pathlib import Path
import pkg_resources
import pandas as pd
import numpy as np
import cv2
from tqdm import tqdm
from typing import Union
import torch

from .base import BaseClassifier
from .utils import wideresnet


class ImageDataset(Dataset):
    # Dataset class for loading images
    def __init__(self, img_paths, transform=None):
        self.img_paths = img_paths
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        image = read_image(str(img_path))
        if self.transform:
            image = self.transform(image)
        return image, str(img_path)

    def collate_fn(self, batch):
        images, paths = zip(*batch)
        images = torch.stack(
            images, dim=0
        )  # This combines the images into a single tensor
        return images, list(paths)


def _returnTF():
    # load the image transformer
    tf = trn.Compose(
        [
            trn.Resize((224, 224)),
            trn.ConvertImageDtype(torch.float),
            trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    return tf


def _recursion_change_bn(module):
    # change the batchnorm2D layer to batchnorm1D
    if isinstance(module, torch.nn.BatchNorm2d):
        module.track_running_stats = 1
    else:
        for i, (name, module1) in enumerate(module._modules.items()):
            module1 = _recursion_change_bn(module1)
    return module


def _returnCAM(feature_conv, weight_softmax, class_idx):
    # generate the class activation maps upsample to 256x256
    size_upsample = (256, 256)
    nc, h, w = feature_conv.shape
    output_cam = []
    for idx in class_idx:
        cam = weight_softmax[class_idx].dot(feature_conv.reshape((nc, h * w)))
        cam = cam.reshape(h, w)
        cam = cam - np.min(cam)
        cam_img = cam / np.max(cam)
        cam_img = np.uint8(255 * cam_img)
        output_cam.append(cv2.resize(cam_img, size_upsample))
    return output_cam


class ClassifierPlaces365(BaseClassifier):
    """
    A classifier for identifying places using the Places365 model. The model is from Zhou et al. (2017) (https://github.com/CSAILVision/places365).

    :param device: The device that the model should be loaded onto. Options are "cpu", "cuda", or "mps".
        If `None`, the model tries to use a GPU if available; otherwise, falls back to CPU.
    :type device: str, optional
    """

    def __init__(self, device=None):
        super().__init__(device)
        self.device = self._get_device(device)
        self.classes, self.labels_IO, self.labels_attribute, self.W_attribute = (
            self._load_labels()
        )
        self.features_blobs = []
        self.model = self._load_model()
        self.model.eval()
        self.model.to(self.device)
        self.weight_softmax = self._load_weight_softmax()

    def _load_labels(self):
        # prepare all the labels
        # scene category relevant
        file_name_category = pkg_resources.resource_filename(
            "zensvi.cv.classification.utils", "categories_places365.txt"
        )
        classes = list()
        with open(file_name_category) as class_file:
            for line in class_file:
                classes.append(line.strip().split(" ")[0][3:])
        classes = tuple(classes)

        # indoor and outdoor relevant
        file_name_IO = pkg_resources.resource_filename(
            "zensvi.cv.classification.utils", "IO_places365.txt"
        )
        with open(file_name_IO) as f:
            lines = f.readlines()
            labels_IO = []
            for line in lines:
                items = line.rstrip().split()
                labels_IO.append(int(items[-1]) - 1)  # 0 is indoor, 1 is outdoor
        labels_IO = np.array(labels_IO)

        # scene attribute relevant
        file_name_attribute = pkg_resources.resource_filename(
            "zensvi.cv.classification.utils", "labels_sunattribute.txt"
        )
        with open(file_name_attribute) as f:
            lines = f.readlines()
            labels_attribute = [item.rstrip() for item in lines]
        file_name_W = pkg_resources.resource_filename(
            "zensvi.cv.classification.utils", "W_sceneattribute_wideresnet18.npy"
        )
        W_attribute = np.load(file_name_W)

        return classes, labels_IO, labels_attribute, W_attribute

    def _hook_feature(self, module, input, output):
        self.features_blobs.append(np.squeeze(output.data.cpu().numpy()))

    def _load_model(self):
        model_file = pkg_resources.resource_filename(
            "zensvi.cv.classification.utils", "wideresnet18_places365.pth.tar"
        )
        model = wideresnet.resnet18(num_classes=365)
        checkpoint = torch.load(model_file, map_location=self.device)
        state_dict = {
            str.replace(k, "module.", ""): v
            for k, v in checkpoint["state_dict"].items()
        }
        model.load_state_dict(state_dict)

        # hacky way to deal with the upgraded batchnorm2D and avgpool layers...
        for i, (name, module) in enumerate(model._modules.items()):
            module = _recursion_change_bn(module)
        model.avgpool = torch.nn.AvgPool2d(kernel_size=14, stride=1, padding=0)

        model.eval()
        # hook the feature extractor
        features_names = [
            "layer4",
            "avgpool",
        ]  # this is the last conv layer of the resnet
        for name in features_names:
            model._modules.get(name).register_forward_hook(self._hook_feature)
        return model

    def _load_weight_softmax(self):
        # get the softmax weight
        params = list(self.model.parameters())
        weight_softmax = params[-2].data.cpu().numpy()
        weight_softmax[weight_softmax < 0] = 0
        return weight_softmax

    def _save_results_to_file(
        self, results, dir_output, file_name, save_format="csv json", csv_format="long"
    ):
        df = pd.DataFrame(results)
        if csv_format == "long":
            # Convert the DataFrame to long format if necessary
            df = pd.melt(
                df,
                id_vars=["filename_key", "environment_type"],
            )
        dir_output = Path(dir_output)
        dir_output.mkdir(parents=True, exist_ok=True)
        if "csv" in save_format:
            file_path = dir_output / f"{file_name}.csv"
            df.to_csv(file_path, index=False)
        if "json" in save_format:
            file_path = dir_output / f"{file_name}.json"
            df.to_json(file_path, orient="records")

    def classify(
        self,
        dir_input: Union[str, Path],
        dir_image_output: Union[str, Path, None] = None,
        dir_summary_output: Union[str, Path, None] = None,
        batch_size: int = 1,
        save_image_options: str = "cam_image blend_image",
        save_format: str = "json csv",
        csv_format: str = "long",  # "long" or "wide"
    ):
        """
        Classifies images based on scene recognition using the Places365 model. The output file can be saved in JSON and/or CSV format and will contain the scene categories, scene attributes, and environment type (indoor or outdoor) for each image.

        A list of categories can be found at https://github.com/CSAILVision/places365/blob/master/categories_places365.txt and a list of attributes can be found at https://github.com/CSAILVision/places365/blob/master/labels_sunattribute.txt

        Scene categories' values range from 0 to 1, where 1 is the highest probability of the scene category. Scene attributes' values are the responses of the scene attributes, which are the dot product of the scene attributes' weight and the features of the image, and higher values indicate a higher presence of the attribute in the image. The environment type is either "indoor" or "outdoor".

        :param dir_input: directory containing input images.
        :type dir_input: Union[str, Path]
        :param dir_image_output: directory to save output images, defaults to None
        :type dir_image_output: Union[str, Path, None], optional
        :param dir_summary_output: directory to save summary output, defaults to None
        :type dir_summary_output: Union[str, Path, None], optional
        :param batch_size: batch size for inference, defaults to 1
        :type batch_size: int, optional
        :param save_image_options: save options for images, defaults to "cam_image blend_image". Options are "cam_image" and "blend_image". Please add a space between options.
        :type save_image_options: str, optional
        :param save_format: save format for the output, defaults to "json csv". Options are "json" and "csv". Please add a space between options.
        :type save_format: str, optional
        :param csv_format: csv format for the output, defaults to "long". Options are "long" and "wide".
        :type csv_format: str, optional
        """
        if not dir_image_output and not dir_summary_output:
            raise ValueError(
                "At least one of dir_image_output and dir_summary_output must be provided"
            )
        # Prepare output directories
        if dir_image_output:
            Path(dir_image_output).mkdir(parents=True, exist_ok=True)
        if dir_summary_output:
            Path(dir_summary_output).mkdir(parents=True, exist_ok=True)

        # get all the images in dir_input
        if Path(dir_input).is_file():
            img_paths = [Path(dir_input)]
        else:
            img_paths = [
                p
                for ext in [
                    "*.jpg",
                    "*.jpeg",
                    "*.png",
                    "*.gif",
                    "*.bmp",
                    "*.tiff",
                    "*.JPG",
                    "*.JPEG",
                    "*.PNG",
                    "*.GIF",
                    "*.BMP",
                    "*.TIFF",
                ]
                for p in Path(dir_input).rglob(ext)
            ]

        # Transform and load the dataset
        transform = _returnTF()
        dataset = ImageDataset(img_paths, transform=transform)
        dataloader = DataLoader(
            dataset, batch_size=batch_size, collate_fn=dataset.collate_fn
        )
        results = []
        # Process images
        for images, paths in tqdm(dataloader, desc="Processing images"):
            self.features_blobs = []
            images = images.to(self.device)
            outputs = self.model(images)
            probs = torch.nn.functional.softmax(outputs, dim=1).cpu().detach().numpy()
            for idx_img_prob, img_prob in enumerate(probs):
                dict_temp = {}
                dict_temp["filename_key"] = str(Path(paths[idx_img_prob]).stem)
                # sort
                top_probs, top_idxs = torch.from_numpy(img_prob).sort(descending=True)
                for idx, prob in zip(top_idxs.numpy(), top_probs.numpy()):
                    dict_temp[self.classes[idx]] = prob
                # Extract features for attributes
                if len(paths) == 1:
                    features_blobs_0 = self.features_blobs[0]
                    features_blobs_1 = self.features_blobs[1]
                else:
                    features_blobs_0 = self.features_blobs[0][idx_img_prob]
                    features_blobs_1 = self.features_blobs[1][idx_img_prob]
                responses_attribute = self.W_attribute.dot(features_blobs_1)
                idx_a = np.argsort(responses_attribute)
                for i, idx in enumerate(idx_a):
                    dict_temp[self.labels_attribute[idx]] = responses_attribute[idx]

                io_image = np.mean(
                    self.labels_IO[top_idxs.numpy()[:10]]
                )  # Assuming labels_IO is correctly shaped
                environment_type = "indoor" if io_image < 0.5 else "outdoor"
                dict_temp["environment_type"] = environment_type
                results.append(dict_temp)

                if len(save_image_options) > 0 and dir_image_output is not None:
                    # Generate class activation mapping
                    CAMs = _returnCAM(
                        features_blobs_0, self.weight_softmax, [top_idxs[0]]
                    )

                    # Render the CAM and output
                    img = cv2.imread(paths[idx_img_prob])
                    height, width, _ = img.shape
                    heatmap = cv2.applyColorMap(
                        cv2.resize(CAMs[0], (width, height)), cv2.COLORMAP_JET
                    )
                    if "cam_image" in save_image_options:
                        output_filename = f"{Path(paths[idx_img_prob]).stem}-cam.jpg"
                        output_filepath = Path(dir_image_output) / output_filename
                        cv2.imwrite(str(output_filepath), heatmap)
                    if "blend_image" in save_image_options:
                        result = heatmap * 0.5 + img * 0.5
                        output_filename = f"{Path(paths[idx_img_prob]).stem}-blend.jpg"
                        output_filepath = Path(dir_image_output) / output_filename
                        cv2.imwrite(str(output_filepath), result)
        # save the results to json
        if dir_summary_output:
            self._save_results_to_file(
                results,
                dir_summary_output,
                "results",
                save_format=save_format,
                csv_format=csv_format,
            )
