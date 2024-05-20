from pathlib import Path
from typing import List, Tuple, Union
from torch.utils.data import Dataset, DataLoader
from huggingface_hub import hf_hub_download
from torchvision import transforms
from PIL import Image
import torch
import pandas as pd
import tqdm

from .utils.global_streetscapes import (
    view_direction_dict2idx,
    GlobalStreetScapesClassificationModel,
)
from .base import BaseClassifier


class ImageDataset(Dataset):
    def __init__(self, image_files: List[Path]):
        self.image_files = [
            image_file
            for image_file in image_files
            if image_file.suffix.lower() in [".jpg", ".jpeg", ".png"]
            and not image_file.name.startswith(".")
        ]

        # Image transformations
        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),  # ImageNet normalization
            ]
        )

    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(self, idx: int) -> Tuple[str, torch.Tensor]:
        image_file = self.image_files[idx]
        img = Image.open(image_file)  # Open image directly using PIL
        img = self.transform(img)  # Apply transformations

        return str(image_file), img

    def collate_fn(
        self, data: List[Tuple[str, torch.Tensor]]
    ) -> Tuple[List[str], torch.Tensor]:
        """
        Custom collate function for the dataset.

        Args:
            data (List[Tuple[str, torch.Tensor]]): List of tuples containing image file path and transformed image tensor.

        Returns:
            Tuple[List[str], torch.Tensor]: Tuple containing lists of image file paths and a batch of image tensors.
        """
        image_files, images = zip(*data)
        images = torch.stack(images)  # Stack images to create a batch
        return list(image_files), images


class ClassifierViewDirection(BaseClassifier):
    """
    A classifier for identifying view_direction.

    :param device: The device that the model should be loaded onto. Options are "cpu", "cuda", or "mps".
        If `None`, the model tries to use a GPU if available; otherwise, falls back to CPU.
    :type device: str, optional
    """

    def __init__(self, device=None):
        super().__init__(device)
        self.device = self._get_device(device)

        file_name = "view_direction_inverse/a0bd363c-60e8-455c-b077-e5852aca9371_view_direction_view_direction_inverse_checkpoint.ckpt"
        checkpoint_path = hf_hub_download(
            repo_id="pihalf/gss-models",
            filename=file_name,
            local_dir=Path(__file__).parent.parent.parent.parent.parent / "models",
        )

        checkpoint = torch.load(
            checkpoint_path, map_location=lambda storage, loc: storage
        )

        # Extract the number of classes
        num_classes = checkpoint["state_dict"]["model.classifier.5.weight"].shape[0]

        # Now load the model
        self.model = GlobalStreetScapesClassificationModel.load_from_checkpoint(
            checkpoint_path, num_classes=num_classes, weight=None, strict=False
        )
        self.model.eval()
        self.model.to(self.device)

    def _save_results_to_file(
        self, results, dir_output, file_name, save_format="csv json"
    ):
        df = pd.DataFrame(results)
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
        dir_summary_output: Union[str, Path],
        batch_size=1,
        save_format="json csv",
    ) -> List[str]:
        """
        Classifies images based on view_direction. The output file can be saved in JSON and/or CSV format and will contain view_direction for each image. The view_direction categories are "front/back" and "side".

        :param dir_input: directory containing input images.
        :type dir_input: Union[str, Path]
        :param dir_summary_output: directory to save summary output.
        :type dir_summary_output: Union[str, Path, None]
        :param batch_size: batch size for inference, defaults to 1
        :type batch_size: int, optional
        :param save_format: save format for the output, defaults to "json csv". Options are "json" and "csv". Please add a space between options.
        :type save_format: str, optional
        """
        # Prepare output directories
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

        dataset = ImageDataset(img_paths)
        dataloader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, collate_fn=dataset.collate_fn
        )

        results = []
        # Using torch.no_grad() to avoid unnecessary gradient computations during inference
        with torch.no_grad():
            results = [
                {
                    "filename_key": str(Path(image_file).stem),
                    "view_direction": view_direction_dict2idx["index2label"][
                        pred.item()
                    ],
                }
                for image_files, images in tqdm.tqdm(
                    dataloader, desc="Classifying view_direction"
                )
                for image_file, pred in zip(
                    image_files,
                    torch.max(
                        self.model(images.to(self.device, dtype=torch.float32)), 1
                    )[1],
                )
            ]

        # save the results to json and csv
        self._save_results_to_file(
            results,
            dir_summary_output,
            "results",
            save_format=save_format,
        )
