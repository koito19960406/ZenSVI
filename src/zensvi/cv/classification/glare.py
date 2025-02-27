from pathlib import Path
from typing import List, Tuple, Union

import pandas as pd
import torch
from huggingface_hub import hf_hub_download
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from zensvi.utils.log import verbosity_tqdm

from .base import BaseClassifier
from .utils.global_streetscapes import GlobalStreetScapesClassificationModel, glare_dict2idx


class ImageDataset(Dataset):
    """Dataset class for loading and preprocessing images.

    Args:
        image_files: List of paths to image files.

    Attributes:
        image_files: Filtered list of valid image file paths.
        transform: Composition of image transformations to apply.
    """

    def __init__(self, image_files: List[Path]):
        self.image_files = [
            image_file
            for image_file in image_files
            if image_file.suffix.lower() in [".jpg", ".jpeg", ".png"] and not image_file.name.startswith(".")
        ]

        # Image transformations
        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # ImageNet normalization
            ]
        )

    def __len__(self) -> int:
        """Returns the number of images in the dataset.

        Returns:
            Number of images.
        """
        return len(self.image_files)

    def __getitem__(self, idx: int) -> Tuple[str, torch.Tensor]:
        """Gets an image and its file path at the given index.

        Args:
            idx: Index of the image to retrieve.

        Returns:
            Tuple containing the image file path as string and the preprocessed image tensor.
        """
        image_file = self.image_files[idx]
        img = Image.open(image_file)  # Open image directly using PIL
        img = self.transform(img)  # Apply transformations

        return str(image_file), img

    def collate_fn(self, data: List[Tuple[str, torch.Tensor]]) -> Tuple[List[str], torch.Tensor]:
        """Custom collate function for batching dataset items.

        Args:
            data: List of tuples containing image file paths and transformed image tensors.

        Returns:
            Tuple containing a list of image file paths and a batch of stacked image tensors.
        """
        image_files, images = zip(*data)
        images = torch.stack(images)  # Stack images to create a batch
        return list(image_files), images


class ClassifierGlare(BaseClassifier):
    """A classifier for identifying glare in images using the model from Hou et al (2024) (https://github.com/ualsg/global-streetscapes).

    Args:
        device (str, optional): The device that the model should be loaded onto. Options are "cpu", "cuda", or "mps".
            If `None`, the model tries to use a GPU if available; otherwise, falls back to CPU.
        verbosity (int, optional): Level of verbosity for progress bars. Defaults to 1.
                                  0 = no progress bars, 1 = outer loops only, 2 = all loops.
    """

    def __init__(self, device=None, verbosity=1):
        super().__init__(device, verbosity)
        self.device = self._get_device(device)

        file_name = "glare_inverse/f6b03038-0831-43ec-88d4-e6c7eb5f8539_glare_glare_inverse_checkpoint.ckpt"
        checkpoint_path = hf_hub_download(
            repo_id="pihalf/gss-models",
            filename=file_name,
            local_dir=Path(__file__).parent.parent.parent.parent.parent / "models",
        )

        checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage, weights_only=False)

        # Extract the number of classes
        num_classes = checkpoint["state_dict"]["model.classifier.5.weight"].shape[0]

        # Now load the model
        self.model = GlobalStreetScapesClassificationModel.load_from_checkpoint(
            checkpoint_path, num_classes=num_classes, weight=None, strict=False
        )
        self.model.eval()
        self.model.to(self.device)

    def _save_results_to_file(self, results, dir_output, file_name, save_format="csv json"):
        """Saves classification results to file(s) in specified format(s).

        Args:
            results: List of dictionaries containing classification results.
            dir_output: Directory to save output files.
            file_name: Base name for output files (without extension).
            save_format: Space-separated string of formats to save in.
                Options are "csv" and "json". Defaults to "csv json".
        """
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
        verbosity: int = None,
    ) -> List[str]:
        """Classifies images based on presence of glare.

        Processes images from the input directory and classifies them as having glare ("True")
        or not having glare ("False"). Results can be saved in JSON and/or CSV format.

        Args:
            dir_input: Directory containing input images or path to a single image.
            dir_summary_output: Directory to save classification results.
            batch_size: Number of images to process simultaneously. Defaults to 1.
            save_format: Space-separated string of output formats.
                Options are "json" and "csv". Defaults to "json csv".
            verbosity (int, optional): Level of verbosity for progress bars.
                If None, uses the instance's verbosity level.
                0 = no progress bars, 1 = outer loops only, 2 = all loops.

        Returns:
            List of glare classifications ("True" or "False") for each image.
        """
        # Use instance verbosity if not specified
        if verbosity is None:
            verbosity = self.verbosity

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
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=dataset.collate_fn)

        results = []
        # Using torch.no_grad() to avoid unnecessary gradient computations during inference
        with torch.no_grad():
            results = [
                {
                    "filename_key": str(Path(image_file).stem),
                    "glare": glare_dict2idx["index2label"][pred.item()],
                }
                for image_files, images in verbosity_tqdm(
                    dataloader, desc="Classifying glare", verbosity=verbosity, level=1
                )
                for image_file, pred in zip(
                    image_files,
                    torch.max(self.model(images.to(self.device, dtype=torch.float32)), 1)[1],
                )
            ]

        # save the results to json and csv
        self._save_results_to_file(
            results,
            dir_summary_output,
            "results",
            save_format=save_format,
        )
