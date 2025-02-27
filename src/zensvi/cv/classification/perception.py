import os
import sys
from pathlib import Path
from typing import List, Tuple, Union

import pandas as pd
import torch
from huggingface_hub import hf_hub_download, snapshot_download
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from zensvi.utils.log import verbosity_tqdm

from .base import BaseClassifier
from .utils.Model_01 import Net
from .utils.place_pulse import PlacePulseClassificationModel


class ImageDataset(Dataset):
    """Dataset class for loading and preprocessing images.

    Args:
        image_files (List[Path]): List of paths to image files.
    """

    def __init__(self, image_files: List[Path], vit=False):
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
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        if vit:  # ViT model has a different image size
            self.transform = transforms.Compose(
                [
                    transforms.Resize((384, 384)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ]
            )
        else:
            self.transform = transforms.Compose(
                [
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ]
            )

    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(self, idx: int) -> Tuple[str, torch.Tensor]:
        image_file = self.image_files[idx]
        img = Image.open(image_file).convert("RGB")
        img = self.transform(img)

        return str(image_file), img

    def collate_fn(self, data: List[Tuple[str, torch.Tensor]]) -> Tuple[List[str], torch.Tensor]:
        """Custom collate function for the dataset.

        Args:
            data (List[Tuple[str, torch.Tensor]]): List of tuples containing image file paths and transformed image tensors.

        Returns:
            Tuple[List[str], torch.Tensor]: Tuple containing a list of image file paths and a batch of stacked image tensors.
        """
        image_files, images = zip(*data)
        images = torch.stack(images)  # Stack images to create a batch
        return list(image_files), images


class ClassifierPerception(BaseClassifier):
    """A classifier for evaluating the perception of streetscape based on a given study.

    Args:
        perception_study (str): The specific perception study for which the model is trained, including
            "safer", "livelier", "wealthier", "more beautiful", "more boring", "more depressing".
            This affects the checkpoint file used.
        device (str, optional): The device that the model should be loaded onto. Options are "cpu", "cuda",
            or "mps". If `None`, the model tries to use a GPU if available; otherwise, falls back to CPU.
        verbosity (int, optional): Level of verbosity for progress bars. Defaults to 1.
                                  0 = no progress bars, 1 = outer loops only, 2 = all loops.
    """

    def __init__(self, perception_study, device=None, verbosity=1):
        super().__init__(device, verbosity)
        self.device = self._get_device(device)
        self.perception_study = perception_study

        file_name = f"{perception_study}.pth"
        checkpoint_path = hf_hub_download(
            repo_id="seshing/placepulse",
            filename=file_name,
            local_dir=Path(__file__).parent.parent.parent.parent.parent / "models",
        )

        # Now load the model
        self.model = PlacePulseClassificationModel(num_classes=5)
        self.model = self._load_checkpoint(self.model, checkpoint_path)
        self.model.eval()
        self.model.to(self.device)

    def _save_results_to_file(self, results, dir_output, file_name, save_format="csv json"):
        """Save results to file in specified format.

        Args:
            results (List[dict]): List of dictionaries containing results to save
            dir_output (Union[str, Path]): Directory to save output files
            file_name (str): Base name for output files
            save_format (str, optional): Format(s) to save results in. Defaults to "csv json".
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

    def _load_checkpoint(self, model, checkpoint_path):
        """Load model weights from checkpoint file.

        Args:
            model (torch.nn.Module): Model to load weights into
            checkpoint_path (Union[str, Path]): Path to checkpoint file

        Returns:
            torch.nn.Module: Model with loaded weights
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        model.load_state_dict(checkpoint)
        return model

    def classify(
        self,
        dir_input: Union[str, Path],
        dir_summary_output: Union[str, Path],
        batch_size=1,
        save_format="json csv",
        verbosity: int = None,
    ) -> List[str]:
        """Classifies images based on human perception of streetscapes from the specified perception study.

        Args:
            dir_input (Union[str, Path]): Directory containing input images.
            dir_summary_output (Union[str, Path]): Directory to save summary output. If None, output is not saved.
            batch_size (int, optional): Batch size for inference. Defaults to 1.
            save_format (str, optional): Save format for the output. Options are "json" and "csv".
                Add a space between options. Defaults to "json csv".
            verbosity (int, optional): Level of verbosity for progress bars.
                If None, uses the instance's verbosity level.
                0 = no progress bars, 1 = outer loops only, 2 = all loops.

        Returns:
            List[dict]: List of dictionaries containing perception scores for each image.
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

        with torch.no_grad():
            for image_files, images in verbosity_tqdm(
                dataloader,
                desc=f"Evaluating human perception of study: {self.perception_study}",
                verbosity=verbosity,
                level=1,
            ):
                images = images.to(self.device, dtype=torch.float32)

                custom_scores = self.model(images)
                for image_file, score in zip(image_files, custom_scores):
                    results.append(
                        {
                            "filename_key": str(Path(image_file).stem),
                            f"{self.perception_study}": score.item(),
                        }
                    )

        # Save the results to JSON and/or CSV
        self._save_results_to_file(
            results,
            dir_summary_output,
            "results",
            save_format=save_format,
        )
        return results


class ClassifierPerceptionViT(BaseClassifier):
    """A classifier for evaluating the perception of streetscape based on a given study

    Args:
        perception_study (str): The specific perception study for which the model is trained, including
            "safer", "livelier", "wealthier", "more beautiful", "more boring", "more depressing".
            This affects the checkpoint file used.
        device (str, optional): The device that the model should be loaded onto. Options are "cpu", "cuda",
            or "mps". If `None`, the model tries to use a GPU if available; otherwise, falls back to CPU.
        verbosity (int, optional): Level of verbosity for progress bars. Defaults to 1.
                                  0 = no progress bars, 1 = outer loops only, 2 = all loops.
    """

    def __init__(self, perception_study, device=None, verbosity=1):
        super().__init__(device, verbosity)
        self.device = self._get_device(device)
        self.perception_study = perception_study

        # directory that stores all the models
        model_load_path = "models"

        # map current models in huggingface
        model_dict = {
            "safer": "safety.pth",
            "livelier": "lively.pth",
            "wealthy": "wealthy.pth",
            "more beautiful": "beautiful.pth",
            "more boring": "boring.pth",
            "more depressing": "depressing.pth",
        }

        # Download model
        file_name = model_dict[perception_study]
        snapshot_download(
            repo_id="Jiani11/human-perception-place-pulse",
            allow_patterns=[file_name, "README.md"],
            local_dir=Path(__file__).parent.parent.parent.parent.parent / "models",
        )
        checkpoint_path = model_load_path + "/" + file_name

        # add the path for model file
        sys.path.append(os.path.dirname(os.path.abspath("src/zensvi/cv/classification/utils/Model_01.py")))

        # Now load the model
        self.model = Net(num_classes=5)
        self.model = self._load_checkpoint(self.model, checkpoint_path)
        self.model.eval()
        self.model.to(self.device)

    def _save_results_to_file(self, results, dir_output, file_name, save_format="csv json"):
        """Save results to file in specified format.

        Args:
            results (List[dict]): List of dictionaries containing results to save
            dir_output (Union[str, Path]): Directory to save output files
            file_name (str): Base name for output files
            save_format (str, optional): Format(s) to save results in. Defaults to "csv json".
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

    def _load_checkpoint(self, model, checkpoint_path):
        model = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        return model

    def classify(
        self,
        dir_input: Union[str, Path],
        dir_summary_output: Union[str, Path],
        batch_size=1,
        save_format="json csv",
        verbosity: int = None,
    ) -> List[str]:
        """Classifies images based on human perception of streetscapes from the specified perception study.

        Args:
            dir_input (Union[str, Path]): Directory containing input images.
            dir_summary_output (Union[str, Path]): Directory to save summary output. If None, output is not saved.
            batch_size (int, optional): Batch size for inference. Defaults to 1.
            save_format (str, optional): Save format for the output. Options are "json" and "csv".
                Add a space between options. Defaults to "json csv".
            verbosity (int, optional): Level of verbosity for progress bars.
                If None, uses the instance's verbosity level.
                0 = no progress bars, 1 = outer loops only, 2 = all loops.

        Returns:
            List[dict]: List of dictionaries containing perception scores for each image.
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

        dataset = ImageDataset(img_paths, vit=True)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=dataset.collate_fn)

        results = []

        with torch.no_grad():
            for image_files, images in verbosity_tqdm(
                dataloader,
                desc=f"Evaluating human perception of study: {self.perception_study}",
                verbosity=verbosity,
                level=1,
            ):
                images = images.to(self.device, dtype=torch.float32)

                custom_scores = self.model(images)
                for image_file, score in zip(image_files, custom_scores):
                    results.append(
                        {
                            "filename_key": str(Path(image_file).stem),
                            f"{self.perception_study}": score.item(),
                        }
                    )

        # Save the results to JSON and/or CSV
        self._save_results_to_file(
            results,
            dir_summary_output,
            "results",
            save_format=save_format,
        )
        return results
