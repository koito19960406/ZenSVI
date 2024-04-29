import cv2
from transformers import DPTImageProcessor, DPTForDepthEstimation
import torch
import numpy as np
from PIL import Image
from pathlib import Path
from typing import List, Tuple, Union
from torch.utils.data import Dataset, DataLoader
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import requests
from torchvision.transforms import Compose

from .zoedepth.models.builder import build_model
from .zoedepth.utils.config import get_config
from .depth_anything.util.transform import Resize, NormalizeImage, PrepareForNet


class ImageDataset(Dataset):
    def __init__(self, image_files: List[Path], task="relative"):
        self.image_files = [
            image_file
            for image_file in image_files
            if image_file.suffix.lower() in [".jpg", ".jpeg", ".png"]
            and not image_file.name.startswith(".")
        ]
        self.task = task
        if self.task == "absolute":
            self.transform = Compose(
                [
                    Resize(
                        width=518,
                        height=518,
                        resize_target=False,
                        keep_aspect_ratio=False,
                        ensure_multiple_of=14,
                        resize_method="lower_bound",
                        image_interpolation_method=cv2.INTER_CUBIC,
                    ),
                    NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    PrepareForNet(),
                ]
            )

    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(self, idx: int) -> Tuple[str, torch.Tensor]:
        image_file = self.image_files[idx]
        if self.task == "absolute":
            image = cv2.cvtColor(cv2.imread(str(image_file)), cv2.COLOR_BGR2RGB) / 255.0
            # get the original image size
            original_size = image.shape[:2]
            image = self.transform({"image": image})["image"]
            image = torch.from_numpy(image)
        else:
            image = Image.open(image_file)
            original_size = (image.size[1], image.size[0])

        return image_file, image, original_size

    def collate_fn(
        self, data: List[Tuple[str, torch.Tensor]]
    ) -> Tuple[List[str], torch.Tensor, List[Tuple[int, int]]]:
        """
        Custom collate function for the dataset.

        Args:
            data (List[Tuple[str, torch.Tensor]]): List of tuples containing image file path and transformed image tensor.

        Returns:
            Tuple[List[str], torch.Tensor]: Tuple containing lists of image file paths and a batch of image tensors.
        """
        image_files, images, original_sizes = zip(*data)
        if self.task == "absolute":
            images = torch.stack(images)
        else:
            images = list(images)
        return list(image_files), images, list(original_sizes)


class DepthEstimator:
    """A class for estimating depth in images. The class uses the DPT model from Hugging Face for relative depth estimation (https://huggingface.co/Intel/dpt-large) and the ZoeDepth model for absolute (metric) depth estimation (https://github.com/LiheYoung/Depth-Anything/tree/1e1c8d373ae6383ef6490a5c2eb5ef29fd085993/metric_depth).

    :param device: device to use for inference, defaults to None
    :type device: str, optional
    :param task: task to perform, either "relative" or "absolute", defaults to "relative"
    :type task: str, optional
    """

    def __init__(self, device=None, task="relative"):
        self.task = task
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        print(f"Using {self.device}")

        if task == "absolute":
            self._setup_absolute_depth()
        else:
            self._setup_relative_depth()

    def _setup_relative_depth(self):
        self.processor = DPTImageProcessor.from_pretrained("Intel/dpt-large")
        self.model = DPTForDepthEstimation.from_pretrained("Intel/dpt-large").to(
            self.device
        )

    def _setup_absolute_depth(self):
        # donwload the model from https://huggingface.co/spaces/LiheYoung/Depth-Anything/resolve/main/checkpoints_metric_depth/depth_anything_metric_depth_outdoor.pt to models/depth_anything_metric_depth_outdoor.pt with request.get
        # Path to the current file (e.g., __file__ in a module)
        current_file_path = Path(__file__)
        # Path to the current file's directory (often the package directory)
        package_directory = current_file_path.parent.parent.parent.parent.parent
        checkpoint_path = (
            package_directory / "models/depth_anything_metric_depth_outdoor.pt"
        )
        checkpoint_path_vit = package_directory / "models/depth_anything_vitl14.pth"
        if Path(checkpoint_path).exists() and Path(checkpoint_path_vit).exists():
            config = get_config(
                "zoedepth",
                mode="infer",
                pretrained_resource="local::" + str(checkpoint_path),
            )
            self.model = build_model(config).to(self.device)
            return
        # make directory
        Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
        url = "https://huggingface.co/spaces/LiheYoung/Depth-Anything/resolve/main/checkpoints_metric_depth/depth_anything_metric_depth_outdoor.pt"
        response = requests.get(url)
        response.raise_for_status()  # This will raise an exception if there is an error
        with open(checkpoint_path, "wb") as f:
            f.write(response.content)

        url_vit = "https://huggingface.co/spaces/LiheYoung/Depth-Anything/resolve/main/checkpoints/depth_anything_vitl14.pth"
        response_vit = requests.get(url_vit)
        response_vit.raise_for_status()  # This will raise an exception if there is an error
        with open(checkpoint_path_vit, "wb") as f:
            f.write(response_vit.content)

        config = get_config(
            "zoedepth", mode="infer", pretrained_resource="local::" + checkpoint_path
        )
        self.model = build_model(config).to(self.device)

    def _process_images(self, image_files, images, original_sizes, dir_output):
        inputs = (
            self.processor(images=images, return_tensors="pt").to(self.device)
            if self.task == "relative"
            else images
        )

        with torch.no_grad():
            outputs = (
                self.model(**inputs)
                if self.task == "relative"
                else self.model(images.to(self.device))
            )
            predicted_depths = (
                outputs.predicted_depth
                if self.task == "relative"
                else outputs["metric_depth"]
            )

            for i, (image_file, predicted_depth, original_size) in enumerate(
                zip(image_files, predicted_depths, original_sizes)
            ):
                if self.device == torch.device("mps"):
                    predicted_depth = predicted_depth.cpu()
                if self.task == "relative":
                    prediction = torch.nn.functional.interpolate(
                        predicted_depth.unsqueeze(0).unsqueeze(0),
                        size=original_size,  # if images[i] is in CxHxW format
                        mode="bicubic",
                        align_corners=False,
                    )
                else:
                    prediction = torch.nn.functional.interpolate(
                        predicted_depth.unsqueeze(0),
                        size=original_size,
                        mode="bicubic",
                        align_corners=False,
                    )
                output = prediction.squeeze().cpu().numpy()
                formatted = (output * 255 / np.max(output)).astype("uint8")
                depth = Image.fromarray(formatted)
                depth.save(Path(dir_output) / image_file.name)

    def estimate_depth(
        self,
        dir_input: Union[str, Path],
        dir_image_output: Union[str, Path],
        batch_size: int = 1,
        max_workers: int = 4,
    ):
        """
        Estimates relative depth in the images. Saves the depth maps in the specified directory.

        :param dir_input: directory containing input images.
        :type dir_input: Union[str, Path]
        :param dir_image_output: directory to save the depth maps.
        :type dir_image_output: Union[str, Path]
        :param batch_size: batch size for inference, defaults to 1
        :type batch_size: int, optional
        :param max_workers: maximum number of workers for parallel processing, defaults to 4
        :type max_workers: int, optional
        """
        # make directory
        dir_input = Path(dir_input)
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
                f for f in Path(dir_input).iterdir() if f.suffix in image_extensions
            ]
        else:
            raise ValueError("dir_input must be either a file or a directory.")
        # skip if there are no image files to process
        if len(image_file_list) == 0:
            print("No image files to process. Skipping segmentation.")
            return
        # make dir_image_output
        Path(dir_image_output).mkdir(parents=True, exist_ok=True)

        dataset = ImageDataset(image_file_list, task=self.task)
        dataloader = DataLoader(
            dataset, batch_size=batch_size, collate_fn=dataset.collate_fn
        )
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []

            for batch in dataloader:
                image_files, images, original_sizes = batch
                futures.append(
                    executor.submit(
                        self._process_images,
                        image_files,
                        images,
                        original_sizes,
                        dir_image_output,
                    )
                )

            for future in tqdm(
                as_completed(futures), total=len(futures), desc="Estimating depth"
            ):
                future.result()
