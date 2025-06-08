import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Tuple, Union

import cv2
import numpy as np
import requests
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


class ImageDataset(Dataset):
    """Dataset class for loading images."""

    def __init__(self, image_files: List[Path], task="relative"):
        self.image_files = [
            image_file
            for image_file in image_files
            if image_file.suffix.lower() in [".jpg", ".jpeg", ".png"] and not image_file.name.startswith(".")
        ]
        self.task = task

    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(self, idx: int) -> Tuple[str, str, Tuple[int, int]]:
        image_file = self.image_files[idx]
        # Read image to get original size
        image = cv2.imread(str(image_file))
        original_size = (image.shape[0], image.shape[1])  # (height, width)
        return image_file, str(image_file), original_size

    def collate_fn(
        self, data: List[Tuple[str, str, Tuple[int, int]]]
    ) -> Tuple[List[str], List[str], List[Tuple[int, int]]]:
        """Collate function for data loader.

        Args:
            data: List of tuples containing (image_file, image_path, original_size).

        Returns:
            Tuple of lists containing image files, image paths, and original sizes.
        """
        image_files, image_paths, original_sizes = zip(*data)
        return list(image_files), list(image_paths), list(original_sizes)


class DepthEstimator:
    """A class for estimating depth in images using DepthAnythingV2."""

    def __init__(self, device=None, task="relative", encoder="vitl", max_depth=80.0):
        self.task = task
        self.encoder = encoder
        self.max_depth = max_depth

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        print(f"Using {self.device}")

        # Model configurations (from the original run.py)
        self.model_configs = {
            "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
            "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
            "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
            "vitg": {"encoder": "vitg", "features": 384, "out_channels": [1536, 1536, 1536, 1536]},
        }

        if task == "absolute":
            self._setup_absolute_depth()
        else:
            self._setup_relative_depth()

    def _setup_relative_depth(self):
        """Sets up the model for relative depth estimation."""
        # Add the DepthAnythingV2 path to sys.path (exactly like the working run.py)
        depth_anything_path = str(Path(__file__).parent / "DepthAnythingV2")
        if depth_anything_path not in sys.path:
            sys.path.insert(0, depth_anything_path)

        from depth_anything_v2.dpt import DepthAnythingV2

        # Path for model checkpoints
        current_file_path = Path(__file__)
        package_directory = current_file_path.parent.parent.parent.parent.parent
        checkpoint_path = package_directory / f"models/depth_anything_v2_{self.encoder}.pth"

        if not checkpoint_path.exists():
            self._download_relative_model(checkpoint_path)

        # Initialize model (exactly like the working run.py)
        config = self.model_configs[self.encoder]
        self.depth_anything = DepthAnythingV2(**config)
        self.depth_anything.load_state_dict(torch.load(str(checkpoint_path), map_location="cpu"))
        self.depth_anything = self.depth_anything.to(self.device).eval()

    def _setup_absolute_depth(self):
        """Sets up the model for absolute depth estimation."""
        # Add the metric depth path to sys.path
        metric_depth_path = str(Path(__file__).parent / "DepthAnythingV2")
        if metric_depth_path not in sys.path:
            sys.path.insert(0, metric_depth_path)

        from metric_depth.depth_anything_v2.dpt import DepthAnythingV2

        # Path for model checkpoints
        current_file_path = Path(__file__)
        package_directory = current_file_path.parent.parent.parent.parent.parent
        checkpoint_path = package_directory / f"models/depth_anything_v2_metric_vkitti_{self.encoder}.pth"

        if not checkpoint_path.exists():
            self._download_absolute_model(checkpoint_path)

        # Initialize model with max_depth parameter (like the working metric run.py)
        config = {**self.model_configs[self.encoder], "max_depth": self.max_depth}
        self.depth_anything = DepthAnythingV2(**config)
        self.depth_anything.load_state_dict(torch.load(str(checkpoint_path), map_location="cpu"))
        self.depth_anything = self.depth_anything.to(self.device).eval()

    def _download_relative_model(self, checkpoint_path: Path):
        """Downloads the relative depth model weights."""
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

        download_urls = {
            "vits": "https://huggingface.co/depth-anything/Depth-Anything-V2-Small/resolve/main/depth_anything_v2_vits.pth?download=true",
            "vitb": "https://huggingface.co/depth-anything/Depth-Anything-V2-Base/resolve/main/depth_anything_v2_vitb.pth?download=true",
            "vitl": "https://huggingface.co/depth-anything/Depth-Anything-V2-Large/resolve/main/depth_anything_v2_vitl.pth?download=true",
            "vitg": "https://huggingface.co/depth-anything/Depth-Anything-V2-Giant/resolve/main/depth_anything_v2_vitg.pth?download=true",
        }

        url = download_urls[self.encoder]
        print(f"Downloading relative depth model weights for {self.encoder}...")

        response = requests.get(url, stream=True)
        response.raise_for_status()

        total_size = int(response.headers.get("content-length", 0))
        with (
            open(checkpoint_path, "wb") as f,
            tqdm(
                desc=f"Downloading {checkpoint_path.name}",
                total=total_size,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
            ) as pbar,
        ):
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))

    def _download_absolute_model(self, checkpoint_path: Path):
        """Downloads the absolute depth model weights."""
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

        download_urls = {
            "vits": "https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-VKITTI-Small/resolve/main/depth_anything_v2_metric_vkitti_vits.pth?download=true",
            "vitb": "https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-VKITTI-Base/resolve/main/depth_anything_v2_metric_vkitti_vitb.pth?download=true",
            "vitl": "https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-VKITTI-Large/resolve/main/depth_anything_v2_metric_vkitti_vitl.pth?download=true",
            "vitg": "https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-VKITTI-Giant/resolve/main/depth_anything_v2_metric_vkitti_vitg.pth?download=true",
        }

        url = download_urls[self.encoder]
        print(f"Downloading absolute depth model weights for {self.encoder}...")

        response = requests.get(url, stream=True)
        response.raise_for_status()

        total_size = int(response.headers.get("content-length", 0))
        with (
            open(checkpoint_path, "wb") as f,
            tqdm(
                desc=f"Downloading {checkpoint_path.name}",
                total=total_size,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
            ) as pbar,
        ):
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))

    def _infer_image_with_device(self, raw_image, input_size=518):
        """Custom inference method that respects our device setting."""
        from torchvision.transforms import Compose

        # Import the transforms from the depth_anything_v2 module
        sys.path.insert(0, str(Path(__file__).parent / "DepthAnythingV2"))
        if self.task == "absolute":
            sys.path.insert(0, str(Path(__file__).parent / "DepthAnythingV2" / "metric_depth"))

        from depth_anything_v2.util.transform import NormalizeImage, PrepareForNet, Resize

        transform = Compose(
            [
                Resize(
                    width=input_size,
                    height=input_size,
                    resize_target=False,
                    keep_aspect_ratio=True,
                    ensure_multiple_of=14,
                    resize_method="lower_bound",
                    image_interpolation_method=cv2.INTER_CUBIC,
                ),
                NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                PrepareForNet(),
            ]
        )

        h, w = raw_image.shape[:2]

        image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB) / 255.0

        image = transform({"image": image})["image"]
        image = torch.from_numpy(image).unsqueeze(0)

        # Use our device instead of hardcoded device detection
        image = image.to(self.device)

        with torch.no_grad():
            depth = self.depth_anything.forward(image)
            depth = torch.nn.functional.interpolate(depth[:, None], (h, w), mode="bilinear", align_corners=True)[0, 0]

        return depth.cpu().numpy()

    def _process_images(self, image_files, image_paths, original_sizes, dir_output):
        """Processes images to estimate depth and save the results."""
        for image_file, image_path, original_size in zip(image_files, image_paths, original_sizes):
            # Read image (exactly like the working run.py)
            raw_image = cv2.imread(image_path)

            # Infer depth using our custom method that respects the device
            depth = self._infer_image_with_device(raw_image, input_size=518)

            # Save raw depth values without normalization
            # Convert to appropriate format for PIL Image
            if depth.dtype != np.float32:
                depth = depth.astype(np.float32)

            # Convert to PIL Image and save with TIFF format to preserve float32 values
            depth_image = Image.fromarray(depth, mode="F")  # 'F' mode for 32-bit floating point

            # Change extension to .tiff to support float32 values
            output_path = Path(dir_output) / image_file.name
            output_path = output_path.with_suffix(".tiff")
            depth_image.save(output_path)

    def estimate_depth(
        self,
        dir_input: Union[str, Path],
        dir_image_output: Union[str, Path],
        batch_size: int = 1,
        max_workers: int = 4,
    ):
        """Estimates depth in the images and saves the depth maps."""
        dir_input = Path(dir_input)

        # Handle both single file and directory inputs
        if dir_input.is_file():
            image_file_list = [dir_input]
        elif dir_input.is_dir():
            image_extensions = [".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"]
            image_file_list = [f for f in Path(dir_input).iterdir() if f.suffix.lower() in image_extensions]
        else:
            raise ValueError("dir_input must be either a file or a directory.")

        if len(image_file_list) == 0:
            print("No image files to process. Skipping depth estimation.")
            return

        Path(dir_image_output).mkdir(parents=True, exist_ok=True)

        dataset = ImageDataset(image_file_list, task=self.task)
        dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=dataset.collate_fn)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []

            for batch in dataloader:
                image_files, image_paths, original_sizes = batch
                futures.append(
                    executor.submit(
                        self._process_images,
                        image_files,
                        image_paths,
                        original_sizes,
                        dir_image_output,
                    )
                )

            for future in tqdm(as_completed(futures), total=len(futures), desc="Estimating depth"):
                future.result()
