from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Tuple, Union

import cv2
import numpy as np
import open3d as o3d
import plotly.graph_objects as go
import torch
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
        if image is None:
            raise ValueError(f"Failed to read image file: {image_file}")
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


class VGGTProcessor:
    """A class for processing images using VGGT model to generate point clouds."""

    def __init__(self, vggt_path: str = "vggt"):
        """Initialize VGGT processor.

        Args:
            vggt_path: Path to VGGT model directory
        """
        import os
        import sys
        from pathlib import Path

        import torch

        print("=== VGGT Processor Initialization Started ===")
        # Add vggt to Python path
        vggt_path = os.path.join(os.path.dirname(__file__), "vggt")
        print(f"VGGT Path: {vggt_path}")
        if vggt_path not in sys.path:
            sys.path.append(vggt_path)
            print("Added VGGT path to sys.path")

        # Add utils path
        utils_path = os.path.join(vggt_path, "vggt", "utils")
        print(f"Utils Path: {utils_path}")
        if utils_path not in sys.path:
            sys.path.append(utils_path)
            print("Added Utils path to sys.path")

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using Device: {self.device}")

        # bfloat16 is supported on Ampere GPUs (Compute Capability 8.0+)
        self.dtype = (
            torch.bfloat16
            if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8
            else torch.float16
        )
        print(f"Data Type: {self.dtype}")

        # Initialize model with optimized local weight management
        try:
            from vggt.models.vggt import VGGT

            # Define local model path
            current_dir = Path(__file__).parent
            models_dir = current_dir.parent.parent.parent / "models"
            models_dir.mkdir(parents=True, exist_ok=True)

            # Use from_pretrained with local cache directory for better performance
            print(f"Loading VGGT model with local cache: {models_dir}")
            self.model = VGGT.from_pretrained("facebook/VGGT-1B", cache_dir=str(models_dir))

            # Move model to device and set to evaluation mode
            self.model = self.model.to(self.device)
            self.model.eval()

            print("=== VGGT Processor Initialization Completed ===")
        except Exception as e:
            print(f"Failed to initialize VGGT model: {str(e)}")
            raise RuntimeError(f"Failed to initialize VGGT model: {str(e)}")

    def process_images(self, image_paths: list) -> dict:
        """Process images and generate predictions.

        Args:
            image_paths: List of paths to input images

        Returns:
            Dictionary containing processed predictions
        """
        try:
            from vggt.utils.load_fn import load_and_preprocess_images

            # Load and preprocess images
            images = load_and_preprocess_images(image_paths).to(self.device)

            # Generate predictions
            with torch.no_grad():
                with torch.cuda.amp.autocast(dtype=self.dtype):
                    predictions = self.model(images)

            return predictions
        except Exception as e:
            raise RuntimeError(f"Failed to process images: {str(e)}")

    def generate_point_cloud(self, predictions: dict) -> tuple:
        """Generate point cloud from model predictions.

        Args:
            predictions: Dictionary containing model predictions

        Returns:
            Tuple containing (points, colors, confidence, camera poses)
        """
        try:
            import numpy as np
            from vggt.utils.geometry import closed_form_inverse_se3
            from vggt.utils.pose_enc import pose_encoding_to_extri_intri

            # Process pose encodings
            extrinsic, intrinsic = pose_encoding_to_extri_intri(
                predictions["pose_enc"], predictions["images"].shape[-2:]
            )
            predictions["extrinsic"] = extrinsic
            predictions["intrinsic"] = intrinsic

            # Convert tensors to numpy
            for key in predictions.keys():
                if isinstance(predictions[key], torch.Tensor):
                    predictions[key] = predictions[key].cpu().numpy().squeeze(0)

            # Extract predictions
            images = predictions["images"]  # (S, 3, H, W)
            world_points = predictions["world_points"]  # (S, H, W, 3)
            conf = predictions["depth_conf"]  # (S, H, W)

            # Process colors
            colors = images.transpose(0, 2, 3, 1)  # (S, H, W, 3)
            S, H, W, _ = world_points.shape

            # Flatten arrays
            points = world_points.reshape(-1, 3)
            colors_flat = (colors.reshape(-1, 3) * 255).astype(np.uint8)
            conf_flat = conf.reshape(-1)

            # Process camera poses
            cam_to_world_mat = closed_form_inverse_se3(predictions["extrinsic"])
            cam_to_world = cam_to_world_mat[:, :3, :]

            # Center points
            scene_center = np.mean(points, axis=0)
            points_centered = points - scene_center

            return points_centered, colors_flat, conf_flat, cam_to_world
        except Exception as e:
            raise RuntimeError(f"Failed to generate point cloud: {str(e)}")

    def _process_batch_to_pointcloud(
        self,
        image_files: List[Path],
        image_paths: List[str],
        original_sizes: List[Tuple[int, int]],
        dir_output: Path,
    ):
        """Process a batch of images to point cloud.

        Args:
            image_files: List of image files
            image_paths: List of image paths
            original_sizes: List of original image sizes
            dir_output: Output directory
        """
        try:
            # Process each image individually to generate separate point clouds
            for i, (image_file, image_path, original_size) in enumerate(zip(image_files, image_paths, original_sizes)):
                # Process single image
                predictions = self.process_images([image_path])

                # Generate point cloud
                points, colors, conf, cam_to_world = self.generate_point_cloud(predictions)

                # Create point cloud
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(points)
                pcd.colors = o3d.utility.Vector3dVector(colors / 255.0)

                # Save point cloud with correct filename for each image
                output_file = dir_output / f"{image_file.stem}.ply"
                o3d.io.write_point_cloud(str(output_file), pcd)

        except Exception as e:
            print(f"Failed to process batch: {str(e)}")

    def process_images_to_pointcloud(
        self,
        dir_input: Union[str, Path],
        dir_output: Union[str, Path],
        batch_size: int = 1,
        max_workers: int = 4,
    ):
        """Process images to generate point clouds using VGGT model.

        Args:
            dir_input: Input directory or file containing images
            dir_output: Output directory for point cloud files
            batch_size: Batch size for processing
            max_workers: Number of worker threads
        """
        dir_input = Path(dir_input)
        dir_output = Path(dir_output)

        # Handle both single file and directory inputs
        if dir_input.is_file():
            image_file_list = [dir_input]
        elif dir_input.is_dir():
            image_extensions = [".jpg", ".jpeg", ".png"]
            image_file_list = [f for f in Path(dir_input).iterdir() if f.suffix.lower() in image_extensions]
        else:
            raise ValueError("dir_input must be either a file or a directory.")

        if len(image_file_list) == 0:
            print("No image files to process. Skipping point cloud generation.")
            return

        dir_output.mkdir(parents=True, exist_ok=True)

        dataset = ImageDataset(image_file_list)
        dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=dataset.collate_fn)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []

            for batch in dataloader:
                image_files, image_paths, original_sizes = batch
                futures.append(
                    executor.submit(
                        self._process_batch_to_pointcloud,
                        image_files,
                        image_paths,
                        original_sizes,
                        dir_output,
                    )
                )

            for future in tqdm(as_completed(futures), total=len(futures), desc="Generating point clouds"):
                future.result()

    def visualize_point_cloud(
        self,
        points,
        colors_flat,
        marker_size=1,
        opacity=0.8,
        sample_rate=0.1,
        camera_eye=dict(x=0, y=0, z=-1),
        camera_up=dict(x=0, y=-1, z=0),
    ):
        """Visualizes a point cloud using Plotly with random sampling.

        Args:
            points (np.ndarray): The point cloud coordinates array.
            colors_flat (np.ndarray): The colors array for the points.
            marker_size (int): Size of point markers.
            opacity (float): Opacity of points.
            sample_rate (float): Percentage of points to sample (0-1).
            camera_eye (dict): Camera position.
            camera_up (dict): Camera up direction.
        """
        # Random sampling to reduce density
        num_points = len(points)
        sample_indices = np.random.choice(num_points, int(num_points * sample_rate), replace=False)

        # Create visualization with sampled points
        fig = go.Figure(
            data=[
                go.Scatter3d(
                    x=points[sample_indices, 0],
                    y=points[sample_indices, 1],
                    z=points[sample_indices, 2],
                    mode="markers",
                    marker=dict(size=marker_size, color=colors_flat[sample_indices], opacity=opacity),
                )
            ]
        )

        # Set layout
        fig.update_layout(
            title=f"3D Point Cloud Visualization ({int(sample_rate*100)}% sampled)",
            scene=dict(
                xaxis_title="X",
                yaxis_title="Y",
                zaxis_title="Z",
                aspectmode="data",
                camera=dict(eye=camera_eye, up=camera_up),
            ),
            showlegend=False,
            margin=dict(l=0, r=0, b=0, t=30),
        )

        fig.show()
