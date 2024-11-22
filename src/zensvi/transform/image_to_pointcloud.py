import copy
from pathlib import Path
from typing import Union

import numpy as np
import open3d as o3d
import pandas as pd
import plotly.graph_objects as go
from PIL import Image

from zensvi.utils.log import Logger


class PointCloudProcessor:
    """A class for processing images and depth maps to generate point clouds.

    This class provides functionality to load image and depth data, convert them into 3D
    point clouds, and perform various operations on the resulting point clouds. It
    supports batch processing of multiple images and offers options for scaling and
    depth normalization.

    Args:
        image_folder (Path): Path to the folder containing color images.
        depth_folder (Path): Path to the folder containing depth images.
        output_coordinate_scale (float): Scaling factor for the output
            coordinates.
        depth_max (float): Maximum depth value for normalization.
        logger (Logger): Optional logger for tracking operations and
            errors.
    """

    def __init__(
        self,
        image_folder: str,
        depth_folder: str,
        output_coordinate_scale: float = 45,
        depth_max: float = 255,
        log_path: Union[str, Path] = None,
    ):
        self.image_folder = Path(image_folder)
        self.depth_folder = Path(depth_folder)
        self.output_coordinate_scale = output_coordinate_scale
        self.depth_max = depth_max
        self._validate_paths()
        # Initialize logger
        if log_path is not None:
            self.logger = Logger(log_path)
        else:
            self.logger = None

    def _validate_paths(self):
        if not self.image_folder.exists():
            raise FileNotFoundError(f"Image folder {self.image_folder} does not exist")
        if not self.depth_folder.exists():
            raise FileNotFoundError(f"Depth folder {self.depth_folder} does not exist")

    def _load_images(self, data):
        """Preloads all images specified in the data DataFrame to optimize the point
        cloud generation process.

        Args:
            data (DataFrame): DataFrame containing image ids for processing.

        Returns:
            Dict: Dictionary containing loaded PIL image objects for depth and color images.
        """
        if self.logger:
            self.logger.log_args("PointCloudProcessor._load_images", data=data)
        images = {}
        for image_id in data["id"].unique():
            depth_path = self.depth_folder / f"{image_id}.jpg"
            color_path = self.image_folder / f"{image_id}.jpg"
            if depth_path.exists() and color_path.exists():
                images[image_id] = {
                    "depth": np.array(Image.open(depth_path).convert("L")),
                    "color": np.array(Image.open(color_path)),
                }
            else:
                print(f"Warning: Missing images for ID {image_id}")
        return images

    def convert_to_point_cloud(self, depth_img, color_img, depth_max):
        """Converts a single depth and color image pair to a point cloud.

        Args:
            depth_img (np.ndarray): The depth image.
            color_img (np.ndarray): The corresponding color image.
            depth_max (float): Maximum value for depth normalization.

        Returns:
            o3d.geometry.PointCloud: The generated point cloud with color.
        """
        if self.logger:
            self.logger.log_args(
                "PointCloudProcessor.convert_to_point_cloud",
                depth_img=depth_img,
                color_img=color_img,
                depth_max=depth_max,
            )

        xs, ys = depth_img.shape[1], depth_img.shape[0]

        da = 2.0 * np.pi / xs
        db = np.pi / ys
        points = []
        colors = []

        for y in range(ys):
            b = -0.5 * np.pi + y * db
            for x in range(xs):
                a = x * da
                r1 = depth_img[y, x]
                r2 = (255 - r1) / depth_max

                xx = 3 * r2**4 * np.cos(a) * np.cos(b) / (np.log(1.1 + 5 * (y / (ys - 1))))
                yy = 3 * r2**4 * np.sin(a) * np.cos(b) / (np.log(1.1 + 5 * (y / (ys - 1))))
                zz = (2 * r2) ** 2 * np.sin(b)

                c = color_img[y, x] / 255.0  # Normalizing color
                points.append([xx, yy, zz])
                colors.append(c)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np.array(points))
        pcd.colors = o3d.utility.Vector3dVector(np.array(colors))
        return pcd

    def transform_point_cloud(self, pcd, origin_x, origin_y, angle, box_extent, box_center):
        """Transforms the point cloud by translating, rotating, and cropping based on
        given parameters.

        Args:
            pcd (o3d.geometry.PointCloud): The point cloud to transform.
            origin_x (float): X coordinate for translation.
            origin_y (float): Y coordinate for translation.
            angle (float): Rotation angle in radians.
            box_extent (List[float]): Extent of the oriented bounding box.
            box_center (List[float]): Center of the oriented bounding box.

        Returns:
            o3d.geometry.PointCloud: Transformed point cloud.
        """
        if self.logger:
            self.logger.log_args(
                "PointCloudProcessor.transform_point_cloud",
                pcd=pcd,
                origin_x=origin_x,
                origin_y=origin_y,
                angle=angle,
                box_extent=box_extent,
                box_center=box_center,
            )
        translation_vector = np.array([origin_x, origin_y, 0])
        pcd_mv = copy.deepcopy(pcd).translate(translation_vector, relative=False)
        pcd_mv.rotate(pcd.get_rotation_matrix_from_xyz((0, 0, angle)))
        obb = o3d.geometry.OrientedBoundingBox(
            center=box_center,
            R=pcd_mv.get_rotation_matrix_from_xyz((0, 0, angle)),
            extent=box_extent,
        )
        pcd_mv = pcd_mv.crop(obb)
        return pcd_mv

    def save_point_cloud_csv(self, pcd, output_path):
        """Saves the point cloud in CSV format.

        Args:
            pcd (o3d.geometry.PointCloud): The point cloud to save.
            output_path (str): Path to save the CSV file.
        """
        points = np.asarray(pcd.points)
        colors = np.asarray(pcd.colors)
        df = pd.DataFrame(np.hstack((points, colors)), columns=["x", "y", "z", "r", "g", "b"])
        df.to_csv(output_path, index=False)

    def save_point_cloud_numpy(self, pcd, output_path):
        """Saves the point cloud as a NumPy array.

        Args:
            pcd (o3d.geometry.PointCloud): The point cloud to save.
            output_path (str): Path to save the NumPy array.
        """
        points = np.asarray(pcd.points)
        colors = np.asarray(pcd.colors)
        np.savez(output_path, points=points, colors=colors)

    def process_multiple_images(self, data, output_dir=None, save_format="pcd"):
        """Generates a point cloud for each entry in the data based on pre-loaded depth
        and color images.

        Args:
            data (DataFrame): DataFrame containing image ids, coordinates, and headings.
            output_dir (str): Path to the output directory to save point clouds.
            save_format (str): Format to save point clouds ('pcd', 'ply', 'npz', 'csv').

        Returns:
            List[o3d.geometry.PointCloud]: List of unprocessed point clouds with color information if output_dir is None.
        """
        if self.logger:
            self.logger.log_args("PointCloudProcessor.process_multiple_images", data=data)
        images = self._load_images(data)
        pcd_list = []

        for idx, row in data.iterrows():
            image_id = row["id"]
            if image_id in images:
                depth_img = images[image_id]["depth"]
                color_img = images[image_id]["color"]

                pcd = self.convert_to_point_cloud(depth_img, color_img, self.depth_max)
                if output_dir:
                    output_path = Path(output_dir) / f"{image_id}.{save_format}"
                    if save_format == "pcd":
                        o3d.io.write_point_cloud(str(output_path), pcd)
                    elif save_format == "ply":
                        o3d.io.write_point_cloud(str(output_path), pcd)
                    elif save_format == "npz":
                        self.save_point_cloud_numpy(pcd, output_path)
                    elif save_format == "csv":
                        self.save_point_cloud_csv(pcd, output_path)
                else:
                    pcd_list.append(pcd)
            else:
                print(f"Image data missing for ID {image_id}, skipping...")

        if not output_dir:
            return pcd_list

    def visualize_point_cloud(self, pcd):
        """Visualizes a point cloud using Plotly.

        Args:
            pcd (o3d.geometry.PointCloud): The point cloud object to visualize.
        """
        if self.logger:
            self.logger.log_args("PointCloudProcessor.visualize_point_cloud", pcd=pcd)
        points = np.asarray(pcd.points)
        colors = np.asarray(pcd.colors)  # Scale colors up as plotly expects colors in [0, 255]

        fig = go.Figure(
            data=[
                go.Scatter3d(
                    x=points[:, 0],
                    y=points[:, 1],
                    z=points[:, 2],
                    mode="markers",
                    marker=dict(
                        size=3,
                        color=colors,  # Color mapping
                        opacity=0.8,  # Slightly transparent
                    ),
                )
            ],
            layout=dict(
                scene=dict(
                    xaxis=dict(visible=True),
                    yaxis=dict(visible=True),
                    zaxis=dict(visible=True),
                    aspectmode="data",  # this controls the scale of the axes
                )
            ),
        )
        # Update layout and camera angles for better initial viewing
        fig.update_layout(
            scene=dict(
                xaxis=dict(visible=True),
                yaxis=dict(visible=True),
                zaxis=dict(visible=True),
                aspectmode="data",  # Keep the scale of axes
            ),
            margin=dict(l=0, r=0, b=0, t=0),  # Reduce padding to maximize plot area
        )

        fig.show()
