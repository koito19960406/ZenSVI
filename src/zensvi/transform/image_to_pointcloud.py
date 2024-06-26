import numpy as np
import open3d as o3d
import copy
from PIL import Image
from pathlib import Path
from typing import List, Dict
import plotly.graph_objects as go


class PointCloudProcessor:
    def __init__(self, image_folder: str, depth_folder: str, output_coordinate_scale: float = 45, depth_max: float = 255):
        """
        Initializes the PointCloudProcessor with necessary parameters.

        Args:
            image_folder (str): Path to the folder containing depth and color images.
            output_coordinate_scale (float): Scaling factor for the coordinates.
            depth_max (float): The maximum depth value to normalize the depth data.
        """
        self.image_folder = Path(image_folder)
        self.depth_folder = Path(depth_folder)
        self.output_coordinate_scale = output_coordinate_scale
        self.depth_max = depth_max
        self.validate_paths()

    def validate_paths(self):
        if not self.image_folder.exists():
            raise FileNotFoundError(f"Image folder {self.image_folder} does not exist")
        if not self.depth_folder.exists():
            raise FileNotFoundError(f"Depth folder {self.depth_folder} does not exist")

    def load_images(self, data):
        """
        Preloads all images specified in the data DataFrame to optimize the point cloud generation process.

        Args:
            data (DataFrame): DataFrame containing image ids for processing.

        Returns:
            Dict: Dictionary containing loaded PIL image objects for depth and color images.
        """
        images = {}
        for image_id in data['id'].unique():
            depth_path = self.depth_folder / f'{image_id}_depth.png'
            color_path = self.image_folder / f'{image_id}_color.jpg'
            if depth_path.exists() and color_path.exists():
                images[image_id] = {
                    'depth': np.array(Image.open(depth_path).convert('L')),
                    'color': np.array(Image.open(color_path))
                }
            else:
                print(f"Warning: Missing images for ID {image_id}")
        return images
    
    def convert_to_point_cloud(self, depth_img, color_img, depth_max):
        """
        Converts a single depth and color image pair to a point cloud.

        Args:
            depth_img (np.ndarray): The depth image.
            color_img (np.ndarray): The corresponding color image.
            depth_max (float): Maximum value for depth normalization.

        Returns:
            o3d.geometry.PointCloud: The generated point cloud with color.
        """

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
                r2 = (255-r1) / depth_max

                # An alternative way to reporject the pixels, to be optimized
                #xx = (r2 * np.cos(a) * np.cos(b) / (np.log10(2 + 6 * (y / (ys - 1)))))
                #yy = (r2 * np.sin(a) * np.cos(b) / (np.log10(2 + 6 * (y / (ys - 1)))))
                #zz = 1.2 * r2 * np.sin(b)

                xx =  3 * r2**4 * np.cos(a) * np.cos(b) /(np.log(1.1 + 5 * (y / (ys - 1))))
                yy =  3 * r2**4 * np.sin(a) * np.cos(b) /(np.log(1.1 + 5 * (y / (ys - 1))))
                zz =  (2 * r2) ** 2 * np.sin(b)

                c = color_img[y, x] / 255.0  # Normalizing color
                points.append([xx, yy, zz])
                colors.append(c)
    
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np.array(points))
        pcd.colors = o3d.utility.Vector3dVector(np.array(colors))
        return pcd

    
    def transform_point_cloud(self, pcd, origin_x, origin_y, angle, box_extent, box_center):
        """
        Transforms the point cloud by translating, rotating, and cropping based on given parameters.

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
        translation_vector = np.array([origin_x, origin_y, 0])
        pcd_mv = copy.deepcopy(pcd).translate(translation_vector, relative=False)
        pcd_mv.rotate(pcd.get_rotation_matrix_from_xyz((0, 0, angle)))
        obb = o3d.geometry.OrientedBoundingBox(center=box_center, R=pcd_mv.get_rotation_matrix_from_xyz((0, 0, angle)), extent=box_extent)
        pcd_mv = pcd_mv.crop(obb)
        return pcd_mv
    

    def process_multiple_images(self, data):
        """
        Generates a point cloud for each entry in the data based on pre-loaded depth and color images.

        Args:
            data (DataFrame): DataFrame containing image ids, coordinates, and headings.

        Returns:
            List[o3d.geometry.PointCloud]: List of unprocessed point clouds with color information.
        """
        images = self.load_images(data)
        pcd_list = []

        for idx, row in data.iterrows():
            image_id = row['id']
            if image_id in images:
                depth_img = images[image_id]['depth']
                color_img = images[image_id]['color']

                pcd = self.convert_to_point_cloud(depth_img, color_img, self.depth_max)
                pcd_list.append(pcd)
            else:
                print(f"Image data missing for ID {image_id}, skipping...")

        return pcd_list

    def visualize_point_cloud(self, pcd):
        """
        Visualizes a point cloud using Plotly.

        Args:
            pcd (o3d.geometry.PointCloud): The point cloud object to visualize.
        """
        points = np.asarray(pcd.points)
        colors = np.asarray(pcd.colors)  # Scale colors up as plotly expects colors in [0, 255]
  

        fig = go.Figure(
            data=[
                go.Scatter3d(
                    x=points[:, 0], y=points[:, 1], z=points[:, 2],
                    mode='markers',
                    marker=dict(
                size=3,  # Increase marker size
                color=colors,  # Color mapping
                opacity=0.8  # Slightly transparent
            )
                )
            ],
            layout=dict(
                scene=dict(
                    xaxis=dict(visible=True),
                    yaxis=dict(visible=True),
                    zaxis=dict(visible=True),
                    aspectmode='data'  # this controls the scale of the axes
                )
            )
        )
        # Update layout and camera angles for better initial viewing
        fig.update_layout(
            scene=dict(
                xaxis=dict(visible=True),
                yaxis=dict(visible=True),
                zaxis=dict(visible=True),
                aspectmode='data',  # Keep the scale of axes

            ),
            margin=dict(l=0, r=0, b=0, t=0)  # Reduce padding to maximize plot area
        )

        fig.show()



