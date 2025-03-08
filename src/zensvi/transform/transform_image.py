import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Union

import cv2
import numpy as np

from zensvi.utils.log import Logger, verbosity_tqdm


def _xyz2lonlat(xyz):
    """Converts 3D Cartesian coordinates (x, y, z) to geographic coordinates (longitude,
    latitude).

    Args:
        xyz (np.ndarray): An array of shape (..., 3) containing 3D Cartesian coordinates.

    Returns:
        np.ndarray: An array of shape (..., 2) containing longitude and latitude coordinates.
    """
    atan2 = np.arctan2
    asin = np.arcsin

    norm = np.linalg.norm(xyz, axis=-1, keepdims=True)
    xyz_norm = xyz / norm
    x = xyz_norm[..., 0:1]
    y = xyz_norm[..., 1:2]
    z = xyz_norm[..., 2:]

    lon = atan2(x, z)
    lat = asin(y)
    lst = [lon, lat]

    out = np.concatenate(lst, axis=-1)
    return out


def _lonlat2XY(lonlat, shape):
    """Converts geographic coordinates (longitude, latitude) to pixel coordinates (X, Y)
    based on an image shape.

    Args:
        lonlat (np.ndarray): An array of shape (..., 2) containing longitude and latitude coordinates.
        shape (tuple): A tuple (height, width) representing the shape of the image.

    Returns:
        np.ndarray: An array of shape (..., 2) containing pixel coordinates.
    """
    X = (lonlat[..., 0:1] / (2 * np.pi) + 0.5) * (shape[1] - 1)
    Y = (lonlat[..., 1:] / (np.pi) + 0.5) * (shape[0] - 1)
    lst = [X, Y]
    out = np.concatenate(lst, axis=-1)

    return out


class ImageTransformer:
    """Transforms images by applying various projections such as fisheye and perspective
    adjustments.

    Args:
        dir_input (Union[str, Path]): Input directory containing images.
        dir_output (Union[str, Path]): Output directory where transformed images will be saved.

    Raises:
        TypeError: If the input or output directories are not specified as string or Path objects.
    """

    def __init__(
        self,
        dir_input: Union[str, Path],
        dir_output: Union[str, Path],
        log_path: Union[str, Path] = None,
        verbosity: int = 1,
    ):
        if isinstance(dir_input, str):
            dir_input = Path(dir_input)
        elif not isinstance(dir_input, Path):
            raise TypeError("dir_input must be a str or Path object.")
        if isinstance(dir_output, str):
            dir_output = Path(dir_output)
        elif not isinstance(dir_output, Path):
            raise TypeError("dir_output must be a str or Path object.")
        self._dir_input = dir_input
        self._dir_output = dir_output
        # initialize the logger
        self.log_path = log_path
        if self.log_path is not None:
            self.logger = Logger(log_path)
        else:
            self.logger = None
        self.verbosity = verbosity

    @property
    def dir_input(self):
        """Property for the input directory.

        Returns:
            Path: dir_input
        """
        return self._dir_input

    @dir_input.setter
    def dir_input(self, value):
        if isinstance(value, str):
            value = Path(value)
        elif not isinstance(value, Path):
            raise TypeError("dir_input must be a str or Path object.")
        self._dir_input = value

    @property
    def dir_output(self):
        """Property for the output directory.

        Returns:
            Path: dir_output
        """
        return self._dir_output

    @dir_output.setter
    def dir_output(self, value):
        if isinstance(value, str):
            value = Path(value)
        elif not isinstance(value, Path):
            raise TypeError("dir_output must be a str or Path object.")
        self._dir_output = value

    def perspective(self, img, FOV, THETA, PHI, height, width):
        """Transforms an image to simulate a perspective view from specific angles.

        Args:
            img (np.ndarray): Source image to transform.
            FOV (float): Field of view in degrees.
            THETA (float): Rotation around the vertical axis in degrees.
            PHI (float): Tilt angle in degrees.
            height (int): Height of the output image.
            width (int): Width of the output image.

        Returns:
            np.ndarray: Transformed image.
        """
        f = 0.5 * width * 1 / np.tan(0.5 * FOV / 180.0 * np.pi)
        cx = (width - 1) / 2.0
        cy = (height - 1) / 2.0
        K = np.array(
            [
                [f, 0, cx],
                [0, f, cy],
                [0, 0, 1],
            ],
            np.float32,
        )
        K_inv = np.linalg.inv(K)

        x = np.arange(width)
        y = np.arange(height)
        x, y = np.meshgrid(x, y)
        z = np.ones_like(x)
        xyz = np.concatenate([x[..., None], y[..., None], z[..., None]], axis=-1)
        xyz = xyz @ K_inv.T

        y_axis = np.array([0.0, 1.0, 0.0], np.float32)
        x_axis = np.array([1.0, 0.0, 0.0], np.float32)
        R1, _ = cv2.Rodrigues(y_axis * np.radians(THETA))
        R2, _ = cv2.Rodrigues(np.dot(R1, x_axis) * np.radians(PHI))
        R = R2 @ R1
        xyz = xyz @ R.T
        lonlat = _xyz2lonlat(xyz)
        XY = _lonlat2XY(lonlat, shape=img.shape).astype(np.float32)
        persp = cv2.remap(img, XY[..., 0], XY[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
        return persp

    def equidistant_fisheye(self, img):
        """Transforms an image to an equidistant fisheye projection.

        Args:
            img (np.ndarray): Source image to transform.

        Returns:
            np.ndarray: Fisheye projected image with transparent background.
        """
        # Implementation
        rows, cols = img.shape[:2]

        # Determine if the input has an alpha channel
        has_alpha = img.shape[2] == 4 if len(img.shape) > 2 else False

        R = cols / (2 * math.pi)
        D = int(2 * R)
        cx, cy = R, R

        x, y = np.meshgrid(np.arange(D), np.arange(D))
        r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
        theta = np.arctan2(y - cy, x - cx)

        xp = np.floor((theta + np.pi) * cols / (2 * np.pi)).astype(int)
        yp = np.floor(r * rows / R).astype(int)

        mask = r < R

        # Create a 4-channel RGBA image with transparent background
        new_img = np.zeros((D, D, 4), dtype=np.uint8)
        # Set the alpha channel to 0 (fully transparent)
        new_img[:, :, 3] = 0

        # Only apply colors inside the mask, and set alpha to 255 (fully opaque)
        if has_alpha:
            # If input has alpha channel, copy it
            new_img[y[mask], x[mask]] = img[yp[mask], xp[mask]]
        else:
            # For RGB input, copy color channels and set alpha to 255
            new_img[y[mask], x[mask], :3] = img[yp[mask], xp[mask]]
            new_img[y[mask], x[mask], 3] = 255

        return new_img

    def orthographic_fisheye(self, img):
        """Transforms an image to an orthographic fisheye projection.

        Args:
            img (np.ndarray): Source image to transform.

        Returns:
            np.ndarray: Fisheye projected image with transparent background.
        """
        rows, cols = img.shape[:2]

        # Determine if the input has an alpha channel
        has_alpha = img.shape[2] == 4 if len(img.shape) > 2 else False

        R = cols / (2 * math.pi)
        D = int(2 * R)
        cx, cy = R, R

        x, y = np.meshgrid(np.arange(D), np.arange(D))
        r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
        theta = np.arctan2(y - cy, x - cx)

        xp = np.floor((theta + np.pi) * cols / (2 * np.pi)).astype(int)
        yp = np.floor((2 / np.pi) * np.arcsin(r / R) * rows).astype(int)

        mask = r < R

        # Create a 4-channel RGBA image with transparent background
        new_img = np.zeros((D, D, 4), dtype=np.uint8)
        # Set the alpha channel to 0 (fully transparent)
        new_img[:, :, 3] = 0

        # Only apply colors inside the mask, and set alpha to 255 (fully opaque)
        if has_alpha:
            # If input has alpha channel, copy it
            new_img[y[mask], x[mask]] = img[yp[mask], xp[mask]]
        else:
            # For RGB input, copy color channels and set alpha to 255
            new_img[y[mask], x[mask], :3] = img[yp[mask], xp[mask]]
            new_img[y[mask], x[mask], 3] = 255

        return new_img

    def stereographic_fisheye(self, img):
        """Transforms an image to a stereographic fisheye projection.

        Args:
            img (np.ndarray): Source image to transform.

        Returns:
            np.ndarray: Fisheye projected image with transparent background.
        """
        rows, cols = img.shape[:2]

        # Determine if the input has an alpha channel
        has_alpha = img.shape[2] == 4 if len(img.shape) > 2 else False

        R = cols / (2 * math.pi)
        D = int(2 * R)
        cx, cy = R, R

        # Create a meshgrid of coordinates
        x, y = np.meshgrid(np.arange(D), np.arange(D))
        r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
        theta = np.arctan2(y - cy, x - cx)

        # Calculate the new positions in the source image
        xp = np.floor((theta + np.pi) * cols / (2 * np.pi)).astype(np.int32)
        yp = np.floor(rows * np.sin(r / R) / np.sin(1)).astype(np.int32)

        # Clip the values of yp and xp to be within the valid range
        yp = np.clip(yp, 0, rows - 1)
        xp = np.clip(xp, 0, cols - 1)

        # Create a mask for the valid fisheye region
        mask = r <= R

        # Create a 4-channel RGBA image with transparent background
        new_img = np.zeros((D, D, 4), dtype=np.uint8)
        # Set the alpha channel to 0 (fully transparent)
        new_img[:, :, 3] = 0

        # Only apply colors inside the mask, and set alpha to 255 (fully opaque)
        if has_alpha:
            # If input has alpha channel, copy it
            new_img[y[mask], x[mask]] = img[yp[mask], xp[mask]]
        else:
            # For RGB input, copy color channels and set alpha to 255
            new_img[y[mask], x[mask], :3] = img[yp[mask], xp[mask]]
            new_img[y[mask], x[mask], 3] = 255

        return new_img

    def equisolid_fisheye(self, img):
        """Transforms an image to an equisolid fisheye projection.

        Args:
            img (np.ndarray): Source image to transform.

        Returns:
            np.ndarray: Fisheye projected image with transparent background.
        """
        rows, cols = img.shape[:2]

        # Determine if the input has an alpha channel
        has_alpha = img.shape[2] == 4 if len(img.shape) > 2 else False

        R = cols / (2 * math.pi)
        D = int(2 * R)
        cx, cy = R, R

        x, y = np.meshgrid(np.arange(D), np.arange(D))
        r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
        theta = np.arctan2(y - cy, x - cx)

        xp = np.floor((theta + np.pi) * cols / (2 * np.pi)).astype(int)
        yp = np.floor(2 * np.sin(r / (2 * R)) * rows).astype(int)

        mask = r < R

        # Create a 4-channel RGBA image with transparent background
        new_img = np.zeros((D, D, 4), dtype=np.uint8)
        # Set the alpha channel to 0 (fully transparent)
        new_img[:, :, 3] = 0

        # Only apply colors inside the mask, and set alpha to 255 (fully opaque)
        if has_alpha:
            # If input has alpha channel, copy it
            new_img[y[mask], x[mask]] = img[yp[mask], xp[mask]]
        else:
            # For RGB input, copy color channels and set alpha to 255
            new_img[y[mask], x[mask], :3] = img[yp[mask], xp[mask]]
            new_img[y[mask], x[mask], 3] = 255

        return new_img

    def transform_images(
        self,
        style_list: str = "perspective equidistant_fisheye orthographic_fisheye stereographic_fisheye equisolid_fisheye",
        FOV: Union[int, float] = 90,
        theta: Union[int, float] = 90,
        phi: Union[int, float] = 0,
        aspects: tuple = (9, 16),
        show_size: Union[int, float] = 100,
        use_upper_half: bool = False,
        verbosity: int = None,
    ):
        """Applies specified transformations to all images in the input directory and
        saves them in the output directory.

        Args:
            style_list (str): Space-separated list of transformation styles to apply. Valid styles include 'perspective',
                            'equidistant_fisheye', 'orthographic_fisheye', 'stereographic_fisheye', and 'equisolid_fisheye'.
            FOV (Union[int, float], optional): Field of view for the 'perspective' style in degrees.
            theta (Union[int, float], optional): Rotation step for generating multiple perspective images in degrees.
            phi (Union[int, float], optional): Tilt angle for the 'perspective' style in degrees.
            aspects (tuple, optional): Aspect ratio of the output images represented as a tuple.
            show_size (Union[int, float], optional): Base size to calculate the dimensions of the output images.
            use_upper_half (bool, optional): If True, only the upper half of the image is used for fisheye transformations.
            verbosity (int, optional): Level of verbosity for progress bars (0=no progress bars, 1=outer loops only, 2=all loops).
                                      If None, uses the instance's verbosity level.

        Raises:
            ValueError: If an invalid style is specified in style_list.

        Notes:
            This method processes images concurrently, leveraging multi-threading to speed up the transformation tasks. It
            automatically splits style_list into individual styles and processes each style, creating appropriate subdirectories
            in the output directory for each style.
        """
        # Use instance verbosity if not specified
        if verbosity is None:
            verbosity = self.verbosity

        if self.logger is not None:
            # record the arguments
            self.logger.log_args(
                "transform_images",
                style_list=style_list,
                FOV=FOV,
                theta=theta,
                phi=phi,
                aspects=aspects,
                show_size=show_size,
                use_upper_half=use_upper_half,
                verbosity=verbosity,
            )
        # raise an error if the style_list is a list
        if isinstance(style_list, list):
            raise ValueError("Please input the correct image style as a string, not a list.")
        # check if there's anything other than "perspective" and "fisheye"
        style_list = style_list.split()
        if not all(
            style
            in [
                "perspective",
                "equidistant_fisheye",
                "orthographic_fisheye",
                "stereographic_fisheye",
                "equisolid_fisheye",
            ]
            for style in style_list
        ):
            raise ValueError(
                "Please input the correct image style. The correct image style should be 'perspective', 'equidistant_fisheye', 'orthographic_fisheye', 'stereographic_fisheye', or 'equisolid_fisheye'"
            )

        # set image file extensions
        image_extensions = [
            ".jpg",
            ".jpeg",
            ".png",
            ".bmp",
            ".gif",
            ".tiff",
            ".tif",
            ".webp",
            ".ico",
            ".jfif",
            ".heic",
            ".heif",
        ]

        def run(path_input, path_output, show_size, style, theta, aspects, FOV):
            # Use IMREAD_UNCHANGED to preserve alpha channel if present
            img_raw = cv2.imread(str(path_input), cv2.IMREAD_UNCHANGED)

            # If the image has no alpha channel, convert it to 3-channel BGR
            if len(img_raw.shape) == 2:  # Grayscale
                img_raw = cv2.cvtColor(img_raw, cv2.COLOR_GRAY2BGR)

            if use_upper_half:
                img_raw = img_raw[: img_raw.shape[0] // 2, :]
            if style == "equidistant_fisheye":
                if not path_output.exists():
                    img_new = self.equidistant_fisheye(img_raw)
                    # Save with transparency (PNG format required for transparency)
                    cv2.imwrite(str(path_output), img_new)

            elif style == "orthographic_fisheye":
                if not path_output.exists():
                    img_new = self.orthographic_fisheye(img_raw)
                    # Save with transparency
                    cv2.imwrite(str(path_output), img_new)

            elif style == "stereographic_fisheye":
                if not path_output.exists():
                    img_new = self.stereographic_fisheye(img_raw)
                    # Save with transparency
                    cv2.imwrite(str(path_output), img_new)

            elif style == "equisolid_fisheye":
                if not path_output.exists():
                    img_new = self.equisolid_fisheye(img_raw)
                    # Save with transparency
                    cv2.imwrite(str(path_output), img_new)

            elif style == "perspective":
                num_images = 360 // theta  # Calculate the number of images based on theta
                thetas = [theta * i for i in range(num_images)]  # Calculate thetas based on step size

                for theta in thetas:
                    height = int(aspects[0] * show_size)
                    width = int(aspects[1] * show_size)
                    aspect_name = "%s--%s" % (aspects[0], aspects[1])
                    path_output_raw = path_output.with_name(
                        f"{path_output.stem}_Direction_{theta}_FOV_{FOV}_aspect_{aspect_name}_raw.png"
                    )
                    if not path_output_raw.exists():
                        img_new = self.perspective(img_raw, FOV, theta, phi, height, width)
                        cv2.imwrite(str(path_output_raw), img_new)

        def process_image(dir_input, dir_output, file_path, show_size, style, theta, aspects, FOV):
            relative_path = file_path.relative_to(dir_input)
            path_output = dir_output / relative_path.with_suffix(".png")
            path_output.parent.mkdir(parents=True, exist_ok=True)
            return file_path, path_output, show_size, style, theta, aspects, FOV

        # Recursive function to get all image files
        def get_image_files(directory):
            for item in directory.iterdir():
                if item.is_file() and item.suffix.lower() in image_extensions:
                    yield item
                elif item.is_dir():
                    yield from get_image_files(item)

        # Check if self.dir_input is a directory or a single file
        if not self.dir_input.is_dir():
            if isinstance(self.dir_input, (str, Path)) and self.dir_input.suffix.lower() in image_extensions:
                dir_input = [Path(self.dir_input)]
                self.dir_input = Path(self.dir_input).parent
            else:
                raise ValueError("Please input a valid directory path or image file.")
        else:
            dir_input = list(get_image_files(self.dir_input))

        for current_style in style_list:
            dir_output = Path(self.dir_output) / current_style
            dir_output.mkdir(parents=True, exist_ok=True)

            with ThreadPoolExecutor() as executor:
                futures = [
                    executor.submit(
                        run,
                        *process_image(
                            self.dir_input,
                            dir_output,
                            file_path,
                            show_size,
                            current_style,
                            theta,
                            aspects,
                            FOV,
                        ),
                    )
                    for file_path in dir_input
                ]
                for future in verbosity_tqdm(
                    as_completed(futures),
                    total=len(futures),
                    desc=f"Converting to {current_style}",
                    verbosity=verbosity,
                    level=1,
                ):
                    future.result()
