import cv2
import numpy as np
import math
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

def xyz2lonlat(xyz):
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


def lonlat2XY(lonlat, shape):
    X = (lonlat[..., 0:1] / (2 * np.pi) + 0.5) * (shape[1] - 1)
    Y = (lonlat[..., 1:] / (np.pi) + 0.5) * (shape[0] - 1)
    lst = [X, Y]
    out = np.concatenate(lst, axis=-1)

    return out


class ImageTransformer:
    def __init__(self, dir_input, dir_output):
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
    
    @property
    def dir_input(self):
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
        return self._dir_output

    @dir_output.setter
    def dir_output(self, value):
        if isinstance(value, str):
            value = Path(value)
        elif not isinstance(value, Path):
            raise TypeError("dir_output must be a str or Path object.")
        self._dir_output = value

    def get_perspective(self, img, FOV, THETA, PHI, height, width):
        #
        # THETA is left/right angle, PHI is up/down angle, both in degree
        #

        f = 0.5 * width * 1 / np.tan(0.5 * FOV / 180.0 * np.pi)
        cx = (width - 1) / 2.0
        cy = (height - 1) / 2.0
        K = np.array([
            [f, 0, cx],
            [0, f, cy],
            [0, 0, 1],
        ], np.float32)
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
        lonlat = xyz2lonlat(xyz)
        XY = lonlat2XY(lonlat, shape=img.shape).astype(np.float32)
        persp = cv2.remap(img, XY[..., 0], XY[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
        return persp
    
    def equidistant_fisheye(self, img):
        rows, cols, c = img.shape
        R = cols / (2 * math.pi)
        D = int(2 * R)
        cx, cy = R, R

        x, y = np.meshgrid(np.arange(D), np.arange(D))
        r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
        theta = np.arctan2(y - cy, x - cx)

        xp = np.floor((theta + np.pi) * cols / (2 * np.pi)).astype(int)
        yp = np.floor(r * rows / R).astype(int)
        
        mask = r < R

        new_img = np.zeros((D, D, c), dtype=np.uint8)
        new_img.fill(255)
        new_img[y[mask], x[mask]] = img[yp[mask], xp[mask]]

        return new_img
    
    def orthographic_fisheye(self, img):
        rows, cols, c = img.shape
        R = cols / (2 * math.pi)
        D = int(2 * R)
        cx, cy = R, R

        x, y = np.meshgrid(np.arange(D), np.arange(D))
        r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
        theta = np.arctan2(y - cy, x - cx)

        xp = np.floor((theta + np.pi) * cols / (2 * np.pi)).astype(int)
        yp = np.floor((2 / np.pi) * np.arcsin(r / R) * rows).astype(int)
        
        mask = r < R

        new_img = np.zeros((D, D, c), dtype=np.uint8)
        new_img.fill(255)
        new_img[y[mask], x[mask]] = img[yp[mask], xp[mask]]

        return new_img

    def stereographic_fisheye(self, img):
        rows, cols, c = img.shape
        R = cols / (2 * math.pi)
        D = int(2 * R)
        cx, cy = R, R

        x, y = np.meshgrid(np.arange(D), np.arange(D))
        r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
        theta = np.arctan2(y - cy, x - cx)

        xp = np.floor((theta + np.pi) * cols / (2 * np.pi)).astype(int)
        yp = np.floor(2 * np.tan(r / (2 * R)) * rows).astype(int)
        
        # Clip the values of yp and xp to be within the valid range
        yp = np.clip(yp, 0, rows-1)
        xp = np.clip(xp, 0, cols-1)
        
        mask = r < R

        new_img = np.zeros((D, D, c), dtype=np.uint8)
        new_img.fill(255)
        new_img[y[mask], x[mask]] = img[yp[mask], xp[mask]]

        return new_img

    def equisolid_fisheye(self, img):
        rows, cols, c = img.shape
        R = cols / (2 * math.pi)
        D = int(2 * R)
        cx, cy = R, R

        x, y = np.meshgrid(np.arange(D), np.arange(D))
        r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
        theta = np.arctan2(y - cy, x - cx)

        xp = np.floor((theta + np.pi) * cols / (2 * np.pi)).astype(int)
        yp = np.floor(2 * np.sin(r / (2 * R)) * rows).astype(int)
        
        mask = r < R

        new_img = np.zeros((D, D, c), dtype=np.uint8)
        new_img.fill(255)
        new_img[y[mask], x[mask]] = img[yp[mask], xp[mask]]

        return new_img

    def get_fisheye(self, img):
        rows, cols, c = img.shape
        R = int(cols / (2 * math.pi))
        D = R * 2
        cx = R
        cy = R

        # Create coordinate grids
        x, y = np.meshgrid(np.arange(D), np.arange(D))
        
        # Compute r and theta in a vectorized way
        r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
        theta = np.arctan2(y - cy, x - cx) % (2 * math.pi)

        # Compute corresponding coordinates in the equirectangular image
        xp = np.floor(theta / (2 * math.pi) * cols).astype(int)
        yp = np.floor(r / R * rows).astype(int) - 1

        # Create a mask for pixels within the circle
        mask = r <= R

        # Apply mask to coordinate grids
        xp = xp[mask]
        yp = yp[mask]

        # Create new image and fill with white
        new_img = np.zeros((D, D, c), dtype=np.uint8)
        new_img.fill(255)

        # Copy pixels from original image to new image
        new_img[y[mask], x[mask]] = img[yp, xp]

        return new_img

    def transform_images(self, style_list=["perspective", "equidistant_fisheye", "orthographic_fisheye", "stereographic_fisheye", "equisolid_fisheye"], 
                        FOV = 90, aspects = (9, 16), show_size=100):
        # FOV validation
        if 360 % FOV != 0:
            raise ValueError("FOV must be a divisor of 360.")

        # check if there's anything other than "perspective" and "fisheye"
        if not all(style in ["perspective", "equidistant_fisheye", "orthographic_fisheye", "stereographic_fisheye", "equisolid_fisheye"] for style in style_list):
            raise ValueError("Please input the correct image style. The correct image style should be 'perspective' or 'fisheye'.")

        # set image file extensions
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.tif', '.webp', '.ico', '.jfif', '.heic', '.heif']

        def run(path_input, path_output, show_size, style):
            img_raw = cv2.imread(str(path_input), cv2.IMREAD_COLOR)
            if style == "equidistant_fisheye":
                if not path_output.exists():
                    img_new = self.equidistant_fisheye(img_raw)
                    cv2.imwrite(str(path_output), img_new)
            
            elif style == "orthographic_fisheye":
                if not path_output.exists():
                    img_new = self.orthographic_fisheye(img_raw)
                    cv2.imwrite(str(path_output), img_new)
                
            elif style == "stereographic_fisheye":
                if not path_output.exists():
                    img_new = self.stereographic_fisheye(img_raw)
                    cv2.imwrite(str(path_output), img_new)
            
            elif style == "equisolid_fisheye":
                if not path_output.exists():
                    img_new = self.equisolid_fisheye(img_raw)
                    cv2.imwrite(str(path_output), img_new)

            elif style == "perspective":
                num_images = 360 // FOV  # Calculate the number of images
                thetas = [FOV * i for i in range(num_images)]  # Calculate thetas based on FOV
                aspects_v = (aspects[0], aspects[1] / num_images)  # Set aspects_v based on the number of images

                for theta in thetas:
                    height = int(aspects_v[0] * show_size)
                    width = int(aspects_v[1] * show_size)
                    aspect_name = '%s--%s' % (aspects[0], aspects[1])
                    path_output_raw = path_output.with_name(f'{path_output.stem}_Direction_{theta}_FOV_{FOV}_aspect_{aspect_name}_raw.png')
                    if not path_output_raw.exists(): 
                        img_new = self.get_perspective(img_raw, FOV, theta, 0, height, width)
                        cv2.imwrite(str(path_output_raw), img_new)

        def process_image(dir_input, dir_output, name, show_size, style):
            path_input = dir_input / name.name
            path_output = dir_output / (name.stem + ".png")
            return path_input, path_output, show_size, style

        for current_style in style_list:
            dir_output = Path(self.dir_output) / current_style
            dir_output.mkdir(parents=True, exist_ok=True)

            with ThreadPoolExecutor() as executor:
                futures = [executor.submit(run, *process_image(self.dir_input, dir_output, name, show_size, current_style)) \
                    for name in self.dir_input.rglob('*') if name.suffix.lower() in image_extensions]
                for future in tqdm(as_completed(futures), total=len(futures), desc=f"Converting to {current_style}"):
                    future.result()