import cv2
import numpy as np
import math

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
    def __init__(self, img):
        self._img = img

    def get_perspective(self, FOV, THETA, PHI, height, width):
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
        XY = lonlat2XY(lonlat, shape=self._img.shape).astype(np.float32)
        persp = cv2.remap(self._img, XY[..., 0], XY[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
        return persp

    # def get_fisheye(self):
    #     rows,cols,c = self._img.shape
    #     R = np.int(cols/2/math.pi)
    #     D = R*2
    #     cx = R
    #     cy = R
    #     new_img = np.zeros((D,D,c),dtype = np.uint8)
    #     new_img[:,:,:] = 255

    #     for i in range(D):
    #         for j in range(D):
    #             r = math.sqrt((i-cx)**2+(j-cy)**2)
    #             if r > R:
    #                 continue
    #             tan_inv = np.arctan((j-cy)/(i-cx+1e-10))
    #             if(i<cx):
    #                 theta = math.pi/2+tan_inv
    #             else:
    #                 theta = math.pi*3/2+tan_inv
    #             xp = np.int(np.floor(theta/2/math.pi*cols))
    #             yp = np.int(np.floor(r/R*rows)-1)
    #             new_img[j,i] = self._img[yp,xp]
    #     return new_img

    def get_fisheye(self):
        rows, cols, c = self._img.shape
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
        new_img[y[mask], x[mask]] = self._img[yp, xp]

        return new_img
