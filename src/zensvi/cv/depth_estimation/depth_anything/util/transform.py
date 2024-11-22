import math

import cv2
import numpy as np
import torch
import torch.nn.functional as F


def apply_min_size(sample, size, image_interpolation_method=cv2.INTER_AREA):
    """Resize the sample to ensure the given size while keeping the aspect ratio.

    Args:
        sample (dict): The sample containing image, disparity, and mask.
        size (tuple): The desired output size as (height, width).
        image_interpolation_method (int, optional): The interpolation method for resizing.
            Defaults to cv2.INTER_AREA.

    Returns:
        tuple: The new size as (width, height).
    """
    shape = list(sample["disparity"].shape)

    if shape[0] >= size[0] and shape[1] >= size[1]:
        return sample

    scale = [0, 0]
    scale[0] = size[0] / shape[0]
    scale[1] = size[1] / shape[1]

    scale = max(scale)

    shape[0] = math.ceil(scale * shape[0])
    shape[1] = math.ceil(scale * shape[1])

    # Resize the image, disparity, and mask
    sample["image"] = cv2.resize(sample["image"], tuple(shape[::-1]), interpolation=image_interpolation_method)

    sample["disparity"] = cv2.resize(sample["disparity"], tuple(shape[::-1]), interpolation=cv2.INTER_NEAREST)
    sample["mask"] = cv2.resize(
        sample["mask"].astype(np.float32),
        tuple(shape[::-1]),
        interpolation=cv2.INTER_NEAREST,
    )
    sample["mask"] = sample["mask"].astype(bool)

    return tuple(shape)


class Resize(object):
    """Resize sample to a given size (width, height)."""

    def __init__(
        self,
        width,
        height,
        resize_target=True,
        keep_aspect_ratio=False,
        ensure_multiple_of=1,
        resize_method="lower_bound",
        image_interpolation_method=cv2.INTER_AREA,
    ):
        """Initialize the Resize object.

        Args:
            width (int): Desired output width.
            height (int): Desired output height.
            resize_target (bool, optional): Whether to resize the full sample (image, mask, target) or just the image.
                Defaults to True.
            keep_aspect_ratio (bool, optional): Whether to preserve the aspect ratio of the input sample. If True,
                output dimensions may differ from specified width/height based on resize_method. Defaults to False.
            ensure_multiple_of (int, optional): Constrains output width and height to be multiples of this value.
                Defaults to 1.
            resize_method (str, optional): Method to use for resizing. Options are:
                - "lower_bound": Output dimensions will be at least as large as specified size
                - "upper_bound": Output dimensions will be at most as large as specified size
                - "minimal": Uses minimal scaling possible (may be smaller than specified size)
                Defaults to "lower_bound".
            image_interpolation_method (int, optional): Interpolation method to use for image resizing.
                Defaults to cv2.INTER_AREA.

        """
        self.__width = width
        self.__height = height

        self.__resize_target = resize_target
        self.__keep_aspect_ratio = keep_aspect_ratio
        self.__multiple_of = ensure_multiple_of
        self.__resize_method = resize_method
        self.__image_interpolation_method = image_interpolation_method

    def constrain_to_multiple_of(self, x, min_val=0, max_val=None):
        """Constrain a value to be a multiple of a specified value.

        Args:
            x (float): The value to constrain.
            min_val (int, optional): The minimum value. Defaults to 0.
            max_val (int, optional): The maximum value. Defaults to None.

        Returns:
            int: The constrained value.
        """
        y = (np.round(x / self.__multiple_of) * self.__multiple_of).astype(int)

        if max_val is not None and y > max_val:
            y = (np.floor(x / self.__multiple_of) * self.__multiple_of).astype(int)

        if y < min_val:
            y = (np.ceil(x / self.__multiple_of) * self.__multiple_of).astype(int)

        return y

    def get_size(self, width, height):
        """Determine the new size based on the desired width and height.

        Args:
            width (int): The original width.
            height (int): The original height.

        Returns:
            tuple: The new size as (new_width, new_height).
        """
        # Determine new height and width
        scale_height = self.__height / height
        scale_width = self.__width / width

        if self.__keep_aspect_ratio:
            if self.__resize_method == "lower_bound":
                # Scale such that output size is lower bound
                if scale_width > scale_height:
                    # Fit width
                    scale_height = scale_width
                else:
                    # Fit height
                    scale_width = scale_height
            elif self.__resize_method == "upper_bound":
                # Scale such that output size is upper bound
                if scale_width < scale_height:
                    # Fit width
                    scale_height = scale_width
                else:
                    # Fit height
                    scale_width = scale_height
            elif self.__resize_method == "minimal":
                # Scale as least as possible
                if abs(1 - scale_width) < abs(1 - scale_height):
                    # Fit width
                    scale_height = scale_width
                else:
                    # Fit height
                    scale_width = scale_height
            else:
                raise ValueError(f"resize_method {self.__resize_method} not implemented")

        if self.__resize_method == "lower_bound":
            new_height = self.constrain_to_multiple_of(scale_height * height, min_val=self.__height)
            new_width = self.constrain_to_multiple_of(scale_width * width, min_val=self.__width)
        elif self.__resize_method == "upper_bound":
            new_height = self.constrain_to_multiple_of(scale_height * height, max_val=self.__height)
            new_width = self.constrain_to_multiple_of(scale_width * width, max_val=self.__width)
        elif self.__resize_method == "minimal":
            new_height = self.constrain_to_multiple_of(scale_height * height)
            new_width = self.constrain_to_multiple_of(scale_width * width)
        else:
            raise ValueError(f"resize_method {self.__resize_method} not implemented")

        return (new_width, new_height)

    def __call__(self, sample):
        """Resize the sample based on the initialized parameters.

        Args:
            sample (dict): The sample containing image, disparity, depth, and mask.

        Returns:
            dict: The resized sample.
        """
        width, height = self.get_size(sample["image"].shape[1], sample["image"].shape[0])

        # Resize sample
        sample["image"] = cv2.resize(
            sample["image"],
            (width, height),
            interpolation=self.__image_interpolation_method,
        )

        if self.__resize_target:
            if "disparity" in sample:
                sample["disparity"] = cv2.resize(
                    sample["disparity"],
                    (width, height),
                    interpolation=cv2.INTER_NEAREST,
                )

            if "depth" in sample:
                sample["depth"] = cv2.resize(sample["depth"], (width, height), interpolation=cv2.INTER_NEAREST)

            if "semseg_mask" in sample:
                sample["semseg_mask"] = F.interpolate(
                    torch.from_numpy(sample["semseg_mask"]).float()[None, None, ...],
                    (height, width),
                    mode="nearest",
                ).numpy()[0, 0]

            if "mask" in sample:
                sample["mask"] = cv2.resize(
                    sample["mask"].astype(np.float32),
                    (width, height),
                    interpolation=cv2.INTER_NEAREST,
                )

        return sample


class NormalizeImage(object):
    """Normalize image by given mean and standard deviation."""

    def __init__(self, mean, std):
        """Initialize the NormalizeImage object.

        Args:
            mean (float): The mean value for normalization.
            std (float): The standard deviation value for normalization.
        """
        self.__mean = mean
        self.__std = std

    def __call__(self, sample):
        """Normalize the image in the sample.

        Args:
            sample (dict): The sample containing the image.

        Returns:
            dict: The sample with the normalized image.
        """
        sample["image"] = (sample["image"] - self.__mean) / self.__std

        return sample


class PrepareForNet(object):
    """Prepare sample for usage as network input."""

    def __init__(self):
        """Initialize the PrepareForNet object."""
        pass

    def __call__(self, sample):
        """Prepare the sample for network input.

        Args:
            sample (dict): The sample containing the image, mask, depth, and semseg_mask.

        Returns:
            dict: The prepared sample for network input.
        """
        image = np.transpose(sample["image"], (2, 0, 1))
        sample["image"] = np.ascontiguousarray(image).astype(np.float32)

        if "mask" in sample:
            sample["mask"] = sample["mask"].astype(np.float32)
            sample["mask"] = np.ascontiguousarray(sample["mask"])

        if "depth" in sample:
            depth = sample["depth"].astype(np.float32)
            sample["depth"] = np.ascontiguousarray(depth)

        if "semseg_mask" in sample:
            sample["semseg_mask"] = sample["semseg_mask"].astype(np.float32)
            sample["semseg_mask"] = np.ascontiguousarray(sample["semseg_mask"])

        return sample
