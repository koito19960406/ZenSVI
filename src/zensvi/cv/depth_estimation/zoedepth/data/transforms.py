# MIT License

# Copyright (c) 2022 Intelligent Systems Lab Org

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# File author: Shariq Farooq Bhat

import math
import random

import cv2
import numpy as np


class RandomFliplr(object):
    """Horizontal flip of the sample with given probability.

    Attributes:
        probability (float): Probability of flipping the sample horizontally.
    """

    def __init__(self, probability=0.5):
        """Initializes RandomFliplr with a specified probability.

        Args:
            probability (float, optional): Flip probability. Defaults to 0.5.
        """
        self.__probability = probability

    def __call__(self, sample):
        """Flips the sample horizontally with the specified probability.

        Args:
            sample (dict): The input sample containing images and other data.

        Returns:
            dict: The modified sample, potentially with flipped images.
        """
        prob = random.random()

        if prob < self.__probability:
            for k, v in sample.items():
                if len(v.shape) >= 2:
                    sample[k] = np.fliplr(v).copy()

        return sample


def apply_min_size(sample, size, image_interpolation_method=cv2.INTER_AREA):
    """Resize the sample to ensure the given size while keeping the aspect ratio.

    Args:
        sample (dict): The input sample containing images and other data.
        size (tuple): Desired image size (height, width).
        image_interpolation_method (int, optional): Interpolation method for resizing. Defaults to cv2.INTER_AREA.

    Returns:
        tuple: The new size of the sample after resizing.
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

    # resize
    sample["image"] = cv2.resize(sample["image"], tuple(shape[::-1]), interpolation=image_interpolation_method)

    sample["disparity"] = cv2.resize(sample["disparity"], tuple(shape[::-1]), interpolation=cv2.INTER_NEAREST)
    sample["mask"] = cv2.resize(
        sample["mask"].astype(np.float32),
        tuple(shape[::-1]),
        interpolation=cv2.INTER_NEAREST,
    )
    sample["mask"] = sample["mask"].astype(bool)

    return tuple(shape)


class RandomCrop(object):
    """Get a random crop of the sample with the given size (width, height).

    Attributes:
        size (tuple): Desired output size (height, width).
        resize_if_needed (bool): Whether to resize the sample if it's smaller than the desired size.
        image_interpolation_method (int): Interpolation method for resizing.
    """

    def __init__(
        self,
        width,
        height,
        resize_if_needed=False,
        image_interpolation_method=cv2.INTER_AREA,
    ):
        """Initializes the RandomCrop object with specified dimensions and options.

        Args:
            width (int): The desired output width of the crop.
            height (int): The desired output height of the crop.
            resize_if_needed (bool, optional): If True, the sample may be upsampled to ensure
                that a crop of size (width, height) is possible. Defaults to False.
            image_interpolation_method (int, optional): The interpolation method to use for resizing.
                Defaults to cv2.INTER_AREA.
        """
        self.__size = (height, width)
        self.__resize_if_needed = resize_if_needed
        self.__image_interpolation_method = image_interpolation_method

    def __call__(self, sample):
        """Crops the sample randomly to the specified size.

        Args:
            sample (dict): The input sample containing images and other data.

        Returns:
            dict: The modified sample with the random crop applied.

        Raises:
            Exception: If the output size is larger than the input size and resizing is not allowed.
        """
        shape = sample["disparity"].shape

        if self.__size[0] > shape[0] or self.__size[1] > shape[1]:
            if self.__resize_if_needed:
                shape = apply_min_size(sample, self.__size, self.__image_interpolation_method)
            else:
                raise Exception("Output size {} bigger than input size {}.".format(self.__size, shape))

        offset = (
            np.random.randint(shape[0] - self.__size[0] + 1),
            np.random.randint(shape[1] - self.__size[1] + 1),
        )

        for k, v in sample.items():
            if k == "code" or k == "basis":
                continue

            if len(sample[k].shape) >= 2:
                sample[k] = v[
                    offset[0] : offset[0] + self.__size[0],
                    offset[1] : offset[1] + self.__size[1],
                ]

        return sample


class Resize(object):
    """Resize sample to given size (width, height).

    Attributes:
        width (int): Desired output width.
        height (int): Desired output height.
        resize_target (bool): Whether to resize the full sample (image, mask, target).
        keep_aspect_ratio (bool): Whether to keep the aspect ratio of the input sample.
        multiple_of (int): Constrain output width and height to be a multiple of this parameter.
        resize_method (str): Method of resizing (lower_bound, upper_bound, minimal).
        image_interpolation_method (int): Interpolation method for resizing.
        letter_box (bool): Whether to apply letterbox padding.
    """

    def __init__(
        self,
        width,
        height,
        resize_target=True,
        keep_aspect_ratio=False,
        ensure_multiple_of=1,
        resize_method="lower_bound",
        image_interpolation_method=cv2.INTER_AREA,
        letter_box=False,
    ):
        """Initializes Resize with specified dimensions and options.

        Args:
            width (int): Desired output width.
            height (int): Desired output height.
            resize_target (bool, optional): True to resize the full sample (image, mask, target). Defaults to True.
            keep_aspect_ratio (bool, optional): True to keep the aspect ratio of the input sample. Defaults to False.
            ensure_multiple_of (int, optional): Output width and height is constrained to be multiple of this parameter. Defaults to 1.
            resize_method (str, optional): Method of resizing. Defaults to "lower_bound".
            image_interpolation_method (int, optional): Interpolation method for resizing. Defaults to cv2.INTER_AREA.
            letter_box (bool, optional): Whether to apply letterbox padding. Defaults to False.
        """
        self.__width = width
        self.__height = height

        self.__resize_target = resize_target
        self.__keep_aspect_ratio = keep_aspect_ratio
        self.__multiple_of = ensure_multiple_of
        self.__resize_method = resize_method
        self.__image_interpolation_method = image_interpolation_method
        self.__letter_box = letter_box

    def constrain_to_multiple_of(self, x, min_val=0, max_val=None):
        """Constrain a value to be a multiple of a specified number.

        Args:
            x (int): The value to constrain.
            min_val (int, optional): Minimum value. Defaults to 0.
            max_val (int, optional): Maximum value. Defaults to None.

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
        """Determine the new size based on the specified resizing parameters.

        Args:
            width (int): Original width.
            height (int): Original height.

        Returns:
            tuple: The new size (new_width, new_height).
        """
        # determine new height and width
        scale_height = self.__height / height
        scale_width = self.__width / width

        if self.__keep_aspect_ratio:
            if self.__resize_method == "lower_bound":
                # scale such that output size is lower bound
                if scale_width > scale_height:
                    # fit width
                    scale_height = scale_width
                else:
                    # fit height
                    scale_width = scale_height
            elif self.__resize_method == "upper_bound":
                # scale such that output size is upper bound
                if scale_width < scale_height:
                    # fit width
                    scale_height = scale_width
                else:
                    # fit height
                    scale_width = scale_height
            elif self.__resize_method == "minimal":
                # scale as least as possible
                if abs(1 - scale_width) < abs(1 - scale_height):
                    # fit width
                    scale_height = scale_width
                else:
                    # fit height
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

    def make_letter_box(self, sample):
        """Apply letterbox padding to the sample.

        Args:
            sample (np.ndarray): The input sample to be padded.

        Returns:
            np.ndarray: The padded sample.
        """
        top = bottom = (self.__height - sample.shape[0]) // 2
        left = right = (self.__width - sample.shape[1]) // 2
        sample = cv2.copyMakeBorder(sample, top, bottom, left, right, cv2.BORDER_CONSTANT, None, 0)
        return sample

    def __call__(self, sample):
        """Resize the sample to the specified dimensions.

        Args:
            sample (dict): The input sample containing images and other data.

        Returns:
            dict: The modified sample with resized images.
        """
        width, height = self.get_size(sample["image"].shape[1], sample["image"].shape[0])

        # resize sample
        sample["image"] = cv2.resize(
            sample["image"],
            (width, height),
            interpolation=self.__image_interpolation_method,
        )

        if self.__letter_box:
            sample["image"] = self.make_letter_box(sample["image"])

        if self.__resize_target:
            if "disparity" in sample:
                sample["disparity"] = cv2.resize(
                    sample["disparity"],
                    (width, height),
                    interpolation=cv2.INTER_NEAREST,
                )

                if self.__letter_box:
                    sample["disparity"] = self.make_letter_box(sample["disparity"])

            if "depth" in sample:
                sample["depth"] = cv2.resize(sample["depth"], (width, height), interpolation=cv2.INTER_NEAREST)

                if self.__letter_box:
                    sample["depth"] = self.make_letter_box(sample["depth"])

            sample["mask"] = cv2.resize(
                sample["mask"].astype(np.float32),
                (width, height),
                interpolation=cv2.INTER_NEAREST,
            )

            if self.__letter_box:
                sample["mask"] = self.make_letter_box(sample["mask"])

            sample["mask"] = sample["mask"].astype(bool)

        return sample


class ResizeFixed(object):
    """Resize the sample to a fixed size.

    Attributes:
        size (tuple): Desired output size (height, width).
    """

    def __init__(self, size):
        """Initializes ResizeFixed with a specified size.

        Args:
            size (tuple): Desired output size (height, width).
        """
        self.__size = size

    def __call__(self, sample):
        """Resize the sample to the fixed size.

        Args:
            sample (dict): The input sample containing images and other data.

        Returns:
            dict: The modified sample with resized images.
        """
        sample["image"] = cv2.resize(sample["image"], self.__size[::-1], interpolation=cv2.INTER_LINEAR)

        sample["disparity"] = cv2.resize(sample["disparity"], self.__size[::-1], interpolation=cv2.INTER_NEAREST)

        sample["mask"] = cv2.resize(
            sample["mask"].astype(np.float32),
            self.__size[::-1],
            interpolation=cv2.INTER_NEAREST,
        )
        sample["mask"] = sample["mask"].astype(bool)

        return sample


class Rescale(object):
    """Rescale target values to the interval [0, max_val].

    If input is constant, values are set to max_val / 2.

    Attributes:
        max_val (float): Max output value.
        use_mask (bool): Whether to only operate on valid pixels (mask == True).
    """

    def __init__(self, max_val=1.0, use_mask=True):
        """Initializes Rescale with specified parameters.

        Args:
            max_val (float, optional): Max output value. Defaults to 1.0.
            use_mask (bool, optional): Only operate on valid pixels (mask == True). Defaults to True.
        """
        self.__max_val = max_val
        self.__use_mask = use_mask

    def __call__(self, sample):
        """Rescale the disparity values in the sample.

        Args:
            sample (dict): The input sample containing images and other data.

        Returns:
            dict: The modified sample with rescaled disparity values.
        """
        disp = sample["disparity"]

        if self.__use_mask:
            mask = sample["mask"]
        else:
            mask = np.ones_like(disp, dtype=np.bool)

        if np.sum(mask) == 0:
            return sample

        min_val = np.min(disp[mask])
        max_val = np.max(disp[mask])

        if max_val > min_val:
            sample["disparity"][mask] = (disp[mask] - min_val) / (max_val - min_val) * self.__max_val
        else:
            sample["disparity"][mask] = np.ones_like(disp[mask]) * self.__max_val / 2.0

        return sample


# mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
class NormalizeImage(object):
    """Normalize image by given mean and std.

    Attributes:
        mean (np.ndarray): Mean values for normalization.
        std (np.ndarray): Standard deviation values for normalization.
    """

    def __init__(self, mean, std):
        """Initializes NormalizeImage with specified mean and std.

        Args:
            mean (np.ndarray): Mean values for normalization.
            std (np.ndarray): Standard deviation values for normalization.
        """
        self.__mean = mean
        self.__std = std

    def __call__(self, sample):
        """Normalize the image in the sample.

        Args:
            sample (dict): The input sample containing images and other data.

        Returns:
            dict: The modified sample with normalized images.
        """
        sample["image"] = (sample["image"] - self.__mean) / self.__std

        return sample


class DepthToDisparity(object):
    """Convert depth to disparity.

    Removes depth from sample.

    Attributes:
        eps (float): Small value to avoid division by zero.
    """

    def __init__(self, eps=1e-4):
        """Initializes DepthToDisparity with a specified epsilon.

        Args:
            eps (float, optional): Small value to avoid division by zero. Defaults to 1e-4.
        """
        self.__eps = eps

    def __call__(self, sample):
        """Convert depth values to disparity in the sample.

        Args:
            sample (dict): The input sample containing images and other data.

        Returns:
            dict: The modified sample with disparity values.
        """
        assert "depth" in sample

        sample["mask"][sample["depth"] < self.__eps] = False

        sample["disparity"] = np.zeros_like(sample["depth"])
        sample["disparity"][sample["depth"] >= self.__eps] = 1.0 / sample["depth"][sample["depth"] >= self.__eps]

        del sample["depth"]

        return sample


class DisparityToDepth(object):
    """Convert disparity to depth.

    Removes disparity from sample.

    Attributes:
        eps (float): Small value to avoid division by zero.
    """

    def __init__(self, eps=1e-4):
        """Initializes DisparityToDepth with a specified epsilon.

        Args:
            eps (float, optional): Small value to avoid division by zero. Defaults to 1e-4.
        """
        self.__eps = eps

    def __call__(self, sample):
        """Convert disparity values to depth in the sample.

        Args:
            sample (dict): The input sample containing images and other data.

        Returns:
            dict: The modified sample with depth values.
        """
        assert "disparity" in sample

        disp = np.abs(sample["disparity"])
        sample["mask"][disp < self.__eps] = False

        sample["depth"] = np.zeros_like(disp)
        sample["depth"][disp >= self.__eps] = 1.0 / disp[disp >= self.__eps]

        del sample["disparity"]

        return sample


class PrepareForNet(object):
    """Prepare sample for usage as network input."""

    def __init__(self):
        """Initializes PrepareForNet."""
        pass

    def __call__(self, sample):
        """Prepare the sample for network input by transposing and converting data types.

        Args:
            sample (dict): The input sample containing images and other data.

        Returns:
            dict: The modified sample ready for network input.
        """
        image = np.transpose(sample["image"], (2, 0, 1))
        sample["image"] = np.ascontiguousarray(image).astype(np.float32)

        if "mask" in sample:
            sample["mask"] = sample["mask"].astype(np.float32)
            sample["mask"] = np.ascontiguousarray(sample["mask"])

        if "disparity" in sample:
            disparity = sample["disparity"].astype(np.float32)
            sample["disparity"] = np.ascontiguousarray(disparity)

        if "depth" in sample:
            depth = sample["depth"].astype(np.float32)
            sample["depth"] = np.ascontiguousarray(depth)

        return sample
