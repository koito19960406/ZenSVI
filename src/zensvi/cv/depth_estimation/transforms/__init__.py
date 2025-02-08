"""Transforms module for depth estimation.

This module contains the transforms used for preprocessing images for depth estimation.
"""

import numpy as np
import torch.nn as nn
from torchvision.transforms import Normalize


class Resize(object):
    """Resize sample to given size (width, height)."""

    def __init__(
        self,
        width,
        height,
        resize_target=True,
        keep_aspect_ratio=False,
        ensure_multiple_of=1,
        resize_method="lower_bound",
    ):
        """Initializes the Resize object.

        Args:
            width (int): Desired output width.
            height (int): Desired output height.
            resize_target (bool, optional): If True, resize the full sample (image, mask, target). Defaults to True.
            keep_aspect_ratio (bool, optional): If True, keep the aspect ratio of the input sample. Defaults to False.
            ensure_multiple_of (int, optional): Output width and height is constrained to be multiple of this parameter. Defaults to 1.
            resize_method (str, optional): Method of resizing. Options are "lower_bound", "upper_bound", "minimal". Defaults to "lower_bound".
        """
        self.__width = width
        self.__height = height
        self.__keep_aspect_ratio = keep_aspect_ratio
        self.__multiple_of = ensure_multiple_of
        self.__resize_method = resize_method

    def constrain_to_multiple_of(self, x, min_val=0, max_val=None):
        """Constrain a value to be a multiple of a specified number.

        Args:
            x (float): Value to constrain.
            min_val (int, optional): Minimum value. Defaults to 0.
            max_val (int, optional): Maximum value. Defaults to None.

        Returns:
            int: Constrained value.
        """
        y = (np.round(x / self.__multiple_of) * self.__multiple_of).astype(int)

        if max_val is not None and y > max_val:
            y = (np.floor(x / self.__multiple_of) * self.__multiple_of).astype(int)

        if y < min_val:
            y = (np.ceil(x / self.__multiple_of) * self.__multiple_of).astype(int)

        return y

    def get_size(self, width, height):
        """Determine the new size based on the resizing method.

        Args:
            width (int): Original width.
            height (int): Original height.

        Returns:
            tuple: New width and height.
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

    def __call__(self, x):
        """Resize the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (N, C, H, W).

        Returns:
            torch.Tensor: Resized tensor.
        """
        width, height = self.get_size(*x.shape[-2:][::-1])
        return nn.functional.interpolate(x, (height, width), mode="bilinear", align_corners=True)


class PrepareForNet(object):
    """Prepares input for network by resizing and normalizing."""

    def __init__(
        self,
        resize_mode="minimal",
        keep_aspect_ratio=True,
        img_size=384,
        do_resize=True,
    ):
        """Initializes the PrepareForNet object.

        Args:
            resize_mode (str, optional): Method of resizing. Defaults to "minimal".
            keep_aspect_ratio (bool, optional): If True, keep the aspect ratio of the input image. Defaults to True.
            img_size (int, tuple, optional): Size of the input image. Defaults to 384.
            do_resize (bool, optional): If True, perform resizing. Defaults to True.
        """
        if isinstance(img_size, int):
            img_size = (img_size, img_size)
        net_h, net_w = img_size
        self.normalization = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.resizer = (
            Resize(
                net_w,
                net_h,
                keep_aspect_ratio=keep_aspect_ratio,
                ensure_multiple_of=14,
                resize_method=resize_mode,
            )
            if do_resize
            else nn.Identity()
        )

    def __call__(self, x):
        """Apply normalization and resizing to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (N, C, H, W).

        Returns:
            torch.Tensor: Normalized and resized tensor.
        """
        return self.normalization(self.resizer(x))


# Alias for backward compatibility
NormalizeImage = Normalize
