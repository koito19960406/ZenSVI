# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math
import random

import numpy as np


class MaskingGenerator:
    """Generates masking patches for images.

    This class is responsible for creating a mask that can be applied to an image,
    allowing for a specified number of patches to be masked out based on various
    parameters such as aspect ratio and size.

    Attributes:
        height (int): The height of the input image.
        width (int): The width of the input image.
        num_patches (int): Total number of patches in the image.
        num_masking_patches (int): Total number of masking patches.
        min_num_patches (int): Minimum number of patches to mask.
        max_num_patches (int): Maximum number of patches to mask.
        log_aspect_ratio (tuple): Logarithmic aspect ratio bounds.
    """

    def __init__(
        self,
        input_size,
        num_masking_patches=None,
        min_num_patches=4,
        max_num_patches=None,
        min_aspect=0.3,
        max_aspect=None,
    ):
        """Initializes the MaskingGenerator.

        Args:
            input_size (tuple or int): Size of the input image. If an int is provided,
                it is treated as both height and width.
            num_masking_patches (int, optional): Total number of patches to mask.
                Defaults to None.
            min_num_patches (int, optional): Minimum number of patches to mask.
                Defaults to 4.
            max_num_patches (int, optional): Maximum number of patches to mask.
                Defaults to None.
            min_aspect (float, optional): Minimum aspect ratio of the patches.
                Defaults to 0.3.
            max_aspect (float, optional): Maximum aspect ratio of the patches.
                Defaults to None.
        """
        if not isinstance(input_size, tuple):
            input_size = (input_size,) * 2
        self.height, self.width = input_size

        self.num_patches = self.height * self.width
        self.num_masking_patches = num_masking_patches

        self.min_num_patches = min_num_patches
        self.max_num_patches = num_masking_patches if max_num_patches is None else max_num_patches

        max_aspect = max_aspect or 1 / min_aspect
        self.log_aspect_ratio = (math.log(min_aspect), math.log(max_aspect))

    def __repr__(self):
        """Returns a string representation of the MaskingGenerator.

        Returns:
            str: A string representation of the generator's configuration.
        """
        repr_str = "Generator(%d, %d -> [%d ~ %d], max = %d, %.3f ~ %.3f)" % (
            self.height,
            self.width,
            self.min_num_patches,
            self.max_num_patches,
            self.num_masking_patches,
            self.log_aspect_ratio[0],
            self.log_aspect_ratio[1],
        )
        return repr_str

    def get_shape(self):
        """Gets the shape of the input image.

        Returns:
            tuple: A tuple containing the height and width of the image.
        """
        return self.height, self.width

    def _mask(self, mask, max_mask_patches):
        """Applies masking to the given mask.

        Args:
            mask (np.ndarray): The mask to apply patches to.
            max_mask_patches (int): The maximum number of patches to mask.

        Returns:
            int: The number of patches that were masked.
        """
        delta = 0
        for _ in range(10):
            target_area = random.uniform(self.min_num_patches, max_mask_patches)
            aspect_ratio = math.exp(random.uniform(*self.log_aspect_ratio))
            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))
            if w < self.width and h < self.height:
                top = random.randint(0, self.height - h)
                left = random.randint(0, self.width - w)

                num_masked = mask[top : top + h, left : left + w].sum()
                # Overlap
                if 0 < h * w - num_masked <= max_mask_patches:
                    for i in range(top, top + h):
                        for j in range(left, left + w):
                            if mask[i, j] == 0:
                                mask[i, j] = 1
                                delta += 1

                if delta > 0:
                    break
        return delta

    def __call__(self, num_masking_patches=0):
        """Generates a mask with the specified number of masking patches.

        Args:
            num_masking_patches (int, optional): The total number of patches to mask.
                Defaults to 0.

        Returns:
            np.ndarray: A boolean mask indicating the masked areas.
        """
        mask = np.zeros(shape=self.get_shape(), dtype=bool)
        mask_count = 0
        while mask_count < num_masking_patches:
            max_mask_patches = num_masking_patches - mask_count
            max_mask_patches = min(max_mask_patches, self.max_num_patches)

            delta = self._mask(mask, max_mask_patches)
            if delta == 0:
                break
            else:
                mask_count += delta

        return mask
