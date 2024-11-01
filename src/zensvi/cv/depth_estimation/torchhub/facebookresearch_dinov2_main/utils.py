# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import itertools
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

_DINOV2_BASE_URL = "https://dl.fbaipublicfiles.com/dinov2"


def _make_dinov2_model_name(arch_name: str, patch_size: int, num_register_tokens: int = 0) -> str:
    """Generates a DINOv2 model name based on architecture name, patch size, and number of register tokens.

    Args:
        arch_name (str): The name of the architecture.
        patch_size (int): The size of the patches.
        num_register_tokens (int, optional): The number of register tokens. Defaults to 0.

    Returns:
        str: The formatted DINOv2 model name.
    """
    compact_arch_name = arch_name.replace("_", "")[:4]
    registers_suffix = f"_reg{num_register_tokens}" if num_register_tokens else ""
    return f"dinov2_{compact_arch_name}{patch_size}{registers_suffix}"


class CenterPadding(nn.Module):
    """A module that applies center padding to input tensors to ensure dimensions are multiples of a specified value."""

    def __init__(self, multiple: int):
        """Initializes the CenterPadding module.

        Args:
            multiple (int): The value to which the dimensions should be a multiple of.
        """
        super().__init__()
        self.multiple = multiple

    def _get_pad(self, size: int) -> tuple:
        """Calculates the padding required to make the size a multiple of the specified value.

        Args:
            size (int): The original size.

        Returns:
            tuple: A tuple containing the left and right padding sizes.
        """
        new_size = math.ceil(size / self.multiple) * self.multiple
        pad_size = new_size - size
        pad_size_left = pad_size // 2
        pad_size_right = pad_size - pad_size_left
        return pad_size_left, pad_size_right

    @torch.inference_mode()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies center padding to the input tensor.

        Args:
            x (torch.Tensor): The input tensor to be padded.

        Returns:
            torch.Tensor: The padded tensor.
        """
        pads = list(itertools.chain.from_iterable(self._get_pad(m) for m in x.shape[:1:-1]))
        output = F.pad(x, pads)
        return output
