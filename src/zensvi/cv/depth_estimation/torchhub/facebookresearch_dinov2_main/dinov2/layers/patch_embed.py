# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# References:
#   https://github.com/facebookresearch/dino/blob/master/vision_transformer.py
#   https://github.com/rwightman/pytorch-image-models/tree/master/timm/layers/patch_embed.py

from typing import Callable, Optional, Tuple, Union

import torch.nn as nn
from torch import Tensor


def make_2tuple(x: Union[int, Tuple[int, int]]) -> Tuple[int, int]:
    """Converts an integer or a tuple into a tuple of two integers.

    Args:
        x (Union[int, Tuple[int, int]]): An integer or a tuple of two integers.

    Returns:
        Tuple[int, int]: A tuple containing two integers.
    """
    if isinstance(x, tuple):
        assert len(x) == 2
        return x

    assert isinstance(x, int)
    return (x, x)


class PatchEmbed(nn.Module):
    """2D image to patch embedding: (B,C,H,W) -> (B,N,D)

    This class converts a 2D image into patch embeddings.

    Args:
        img_size (Union[int, Tuple[int, int]]): Image size.
        patch_size (Union[int, Tuple[int, int]]): Patch token size.
        in_chans (int): Number of input image channels.
        embed_dim (int): Number of linear projection output channels.
        norm_layer (Optional[Callable]): Normalization layer.
        flatten_embedding (bool): Whether to flatten the embedding. Defaults to True.

    Attributes:
        img_size (Tuple[int, int]): The size of the input image.
        patch_size (Tuple[int, int]): The size of each patch.
        patches_resolution (Tuple[int, int]): The resolution of the patches.
        num_patches (int): The total number of patches.
        in_chans (int): Number of input channels.
        embed_dim (int): Dimension of the embedding.
        flatten_embedding (bool): Whether to flatten the embedding.
        proj (nn.Conv2d): Convolutional layer for patch projection.
        norm (nn.Module): Normalization layer.
    """

    def __init__(
        self,
        img_size: Union[int, Tuple[int, int]] = 224,
        patch_size: Union[int, Tuple[int, int]] = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        norm_layer: Optional[Callable] = None,
        flatten_embedding: bool = True,
    ) -> None:
        super().__init__()

        image_HW = make_2tuple(img_size)
        patch_HW = make_2tuple(patch_size)
        patch_grid_size = (
            image_HW[0] // patch_HW[0],
            image_HW[1] // patch_HW[1],
        )

        self.img_size = image_HW
        self.patch_size = patch_HW
        self.patches_resolution = patch_grid_size
        self.num_patches = patch_grid_size[0] * patch_grid_size[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.flatten_embedding = flatten_embedding

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_HW, stride=patch_HW)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass for the patch embedding layer.

        Args:
            x (Tensor): Input tensor of shape (B, C, H, W).

        Returns:
            Tensor: Output tensor of shape (B, H', W', D) if flatten_embedding is False,
                    otherwise (B, HW, D).
        """
        _, _, H, W = x.shape
        patch_H, patch_W = self.patch_size

        assert H % patch_H == 0, f"Input image height {H} is not a multiple of patch height {patch_H}"
        assert W % patch_W == 0, f"Input image width {W} is not a multiple of patch width: {patch_W}"

        x = self.proj(x)  # B C H W
        H, W = x.size(2), x.size(3)
        x = x.flatten(2).transpose(1, 2)  # B HW C
        x = self.norm(x)
        if not self.flatten_embedding:
            x = x.reshape(-1, H, W, self.embed_dim)  # B H W C
        return x

    def flops(self) -> float:
        """Calculates the number of floating point operations (FLOPs).

        Returns:
            float: The number of FLOPs for the patch embedding operation.
        """
        Ho, Wo = self.patches_resolution
        flops = Ho * Wo * self.embed_dim * self.in_chans * (self.patch_size[0] * self.patch_size[1])
        if self.norm is not None:
            flops += Ho * Wo * self.embed_dim
        return flops
