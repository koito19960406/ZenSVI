# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Modified from: https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/vision_transformer.py#L103-L110

from typing import Union

import torch
from torch import Tensor, nn


class LayerScale(nn.Module):
    """Applies layer scaling to the input tensor.

    This module scales the input tensor by a learnable parameter (gamma),
    which can be initialized to a specific value. The scaling can be done
    in-place or not, depending on the `inplace` parameter.

    Attributes:
        gamma (nn.Parameter): The learnable scaling parameter.

    Args:
        dim (int): The dimension of the input tensor.
        init_values (Union[float, Tensor], optional): Initial value(s) for gamma. Defaults to 1e-5.
        inplace (bool, optional): If True, performs the operation in-place. Defaults to False.
    """

    def __init__(
        self,
        dim: int,
        init_values: Union[float, Tensor] = 1e-5,
        inplace: bool = False,
    ) -> None:
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass for the LayerScale module.

        Args:
            x (Tensor): The input tensor to be scaled.

        Returns:
            Tensor: The scaled tensor.
        """
        return x.mul_(self.gamma) if self.inplace else x * self.gamma
