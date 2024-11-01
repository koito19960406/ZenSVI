# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Callable, Optional

import torch.nn.functional as F
from torch import Tensor, nn


class SwiGLUFFN(nn.Module):
    """SwiGLU Feed Forward Network.

    This class implements a feed-forward neural network using the SwiGLU activation function.

    Args:
        in_features (int): Number of input features.
        hidden_features (Optional[int]): Number of hidden features. If None, defaults to in_features.
        out_features (Optional[int]): Number of output features. If None, defaults to in_features.
        act_layer (Callable[..., nn.Module], optional): Activation layer to use. Defaults to None.
        drop (float, optional): Dropout rate. Defaults to 0.0.
        bias (bool, optional): Whether to use bias in the linear layers. Defaults to True.
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: Callable[..., nn.Module] = None,
        drop: float = 0.0,
        bias: bool = True,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.w12 = nn.Linear(in_features, 2 * hidden_features, bias=bias)
        self.w3 = nn.Linear(hidden_features, out_features, bias=bias)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through the network.

        Args:
            x (Tensor): Input tensor of shape (batch_size, in_features).

        Returns:
            Tensor: Output tensor of shape (batch_size, out_features).
        """
        x12 = self.w12(x)
        x1, x2 = x12.chunk(2, dim=-1)
        hidden = F.silu(x1) * x2
        return self.w3(hidden)


try:
    from xformers.ops import SwiGLU

    XFORMERS_AVAILABLE = True
except ImportError:
    SwiGLU = SwiGLUFFN
    XFORMERS_AVAILABLE = False


class SwiGLUFFNFused(SwiGLU):
    """Fused SwiGLU Feed Forward Network.

    This class extends the SwiGLU class for optimized performance.

    Args:
        in_features (int): Number of input features.
        hidden_features (Optional[int]): Number of hidden features. If None, defaults to in_features.
        out_features (Optional[int]): Number of output features. If None, defaults to in_features.
        act_layer (Callable[..., nn.Module], optional): Activation layer to use. Defaults to None.
        drop (float, optional): Dropout rate. Defaults to 0.0.
        bias (bool, optional): Whether to use bias in the linear layers. Defaults to True.
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: Callable[..., nn.Module] = None,
        drop: float = 0.0,
        bias: bool = True,
    ) -> None:
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        hidden_features = (int(hidden_features * 2 / 3) + 7) // 8 * 8
        super().__init__(
            in_features=in_features,
            hidden_features=hidden_features,
            out_features=out_features,
            bias=bias,
        )
