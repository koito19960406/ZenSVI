# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# References:
#   https://github.com/facebookresearch/dino/blob/master/vision_transformer.py
#   https://github.com/rwightman/pytorch-image-models/tree/master/timm/layers/mlp.py


from typing import Callable, Optional

from torch import Tensor, nn


class Mlp(nn.Module):
    """Multi-layer perceptron (MLP) module.

    This module implements a simple feedforward neural network with
    one hidden layer, an activation function, and dropout.

    Attributes:
        fc1 (nn.Linear): The first linear layer.
        act (Callable): The activation function.
        fc2 (nn.Linear): The second linear layer.
        drop (nn.Dropout): The dropout layer.

    Args:
        in_features (int): Number of input features.
        hidden_features (Optional[int]): Number of hidden features. If None, defaults to in_features.
        out_features (Optional[int]): Number of output features. If None, defaults to in_features.
        act_layer (Callable[..., nn.Module]): Activation layer to use. Defaults to nn.GELU.
        drop (float): Dropout probability. Defaults to 0.0.
        bias (bool): Whether to use bias in the linear layers. Defaults to True.
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: Callable[..., nn.Module] = nn.GELU,
        drop: float = 0.0,
        bias: bool = True,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
        self.drop = nn.Dropout(drop)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through the MLP.

        Args:
            x (Tensor): Input tensor of shape (N, in_features), where N is the batch size.

        Returns:
            Tensor: Output tensor of shape (N, out_features).
        """
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
