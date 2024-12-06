# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
from torch.nn.init import trunc_normal_
from torch.nn.utils import weight_norm


class DINOHead(nn.Module):
    """DINO Head for the DINO model.

    This class implements the DINO head which consists of a multi-layer perceptron (MLP)
    followed by a normalization layer and a final linear layer.

    Args:
        in_dim (int): The input dimension of the MLP.
        out_dim (int): The output dimension of the final layer.
        use_bn (bool, optional): Whether to use batch normalization in the MLP. Defaults to False.
        nlayers (int, optional): The number of layers in the MLP. Defaults to 3.
        hidden_dim (int, optional): The hidden dimension of the MLP. Defaults to 2048.
        bottleneck_dim (int, optional): The bottleneck dimension of the MLP. Defaults to 256.
        mlp_bias (bool, optional): Whether to use bias in the MLP layers. Defaults to True.
    """

    def __init__(
        self,
        in_dim,
        out_dim,
        use_bn=False,
        nlayers=3,
        hidden_dim=2048,
        bottleneck_dim=256,
        mlp_bias=True,
    ):
        super().__init__()
        nlayers = max(nlayers, 1)
        self.mlp = _build_mlp(
            nlayers,
            in_dim,
            bottleneck_dim,
            hidden_dim=hidden_dim,
            use_bn=use_bn,
            bias=mlp_bias,
        )
        self.apply(self._init_weights)
        self.last_layer = weight_norm(nn.Linear(bottleneck_dim, out_dim, bias=False))
        self.last_layer.weight_g.data.fill_(1)

    def _init_weights(self, m):
        """Initializes the weights of the model.

        This method applies weight initialization to the linear layers.

        Args:
            m (nn.Module): The module to initialize.
        """
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """Forward pass for the DINO head.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor after applying the MLP and final linear layer.
        """
        x = self.mlp(x)
        eps = 1e-6 if x.dtype == torch.float16 else 1e-12
        x = nn.functional.normalize(x, dim=-1, p=2, eps=eps)
        x = self.last_layer(x)
        return x


def _build_mlp(nlayers, in_dim, bottleneck_dim, hidden_dim=None, use_bn=False, bias=True):
    """Builds a multi-layer perceptron (MLP).

    This function constructs an MLP with the specified number of layers, input dimension,
    bottleneck dimension, and optional batch normalization.

    Args:
        nlayers (int): The number of layers in the MLP.
        in_dim (int): The input dimension of the MLP.
        bottleneck_dim (int): The bottleneck dimension of the MLP.
        hidden_dim (int, optional): The hidden dimension of the MLP. Defaults to None.
        use_bn (bool, optional): Whether to use batch normalization in the MLP. Defaults to False.
        bias (bool, optional): Whether to use bias in the MLP layers. Defaults to True.

    Returns:
        nn.Sequential: A sequential container of the MLP layers.
    """
    if nlayers == 1:
        return nn.Linear(in_dim, bottleneck_dim, bias=bias)
    else:
        layers = [nn.Linear(in_dim, hidden_dim, bias=bias)]
        if use_bn:
            layers.append(nn.BatchNorm1d(hidden_dim))
        layers.append(nn.GELU())
        for _ in range(nlayers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim, bias=bias))
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.GELU())
        layers.append(nn.Linear(hidden_dim, bottleneck_dim, bias=bias))
        return nn.Sequential(*layers)
