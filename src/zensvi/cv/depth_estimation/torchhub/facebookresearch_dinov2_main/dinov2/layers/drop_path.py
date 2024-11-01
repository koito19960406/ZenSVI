# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# References:
#   https://github.com/facebookresearch/dino/blob/master/vision_transformer.py
#   https://github.com/rwightman/pytorch-image-models/tree/master/timm/layers/drop.py


from torch import nn


def drop_path(x, drop_prob: float = 0.0, training: bool = False):
    """Applies Drop Path (Stochastic Depth) to the input tensor.

    This function randomly drops paths during training based on the specified drop probability.

    Args:
        x (torch.Tensor): The input tensor to which Drop Path will be applied.
        drop_prob (float, optional): The probability of dropping a path. Defaults to 0.0.
        training (bool, optional): Indicates whether the model is in training mode. Defaults to False.

    Returns:
        torch.Tensor: The output tensor after applying Drop Path.
    """
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0:
        random_tensor.div_(keep_prob)
    output = x * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample when applied in the main path of residual blocks.

    Args:
        drop_prob (float, optional): The probability of dropping a path. Defaults to None.
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        """Forward pass for the DropPath module.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor after applying Drop Path.
        """
        return drop_path(x, self.drop_prob, self.training)
