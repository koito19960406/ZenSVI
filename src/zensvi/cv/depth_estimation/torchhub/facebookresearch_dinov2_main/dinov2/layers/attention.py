# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# References:
#   https://github.com/facebookresearch/dino/blob/master/vision_transformer.py
#   https://github.com/rwightman/pytorch-image-models/tree/master/timm/models/vision_transformer.py

import logging

from torch import Tensor, nn

logger = logging.getLogger("dinov2")


try:
    from xformers.ops import memory_efficient_attention, unbind

    XFORMERS_AVAILABLE = True
except ImportError:
    logger.warning("xFormers not available")
    XFORMERS_AVAILABLE = False


class Attention(nn.Module):
    """Implements the multi-head attention mechanism.

    This class applies the attention mechanism to the input tensor, allowing the model
    to focus on different parts of the input sequence.

    Args:
        dim (int): The number of input features.
        num_heads (int, optional): The number of attention heads. Defaults to 8.
        qkv_bias (bool, optional): If True, adds a learnable bias to the query, key, and value projections. Defaults to False.
        proj_bias (bool, optional): If True, adds a learnable bias to the output projection. Defaults to True.
        attn_drop (float, optional): Dropout rate for the attention weights. Defaults to 0.0.
        proj_drop (float, optional): Dropout rate for the output projection. Defaults to 0.0.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass for the attention layer.

        Args:
            x (Tensor): The input tensor of shape (B, N, C), where B is the batch size,
                        N is the number of tokens, and C is the number of features.

        Returns:
            Tensor: The output tensor after applying attention and projection.
        """
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)

        q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
        attn = q @ k.transpose(-2, -1)

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class MemEffAttention(Attention):
    """Implements memory-efficient multi-head attention.

    This class extends the Attention class to utilize xFormers for memory-efficient
    attention computation when available.

    Args:
        Attention (nn.Module): The base attention class.
    """

    def forward(self, x: Tensor, attn_bias=None) -> Tensor:
        """Forward pass for the memory-efficient attention layer.

        Args:
            x (Tensor): The input tensor of shape (B, N, C).
            attn_bias (optional): An optional bias to add to the attention scores. Defaults to None.

        Returns:
            Tensor: The output tensor after applying memory-efficient attention and projection.
        """
        if not XFORMERS_AVAILABLE:
            assert attn_bias is None, "xFormers is required for nested tensors usage"
            return super().forward(x)

        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)

        q, k, v = unbind(qkv, 2)

        x = memory_efficient_attention(q, k, v, attn_bias=attn_bias)
        x = x.reshape([B, N, C])

        x = self.proj(x)
        x = self.proj_drop(x)
        return x
