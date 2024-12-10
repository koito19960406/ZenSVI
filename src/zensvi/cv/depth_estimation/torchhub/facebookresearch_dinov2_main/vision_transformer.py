# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

# References:
#   https://github.com/facebookresearch/dino/blob/main/vision_transformer.py
#   https://github.com/rwightman/pytorch-image-models/tree/master/timm/models/vision_transformer.py

import logging
import math
from functools import partial
from typing import Callable, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.utils.checkpoint
from dinov2.layers import MemEffAttention, Mlp
from dinov2.layers import NestedTensorBlock as Block
from dinov2.layers import PatchEmbed, SwiGLUFFNFused
from torch.nn.init import trunc_normal_

logger = logging.getLogger("dinov2")


def named_apply(fn: Callable, module: nn.Module, name="", depth_first=True, include_root=False) -> nn.Module:
    """Applies a function to a module and its children.

    Args:
        fn (Callable): The function to apply.
        module (nn.Module): The module to apply the function to.
        name (str, optional): The name of the module. Defaults to "".
        depth_first (bool, optional): If True, apply function to children before the module itself. Defaults to True.
        include_root (bool, optional): If True, include the root module in the application. Defaults to False.

    Returns:
        nn.Module: The module after applying the function.
    """
    if not depth_first and include_root:
        fn(module=module, name=name)
    for child_name, child_module in module.named_children():
        child_name = ".".join((name, child_name)) if name else child_name
        named_apply(
            fn=fn,
            module=child_module,
            name=child_name,
            depth_first=depth_first,
            include_root=True,
        )
    if depth_first and include_root:
        fn(module=module, name=name)
    return module


class BlockChunk(nn.ModuleList):
    """A chunk of transformer blocks."""

    def forward(self, x):
        """Forward pass through the chunk of blocks.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after passing through the blocks.
        """
        for b in self:
            x = b(x)
        return x


class DinoVisionTransformer(nn.Module):
    """Vision Transformer model for DINO."""

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        ffn_bias=True,
        proj_bias=True,
        drop_path_rate=0.0,
        drop_path_uniform=False,
        init_values=None,  # for layerscale: None or 0 => no layerscale
        embed_layer=PatchEmbed,
        act_layer=nn.GELU,
        block_fn=Block,
        ffn_layer="mlp",
        block_chunks=1,
        num_register_tokens=0,
        interpolate_antialias=False,
        interpolate_offset=0.1,
    ):
        """Initializes the DinoVisionTransformer.

        Args:
            img_size (int, optional): Input image size. Defaults to 224.
            patch_size (int, optional): Patch size. Defaults to 16.
            in_chans (int, optional): Number of input channels. Defaults to 3.
            embed_dim (int, optional): Embedding dimension. Defaults to 768.
            depth (int, optional): Depth of transformer. Defaults to 12.
            num_heads (int, optional): Number of attention heads. Defaults to 12.
            mlp_ratio (float, optional): Ratio of MLP hidden dimension to embedding dimension. Defaults to 4.0.
            qkv_bias (bool, optional): Enable bias for QKV if True. Defaults to True.
            ffn_bias (bool, optional): Enable bias for FFN if True. Defaults to True.
            proj_bias (bool, optional): Enable bias for projection in attention if True. Defaults to True.
            drop_path_rate (float, optional): Stochastic depth rate. Defaults to 0.0.
            drop_path_uniform (bool, optional): Apply uniform drop rate across blocks. Defaults to False.
            init_values (float, optional): Layer-scale initialization values. Defaults to None.
            embed_layer (nn.Module, optional): Patch embedding layer. Defaults to PatchEmbed.
            act_layer (nn.Module, optional): MLP activation layer. Defaults to nn.GELU.
            block_fn (nn.Module, optional): Transformer block class. Defaults to Block.
            ffn_layer (str, optional): Type of FFN layer. Options: "mlp", "swiglu", "swiglufused", or "identity". Defaults to "mlp".
            block_chunks (int, optional): Split block sequence into block_chunks units for FSDP wrap. Defaults to 1.
            num_register_tokens (int, optional): Number of extra CLS tokens (so-called "registers"). Defaults to 0.
            interpolate_antialias (bool, optional): Flag to apply anti-aliasing when interpolating positional embeddings. Defaults to False.
            interpolate_offset (float, optional): Work-around offset to apply when interpolating positional embeddings. Defaults to 0.1.
        """
        super().__init__()
        norm_layer = partial(nn.LayerNorm, eps=1e-6)

        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_tokens = 1
        self.n_blocks = depth
        self.num_heads = num_heads
        self.patch_size = patch_size
        self.num_register_tokens = num_register_tokens
        self.interpolate_antialias = interpolate_antialias
        self.interpolate_offset = interpolate_offset

        self.patch_embed = embed_layer(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
        )
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
        assert num_register_tokens >= 0
        self.register_tokens = (
            nn.Parameter(torch.zeros(1, num_register_tokens, embed_dim)) if num_register_tokens else None
        )

        if drop_path_uniform is True:
            dpr = [drop_path_rate] * depth
        else:
            dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule

        if ffn_layer == "mlp":
            logger.info("using MLP layer as FFN")
            ffn_layer = Mlp
        elif ffn_layer == "swiglufused" or ffn_layer == "swiglu":
            logger.info("using SwiGLU layer as FFN")
            ffn_layer = SwiGLUFFNFused
        elif ffn_layer == "identity":
            logger.info("using Identity layer as FFN")

            def f(*args, **kwargs):
                """Identity function for FFN layer.

                Args:
                    *args: Variable length argument list.
                    **kwargs: Arbitrary keyword arguments.

                Returns:
                    nn.Identity: Identity layer.
                """
                return nn.Identity()

            ffn_layer = f
        else:
            raise NotImplementedError

        blocks_list = [
            block_fn(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                proj_bias=proj_bias,
                ffn_bias=ffn_bias,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                act_layer=act_layer,
                ffn_layer=ffn_layer,
                init_values=init_values,
            )
            for i in range(depth)
        ]
        if block_chunks > 0:
            self.chunked_blocks = True
            chunked_blocks = []
            chunksize = depth // block_chunks
            for i in range(0, depth, chunksize):
                # this is to keep the block index consistent if we chunk the block list
                chunked_blocks.append([nn.Identity()] * i + blocks_list[i : i + chunksize])
            self.blocks = nn.ModuleList([BlockChunk(p) for p in chunked_blocks])
        else:
            self.chunked_blocks = False
            self.blocks = nn.ModuleList(blocks_list)

        self.norm = norm_layer(embed_dim)
        self.head = nn.Identity()

        self.mask_token = nn.Parameter(torch.zeros(1, embed_dim))

        self.init_weights()

    def init_weights(self):
        """Initializes the weights of the model."""
        trunc_normal_(self.pos_embed, std=0.02)
        nn.init.normal_(self.cls_token, std=1e-6)
        if self.register_tokens is not None:
            nn.init.normal_(self.register_tokens, std=1e-6)
        named_apply(init_weights_vit_timm, self)

    def interpolate_pos_encoding(self, x, w, h):
        """Interpolates the positional encoding based on input dimensions.

        Args:
            x (torch.Tensor): Input tensor.
            w (int): Width of the input.
            h (int): Height of the input.

        Returns:
            torch.Tensor: Interpolated positional encoding.
        """
        previous_dtype = x.dtype
        npatch = x.shape[1] - 1
        N = self.pos_embed.shape[1] - 1
        if npatch == N and w == h:
            return self.pos_embed
        pos_embed = self.pos_embed.float()
        class_pos_embed = pos_embed[:, 0]
        patch_pos_embed = pos_embed[:, 1:]
        dim = x.shape[-1]
        w0 = w // self.patch_size
        h0 = h // self.patch_size
        # we add a small number to avoid floating point error in the interpolation
        # see discussion at https://github.com/facebookresearch/dino/issues/8
        # DINOv2 with register modify the interpolate_offset from 0.1 to 0.0
        w0, h0 = w0 + self.interpolate_offset, h0 + self.interpolate_offset

        sqrt_N = math.sqrt(N)
        sx, sy = float(w0) / sqrt_N, float(h0) / sqrt_N
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, int(sqrt_N), int(sqrt_N), dim).permute(0, 3, 1, 2),
            scale_factor=(sx, sy),
            mode="bicubic",
            antialias=self.interpolate_antialias,
        )

        assert int(w0) == patch_pos_embed.shape[-2]
        assert int(h0) == patch_pos_embed.shape[-1]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1).to(previous_dtype)

    def prepare_tokens_with_masks(self, x, masks=None):
        """Prepares input tokens with optional masks.

        Args:
            x (torch.Tensor): Input tensor.
            masks (torch.Tensor, optional): Masks to apply. Defaults to None.

        Returns:
            torch.Tensor: Prepared tokens.
        """
        B, nc, w, h = x.shape
        x = self.patch_embed(x)
        if masks is not None:
            x = torch.where(masks.unsqueeze(-1), self.mask_token.to(x.dtype).unsqueeze(0), x)

        x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = x + self.interpolate_pos_encoding(x, w, h)

        if self.register_tokens is not None:
            x = torch.cat(
                (
                    x[:, :1],
                    self.register_tokens.expand(x.shape[0], -1, -1),
                    x[:, 1:],
                ),
                dim=1,
            )

        return x

    def forward_features_list(self, x_list, masks_list):
        """Processes a list of inputs through the transformer blocks.

        Args:
            x_list (list): List of input tensors.
            masks_list (list): List of masks corresponding to the inputs.

        Returns:
            list: List of outputs from the transformer blocks.
        """
        x = [self.prepare_tokens_with_masks(x, masks) for x, masks in zip(x_list, masks_list)]
        for blk in self.blocks:
            x = blk(x)

        all_x = x
        output = []
        for x, masks in zip(all_x, masks_list):
            x_norm = self.norm(x)
            output.append(
                {
                    "x_norm_clstoken": x_norm[:, 0],
                    "x_norm_regtokens": x_norm[:, 1 : self.num_register_tokens + 1],
                    "x_norm_patchtokens": x_norm[:, self.num_register_tokens + 1 :],
                    "x_prenorm": x,
                    "masks": masks,
                }
            )
        return output

    def forward_features(self, x, masks=None):
        """Processes the input through the transformer blocks.

        Args:
            x (torch.Tensor): Input tensor.
            masks (torch.Tensor, optional): Masks to apply. Defaults to None.

        Returns:
            dict: Output containing normalized class token, register tokens, patch tokens, prenormalized output, and masks.
        """
        if isinstance(x, list):
            return self.forward_features_list(x, masks)

        x = self.prepare_tokens_with_masks(x, masks)

        for blk in self.blocks:
            x = blk(x)

        x_norm = self.norm(x)
        return {
            "x_norm_clstoken": x_norm[:, 0],
            "x_norm_regtokens": x_norm[:, 1 : self.num_register_tokens + 1],
            "x_norm_patchtokens": x_norm[:, self.num_register_tokens + 1 :],
            "x_prenorm": x,
            "masks": masks,
        }

    def _get_intermediate_layers_not_chunked(self, x, n=1):
        """Retrieves intermediate layers from the transformer blocks.

        Args:
            x (torch.Tensor): Input tensor.
            n (int, optional): Number of last blocks to take. Defaults to 1.

        Returns:
            list: List of intermediate outputs.
        """
        x = self.prepare_tokens_with_masks(x)
        output, total_block_len = [], len(self.blocks)
        blocks_to_take = range(total_block_len - n, total_block_len) if isinstance(n, int) else n
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if i in blocks_to_take:
                output.append(x)
        assert len(output) == len(blocks_to_take), f"only {len(output)} / {len(blocks_to_take)} blocks found"
        return output

    def _get_intermediate_layers_chunked(self, x, n=1):
        """Retrieves intermediate layers from chunked transformer blocks.

        Args:
            x (torch.Tensor): Input tensor.
            n (int, optional): Number of last blocks to take. Defaults to 1.

        Returns:
            list: List of intermediate outputs.
        """
        x = self.prepare_tokens_with_masks(x)
        output, i, total_block_len = [], 0, len(self.blocks[-1])
        blocks_to_take = range(total_block_len - n, total_block_len) if isinstance(n, int) else n
        for block_chunk in self.blocks:
            for blk in block_chunk[i:]:  # Passing the nn.Identity()
                x = blk(x)
                if i in blocks_to_take:
                    output.append(x)
                i += 1
        assert len(output) == len(blocks_to_take), f"only {len(output)} / {len(blocks_to_take)} blocks found"
        return output

    def get_intermediate_layers(
        self,
        x: torch.Tensor,
        n: Union[int, Sequence] = 1,  # Layers or n last layers to take
        reshape: bool = False,
        return_class_token: bool = False,
        norm=True,
    ) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor]]]:
        """Gets intermediate layers from the transformer.

        Args:
            x (torch.Tensor): Input tensor.
            n (Union[int, Sequence], optional): Number of layers or specific layers to take. Defaults to 1.
            reshape (bool, optional): If True, reshape the output. Defaults to False.
            return_class_token (bool, optional): If True, return the class token along with outputs. Defaults to False.
            norm (bool, optional): If True, apply normalization to outputs. Defaults to True.

        Returns:
            Tuple[Union[torch.Tensor, Tuple[torch.Tensor]]]: Outputs from the transformer.
        """
        if self.chunked_blocks:
            outputs = self._get_intermediate_layers_chunked(x, n)
        else:
            outputs = self._get_intermediate_layers_not_chunked(x, n)
        if norm:
            outputs = [self.norm(out) for out in outputs]
        class_tokens = [out[:, 0] for out in outputs]
        outputs = [out[:, 1 + self.num_register_tokens :] for out in outputs]
        if reshape:
            B, _, w, h = x.shape
            outputs = [
                out.reshape(B, w // self.patch_size, h // self.patch_size, -1).permute(0, 3, 1, 2).contiguous()
                for out in outputs
            ]
        if return_class_token:
            return tuple(zip(outputs, class_tokens))
        return tuple(outputs)

    def forward(self, *args, is_training=False, **kwargs):
        """Forward pass through the model.

        Args:
            *args: Variable length argument list.
            is_training (bool, optional): If True, return training output. Defaults to False.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            dict or torch.Tensor: Output from the model.
        """
        ret = self.forward_features(*args, **kwargs)
        if is_training:
            return ret
        else:
            return self.head(ret["x_norm_clstoken"])


def init_weights_vit_timm(module: nn.Module, name: str = ""):
    """Initializes weights for ViT model.

    Args:
        module (nn.Module): The module to initialize.
        name (str, optional): Name of the module. Defaults to "".
    """
    if isinstance(module, nn.Linear):
        trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


def vit_small(patch_size=16, num_register_tokens=0, **kwargs):
    """Creates a small Vision Transformer model.

    Args:
        patch_size (int, optional): Patch size. Defaults to 16.
        num_register_tokens (int, optional): Number of extra CLS tokens. Defaults to 0.
        **kwargs: Additional keyword arguments.

    Returns:
        DinoVisionTransformer: The constructed model.
    """
    model = DinoVisionTransformer(
        patch_size=patch_size,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4,
        block_fn=partial(Block, attn_class=MemEffAttention),
        num_register_tokens=num_register_tokens,
        **kwargs,
    )
    return model


def vit_base(patch_size=16, num_register_tokens=0, **kwargs):
    """Creates a base Vision Transformer model.

    Args:
        patch_size (int, optional): Patch size. Defaults to 16.
        num_register_tokens (int, optional): Number of extra CLS tokens. Defaults to 0.
        **kwargs: Additional keyword arguments.

    Returns:
        DinoVisionTransformer: The constructed model.
    """
    model = DinoVisionTransformer(
        patch_size=patch_size,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        block_fn=partial(Block, attn_class=MemEffAttention),
        num_register_tokens=num_register_tokens,
        **kwargs,
    )
    return model


def vit_large(patch_size=16, num_register_tokens=0, **kwargs):
    """Creates a large Vision Transformer model.

    Args:
        patch_size (int, optional): Patch size. Defaults to 16.
        num_register_tokens (int, optional): Number of extra CLS tokens. Defaults to 0.
        **kwargs: Additional keyword arguments.

    Returns:
        DinoVisionTransformer: The constructed model.
    """
    model = DinoVisionTransformer(
        patch_size=patch_size,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4,
        block_fn=partial(Block, attn_class=MemEffAttention),
        num_register_tokens=num_register_tokens,
        **kwargs,
    )
    return model


def vit_giant2(patch_size=16, num_register_tokens=0, **kwargs):
    """Creates a giant Vision Transformer model.

    Args:
        patch_size (int, optional): Patch size. Defaults to 16.
        num_register_tokens (int, optional): Number of extra CLS tokens. Defaults to 0.
        **kwargs: Additional keyword arguments.

    Returns:
        DinoVisionTransformer: The constructed model.
    """
    model = DinoVisionTransformer(
        patch_size=patch_size,
        embed_dim=1536,
        depth=40,
        num_heads=24,
        mlp_ratio=4,
        block_fn=partial(Block, attn_class=MemEffAttention),
        num_register_tokens=num_register_tokens,
        **kwargs,
    )
    return model
