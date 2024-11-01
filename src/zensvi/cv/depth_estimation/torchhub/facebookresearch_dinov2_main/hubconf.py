# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

from enum import Enum
from typing import Union

import torch

_DINOV2_BASE_URL = "https://dl.fbaipublicfiles.com/dinov2"


def _make_dinov2_model_name(arch_name: str, patch_size: int, num_register_tokens: int = 0) -> str:
    """Generates the model name for DINOv2 based on architecture name, patch size, and number of register tokens.

    Args:
        arch_name (str): The name of the architecture.
        patch_size (int): The size of the patches.
        num_register_tokens (int, optional): The number of register tokens. Defaults to 0.

    Returns:
        str: The generated model name.
    """
    compact_arch_name = arch_name.replace("_", "")[:4]
    registers_suffix = f"_reg{num_register_tokens}" if num_register_tokens else ""
    return f"dinov2_{compact_arch_name}{patch_size}{registers_suffix}"


class Weights(Enum):
    """Enum for model weights."""

    LVD142M = "LVD142M"


def _make_dinov2_model(
    *,
    arch_name: str = "vit_large",
    img_size: int = 518,
    patch_size: int = 14,
    init_values: float = 1.0,
    ffn_layer: str = "mlp",
    block_chunks: int = 0,
    num_register_tokens: int = 0,
    interpolate_antialias: bool = False,
    interpolate_offset: float = 0.1,
    pretrained: bool = True,
    weights: Union[Weights, str] = Weights.LVD142M,
    **kwargs,
):
    """Creates a DINOv2 model with specified parameters.

    Args:
        arch_name (str, optional): The architecture name. Defaults to "vit_large".
        img_size (int, optional): The size of the input images. Defaults to 518.
        patch_size (int, optional): The size of the patches. Defaults to 14.
        init_values (float, optional): Initial values for the model. Defaults to 1.0.
        ffn_layer (str, optional): The type of feedforward layer. Defaults to "mlp".
        block_chunks (int, optional): Number of chunks for blocks. Defaults to 0.
        num_register_tokens (int, optional): Number of register tokens. Defaults to 0.
        interpolate_antialias (bool, optional): Whether to use antialiasing during interpolation. Defaults to False.
        interpolate_offset (float, optional): Offset for interpolation. Defaults to 0.1.
        pretrained (bool, optional): Whether to load pretrained weights. Defaults to True.
        weights (Union[Weights, str], optional): The weights to use. Defaults to Weights.LVD142M.
        **kwargs: Additional keyword arguments.

    Returns:
        nn.Module: The constructed DINOv2 model.
    """
    import vision_transformer as vits

    if isinstance(weights, str):
        try:
            weights = Weights[weights]
        except KeyError:
            raise AssertionError(f"Unsupported weights: {weights}")

    model_base_name = _make_dinov2_model_name(arch_name, patch_size)
    vit_kwargs = dict(
        img_size=img_size,
        patch_size=patch_size,
        init_values=init_values,
        ffn_layer=ffn_layer,
        block_chunks=block_chunks,
        num_register_tokens=num_register_tokens,
        interpolate_antialias=interpolate_antialias,
        interpolate_offset=interpolate_offset,
    )
    vit_kwargs.update(**kwargs)
    model = vits.__dict__[arch_name](**vit_kwargs)

    if pretrained:
        model_full_name = _make_dinov2_model_name(arch_name, patch_size, num_register_tokens)
        url = _DINOV2_BASE_URL + f"/{model_base_name}/{model_full_name}_pretrain.pth"
        state_dict = torch.hub.load_state_dict_from_url(url, map_location="cpu")
        model.load_state_dict(state_dict, strict=True)

    return model


def dinov2_vits14(*, pretrained: bool = True, weights: Union[Weights, str] = Weights.LVD142M, **kwargs):
    """Creates a DINOv2 ViT-S/14 model (optionally) pretrained on the LVD-142M dataset.

    Args:
        pretrained (bool, optional): Whether to load pretrained weights. Defaults to True.
        weights (Union[Weights, str], optional): The weights to use. Defaults to Weights.LVD142M.
        **kwargs: Additional keyword arguments.

    Returns:
        nn.Module: The constructed DINOv2 ViT-S/14 model.
    """
    return _make_dinov2_model(arch_name="vit_small", pretrained=pretrained, weights=weights, **kwargs)


def dinov2_vitb14(*, pretrained: bool = True, weights: Union[Weights, str] = Weights.LVD142M, **kwargs):
    """Creates a DINOv2 ViT-B/14 model (optionally) pretrained on the LVD-142M dataset.

    Args:
        pretrained (bool, optional): Whether to load pretrained weights. Defaults to True.
        weights (Union[Weights, str], optional): The weights to use. Defaults to Weights.LVD142M.
        **kwargs: Additional keyword arguments.

    Returns:
        nn.Module: The constructed DINOv2 ViT-B/14 model.
    """
    return _make_dinov2_model(arch_name="vit_base", pretrained=pretrained, weights=weights, **kwargs)


def dinov2_vitl14(*, pretrained: bool = True, weights: Union[Weights, str] = Weights.LVD142M, **kwargs):
    """Creates a DINOv2 ViT-L/14 model (optionally) pretrained on the LVD-142M dataset.

    Args:
        pretrained (bool, optional): Whether to load pretrained weights. Defaults to True.
        weights (Union[Weights, str], optional): The weights to use. Defaults to Weights.LVD142M.
        **kwargs: Additional keyword arguments.

    Returns:
        nn.Module: The constructed DINOv2 ViT-L/14 model.
    """
    return _make_dinov2_model(arch_name="vit_large", pretrained=pretrained, weights=weights, **kwargs)


def dinov2_vitg14(*, pretrained: bool = True, weights: Union[Weights, str] = Weights.LVD142M, **kwargs):
    """Creates a DINOv2 ViT-g/14 model (optionally) pretrained on the LVD-142M dataset.

    Args:
        pretrained (bool, optional): Whether to load pretrained weights. Defaults to True.
        weights (Union[Weights, str], optional): The weights to use. Defaults to Weights.LVD142M.
        **kwargs: Additional keyword arguments.

    Returns:
        nn.Module: The constructed DINOv2 ViT-g/14 model.
    """
    return _make_dinov2_model(
        arch_name="vit_giant2",
        ffn_layer="swiglufused",
        weights=weights,
        pretrained=pretrained,
        **kwargs,
    )


def dinov2_vits14_reg(*, pretrained: bool = True, weights: Union[Weights, str] = Weights.LVD142M, **kwargs):
    """Creates a DINOv2 ViT-S/14 model with registers (optionally) pretrained on the LVD-142M dataset.

    Args:
        pretrained (bool, optional): Whether to load pretrained weights. Defaults to True.
        weights (Union[Weights, str], optional): The weights to use. Defaults to Weights.LVD142M.
        **kwargs: Additional keyword arguments.

    Returns:
        nn.Module: The constructed DINOv2 ViT-S/14 model with registers.
    """
    return _make_dinov2_model(
        arch_name="vit_small",
        pretrained=pretrained,
        weights=weights,
        num_register_tokens=4,
        interpolate_antialias=True,
        interpolate_offset=0.0,
        **kwargs,
    )


def dinov2_vitb14_reg(*, pretrained: bool = True, weights: Union[Weights, str] = Weights.LVD142M, **kwargs):
    """Creates a DINOv2 ViT-B/14 model with registers (optionally) pretrained on the LVD-142M dataset.

    Args:
        pretrained (bool, optional): Whether to load pretrained weights. Defaults to True.
        weights (Union[Weights, str], optional): The weights to use. Defaults to Weights.LVD142M.
        **kwargs: Additional keyword arguments.

    Returns:
        nn.Module: The constructed DINOv2 ViT-B/14 model with registers.
    """
    return _make_dinov2_model(
        arch_name="vit_base",
        pretrained=pretrained,
        weights=weights,
        num_register_tokens=4,
        interpolate_antialias=True,
        interpolate_offset=0.0,
        **kwargs,
    )


def dinov2_vitl14_reg(*, pretrained: bool = True, weights: Union[Weights, str] = Weights.LVD142M, **kwargs):
    """Creates a DINOv2 ViT-L/14 model with registers (optionally) pretrained on the LVD-142M dataset.

    Args:
        pretrained (bool, optional): Whether to load pretrained weights. Defaults to True.
        weights (Union[Weights, str], optional): The weights to use. Defaults to Weights.LVD142M.
        **kwargs: Additional keyword arguments.

    Returns:
        nn.Module: The constructed DINOv2 ViT-L/14 model with registers.
    """
    return _make_dinov2_model(
        arch_name="vit_large",
        pretrained=pretrained,
        weights=weights,
        num_register_tokens=4,
        interpolate_antialias=True,
        interpolate_offset=0.0,
        **kwargs,
    )


def dinov2_vitg14_reg(*, pretrained: bool = True, weights: Union[Weights, str] = Weights.LVD142M, **kwargs):
    """Creates a DINOv2 ViT-g/14 model with registers (optionally) pretrained on the LVD-142M dataset.

    Args:
        pretrained (bool, optional): Whether to load pretrained weights. Defaults to True.
        weights (Union[Weights, str], optional): The weights to use. Defaults to Weights.LVD142M.
        **kwargs: Additional keyword arguments.

    Returns:
        nn.Module: The constructed DINOv2 ViT-g/14 model with registers.
    """
    return _make_dinov2_model(
        arch_name="vit_giant2",
        ffn_layer="swiglufused",
        weights=weights,
        pretrained=pretrained,
        num_register_tokens=4,
        interpolate_antialias=True,
        interpolate_offset=0.0,
        **kwargs,
    )
