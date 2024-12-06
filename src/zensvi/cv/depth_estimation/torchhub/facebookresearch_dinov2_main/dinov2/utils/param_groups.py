# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
from collections import defaultdict

logger = logging.getLogger("dinov2")


def get_vit_lr_decay_rate(
    name: str,
    lr_decay_rate: float = 1.0,
    num_layers: int = 12,
    force_is_backbone: bool = False,
    chunked_blocks: bool = False,
) -> float:
    """Calculate the learning rate decay rate for different ViT blocks.

    Args:
        name (str): Parameter name.
        lr_decay_rate (float, optional): Base learning rate decay rate. (Default value = 1.0)
        num_layers (int, optional): Number of layers. (Default value = 12)
        force_is_backbone (bool, optional): Flag to force treating as backbone. (Default value = False)
        chunked_blocks (bool, optional): Flag to indicate if blocks are chunked. (Default value = False)

    Returns:
        float: Learning rate decay rate for the given parameter.
    """
    layer_id = num_layers + 1
    if name.startswith("backbone") or force_is_backbone:
        if ".pos_embed" in name or ".patch_embed" in name or ".mask_token" in name or ".cls_token" in name:
            layer_id = 0
        elif force_is_backbone and (
            "pos_embed" in name or "patch_embed" in name or "mask_token" in name or "cls_token" in name
        ):
            layer_id = 0
        elif ".blocks." in name and ".residual." not in name:
            layer_id = int(name[name.find(".blocks.") :].split(".")[2]) + 1
        elif chunked_blocks and "blocks." in name and "residual." not in name:
            layer_id = int(name[name.find("blocks.") :].split(".")[2]) + 1
        elif "blocks." in name and "residual." not in name:
            layer_id = int(name[name.find("blocks.") :].split(".")[1]) + 1

    return lr_decay_rate ** (num_layers + 1 - layer_id)


def get_params_groups_with_decay(model, lr_decay_rate: float = 1.0, patch_embed_lr_mult: float = 1.0):
    """Get parameter groups with decay rates for a given model.

    Args:
        model: The model from which to extract parameters.
        lr_decay_rate (float, optional): Base learning rate decay rate. (Default value = 1.0)
        patch_embed_lr_mult (float, optional): Learning rate multiplier for patch embedding. (Default value = 1.0)

    Returns:
        list: A list of parameter groups with their respective decay rates.
    """
    chunked_blocks = False
    if hasattr(model, "n_blocks"):
        logger.info("chunked fsdp")
        n_blocks = model.n_blocks
        chunked_blocks = model.chunked_blocks
    elif hasattr(model, "blocks"):
        logger.info("first code branch")
        n_blocks = len(model.blocks)
    elif hasattr(model, "backbone"):
        logger.info("second code branch")
        n_blocks = len(model.backbone.blocks)
    else:
        logger.info("else code branch")
        n_blocks = 0
    all_param_groups = []

    for name, param in model.named_parameters():
        name = name.replace("_fsdp_wrapped_module.", "")
        if not param.requires_grad:
            continue
        decay_rate = get_vit_lr_decay_rate(
            name,
            lr_decay_rate,
            num_layers=n_blocks,
            force_is_backbone=n_blocks > 0,
            chunked_blocks=chunked_blocks,
        )
        d = {
            "params": param,
            "is_last_layer": False,
            "lr_multiplier": decay_rate,
            "wd_multiplier": 1.0,
            "name": name,
        }

        if "last_layer" in name:
            d.update({"is_last_layer": True})

        if name.endswith(".bias") or "norm" in name or "gamma" in name:
            d.update({"wd_multiplier": 0.0})

        if "patch_embed" in name:
            d.update({"lr_multiplier": d["lr_multiplier"] * patch_embed_lr_mult})

        all_param_groups.append(d)
        logger.info(f"""{name}: lr_multiplier: {d["lr_multiplier"]}, wd_multiplier: {d["wd_multiplier"]}""")

    return all_param_groups


def fuse_params_groups(all_params_groups, keys=("lr_multiplier", "wd_multiplier", "is_last_layer")):
    """Fuse parameter groups based on specified keys.

    Args:
        all_params_groups: A list of parameter groups to be fused.
        keys (tuple, optional): Keys to use for fusion. (Default value = ("lr_multiplier", "wd_multiplier", "is_last_layer"))

    Returns:
        dict_values: Fused parameter groups.
    """
    fused_params_groups = defaultdict(lambda: {"params": []})
    for d in all_params_groups:
        identifier = ""
        for k in keys:
            identifier += k + str(d[k]) + "_"

        for k in keys:
            fused_params_groups[identifier][k] = d[k]
        fused_params_groups[identifier]["params"].append(d["params"])

    return fused_params_groups.values()
