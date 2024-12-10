# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging

from . import vision_transformer as vits

logger = logging.getLogger("dinov2")


def build_model(args, only_teacher=False, img_size=224):
    """Builds a model based on the provided arguments.

    Args:
        args: The arguments containing model configuration.
        only_teacher (bool, optional): If True, only the teacher model is returned. Defaults to False.
        img_size (int, optional): The size of the input images. Defaults to 224.

    Returns:
        tuple: A tuple containing the student model, teacher model, and the embedding dimension.
    """
    args.arch = args.arch.removesuffix("_memeff")
    if "vit" in args.arch:
        vit_kwargs = dict(
            img_size=img_size,
            patch_size=args.patch_size,
            init_values=args.layerscale,
            ffn_layer=args.ffn_layer,
            block_chunks=args.block_chunks,
            qkv_bias=args.qkv_bias,
            proj_bias=args.proj_bias,
            ffn_bias=args.ffn_bias,
        )
        teacher = vits.__dict__[args.arch](**vit_kwargs)
        if only_teacher:
            return teacher, teacher.embed_dim
        student = vits.__dict__[args.arch](
            **vit_kwargs,
            drop_path_rate=args.drop_path_rate,
            drop_path_uniform=args.drop_path_uniform,
        )
        embed_dim = student.embed_dim
    return student, teacher, embed_dim


def build_model_from_cfg(cfg, only_teacher=False):
    """Builds a model from the given configuration.

    Args:
        cfg: The configuration object containing model parameters.
        only_teacher (bool, optional): If True, only the teacher model is returned. Defaults to False.

    Returns:
        tuple: A tuple containing the student model, teacher model, and the embedding dimension.
    """
    return build_model(cfg.student, only_teacher=only_teacher, img_size=cfg.crops.global_crops_size)
