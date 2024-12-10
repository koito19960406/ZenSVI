# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
from typing import Any, List, Optional, Tuple

import dinov2.utils.utils as dinov2_utils
import torch
import torch.backends.cudnn as cudnn
from dinov2.models import build_model_from_cfg
from dinov2.utils.config import setup


def get_args_parser(
    description: Optional[str] = None,
    parents: Optional[List[argparse.ArgumentParser]] = None,
    add_help: bool = True,
) -> argparse.ArgumentParser:
    """Creates an argument parser for the evaluation setup.

    Args:
        description (Optional[str]): Description of the parser.
        parents (Optional[List[argparse.ArgumentParser]]): Parent parsers to inherit arguments from.
        add_help (bool): Whether to add a help option to the parser.

    Returns:
        argparse.ArgumentParser: The configured argument parser.
    """
    parser = argparse.ArgumentParser(
        description=description,
        parents=parents or [],
        add_help=add_help,
    )
    parser.add_argument(
        "--config-file",
        type=str,
        help="Model configuration file",
    )
    parser.add_argument(
        "--pretrained-weights",
        type=str,
        help="Pretrained model weights",
    )
    parser.add_argument(
        "--output-dir",
        default="",
        type=str,
        help="Output directory to write results and logs",
    )
    parser.add_argument(
        "--opts",
        help="Extra configuration options",
        default=[],
        nargs="+",
    )
    return parser


def get_autocast_dtype(config) -> torch.dtype:
    """Determines the appropriate autocast data type based on the configuration.

    Args:
        config: The configuration object containing precision settings.

    Returns:
        torch.dtype: The data type to use for autocasting.
    """
    teacher_dtype_str = config.compute_precision.teacher.backbone.mixed_precision.param_dtype
    if teacher_dtype_str == "fp16":
        return torch.half
    elif teacher_dtype_str == "bf16":
        return torch.bfloat16
    else:
        return torch.float


def build_model_for_eval(config, pretrained_weights) -> torch.nn.Module:
    """Builds the model for evaluation using the provided configuration and weights.

    Args:
        config: The configuration object for the model.
        pretrained_weights: Path to the pretrained model weights.

    Returns:
        torch.nn.Module: The constructed model ready for evaluation.
    """
    model, _ = build_model_from_cfg(config, only_teacher=True)
    dinov2_utils.load_pretrained_weights(model, pretrained_weights, "teacher")
    model.eval()
    model.cuda()
    return model


def setup_and_build_model(args) -> Tuple[Any, torch.dtype]:
    """Sets up the model and returns it along with the autocast data type.

    Args:
        args: The command line arguments containing configuration and weights.

    Returns:
        Tuple[Any, torch.dtype]: The model and the autocast data type.
    """
    cudnn.benchmark = True
    config = setup(args)
    model = build_model_for_eval(config, args.pretrained_weights)
    autocast_dtype = get_autocast_dtype(config)
    return model, autocast_dtype
