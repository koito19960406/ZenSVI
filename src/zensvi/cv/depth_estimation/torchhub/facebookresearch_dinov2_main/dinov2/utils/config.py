# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import math
import os

import dinov2.distributed as distributed
from dinov2.configs import dinov2_default_config
from dinov2.logging import setup_logging
from dinov2.utils import utils
from omegaconf import OmegaConf

logger = logging.getLogger("dinov2")


def apply_scaling_rules_to_cfg(cfg):
    """Applies scaling rules to the configuration.

    Args:
        cfg: The configuration object containing optimization settings.

    Returns:
        The updated configuration object with applied scaling rules.

    Raises:
        NotImplementedError: If the scaling rule is not implemented.
    """
    if cfg.optim.scaling_rule == "sqrt_wrt_1024":
        base_lr = cfg.optim.base_lr
        cfg.optim.lr = base_lr
        cfg.optim.lr *= math.sqrt(cfg.train.batch_size_per_gpu * distributed.get_global_size() / 1024.0)
        logger.info(f"sqrt scaling learning rate; base: {base_lr}, new: {cfg.optim.lr}")
    else:
        raise NotImplementedError
    return cfg


def write_config(cfg, output_dir, name="config.yaml"):
    """Writes the configuration to a YAML file.

    Args:
        cfg: The configuration object to be saved.
        output_dir: The directory where the configuration file will be saved.
        name: The name of the configuration file (default is "config.yaml").

    Returns:
        The path to the saved configuration file.
    """
    logger.info(OmegaConf.to_yaml(cfg))
    saved_cfg_path = os.path.join(output_dir, name)
    with open(saved_cfg_path, "w") as f:
        OmegaConf.save(config=cfg, f=f)
    return saved_cfg_path


def get_cfg_from_args(args):
    """Generates a configuration object from command line arguments.

    Args:
        args: The command line arguments containing configuration options.

    Returns:
        The merged configuration object.
    """
    args.output_dir = os.path.abspath(args.output_dir)
    args.opts += [f"train.output_dir={args.output_dir}"]
    default_cfg = OmegaConf.create(dinov2_default_config)
    cfg = OmegaConf.load(args.config_file)
    cfg = OmegaConf.merge(default_cfg, cfg, OmegaConf.from_cli(args.opts))
    return cfg


def default_setup(args):
    """Performs the default setup for distributed training.

    Args:
        args: The command line arguments containing setup options.

    Returns:
        None
    """
    distributed.enable(overwrite=True)
    seed = getattr(args, "seed", 0)
    rank = distributed.get_global_rank()

    global logger
    setup_logging(output=args.output_dir, level=logging.INFO)
    logger = logging.getLogger("dinov2")

    utils.fix_random_seeds(seed + rank)
    logger.info("git:\n  {}\n".format(utils.get_sha()))
    logger.info("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))


def setup(args):
    """Creates configurations and performs basic setups.

    Args:
        args: The command line arguments containing setup options.

    Returns:
        The configuration object after setup.
    """
    cfg = get_cfg_from_args(args)
    os.makedirs(args.output_dir, exist_ok=True)
    default_setup(args)
    apply_scaling_rules_to_cfg(cfg)
    write_config(cfg, args.output_dir)
    return cfg
