# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import pathlib

from omegaconf import OmegaConf


def load_config(config_name: str):
    """Load a configuration file.

    Args:
        config_name (str): The name of the configuration file (without the .yaml extension).

    Returns:
        OmegaConf: The loaded configuration as an OmegaConf object.
    """
    config_filename = config_name + ".yaml"
    return OmegaConf.load(pathlib.Path(__file__).parent.resolve() / config_filename)


dinov2_default_config = load_config("ssl_default_config")


def load_and_merge_config(config_name: str):
    """Load a configuration file and merge it with the default configuration.

    Args:
        config_name (str): The name of the configuration file (without the .yaml extension).

    Returns:
        OmegaConf: The merged configuration as an OmegaConf object.
    """
    default_config = OmegaConf.create(dinov2_default_config)
    loaded_config = load_config(config_name)
    return OmegaConf.merge(default_config, loaded_config)
