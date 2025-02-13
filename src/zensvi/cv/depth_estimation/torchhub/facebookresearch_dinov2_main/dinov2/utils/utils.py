# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
import random
import subprocess
from urllib.parse import urlparse

import numpy as np
import torch
from torch import nn

logger = logging.getLogger("dinov2")


def load_pretrained_weights(model, pretrained_weights, checkpoint_key):
    """Load pretrained weights into the model.

    Args:
        model (torch.nn.Module): The model to load weights into.
        pretrained_weights (str): The path or URL to the pretrained weights.
        checkpoint_key (str, optional): The key to extract from the state_dict if provided.

    Returns:
        None
    """
    if urlparse(pretrained_weights).scheme:  # If it looks like an URL
        state_dict = torch.hub.load_state_dict_from_url(pretrained_weights, map_location="cpu")
    else:
        state_dict = torch.load(pretrained_weights, map_location="cpu", weights_only=False)
    if checkpoint_key is not None and checkpoint_key in state_dict:
        logger.info(f"Take key {checkpoint_key} in provided checkpoint dict")
        state_dict = state_dict[checkpoint_key]
    # remove `module.` prefix
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    # remove `backbone.` prefix induced by multicrop wrapper
    state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
    msg = model.load_state_dict(state_dict, strict=False)
    logger.info("Pretrained weights found at {} and loaded with msg: {}".format(pretrained_weights, msg))


def fix_random_seeds(seed=31):
    """Fix random seeds for reproducibility.

    Args:
        seed (int, optional): The seed value to set. Default is 31.

    Returns:
        None
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def get_sha():
    """Get the current git commit SHA, branch, and status.

    Returns:
        str: A string containing the SHA, status of the working directory, and current branch.
    """
    cwd = os.path.dirname(os.path.abspath(__file__))

    def _run(command):
        """Run a command in the specified directory.

        Args:
            command (list): The command to run as a list of strings.

        Returns:
            str: The output of the command.
        """
        return subprocess.check_output(command, cwd=cwd).decode("ascii").strip()

    sha = "N/A"
    diff = "clean"
    branch = "N/A"
    try:
        sha = _run(["git", "rev-parse", "HEAD"])
        subprocess.check_output(["git", "diff"], cwd=cwd)
        diff = _run(["git", "diff-index", "HEAD"])
        diff = "has uncommitted changes" if diff else "clean"
        branch = _run(["git", "rev-parse", "--abbrev-ref", "HEAD"])
    except Exception:
        pass
    message = f"sha: {sha}, status: {diff}, branch: {branch}"
    return message


class CosineScheduler(object):
    """A scheduler that follows a cosine annealing schedule."""

    def __init__(
        self,
        base_value,
        final_value,
        total_iters,
        warmup_iters=0,
        start_warmup_value=0,
        freeze_iters=0,
    ):
        """Initialize the CosineScheduler.

        Args:
            base_value (float): The initial value of the schedule.
            final_value (float): The final value of the schedule.
            total_iters (int): The total number of iterations for the schedule.
            warmup_iters (int, optional): The number of warmup iterations. Default is 0.
            start_warmup_value (float, optional): The starting value during warmup. Default is 0.
            freeze_iters (int, optional): The number of iterations to freeze the value. Default is 0.
        """
        super().__init__()
        self.final_value = final_value
        self.total_iters = total_iters

        freeze_schedule = np.zeros((freeze_iters))

        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

        iters = np.arange(total_iters - warmup_iters - freeze_iters)
        schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))
        self.schedule = np.concatenate((freeze_schedule, warmup_schedule, schedule))

        assert len(self.schedule) == self.total_iters

    def __getitem__(self, it):
        """Get the value at a specific iteration.

        Args:
            it (int): The iteration index.

        Returns:
            float: The value at the specified iteration.
        """
        if it >= self.total_iters:
            return self.final_value
        else:
            return self.schedule[it]


def has_batchnorms(model):
    """Check if the model contains any BatchNorm layers.

    Args:
        model (torch.nn.Module): The model to check.

    Returns:
        bool: True if the model contains BatchNorm layers, False otherwise.
    """
    bn_types = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm)
    for name, module in model.named_modules():
        if isinstance(module, bn_types):
            return True
    return False
