# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
from functools import partial
from typing import Any

import dinov2.distributed as distributed
import torch
from fvcore.common.checkpoint import Checkpointer
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision, ShardingStrategy, StateDictType
from torch.distributed.fsdp._runtime_utils import _reshard
from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler
from torch.distributed.fsdp.wrap import ModuleWrapPolicy


def get_fsdp_wrapper(model_cfg, modules_to_wrap=set()):
    """Creates a wrapper for the Fully Sharded Data Parallel (FSDP) model.

    Args:
        model_cfg: Configuration object containing model settings.
        modules_to_wrap (set, optional): A set of modules to wrap with FSDP. Defaults to an empty set.

    Returns:
        Callable: A partial function that wraps the FSDP model with the specified configuration.
    """
    sharding_strategy_dict = {
        "NO_SHARD": ShardingStrategy.NO_SHARD,
        "SHARD_GRAD_OP": ShardingStrategy.SHARD_GRAD_OP,
        "FULL_SHARD": ShardingStrategy.FULL_SHARD,
    }

    dtype_dict = {
        "fp32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }

    mixed_precision_config = MixedPrecision(
        param_dtype=dtype_dict[model_cfg.mixed_precision.param_dtype],
        reduce_dtype=dtype_dict[model_cfg.mixed_precision.reduce_dtype],
        buffer_dtype=dtype_dict[model_cfg.mixed_precision.buffer_dtype],
    )

    sharding_strategy_config = sharding_strategy_dict[model_cfg.sharding_strategy]

    local_rank = distributed.get_local_rank()

    fsdp_wrapper = partial(
        FSDP,
        sharding_strategy=sharding_strategy_config,
        mixed_precision=mixed_precision_config,
        device_id=local_rank,
        sync_module_states=True,
        use_orig_params=True,
        auto_wrap_policy=ModuleWrapPolicy(modules_to_wrap),
    )
    return fsdp_wrapper


def is_fsdp(x):
    """Checks if the given object is an instance of Fully Sharded Data Parallel (FSDP).

    Args:
        x: The object to check.

    Returns:
        bool: True if the object is an instance of FSDP, False otherwise.
    """
    return isinstance(x, FSDP)


def is_sharded_fsdp(x):
    """Checks if the given FSDP instance is using sharding.

    Args:
        x: The FSDP instance to check.

    Returns:
        bool: True if the FSDP instance is sharded, False otherwise.
    """
    return is_fsdp(x) and x.sharding_strategy is not ShardingStrategy.NO_SHARD


def free_if_fsdp(x):
    """Frees the resources of the FSDP instance if it is sharded.

    Args:
        x: The FSDP instance to free resources from.
    """
    if is_sharded_fsdp(x):
        handles = x._handles
        true_list = [True for h in handles]
        _reshard(x, handles, true_list)


def get_fsdp_modules(x):
    """Retrieves the FSDP modules from the given model.

    Args:
        x: The model from which to retrieve FSDP modules.

    Returns:
        List: A list of FSDP modules contained in the model.
    """
    return FSDP.fsdp_modules(x)


def reshard_fsdp_model(x):
    """Reshards the FSDP model by freeing its modules.

    Args:
        x: The FSDP model to reshard.
    """
    for m in get_fsdp_modules(x):
        free_if_fsdp(m)


def rankstr():
    """Generates a string representation of the current global rank.

    Returns:
        str: A string indicating the current global rank.
    """
    return f"rank_{distributed.get_global_rank()}"


class FSDPCheckpointer(Checkpointer):
    """Checkpointer for saving and loading FSDP models."""

    def save(self, name: str, **kwargs: Any) -> None:
        """Saves the model and checkpointables to a file.

        Args:
            name (str): The name of the file to save.
            **kwargs: Additional arbitrary data to save.
        """
        if not self.save_dir or not self.save_to_disk:
            return

        data = {}
        with FSDP.state_dict_type(self.model, StateDictType.LOCAL_STATE_DICT):
            data["model"] = self.model.state_dict()

        for key, obj in self.checkpointables.items():
            data[key] = obj.state_dict()
        data.update(kwargs)

        basename = f"{name}.{rankstr()}.pth"
        save_file = os.path.join(self.save_dir, basename)
        assert os.path.basename(save_file) == basename, basename
        self.logger.info("Saving checkpoint to {}".format(save_file))
        with self.path_manager.open(save_file, "wb") as f:
            torch.save(data, f)
        self.tag_last_checkpoint(basename)

    def load(self, *args, **kwargs):
        """Loads the model and checkpointables from a file.

        Args:
            *args: Positional arguments for loading.
            **kwargs: Keyword arguments for loading.

        Returns:
            The loaded checkpoint data.
        """
        with FSDP.state_dict_type(self.model, StateDictType.LOCAL_STATE_DICT):
            return super().load(*args, **kwargs)

    def has_checkpoint(self) -> bool:
        """Checks if a checkpoint exists in the target directory.

        Returns:
            bool: True if a checkpoint exists, False otherwise.
        """
        save_file = os.path.join(self.save_dir, f"last_checkpoint.{rankstr()}")
        return self.path_manager.exists(save_file)

    def get_checkpoint_file(self) -> str:
        """Retrieves the latest checkpoint file in the target directory.

        Returns:
            str: The path to the latest checkpoint file.
        """
        save_file = os.path.join(self.save_dir, f"last_checkpoint.{rankstr()}")
        try:
            with self.path_manager.open(save_file, "r") as f:
                last_saved = f.read().strip()
        except IOError:
            return ""
        return os.path.join(self.save_dir, last_saved)

    def tag_last_checkpoint(self, last_filename_basename: str) -> None:
        """Tags the last checkpoint with its basename.

        Args:
            last_filename_basename (str): The basename of the last checkpoint file.
        """
        if distributed.is_enabled():
            torch.distributed.barrier()
        save_file = os.path.join(self.save_dir, f"last_checkpoint.{rankstr()}")
        with self.path_manager.open(save_file, "w") as f:
            f.write(last_filename_basename)  # pyre-ignore


ShardedGradScaler = ShardedGradScaler
