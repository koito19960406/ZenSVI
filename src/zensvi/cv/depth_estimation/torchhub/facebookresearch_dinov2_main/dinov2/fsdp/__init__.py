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
    """

    Args:
      model_cfg:
      modules_to_wrap: (Default value = set())

    Returns:

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
    """

    Args:
      x:

    Returns:

    """
    return isinstance(x, FSDP)


def is_sharded_fsdp(x):
    """

    Args:
      x:

    Returns:

    """
    return is_fsdp(x) and x.sharding_strategy is not ShardingStrategy.NO_SHARD


def free_if_fsdp(x):
    """

    Args:
      x:

    Returns:

    """
    if is_sharded_fsdp(x):
        handles = x._handles
        true_list = [True for h in handles]
        _reshard(x, handles, true_list)


def get_fsdp_modules(x):
    """

    Args:
      x:

    Returns:

    """
    return FSDP.fsdp_modules(x)


def reshard_fsdp_model(x):
    """

    Args:
      x:

    Returns:

    """
    for m in get_fsdp_modules(x):
        free_if_fsdp(m)


def rankstr():
    """"""
    return f"rank_{distributed.get_global_rank()}"


class FSDPCheckpointer(Checkpointer):
    """"""

    def save(self, name: str, **kwargs: Any) -> None:
        """Dump model and checkpointables to a file.

        Args:
          name(str): name of the file.
          kwargs(dict): extra arbitrary data to save.
          name: str:
          **kwargs: Any:
          name: str:
          **kwargs: Any:

        Returns:
        """
        if not self.save_dir or not self.save_to_disk:
            return

        data = {}
        with FSDP.state_dict_type(self.model, StateDictType.LOCAL_STATE_DICT):
            data["model"] = self.model.state_dict()

        # data["model"] = self.model.state_dict()
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
        """

        Args:
          *args:
          **kwargs:

        Returns:

        """
        with FSDP.state_dict_type(self.model, StateDictType.LOCAL_STATE_DICT):
            return super().load(*args, **kwargs)

    def has_checkpoint(self) -> bool:
        """

        Args:

        Returns:
          bool: whether a checkpoint exists in the target directory.

        """
        save_file = os.path.join(self.save_dir, f"last_checkpoint.{rankstr()}")
        return self.path_manager.exists(save_file)

    def get_checkpoint_file(self) -> str:
        """

        Args:

        Returns:
          str: The latest checkpoint file in target directory.

        """
        save_file = os.path.join(self.save_dir, f"last_checkpoint.{rankstr()}")
        try:
            with self.path_manager.open(save_file, "r") as f:
                last_saved = f.read().strip()
        except IOError:
            # if file doesn't exist, maybe because it has just been
            # deleted by a separate process
            return ""
        # pyre-fixme[6]: For 2nd param expected `Union[PathLike[str], str]` but got
        #  `Union[bytes, str]`.
        return os.path.join(self.save_dir, last_saved)

    def tag_last_checkpoint(self, last_filename_basename: str) -> None:
        """Tag the last checkpoint.

        Args:
          last_filename_basename(str): the basename of the last filename.
          last_filename_basename: str:
          last_filename_basename: str:

        Returns:
        """
        if distributed.is_enabled():
            torch.distributed.barrier()
        save_file = os.path.join(self.save_dir, f"last_checkpoint.{rankstr()}")
        with self.path_manager.open(save_file, "w") as f:
            f.write(last_filename_basename)  # pyre-ignore


ShardedGradScaler = ShardedGradScaler
