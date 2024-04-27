:py:mod:`dinov2.distributed`
============================

.. py:module:: dinov2.distributed


Package Contents
----------------


Functions
~~~~~~~~~

.. autoapisummary::

   dinov2.distributed.is_enabled
   dinov2.distributed.get_global_size
   dinov2.distributed.get_global_rank
   dinov2.distributed.get_local_rank
   dinov2.distributed.get_local_size
   dinov2.distributed.is_main_process
   dinov2.distributed.enable



.. py:function:: is_enabled() -> bool

   :returns: True if distributed training is enabled


.. py:function:: get_global_size() -> int

   :returns: The number of processes in the process group


.. py:function:: get_global_rank() -> int

   :returns: The rank of the current process within the global process group.


.. py:function:: get_local_rank() -> int

   :returns: The rank of the current process within the local (per-machine) process group.


.. py:function:: get_local_size() -> int

   :returns: The size of the per-machine process group,
             i.e. the number of processes per machine.


.. py:function:: is_main_process() -> bool

   :returns: True if the current process is the main one.


.. py:function:: enable(*, set_cuda_current_device: bool = True, overwrite: bool = False, allow_nccl_timeout: bool = False)

   Enable distributed mode

   :param set_cuda_current_device: If True, call torch.cuda.set_device() to set the
                                   current PyTorch CUDA device to the one matching the local rank.
   :param overwrite: If True, overwrites already set variables. Else fails.


