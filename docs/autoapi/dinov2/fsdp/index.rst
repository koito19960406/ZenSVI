:py:mod:`dinov2.fsdp`
=====================

.. py:module:: dinov2.fsdp


Package Contents
----------------

Classes
~~~~~~~

.. autoapisummary::

   dinov2.fsdp.FSDPCheckpointer



Functions
~~~~~~~~~

.. autoapisummary::

   dinov2.fsdp.get_fsdp_wrapper
   dinov2.fsdp.is_fsdp
   dinov2.fsdp.is_sharded_fsdp
   dinov2.fsdp.free_if_fsdp
   dinov2.fsdp.get_fsdp_modules
   dinov2.fsdp.reshard_fsdp_model
   dinov2.fsdp.rankstr



Attributes
~~~~~~~~~~

.. autoapisummary::

   dinov2.fsdp.ShardedGradScaler


.. py:function:: get_fsdp_wrapper(model_cfg, modules_to_wrap=set())


.. py:function:: is_fsdp(x)


.. py:function:: is_sharded_fsdp(x)


.. py:function:: free_if_fsdp(x)


.. py:function:: get_fsdp_modules(x)


.. py:function:: reshard_fsdp_model(x)


.. py:function:: rankstr()


.. py:class:: FSDPCheckpointer


   Bases: :py:obj:`fvcore.common.checkpoint.Checkpointer`

   .. py:method:: save(name: str, **kwargs: Any) -> None

      Dump model and checkpointables to a file.

      :param name: name of the file.
      :type name: str
      :param kwargs: extra arbitrary data to save.
      :type kwargs: dict


   .. py:method:: load(*args, **kwargs)


   .. py:method:: has_checkpoint() -> bool

      :returns: whether a checkpoint exists in the target directory.
      :rtype: bool


   .. py:method:: get_checkpoint_file() -> str

      :returns: The latest checkpoint file in target directory.
      :rtype: str


   .. py:method:: tag_last_checkpoint(last_filename_basename: str) -> None

      Tag the last checkpoint.

      :param last_filename_basename: the basename of the last filename.
      :type last_filename_basename: str



.. py:data:: ShardedGradScaler

   

