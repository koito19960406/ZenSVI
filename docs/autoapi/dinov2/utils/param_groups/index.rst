:py:mod:`dinov2.utils.param_groups`
===================================

.. py:module:: dinov2.utils.param_groups


Module Contents
---------------


Functions
~~~~~~~~~

.. autoapisummary::

   dinov2.utils.param_groups.get_vit_lr_decay_rate
   dinov2.utils.param_groups.get_params_groups_with_decay
   dinov2.utils.param_groups.fuse_params_groups



Attributes
~~~~~~~~~~

.. autoapisummary::

   dinov2.utils.param_groups.logger


.. py:data:: logger

   

.. py:function:: get_vit_lr_decay_rate(name, lr_decay_rate=1.0, num_layers=12, force_is_backbone=False, chunked_blocks=False)

   Calculate lr decay rate for different ViT blocks.
   :param name: parameter name.
   :type name: string
   :param lr_decay_rate: base lr decay rate.
   :type lr_decay_rate: float
   :param num_layers: number of ViT blocks.
   :type num_layers: int

   :returns: lr decay rate for the given parameter.


.. py:function:: get_params_groups_with_decay(model, lr_decay_rate=1.0, patch_embed_lr_mult=1.0)


.. py:function:: fuse_params_groups(all_params_groups, keys=('lr_multiplier', 'wd_multiplier', 'is_last_layer'))


