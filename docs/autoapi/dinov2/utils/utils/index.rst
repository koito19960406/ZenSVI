:py:mod:`dinov2.utils.utils`
============================

.. py:module:: dinov2.utils.utils


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   dinov2.utils.utils.CosineScheduler



Functions
~~~~~~~~~

.. autoapisummary::

   dinov2.utils.utils.load_pretrained_weights
   dinov2.utils.utils.fix_random_seeds
   dinov2.utils.utils.get_sha
   dinov2.utils.utils.has_batchnorms



Attributes
~~~~~~~~~~

.. autoapisummary::

   dinov2.utils.utils.logger


.. py:data:: logger

   

.. py:function:: load_pretrained_weights(model, pretrained_weights, checkpoint_key)


.. py:function:: fix_random_seeds(seed=31)

   Fix random seeds.


.. py:function:: get_sha()


.. py:class:: CosineScheduler(base_value, final_value, total_iters, warmup_iters=0, start_warmup_value=0, freeze_iters=0)


   Bases: :py:obj:`object`

   .. py:method:: __getitem__(it)



.. py:function:: has_batchnorms(model)


