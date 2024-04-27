:py:mod:`zensvi.cv.depth_estimation.zoedepth.models.builder`
============================================================

.. py:module:: zensvi.cv.depth_estimation.zoedepth.models.builder


Module Contents
---------------


Functions
~~~~~~~~~

.. autoapisummary::

   zensvi.cv.depth_estimation.zoedepth.models.builder.build_model



.. py:function:: build_model(config) -> zensvi.cv.depth_estimation.zoedepth.models.depth_model.DepthModel

   Builds a model from a config. The model is specified by the model name and version in the config. The model is then constructed using the build_from_config function of the model interface.
   This function should be used to construct models for training and evaluation.

   :param config: Config dict. Config is constructed in utils/config.py. Each model has its own config file(s) saved in its root model folder.
   :type config: dict

   :returns: Model corresponding to name and version as specified in config
   :rtype: torch.nn.Module


