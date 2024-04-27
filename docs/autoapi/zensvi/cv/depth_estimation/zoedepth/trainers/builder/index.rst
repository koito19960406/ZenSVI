:py:mod:`zensvi.cv.depth_estimation.zoedepth.trainers.builder`
==============================================================

.. py:module:: zensvi.cv.depth_estimation.zoedepth.trainers.builder


Module Contents
---------------


Functions
~~~~~~~~~

.. autoapisummary::

   zensvi.cv.depth_estimation.zoedepth.trainers.builder.get_trainer



.. py:function:: get_trainer(config)

   Builds and returns a trainer based on the config.

   :param config: the config dict (typically constructed using utils.config.get_config)
                  config.trainer (str): the name of the trainer to use. The module named "{config.trainer}_trainer" must exist in trainers root module
   :type config: dict

   :raises ValueError: If the specified trainer does not exist under trainers/ folder

   :returns: The Trainer object
   :rtype: Trainer (inherited from zoedepth.trainers.BaseTrainer)


