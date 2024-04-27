:orphan:

:py:mod:`zensvi.cv.depth_estimation.zoedepth.trainers.base_trainer`
===================================================================

.. py:module:: zensvi.cv.depth_estimation.zoedepth.trainers.base_trainer


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   zensvi.cv.depth_estimation.zoedepth.trainers.base_trainer.BaseTrainer



Functions
~~~~~~~~~

.. autoapisummary::

   zensvi.cv.depth_estimation.zoedepth.trainers.base_trainer.is_rank_zero



.. py:function:: is_rank_zero(args)


.. py:class:: BaseTrainer(config, model, train_loader, test_loader=None, device=None)


   .. py:property:: iters_per_epoch


   .. py:property:: total_iters


   .. py:method:: resize_to_target(prediction, target)


   .. py:method:: load_ckpt(checkpoint_dir='./checkpoints', ckpt_type='best')


   .. py:method:: init_optimizer()


   .. py:method:: init_scheduler()


   .. py:method:: train_on_batch(batch, train_step)
      :abstractmethod:


   .. py:method:: validate_on_batch(batch, val_step)
      :abstractmethod:


   .. py:method:: raise_if_nan(losses)


   .. py:method:: should_early_stop()


   .. py:method:: train()


   .. py:method:: validate()


   .. py:method:: save_checkpoint(filename)


   .. py:method:: log_images(rgb: Dict[str, list] = {}, depth: Dict[str, list] = {}, scalar_field: Dict[str, list] = {}, prefix='', scalar_cmap='jet', min_depth=None, max_depth=None)


   .. py:method:: log_line_plot(data)


   .. py:method:: log_bar_plot(title, labels, values)



