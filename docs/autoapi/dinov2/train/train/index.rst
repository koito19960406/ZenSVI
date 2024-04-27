:py:mod:`dinov2.train.train`
============================

.. py:module:: dinov2.train.train


Module Contents
---------------


Functions
~~~~~~~~~

.. autoapisummary::

   dinov2.train.train.get_args_parser
   dinov2.train.train.build_optimizer
   dinov2.train.train.build_schedulers
   dinov2.train.train.apply_optim_scheduler
   dinov2.train.train.do_test
   dinov2.train.train.do_train
   dinov2.train.train.main



Attributes
~~~~~~~~~~

.. autoapisummary::

   dinov2.train.train.logger
   dinov2.train.train.args


.. py:data:: logger

   

.. py:function:: get_args_parser(add_help: bool = True)


.. py:function:: build_optimizer(cfg, params_groups)


.. py:function:: build_schedulers(cfg)


.. py:function:: apply_optim_scheduler(optimizer, lr, wd, last_layer_lr)


.. py:function:: do_test(cfg, model, iteration)


.. py:function:: do_train(cfg, model, resume=False)


.. py:function:: main(args)


.. py:data:: args

   

