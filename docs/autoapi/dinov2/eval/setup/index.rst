:py:mod:`dinov2.eval.setup`
===========================

.. py:module:: dinov2.eval.setup


Module Contents
---------------


Functions
~~~~~~~~~

.. autoapisummary::

   dinov2.eval.setup.get_args_parser
   dinov2.eval.setup.get_autocast_dtype
   dinov2.eval.setup.build_model_for_eval
   dinov2.eval.setup.setup_and_build_model



.. py:function:: get_args_parser(description: Optional[str] = None, parents: Optional[List[argparse.ArgumentParser]] = None, add_help: bool = True)


.. py:function:: get_autocast_dtype(config)


.. py:function:: build_model_for_eval(config, pretrained_weights)


.. py:function:: setup_and_build_model(args) -> Tuple[Any, torch.dtype]


