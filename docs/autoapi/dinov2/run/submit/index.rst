:py:mod:`dinov2.run.submit`
===========================

.. py:module:: dinov2.run.submit


Module Contents
---------------


Functions
~~~~~~~~~

.. autoapisummary::

   dinov2.run.submit.get_args_parser
   dinov2.run.submit.get_shared_folder
   dinov2.run.submit.submit_jobs



Attributes
~~~~~~~~~~

.. autoapisummary::

   dinov2.run.submit.logger


.. py:data:: logger

   

.. py:function:: get_args_parser(description: Optional[str] = None, parents: Optional[List[argparse.ArgumentParser]] = None, add_help: bool = True) -> argparse.ArgumentParser


.. py:function:: get_shared_folder() -> pathlib.Path


.. py:function:: submit_jobs(task_class, args, name: str)


