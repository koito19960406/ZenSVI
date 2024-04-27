:py:mod:`zensvi.utils.log`
==========================

.. py:module:: zensvi.utils.log


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   zensvi.utils.log.Logger




.. py:class:: Logger


   .. py:method:: log_info(message)


   .. py:method:: log_error(message)


   .. py:method:: log_warning(message)


   .. py:method:: log_debug(message)


   .. py:method:: log_args(function_name, *args, **kwargs)

      Logs the arguments of a function call along with the function's name.


   .. py:method:: log_failed_tiles(failed_tile_name)

      Logs the failed tiles to a log file.


   .. py:method:: log_failed_pids(failed_pid)

      Logs the failed pids to a log file.



