:py:mod:`dinov2.logging`
========================

.. py:module:: dinov2.logging


Submodules
----------
.. toctree::
   :titlesonly:
   :maxdepth: 1

   helpers/index.rst


Package Contents
----------------

Classes
~~~~~~~

.. autoapisummary::

   dinov2.logging.MetricLogger
   dinov2.logging.SmoothedValue



Functions
~~~~~~~~~

.. autoapisummary::

   dinov2.logging.setup_logging



.. py:class:: MetricLogger(delimiter='\t', output_file=None)


   Bases: :py:obj:`object`

   .. py:method:: update(**kwargs)


   .. py:method:: __getattr__(attr)


   .. py:method:: __str__()

      Return str(self).


   .. py:method:: synchronize_between_processes()


   .. py:method:: add_meter(name, meter)


   .. py:method:: dump_in_output_file(iteration, iter_time, data_time)


   .. py:method:: log_every(iterable, print_freq, header=None, n_iterations=None, start_iteration=0)



.. py:class:: SmoothedValue(window_size=20, fmt=None)


   Track a series of values and provide access to smoothed values over a
   window or the global series average.

   .. py:property:: median


   .. py:property:: avg


   .. py:property:: global_avg


   .. py:property:: max


   .. py:property:: value


   .. py:method:: update(value, num=1)


   .. py:method:: synchronize_between_processes()

      Distributed synchronization of the metric
      Warning: does not synchronize the deque!


   .. py:method:: __str__()

      Return str(self).



.. py:function:: setup_logging(output: Optional[str] = None, *, name: Optional[str] = None, level: int = logging.DEBUG, capture_warnings: bool = True) -> None

   Setup logging.

   :param output: A file name or a directory to save log files. If None, log
                  files will not be saved. If output ends with ".txt" or ".log", it
                  is assumed to be a file name.
                  Otherwise, logs will be saved to `output/log.txt`.
   :param name: The name of the logger to configure, by default the root logger.
   :param level: The logging level to use.
   :param capture_warnings: Whether warnings should be captured as logs.


