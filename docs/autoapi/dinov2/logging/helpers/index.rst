:py:mod:`dinov2.logging.helpers`
================================

.. py:module:: dinov2.logging.helpers


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   dinov2.logging.helpers.MetricLogger
   dinov2.logging.helpers.SmoothedValue




Attributes
~~~~~~~~~~

.. autoapisummary::

   dinov2.logging.helpers.logger


.. py:data:: logger

   

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



