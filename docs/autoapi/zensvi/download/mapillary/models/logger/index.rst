:py:mod:`zensvi.download.mapillary.models.logger`
=================================================

.. py:module:: zensvi.download.mapillary.models.logger

.. autoapi-nested-parse::

   mapillary.utils.logger
   ~~~~~~~~~~~~~~~~~~~~~~

   This module implements the logger for mapillary, which is a wrapper of the logger package and
   the default configuration for each of the loggers per file.



Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   zensvi.download.mapillary.models.logger.Logger




.. py:class:: Logger


   .. py:attribute:: format_string
      :type: str
      :value: '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

      

   .. py:attribute:: level
      :type: int

      

   .. py:method:: setup_logger(name: str, level: int = logging.INFO) -> logging.Logger
      :staticmethod:

      Function setup as many loggers as you want. To be used at the top of the file.

      Usage::

          >>> Logger.setup_logger(name='mapillary.xxxx.yyyy', level=logging.INFO)
          logger.Logger

      :param name: The name of the logger
      :type name: str

      :param level: The level of the logger
      :type level: int

      :return: The logger object
      :rtype: logging.Logger


   .. py:method:: get_os_log_path(log_file: str) -> str
      :staticmethod:

      Get the path of the log file based on the OS

      :param log_file: The name of the log file
      :type log_file: str

      :return: The path where the logs will be stored
      :rtype: str



