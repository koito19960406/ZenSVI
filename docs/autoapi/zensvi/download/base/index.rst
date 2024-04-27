:orphan:

:py:mod:`zensvi.download.base`
==============================

.. py:module:: zensvi.download.base


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   zensvi.download.base.BaseDownloader




.. py:class:: BaseDownloader(log_path=None)


   Bases: :py:obj:`abc.ABC`

   Helper class that provides a standard way to create an ABC using
   inheritance.

   .. py:property:: log_path

      Property for log_path.

      :return: log_path
      :rtype: str

   .. py:method:: download_svi(dir_output, lat=None, lon=None, input_csv_file='', input_shp_file='', input_place_name='', id_columns=None, buffer=0, update_pids=False, start_date=None, end_date=None, metadata_only=False)
      :abstractmethod:



