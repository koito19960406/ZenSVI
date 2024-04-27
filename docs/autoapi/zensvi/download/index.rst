:py:mod:`zensvi.download`
=========================

.. py:module:: zensvi.download


Submodules
----------
.. toctree::
   :titlesonly:
   :maxdepth: 1

   mly/index.rst


Package Contents
----------------

Classes
~~~~~~~

.. autoapisummary::

   zensvi.download.MLYDownloader




.. py:class:: MLYDownloader(mly_api_key, log_path=None, max_workers=None)


   Bases: :py:obj:`zensvi.download.base.BaseDownloader`

   Mapillary Downloader class.

   :param mly_api_key: Mapillary API key. Defaults to None.
   :type mly_api_key: str, optional
   :param log_path: Path to the log file. Defaults to None.
   :type log_path: str, optional
   :param max_workers: Number of workers for parallel processing. Defaults to None.
   :type max_workers: int, optional

   .. py:property:: mly_api_key

      Property for Mapillary API key.

      :return: mly_api_key
      :rtype: str

   .. py:property:: max_workers

      Property for the number of workers for parallel processing.

      :return: max_workers
      :rtype: int

   .. py:property:: log_path

      Property for log_path.

      :return: log_path
      :rtype: str

   .. py:method:: download_svi(dir_output, path_pid=None, lat=None, lon=None, input_csv_file='', input_shp_file='', input_place_name='', buffer=0, update_pids=False, resolution=1024, cropped=False, batch_size=1000, start_date=None, end_date=None, metadata_only=False, use_cache=True, **kwargs)

      Downloads street view images from Mapillary using specified parameters.

      :param dir_output: Directory where output files and images will be stored.
      :type dir_output: str
      :param path_pid: Path to a file containing panorama IDs. If not provided, IDs will be fetched based on other parameters.
      :type path_pid: str, optional
      :param lat: Latitude to fetch panorama IDs around this point. Must be used with `lon`.
      :type lat: float, optional
      :param lon: Longitude to fetch panorama IDs around this point. Must be used with `lat`.
      :type lon: float, optional
      :param input_csv_file: Path to a CSV file containing locations for which to fetch panorama IDs.
      :type input_csv_file: str, optional
      :param input_shp_file: Path to a shapefile containing geographic locations for fetching panorama IDs.
      :type input_shp_file: str, optional
      :param input_place_name: A place name for geocoding to fetch panorama IDs.
      :type input_place_name: str, optional
      :param buffer: Buffer size in meters to expand the geographic area for panorama ID fetching.
      :type buffer: int, optional
      :param update_pids: If True, will update panorama IDs even if a valid `path_pid` is provided. Defaults to False.
      :type update_pids: bool, optional
      :param resolution: The resolution of the images to download. Defaults to 1024.
      :type resolution: int, optional
      :param cropped: If True, images will be cropped to the upper half. Defaults to False.
      :type cropped: bool, optional
      :param batch_size: Number of images to process in each batch. Defaults to 1000.
      :type batch_size: int, optional
      :param start_date: Start date (YYYY-MM-DD) to filter images by capture date.
      :type start_date: str, optional
      :param end_date: End date (YYYY-MM-DD) to filter images by capture date.
      :type end_date: str, optional
      :param metadata_only: If True, skips downloading images and only fetches metadata. Defaults to False.
      :type metadata_only: bool, optional
      :param use_cache: If True, uses cached data to speed up the operation. Defaults to True.
      :type use_cache: bool, optional
      :param \*\*kwargs: Additional keyword arguments that are passed to the API.

      :returns: This method does not return a value but will save files directly to the specified output directory.
      :rtype: None

      :raises ValueError: If required parameters for fetching panorama IDs are not adequately specified.
      :raises FileNotFoundError: If `path_pid` is specified but the file does not exist.

      .. rubric:: Notes

      This method logs significant events and errors, making it suitable for both interactive usage and automated workflows.



