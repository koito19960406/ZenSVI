:py:mod:`zensvi.download.mapillary.controller.save`
===================================================

.. py:module:: zensvi.download.mapillary.controller.save

.. autoapi-nested-parse::

   mapillary.controllers.save
   ~~~~~~~~~~~~~~~~~~~~~~~~~~

   This module implements the saving business logic functionalities of the Mapillary Python SDK.

   For more information, please check out https://www.mapillary.com/developer/api-documentation/

   - Copyright: (c) 2021 Facebook
   - License: MIT LICENSE



Module Contents
---------------


Functions
~~~~~~~~~

.. autoapisummary::

   zensvi.download.mapillary.controller.save.save_as_csv_controller
   zensvi.download.mapillary.controller.save.save_as_geojson_controller



.. py:function:: save_as_csv_controller(data: str, path: str, file_name: str) -> None

   Save data as CSV to given file path

   :param data: The data to save as CSV
   :type data: str

   :param path: The path to save to
   :type path: str

   :param file_name: The file name to save as
   :type file_name: str

   :return: None
   :rtype: None


.. py:function:: save_as_geojson_controller(data: str, path: str, file_name: str) -> None

   Save data as GeoJSON to given file path

   :param data: The data to save as GeoJSON
   :type data: str

   :param path: The path to save to
   :type path: str

   :param file_name: The file name to save as
   :type file_name: str

   :return: None
   :rtype: None


