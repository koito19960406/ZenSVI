:py:mod:`zensvi.download.mapillary.controller.detection`
========================================================

.. py:module:: zensvi.download.mapillary.controller.detection

.. autoapi-nested-parse::

   mapillary.controllers.detection
   ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

   This module implements the detection based business logic functionalities of the Mapillary
   Python SDK.

   For more information, please check out https://www.mapillary.com/developer/api-documentation/

   - Copyright: (c) 2021 Facebook
   - License: MIT LICENSE



Module Contents
---------------


Functions
~~~~~~~~~

.. autoapisummary::

   zensvi.download.mapillary.controller.detection.get_image_detections_controller
   zensvi.download.mapillary.controller.detection.get_map_feature_detections_controller



.. py:function:: get_image_detections_controller(image_id: Union[str, int], fields: list = []) -> zensvi.download.mapillary.models.geojson.GeoJSON

   Get image detections with given (image) key

   :param image_id: The image id
   :type image_id: str

   :param fields: The fields possible for the detection endpoint. Please see
       https://www.mapillary.com/developer/api-documentation for more information
   :type fields: list

   :return: GeoJSON
   :rtype: dict


.. py:function:: get_map_feature_detections_controller(map_feature_id: Union[str, int], fields: list = []) -> zensvi.download.mapillary.models.geojson.GeoJSON

   Get image detections with given (map feature) key

   :param map_feature_id: The map feature id
   :type map_feature_id: str

   :param fields: The fields possible for the detection endpoint. Please see
       https://www.mapillary.com/developer/api-documentation for more information
   :type fields: list

   :return: GeoJSON
   :rtype: dict


