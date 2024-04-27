:py:mod:`zensvi.download.mapillary.controller.feature`
======================================================

.. py:module:: zensvi.download.mapillary.controller.feature

.. autoapi-nested-parse::

   mapillary.controllers.feature
   ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

   This module implements the feature extraction business logic functionalities of the Mapillary
   Python SDK.

   For more information, please check out https://www.mapillary.com/developer/api-documentation/

   - Copyright: (c) 2021 Facebook
   - License: MIT LICENSE



Module Contents
---------------


Functions
~~~~~~~~~

.. autoapisummary::

   zensvi.download.mapillary.controller.feature.get_feature_from_key_controller
   zensvi.download.mapillary.controller.feature.get_map_features_in_bbox_controller



.. py:function:: get_feature_from_key_controller(key: int, fields: list) -> str

   A controller for getting properties of a certain image given the image key and
   the list of fields/properties to be returned

   :param key: The image key
   :type key: int

   :param fields: List of possible fields
   :type fields: list

   :return: The requested feature properties in GeoJSON format
   :rtype: str


.. py:function:: get_map_features_in_bbox_controller(bbox: dict, filter_values: list, filters: dict, layer: str = 'points') -> str

   For extracting either map feature points or traffic signs within a bounding box

   :param bbox: Bounding box coordinates as argument
   :type bbox: dict

   :param layer: 'points' or 'traffic_signs'
   :type layer: str

   :param filter_values: a list of filter values supported by the API.
   :type filter_values: list

   :param filters: Chronological filters
   :type filters: dict

   :return: GeoJSON
   :rtype: str


