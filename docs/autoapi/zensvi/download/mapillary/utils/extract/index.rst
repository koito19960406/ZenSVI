:py:mod:`zensvi.download.mapillary.utils.extract`
=================================================

.. py:module:: zensvi.download.mapillary.utils.extract

.. autoapi-nested-parse::

   mapillary.utils.extract
   =======================

   This module deals with extracting multiple fields nested within a GeoJSON packet.

   - Copyright: (c) 2021 Facebook
   - License: MIT LICENSE



Module Contents
---------------


Functions
~~~~~~~~~

.. autoapisummary::

   zensvi.download.mapillary.utils.extract.extract_properties



.. py:function:: extract_properties(geojson: dict, properties: list) -> dict

   Extracts specific properties from a complete GeoJSON

   :param geojson: GeoJSON object
   :type geojson: dict

   :param properties: A list of properties to extract
   :type properties: list

   :return: The extracted fields as a dict
   :rtype: dict

   Usage::

       >>> from utils.extract import extract_properties
       >>> extract_properties(geojson={"type":"FeatureCollection","features":[{"type":"Feature",
       ... "geometry":{"type":"Point","coordinates":[-80.12991070747375,25.787652114106322]},
       ... "properties":{"captured_at":1540386861135, "compass_angle":252.04260253906,"id":
       ... 1274987139570038,"is_pano":'False',"sequence_id":"Vf8Iwxx5SemxI7_b_7J5Kw"}},{"type":
       ... "Feature","geometry":{"type":"Point","coordinates":[-80.13223886489868,
       ... 25.78756517066695]}, "properties":{"captured_at":1422989164000,"compass_angle":
       ... 89.781,"id":169629268373019,"is_pano": "True","sequence_id":"dqjuprkOwUnmdEVt5gx-Iw"}}]}
       ... , properties=['id']) # id most likely exists


