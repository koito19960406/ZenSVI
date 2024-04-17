:py:mod:`zensvi.download.mapillary.config.api.general`
======================================================

.. py:module:: zensvi.download.mapillary.config.api.general

.. autoapi-nested-parse::

   mapillary.config.api.general
   ~~~~~~~~~~~~~~~~~~~~~~~~~~~~

   This module contains the class implementation of the
   general metadata functionalities for the API v4 of Mapillary.

   For more information, please check out https://www.mapillary.com/developer/api-documentation/.

   - Copyright: (c) 2021 Facebook
   - License: MIT LICENSE



Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   zensvi.download.mapillary.config.api.general.General




.. py:class:: General


   A general list of metadata API endpoints for API v4

   .. py:method:: get_tile_metadata()
      :staticmethod:

      Root endpoint for metadata


   .. py:method:: get_vector_tiles()
      :staticmethod:

      Root endpoint for vector tiles


   .. py:method:: get_image_type_tiles(x: float, y: float, z: float) -> str
      :staticmethod:

      image_type tiles


   .. py:method:: get_computed_image_type_tiles(x: float, y: float, z: float) -> str
      :staticmethod:

      Computed image_type tiles


   .. py:method:: get_map_features_points_tiles(x: float, y: float, z: float) -> str
      :staticmethod:

      Map features (points) tiles


   .. py:method:: get_map_features_traffic_signs_tiles(x: float, y: float, z: float) -> str
      :staticmethod:

      Map features (traffic signs) tiles



