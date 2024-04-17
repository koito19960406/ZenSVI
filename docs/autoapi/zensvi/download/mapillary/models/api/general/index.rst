:py:mod:`zensvi.download.mapillary.models.api.general`
======================================================

.. py:module:: zensvi.download.mapillary.models.api.general

.. autoapi-nested-parse::

   mapillary.models.api.entities
   ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

   This module contains the Adapter design for the Entities API of Mapillary API v4.

   For more information, please check out https://www.mapillary.com/developer/api-documentation/.

   - Copyright: (c) 2021 Facebook
   - License: MIT LICENSE



Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   zensvi.download.mapillary.models.api.general.GeneralAdapter




.. py:class:: GeneralAdapter(*args: object)


   Bases: :py:obj:`object`

   General adaptor for using the API calls defined for the general module
   (mapillary.config.api.general)

   The GeneralAdaptor provides functions for getting preprocessed data from the API, through the
   API calls mentioned in the previously mentioned config.

   It performs parsing, property handling, easier logic for extracing information, focusing on
   adding a layer of abstraction by removing details of using `mercantile`, `ast`, et al, to
   focus on the inputs and outputs of functions

   Usage::

       >>> import mapillary

   .. py:method:: fetch_image_tiles(zoom: int, longitude: float, latitude: float, layer: str = 'image') -> dict

      Get the tiles for a given image.

      :param zoom: Zoom level of the image.
      :type zoom: int

      :param longitude: Longitude of the image
      :type longitude: float

      :param latitude: Latitude of the image
      :type latitude: float

      :return: A dictionary containing the tiles for the image.
      :rtype: dict


   .. py:method:: fetch_computed_image_tiles(zoom: int, longitude: float, latitude: float, layer: str = 'image') -> dict

      Get the image type for a given image.

      :param zoom: The zoom to get the image type for.
      :type zoom: int

      :param longitude: The longitude of the image.
      :type longitude: float

      :param latitude: The latitude of the image.
      :type latitude: float

      :return: A dictionary containing the image type for the image.
      :rtype: dict


   .. py:method:: fetch_map_features_tiles(zoom: int, longitude: float, latitude: float, layer: str = 'image')

      Get the map features for a given coordinate set

      :param zoom: The zoom value to get the map features for
      :type zoom: int

      :param longitude: The longitude of the image
      :type longitude: float

      :param latitude: The latitude of the image
      :type latitude: float

      :return: A dictionary containing the map features for the image.
      :rtype: dict


   .. py:method:: fetch_map_features_traffic_tiles(zoom: int, longitude: float, latitude: float, layer: str)

      Get the map feature traffic for a given coordinate set

      :param zoom: The zoom value to get the map features for
      :type zoom: int

      :param longitude: The longitude of the image
      :type longitude: float

      :param latitude: The latitude of the image
      :type latitude: float

      :return: A dictionary containing the map features for the image.
      :rtype: dict



