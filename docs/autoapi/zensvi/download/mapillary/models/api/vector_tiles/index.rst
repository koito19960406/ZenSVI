:py:mod:`zensvi.download.mapillary.models.api.vector_tiles`
===========================================================

.. py:module:: zensvi.download.mapillary.models.api.vector_tiles

.. autoapi-nested-parse::

   mapillary.models.api.vector_tiles
   ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

   This module contains the adapter design for the Vector Tiles API of Mapillary API v4, Vector tiles
   provide an easy way to visualize vast amounts of data. Vector tiles support filtering and querying
   rendered features. Mapillary vector tiles follow the Mapbox tile specification.

   For more information, please check out https://www.mapillary.com/developer/api-documentation/.

   - Copyright: (c) 2021 Facebook
   - License: MIT LICENSE



Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   zensvi.download.mapillary.models.api.vector_tiles.VectorTilesAdapter




.. py:class:: VectorTilesAdapter


   Bases: :py:obj:`object`

   Adapter model for dealing with the VectorTiles API, through the DRY principle. The
   VectorTilesAdapter class can be instantiated in the controller modules, providing an
   abstraction layer that uses the Client class, endpoints provided by the API v4 under
   `/config/api/vector_tiles.py`.

   It performs parsing, handling of layers, properties, and fields to make it easier to
   write higher level logic for extracing information, and lets developers to focus only
   on writing the high level business logic without having to repeat the process of parsing
   and using libraries such as `mercantile`, 'vt2geojson' and others, caring only about
   inputs/outputs

   Usage::

       >>> import mapillary
       >>> from mapillary.models.api.vector_tiles import VectorTilesAdapter
       >>> latitude, longitude = 30, 31
       >>> VectorTilesAdapter().fetch_layer(layer="image", zoom=14, longitude=longitude,
       ...     latitude=latitude,
       ... )
       >>> VectorTilesAdapter().fetch_layer(layer="sequence", zoom=10, longitude=longitude,
       ...     latitude=latitude,
       ... )
       >>> VectorTilesAdapter().fetch_layer(layer="overview", zoom=3, longitude=longitude,
       ...     latitude=latitude,
       ... )

   .. py:method:: fetch_layer(layer: str, longitude: float, latitude: float, zoom: int = 14) -> dict

      Fetches an image tile layer depending on the coordinates, and the layer selected
      along with the zoom level

      :param layer: Either 'overview', 'sequence', 'image'
      :type layer: str

      :param longitude: The longitude of the coordinates
      :type longitude: float

      :param latitude: The latitude of the coordinates
      :type latitude: float

      :param zoom: The zoom level, [0, 14], inclusive
      :type zoom: int

      :return: A GeoJSON for that specific layer and the specified zoom level
      :rtype: dict


   .. py:method:: fetch_computed_layer(layer: str, zoom: int, longitude: float, latitude: float)

      Same as `fetch_layer`, but gets in return computed tiles only.
      Depends on the layer, zoom level, longitude and the latitude specifications

      :param layer: Either 'overview', 'sequence', 'image'
      :type layer: str

      :param zoom: The zoom level, [0, 14], inclusive
      :type zoom: int

      :param longitude: The longitude of the coordinates
      :type longitude: float

      :param latitude: The latitude of the coordinates
      :type latitude: float

      :return: A GeoJSON for that specific layer and the specified zoom level
      :rtype: dict


   .. py:method:: fetch_features(feature_type: str, zoom: int, longitude: float, latitude: float)

      Fetches specified features from the coordinates with the appropriate zoom level

      :param feature_type: Either `point`, or `tiles`
      :type feature_type: str

      :param zoom: The zoom level
      :type zoom: int

      :param longitude: The longitude of the coordinates
      :type longitude: float

      :param latitude: The latitude of the coordinates
      :type latitude: float

      :return: A GeoJSON for that specific layer and the specified zoom level
      :rtype: dict


   .. py:method:: fetch_layers(coordinates: list[list], layer: str = 'image', zoom: int = 14, is_computed: bool = False, **kwargs) -> zensvi.download.mapillary.models.geojson.GeoJSON

      Fetches multiple vector tiles based on a list of multiple coordinates in a listed format

      :param coordinates: A list of lists of coordinates to get the vector tiles for
      :type coordinates: "list[list]"

      :param layer: Either "overview", "sequence", "image", "traffic_sign", or "map_feature",
          defaults to "image"
      :type layer: str

      :param zoom: the zoom level [0, 14], inclusive. Defaults to 14
      :type zoom: int

      :param is_computed: Will to be fetched layers be computed? Defaults to False
      :type is_computed: bool

      :return: A geojson with merged features from all unique vector tiles
      :rtype: dict


   .. py:method:: fetch_map_features(coordinates: list[list], feature_type: str, zoom: int = 14) -> zensvi.download.mapillary.models.geojson.GeoJSON

      Fetches map features based on a list Polygon object

      :param coordinates: A list of lists of coordinates to get the map features for
      :type coordinates: "list[list]"

      :param feature_type: Either "point", "traffic_signs", defaults to "point"
      :type feature_type: str

      :param zoom: the zoom level [0, 14], inclusive. Defaults to 14
      :type zoom: int

      :return: A geojson with merged features from all unique vector tiles
      :rtype: dict



