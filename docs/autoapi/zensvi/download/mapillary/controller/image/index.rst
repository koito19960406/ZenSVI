:py:mod:`zensvi.download.mapillary.controller.image`
====================================================

.. py:module:: zensvi.download.mapillary.controller.image

.. autoapi-nested-parse::

   mapillary.controllers.image
   ~~~~~~~~~~~~~~~~~~~~~~~~~~~

   This module implements the image filtering and analysis business logic functionalities of the
   Mapillary Python SDK.

   For more information, please check out https://www.mapillary.com/developer/api-documentation/

   - Copyright: (c) 2021 Facebook
   - License: MIT LICENSE



Module Contents
---------------


Functions
~~~~~~~~~

.. autoapisummary::

   zensvi.download.mapillary.controller.image.get_image_close_to_controller
   zensvi.download.mapillary.controller.image.get_image_looking_at_controller
   zensvi.download.mapillary.controller.image.is_image_being_looked_at_controller
   zensvi.download.mapillary.controller.image.get_image_thumbnail_controller
   zensvi.download.mapillary.controller.image.get_images_in_bbox_controller
   zensvi.download.mapillary.controller.image.get_image_from_key_controller
   zensvi.download.mapillary.controller.image.geojson_features_controller
   zensvi.download.mapillary.controller.image.shape_features_controller



.. py:function:: get_image_close_to_controller(longitude: float, latitude: float, kwargs: dict) -> zensvi.download.mapillary.models.geojson.GeoJSON

   Extracting the GeoJSON for the image data near the [longitude, latitude] coordinates

   :param kwargs: The kwargs for the filter
   :type kwargs: dict

   :param longitude: The longitude
   :type longitude: float

   :param latitude: The latitude
   :type latitude: float

   :param kwargs.zoom: The zoom level of the tiles to obtain, defaults to 14
   :type kwargs.zoom: int

   :param kwargs.min_captured_at: The minimum date to filter till
   :type kwargs.min_captured_at: str

   :param kwargs.max_captured_at: The maximum date to filter upto
   :type kwargs.max_captured_at: str

   :param kwargs.image_type: Either 'pano', 'flat' or 'all'
   :type kwargs.image_type: str

   :param kwargs.organization_id: The organization to retrieve the data for
   :type kwargs.organization_id: str

   :param kwargs.radius: The radius that the geometry points will lie in
   :type kwargs.radius: float

   :return: GeoJSON
   :rtype: dict


.. py:function:: get_image_looking_at_controller(at: Union[dict, zensvi.download.mapillary.models.geojson.Coordinates, list], filters: dict) -> zensvi.download.mapillary.models.geojson.GeoJSON

   Checks if the image with coordinates 'at' is looked with the given filters.

   :param filters: Filters to pass the data through
   :type filters: dict

   :param at: The dict of coordinates of the position of the looking at
       coordinates. Format::

           >>> {
           >>>     'lng': 'longitude',
           >>>     'lat': 'latitude'
           >>> }

   :type at: dict

   :param filters.zoom: The zoom level of the tiles to obtain, defaults to 14
   :type filters.zoom: int

   :param filters.min_captured_at: The minimum date to filter till
   :type filters.min_captured_at: str

   :param filters.max_captured_at: The maximum date to filter upto
   :type filters.max_captured_at: str

   :param filters.radius: The radius that the geometry points will lie in
   :type filters.radius: float

   :param filters.image_type: Either 'pano', 'flat' or 'all'
   :type filters.image_type: str

   :param filters.organization_id: The organization to retrieve the data for
   :type filters.organization_id: str

   :return: GeoJSON
   :rtype: dict


.. py:function:: is_image_being_looked_at_controller(at: Union[dict, zensvi.download.mapillary.models.geojson.Coordinates, list], filters: dict) -> bool

   Checks if the image with coordinates 'at' is looked with the given filters.

   :param at: The dict of coordinates of the position of the looking at coordinates.

       Format::

           >>> at_dict = {
           ...     'lng': 'longitude',
           ...     'lat': 'latitude'
           ... }
           >>> at_list = [12.954940544167, 48.0537894275]
           >>> from mapillary.models.geojson import Coordinates
           >>> at_coord: Coordinates = Coordinates(lng=12.954940544167, lat=48.0537894275)

   :type at: Union[dict, mapillary.models.geojson.Coordinates, list]

   :param filters.zoom: The zoom level of the tiles to obtain, defaults to 14
   :type filter.zoom: int

   :param filters.min_captured_at: The minimum date to filter till
   :type filters.min_captured_at: str

   :param filters.max_captured_at: The maximum date to filter upto
   :type filters.max_captured_at: str

   :param filters.radius: The radius that the geometry points will lie in
   :type filters.radius: float

   :param filters.image_type: Either 'pano', 'flat' or 'all'
   :type filters.image_type: str

   :param filters.organization_id: The organization to retrieve the data for
   :type filters.organization_id: str

   :return: True if the image is looked at by the given looker and at coordinates, False otherwise
   :rtype: bool


.. py:function:: get_image_thumbnail_controller(image_id: str, resolution: int) -> str

   This controller holds the business logic for retrieving
   an image thumbnail with a specific resolution (256, 1024, or 2048)
   using an image ID/key

   :param image_id: Image key as the argument
   :type image_id: str

   :param resolution: Option for the thumbnail size, with available resolutions:
       256, 1024, and 2048
   :type resolution: int

   :return: A URL for the thumbnail
   :rtype: str


.. py:function:: get_images_in_bbox_controller(bounding_box: dict, layer: str, zoom: int, filters: dict) -> str

   For getting a complete list of images that lie within a bounding box,
   that can be filtered via the filters argument

   :param bounding_box: A bounding box representation
       Example::

           >>> {
           ...     'west': 'BOUNDARY_FROM_WEST',
           ...     'south': 'BOUNDARY_FROM_SOUTH',
           ...     'east': 'BOUNDARY_FROM_EAST',
           ...     'north': 'BOUNDARY_FROM_NORTH'
           ... }

   :type bounding_box: dict

   :param zoom: The zoom level
   :param zoom: int

   :param layer: Either 'image', 'sequence', 'overview'
   :type layer: str

   :param filters: Filters to pass the data through
   :type filters: dict

   :param filters.max_captured_at: The max date that can be filtered upto
   :type filters.max_captured_at: str

   :param filters.min_captured_at: The min date that can be filtered from
   :type filters.min_captured_at: str

   :param filters.image_type: Either 'pano', 'flat' or 'all'
   :type filters.image_type: str

   :param filters.compass_angle:
   :type filters.compass_angle: float

   :param filters.organization_id:
   :type filters.organization_id: int

   :param filters.sequence_id:
   :type filters.sequence_id: str

   :raises InvalidKwargError: Raised when a function is called with the invalid keyword argument(s)
       that do not belong to the requested API end call

   :return: GeoJSON
   :rtype: str

   Reference,

   - https://www.mapillary.com/developer/api-documentation/#coverage-tiles


.. py:function:: get_image_from_key_controller(key: int, fields: list) -> str

   A controller for getting properties of a certain image given the image key and
   the list of fields/properties to be returned

   :param key: The image key
   :type key: int

   :param fields: The list of fields to be returned
   :type fields: list

   :return: The requested image properties in GeoJSON format
   :rtype: str


.. py:function:: geojson_features_controller(geojson: dict, is_image: bool = True, filters: dict = None, **kwargs) -> zensvi.download.mapillary.models.geojson.GeoJSON

   For extracting images that lie within a GeoJSON and merges the results of the found
   GeoJSON(s) into a single object - by merging all the features into one feature list.

   :param geojson: The geojson to act as the query extent
   :type geojson: dict

   :param is_image: Is the feature extraction for images? True for images, False for map features
       Defaults to True
   :type is_image: bool

   :param filters: Different filters that may be applied to the output, defaults to {}
   :type filters: dict (kwargs)

   :param filters.zoom: The zoom level to obtain vector tiles for, defaults to 14
   :type filters.zoom: int

   :param filters.max_captured_at: The max date. Format from 'YYYY', to 'YYYY-MM-DDTHH:MM:SS'
   :type filters.max_captured_at: str

   :param filters.min_captured_at: The min date. Format from 'YYYY', to 'YYYY-MM-DDTHH:MM:SS'
   :type filters.min_captured_at: str

   :param filters.image_type: The tile image_type to be obtained, either as 'flat', 'pano'
       (panoramic), or 'all'. See https://www.mapillary.com/developer/api-documentation/ under
       'image_type Tiles' for more information
   :type filters.image_type: str

   :param filters.compass_angle: The compass angle of the image
   :type filters.compass_angle: int

   :param filters.sequence_id: ID of the sequence this image belongs to
   :type filters.sequence_id: str

   :param filters.organization_id: ID of the organization this image belongs to. It can be absent
   :type filters.organization_id: str

   :param filters.layer: The specified image layer, either 'overview', 'sequence', 'image'
       if is_image is True, defaults to 'image'
   :type filters.layer: str

   :param filters.feature_type: The specified map features, either 'point' or 'traffic_signs'
       if is_image is False, defaults to 'point'
   :type filters.feature_type: str

   :raises InvalidKwargError: Raised when a function is called with the invalid keyword argument(s)
       that do not belong to the requested API end call

   :return: A feature collection as a GeoJSON
   :rtype: dict


.. py:function:: shape_features_controller(shape, is_image: bool = True, filters: dict = None) -> zensvi.download.mapillary.models.geojson.GeoJSON

   For extracting images that lie within a shape, merging the results of the found features
   into a single object - by merging all the features into one list in a feature collection.

   The shape format is as follows::

       >>> {
       ...     "type": "FeatureCollection",
       ...     "features": [
       ...         {
       ...             "type": "Feature",
       ...             "properties": {},
       ...             "geometry": {
       ...                 "type": "Polygon",
       ...                 "coordinates": [
       ...                     [
       ...                        [
       ...                              7.2564697265625,
       ...                             43.69716905314008
       ...                         ],
       ...                         ...
       ...                     ]
       ...                 ]
       ...             }
       ...         }
       ...     ]
       ... }

   :param shape: A shape that describes features, formatted as a geojson
   :type shape: dict

   :param is_image: Is the feature extraction for images? True for images, False for map features
       Defaults to True
   :type is_image: bool

   :param filters: Different filters that may be applied to the output, defaults to {}
   :type filters: dict (kwargs)

   :param filters.max_captured_at: The max date. Format from 'YYYY', to 'YYYY-MM-DDTHH:MM:SS'
   :type filters.max_captured_at: str

   :param filters.min_captured_at: The min date. Format from 'YYYY', to 'YYYY-MM-DDTHH:MM:SS'
   :type filters.min_captured_at: str

   :param filters.image_type: The tile image_type to be obtained, either as 'flat', 'pano'
       (panoramic), or 'all'. See https://www.mapillary.com/developer/api-documentation/ under
       'image_type Tiles' for more information
   :type filters.image_type: str

   :param filters.compass_angle: The compass angle of the image
   :type filters.compass_angle: int

   :param filters.sequence_id: ID of the sequence this image belongs to
   :type filters.sequence_id: str

   :param filters.organization_id: ID of the organization this image belongs to. It can be absent
   :type filters.organization_id: str

   :param filters.layer: The specified image layer, either 'overview', 'sequence', 'image'
       if is_image is True, defaults to 'image'
   :type filters.layer: str

   :param filters.feature_type: The specified map features, either 'point' or 'traffic_signs'
       if is_image is False, defaults to 'point'
   :type filters.feature_type: str

   :raises InvalidKwargError: Raised when a function is called with the invalid keyword argument(s)
       that do not belong to the requested API end call

   :return: A feature collection as a GeoJSON
   :rtype: dict


