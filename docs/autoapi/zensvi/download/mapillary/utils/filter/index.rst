:py:mod:`zensvi.download.mapillary.utils.filter`
================================================

.. py:module:: zensvi.download.mapillary.utils.filter

.. autoapi-nested-parse::

   mapillary.utils.filter
   ======================

   This module contains the filter utilies for high level filtering logic

   - Copyright: (c) 2021 Facebook
   - License: MIT LICENSE



Module Contents
---------------


Functions
~~~~~~~~~

.. autoapisummary::

   zensvi.download.mapillary.utils.filter.pipeline_component
   zensvi.download.mapillary.utils.filter.pipeline
   zensvi.download.mapillary.utils.filter.max_captured_at
   zensvi.download.mapillary.utils.filter.min_captured_at
   zensvi.download.mapillary.utils.filter.features_in_bounding_box
   zensvi.download.mapillary.utils.filter.filter_values
   zensvi.download.mapillary.utils.filter.existed_at
   zensvi.download.mapillary.utils.filter.existed_before
   zensvi.download.mapillary.utils.filter.haversine_dist
   zensvi.download.mapillary.utils.filter.image_type
   zensvi.download.mapillary.utils.filter.organization_id
   zensvi.download.mapillary.utils.filter.sequence_id
   zensvi.download.mapillary.utils.filter.compass_angle
   zensvi.download.mapillary.utils.filter.is_looking_at
   zensvi.download.mapillary.utils.filter.by_look_at_feature
   zensvi.download.mapillary.utils.filter.hits_by_look_at
   zensvi.download.mapillary.utils.filter.in_shape
   zensvi.download.mapillary.utils.filter.pipeline



Attributes
~~~~~~~~~~

.. autoapisummary::

   zensvi.download.mapillary.utils.filter.logger


.. py:data:: logger

   

.. py:function:: pipeline_component(func, data: list, exception_message: str, args: list) -> list

   A pipeline component which is respnonsible for sending functional arguments over
   to the selected target function - throwing a warning in case of an exception

   :param func: The filter to apply
   :type func: function

   :param data: The list of features to filter
   :type data: list

   :param exception_message: The exception message to print
   :type exception_message: str

   :param args: Arguments
   :type args: list

   :return: The filtered feature list
   :rtype: list

   Usage::

       >>> # internally used in mapillary.utils.pipeline


.. py:function:: pipeline(data: dict, components: list) -> list

   A pipeline component that helps with making filtering easier. It provides
   access to different filtering mechanism by simplying letting users
   pass in what filter they want to apply, and the arguments for that filter

   :param data: The GeoJSON to be filtered
   :type data: dict

   :param components: The list of filters to apply
   :type components: list

   :return: The filtered feature list
   :rtype: list

   Usage::

       >>> # assume variables 'data', 'kwargs'
       >>> pipeline(
       ...     data=data,
       ...     components=[
       ...         {"filter": "image_type", "tile": kwargs["image_type"]}
       ...         if "image_type" in kwargs
       ...         else {},
       ...         {"filter": "organization_id", "organization_ids": kwargs["org_id"]}
       ...         if "org_id" in kwargs
       ...         else {},
       ...         {
       ...             "filter": "haversine_dist",
       ...             "radius": kwargs["radius"],
       ...             "coords": [longitude, latitude],
       ...         }
       ...         if "radius" in kwargs
       ...         else 1000
       ...     ]
       ... )


.. py:function:: max_captured_at(data: list, max_timestamp: str) -> list

   Selects only the feature items that are less
   than the max_timestamp

   :param data: The feature list
   :type data: list

   :param max_timestamp: The UNIX timestamp as the max time
   :type max_timestamp: str

   :return: Filtered feature list
   :rtype: list

   Usage::

       >>> max_captured_at([{'type': 'Feature', 'geometry':
       ... {'type': 'Point', 'coordinates': [30.98594605922699, 30.003757307208872]}, 'properties':
       ... { ... }, ...}], '2020-05-23')


.. py:function:: min_captured_at(data: list, min_timestamp: str) -> list

   Selects only the feature items that are less
   than the min_timestamp

   :param data: The feature list
   :type data: list

   :param min_timestamp: The UNIX timestamp as the max time
   :type min_timestamp: str

   :return: Filtered feature list
   :rtype: list

   Usage::

       >>> max_captured_at([{'type': 'Feature', 'geometry':
       ... {'type': 'Point', 'coordinates': [30.98594605922699, 30.003757307208872]}, 'properties':
       ... { ... }, ...}], '2020-05-23')


.. py:function:: features_in_bounding_box(data: list, bbox: dict) -> list

   Filter for extracting features only in a bounding box

   :param data: the features list to be checked
   :type data: list

   :param bbox: Bounding box coordinates

       Example::
           >>> {
           ...     'west': 'BOUNDARY_FROM_WEST',
           ...     'south': 'BOUNDARY_FROM_SOUTH',
           ...     'east': 'BOUNDARY_FROM_EAST',
           ...     'north': 'BOUNDARY_FROM_NORTH'
           ... }

   :type bbox: dict

   :return: Features that only exist within the bounding box selected for the given features list
       provided in the BBox functon
   :rtype: list


.. py:function:: filter_values(data: list, values: list, property: str = 'value') -> list

   Filter the features based on the existence of a specified value
   in one of the property.

   *TODO*: Need documentation that lists the 'values', specifically, it refers to 'value'
   *TODO*: under 'Detection', and 'Map feature', related to issue #65

   :param data: The data to be filtered
   :type data: dict

   :param values: A list of values to filter by
   :type values: list

   :param property: The specific parameter to look into
   :type property: str

   :return: A feature list
   :rtype: dict


.. py:function:: existed_at(data: list, existed_at: str) -> list

   Whether the first_seen_at properly existed after a specified time period

   :param data: The feature list
   :type data: list

   :param existed_at: The UNIX timestamp
   :type existed_at: str

   :return: The feature list
   :rtype: list


.. py:function:: existed_before(data: list, existed_before: str) -> list

   Whether the first_seen_at properly existed before a specified time period

   :param data: The feature list
   :type data: list

   :param existed_before: The UNIX timestamp
   :type existed_before: str

   :return: A feature list
   :rtype: list


.. py:function:: haversine_dist(data: dict, radius: float, coords: list, unit: str = 'm') -> list

   Returns features that are only in the radius specified using the Haversine distance, from
   the haversine package

   :param data: The data to be filtered
   :type data: dict

   :param radius: Radius for coordinates to fall into
   :type radius: float

   :param coords: The input coordinates (long, lat)
   :type coords: list

   :param unit: Either 'ft', 'km', 'm', 'mi', 'nmi', see here https://pypi.org/project/haversine/
   :type unit: str

   :return: A feature list
   :rtype: list


.. py:function:: image_type(data: list, image_type: str) -> list

   The parameter might be 'all' (both is_pano == true and false), 'pano' (is_pano == true only),
   or 'flat' (is_pano == false only)

   :param data: The data to be filtered
   :type data: list

   :param image_type: Either 'pano' (True), 'flat' (False), or 'all' (None)
   :type image_type: str

   :return: A feature list
   :rtype: list


.. py:function:: organization_id(data: list, organization_ids: list) -> list

   Select only features that contain the specific organization_id

   :param data: The data to be filtered
   :type data: dict

   :param organization_ids: The oragnization id(s) to filter through
   :type organization_ids: list

   :return: A feature list
   :rtype: dict


.. py:function:: sequence_id(data: list, ids: list) -> list

   Filter out images that do not have the sequence_id in the list of ids

   :param data: The data to be filtered
   :type data: list

   :param ids: The sequence id(s) to filter through
   :type ids: list

   :return: A feature list
   :rtype: list


.. py:function:: compass_angle(data: list, angles: tuple = (0.0, 360.0)) -> list

   Filter out images that do not lie within compass angle range

   :param data: The data to be filtered
   :type data: list

   :param angles: The compass angle range to filter through
   :type angle: tuple of floats

   :return: A feature list
   :rtype: list


.. py:function:: is_looking_at(image_feature: geojson.Feature, look_at_feature: geojson.Feature) -> bool

   Return whether the image_feature is looking at the look_at_feature

   :param image_feature: The feature set of the image
   :type image_feature: dict

   :param look_at_feature: The feature that is being looked at
   :type look_at_feature: dict

   :return: Whether the diff is greater than 310, or less than 50
   :rtype: bool


.. py:function:: by_look_at_feature(image: dict, look_at_feature: geojson.Feature) -> bool

   Filter through the given image features and return only features with the look_at_feature

   :param image: The feature dictionary
   :type image: dict

   :param look_at_feature: Feature description
   :type look_at_feature: A WGS84 GIS feature, TurfPy

   :return: Whether the given feature is looking at the `look_at_features`
   :rtype: bool


.. py:function:: hits_by_look_at(data: list, at: dict) -> list

   Whether the given data have any feature that look at the `at` coordinates

   :param data: List of features with an Image entity
   :type data: list

   :param at: The lng and lat coordinates

       Example::

           >>> {
           ...     'lng': 'longitude',
           ...     'lat': 'latitude'
           ... }

   :type at: dict

   :return: Filtered results of features only looking at `at`
   :rtype: list


.. py:function:: in_shape(data: list, boundary) -> list

   Whether the given feature list lies within the shape

   :param data: A feature list to be filtered
   :type data: list

   :param boundary: Shapely helper for determining existence of point within a boundary
   :type boundary:

   :return: A feature list
   :rtype: list


.. py:function:: pipeline(data: dict, components: list, **kwargs) -> list


