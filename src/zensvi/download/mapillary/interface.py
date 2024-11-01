# Copyright (c) Facebook, Inc. and its affiliates. (http://www.facebook.com)
# -*- coding: utf-8 -*-
"""mapillary.interface
~~~~~~~~~~~~~~~~~~~

This module implements the basic functionalities of the Mapillary Python SDK, a Python
implementation of the Mapillary API v4. For more information, please check out
https://www.mapillary.com/developer/api-documentation/

- Copyright: (c) 2021 Facebook
- License: MIT LICENSE
"""
import json
import os
from typing import Union

import requests

import zensvi.download.mapillary.controller.detection as detection
import zensvi.download.mapillary.controller.feature as feature
import zensvi.download.mapillary.controller.image as image
import zensvi.download.mapillary.controller.save as save
from zensvi.download.mapillary.models.config import Config
from zensvi.download.mapillary.models.exceptions import InvalidOptionError
from zensvi.download.mapillary.models.geojson import Coordinates, GeoJSON
from zensvi.download.mapillary.utils.auth import auth, set_token


def configure_mapillary_settings(**kwargs):
    """A function allowing the user to configure the Mapillary settings for the session.
    Takes no arguments and sets a global variable used by other functions making API requests.
    For more information what the details of authentication, please check out the blog post at Mapillary.
    https://blog.mapillary.com/update/2021/06/23/getting-started-with-the-new-mapillary-api-v4.html

    Args:
      kwargs: Configuration options
      **kwargs:

    Returns:
      : None
      Usage: :

    >>> import mapillary as mly
        >>> mly.interface.configure_mapillary_settings()
        >>> mly.interface.configure_mapillary_settings(use_strict=True)
    """
    return Config(kwargs)


def set_access_token(token: str):
    """A function allowing the user to set an access token for the session.
    Takes token as an argument and sets a global variable used by other functions making API requests.
    For more information what the details of authentication, please check out the blog post at Mapillary.
    https://blog.mapillary.com/update/2021/06/23/getting-started-with-the-new-mapillary-api-v4.html

    Args:
      token: The access token to set
      token: str:
      token: str:

    Returns:
      : None
      Usage: :

    >>> import mapillary as mly
        >>> mly.interface.set_access_token('CLIENT_TOKEN_HERE')
    """
    return set_token(token)


@auth()
def get_image_close_to(latitude=-122.1504711, longitude=37.485073, **kwargs):
    """Function that takes a longitude, latitude as argument and outputs the near
    images. This makes an API call with the token set in set_access_token and returns a
    JSON object.

    Args:
      longitude(float or double, optional): The longitude (Default value = 37.485073)
      latitude(float or double, optional): The latitude (Default value = -122.1504711)
      kwargs.fields(list): A list of options, either as ['all'], or a
    list of fields. See https://www.mapillary.com/developer/api-
    documentation/, under 'Fields' for more insight.
      kwargs.zoom(int): The zoom level of the tiles to obtain,
    defaults to 14
      kwargs.radius(float or int or double): The radius of the images
    obtained from a center center
      kwargs.image_type(str): The tile image_type to be obtained,
    either as 'flat', 'pano' (panoramic), or 'both'. See
    https://www.mapillary.com/developer/api-documentation/ under
    'image_type Tiles' for more information
      kwargs.min_captured_at(str): The min date. Format from 'YYYY',
    to 'YYYY-MM-DDTHH:MM:SS'
      kwargs.max_captured_at(str): The max date. Format from 'YYYY',
    to 'YYYY-MM-DDTHH:MM:SS'
      kwargs.org_id(int): The organization id, ID of the organization
    this image (or sets of images) belong to. It can be absent.
    Thus, default is -1 (None)
      **kwargs:

    Returns:
      dict: GeoJSON
      Usage: :

    >>> import mapillary as mly
        >>> mly.interface.set_access_token('CLIENT_TOKEN_HERE')
        >>> mly.interface.get_image_close_to(longitude=31, latitude=30)
        ... {'type': 'FeatureCollection', 'features': [{'type': 'Feature',
        'geometry': {'type': 'Point', 'coordinates': [30.9912246465683,
        29.99794091267283]}, 'properties': {'captured_at': 1621008070596,
        'compass_angle': 322.56726074219, 'id': 499412381300321, 'is_pano':
        False, 'sequence_id': '94afmyyhq85xd9bi8p44ve'}} ...
    """
    return image.get_image_close_to_controller(
        latitude=latitude,
        longitude=longitude,
        kwargs=kwargs,
    )


@auth()
def get_image_looking_at(
    at: dict,
    **filters: dict,
) -> GeoJSON:
    """Function that takes two sets of latitude and longitude, where the 2nd set is the
    "looking at" location from 1st set's perspective argument and outputs the near
    images. This makes an API call with the token set in set_access_token and returns a
    JSON object.

    Args:
      at(dict): The coordinate sets to where a certain point is being
    looked at
    Format::
      at: dict:
      **filters: dict:
      at: dict:
      **filters: dict:
      at: dict:
      **filters: dict:

    Returns:
      GeoJSON: The GeoJSON response containing relevant features
      Usage: :

    >>> {
                ...     'lng': 'longitude',
                ...     'lat': 'latitude'
                ... }
        filters.min_captured_at (str): The minimum date to filter till
        filters.max_captured_at (str): The maximum date to filter upto
        filters.radius (float): The radius that the geometry points will
            lie in
        filters.image_type (str): Either 'pano', 'flat' or 'all'
        filters.organization_id (str): The organization to retrieve the
            data for

        >>> import mapillary as mly
        >>> mly.interface.set_access_token('MLY|XXX')
        >>> data = mly.interface.get_image_looking_at(
        ...         at={
        ...             'lng': 12.955075073889,
        ...             'lat': 48.053805939722,
        ...         },
        ...         radius = 5000,
        ...     )
        >>> data
        ... {'type': 'FeatureCollection', 'features': [{'type': 'Feature', 'geometry': {'type':
        ... 'Point', 'coordinates': [12.954479455947876, 48.05091893670834]}, 'properties':
        ... {'captured_at': 1612606959408, 'compass_angle': 21.201110839844, 'id': 1199705400459580,
        ... 'is_pano': False, 'sequence_id': 'qrrqtke4a6vtygyc7w8rzc'}}, ... }
    """
    return image.get_image_looking_at_controller(
        at=at,
        filters=filters,
    )


@auth()
def is_image_being_looked_at(
    at: Union[dict, Coordinates, list],
    **filters: dict,
) -> bool:
    """Function that two sets of coordinates and returns whether the image  with
    coordinates of "at" is looked at or not by the image with coordinates of "looker".

    Args:
      at(Union[dict): The coordinate sets to where a certain point is being looked
    at
    Format::
      at: Union[dict:
      Coordinates:
      list]:
      **filters: dict:
      at: Union[dict:
      **filters: dict:
      at: Union[dict:
      **filters: dict:

    Returns:
      bool: True if the image is looked at, False otherwise
      Usage: :

    >>> at_dict = {
                ...     'lng': 'longitude',
                ...     'lat': 'latitude'
                ... }
                >>> at_list = [12.954940544167, 48.0537894275]
                >>> from mapillary.models.geojson import Coordinates
                >>> at_coord: Coordinates = Coordinates(lng=12.954940544167, lat=48.0537894275)

        >>> import mapillary as mly
        >>> mly.interface.set_access_token('MLY|XXX')
        >>> mly.interface.is_image_being_looked_at(
        ...         at={
        ...             'lng': 12.955075073889,
        ...             'lat': 48.053805939722,
        ...         },
        ...         radius=50,
        ...     )
        ... True
        >>> # OR
        >>> from mapillary.models.geojson import Coordinates
        >>> mly.interface.is_image_looked_at(
        ...         at=Coordinates(lng=11.954940544167, lat=46.0537894275),
        ...         radius=50,
        ...     )
        ... True
    """
    return image.is_image_being_looked_at_controller(at=at, filters=filters)


@auth()
def get_detections_with_image_id(image_id: int, fields: list = []):
    """Extracting all the detections within an image using an image key.

    Args:
      image_id(int): The image key as the argument
      fields(list): The fields possible for the detection endpoint.
    Please see https://www.mapillary.com/developer/api-
    documentation for more information
      image_id: int:
      fields: list:  (Default value = [])
      image_id: int:
      fields: list:  (Default value = [])
      image_id: int:
      fields: list:  (Default value = [])

    Returns:
      dict: The GeoJSON in response
      Usage: :

    >>> import mapillary as mly
        >>> mly.interface.set_access_token('CLIENT_TOKEN_HERE')
        >>> mly.interface.get_detections_with_image_id(image_id=1933525276802129)
        ... {"data":[{"created_at":"2021-05-20T17:49:01+0000","geometry":
        ... "GjUKBm1weS1vchIVEgIAABgDIg0JhiekKBoqAABKKQAPGgR0eXBlIgkKB3BvbHlnb24ogCB4AQ==","image"
        ... :{"geometry":{"type":"Point","coordinates":[-97.743279722222,30.270651388889]},"id":
        ... "1933525276802129"},"value":"regulatory--no-parking--g2","id":"1942105415944115"},
        ... {"created_at":"2021-05-20T18:40:21+0000","geometry":
        ... "GjYKBm1weS1vchIWEgIAABgDIg4J7DjqHxpWAADiAVUADxoEdHlwZSIJCgdwb2x5Z29uKIAgeAE=",
        ... "image":{"geometry":{"type":"Point","coordinates":[-97.743279722222,30.270651388889]},
        ... "id":"1933525276802129"},"value":"information--parking--g1","id":"1942144785940178"},
        ... , ...}
    """
    return detection.get_image_detections_controller(
        image_id=image_id,
        fields=fields,
    )


@auth()
def get_detections_with_map_feature_id(map_feature_id: str, fields: list = None) -> GeoJSON:
    """Extracting all detections made for a map feature key.

    Args:
      map_feature_id(int): A map feature key as the argument
      fields(list): The fields possible for the detection endpoint.
    Please see https://www.mapillary.com/developer/api-
    documentation for more information
      map_feature_id: str:
      fields: list:  (Default value = None)
      map_feature_id: str:
      fields: list:  (Default value = None)
      map_feature_id: str:
      fields: list:  (Default value = None)

    Returns:
      GeoJSON: The GeoJSON in response
      Usage: :

    >>> import mapillary as mly
        >>> mly.interface.set_access_token('MLY|XXX')
        >>> mly.interface.get_detections_with_map_feature_id(map_feature_id='1933525276802129')
        ...     File "/home/saif/MLH/mapillary-python-sdk/mapillary/controller/rules/verify.py",
        ...         line 227, in valid_id
        ...             raise InvalidOptionError(
        ... mly.models.exceptions.InvalidOptionError: InvalidOptionError: Given id value,
        ...     "Id: 1933525276802129, image: False" while possible id options, [Id is image_id
        ...     AND image is True, key is map_feature_id ANDimage is False]
    """
    return detection.get_map_feature_detections_controller(
        map_feature_id=map_feature_id,
        fields=fields,
    )


@auth()
def image_thumbnail(image_id: str, resolution: int = 1024, additional_fields: list = None) -> str:
    """Gets the thumbnails of images from the API.

    Args:
      image_id: Image key as the argument
      resolution: Option for the thumbnail size, with available
    resolutions: 256, 1024, and 2048
      image_id: str:
      resolution: int:  (Default value = 1024)
      image_id: str:
      resolution: int:  (Default value = 1024)
      additional_fields: list:  (Default value = None)
      image_id: str:
      resolution: int:  (Default value = 1024)
      additional_fields: list:  (Default value = None)

    Returns:
      str: A URL for the thumbnail
      Usage: :

    >>> import mapillary as mly
        >>> mly.interface.set_access_token('MLY|XXX')
        >>> mly.interface.image_thumbnail(
        ...     image_id='IMAGE_ID_HERE',
        ...     resolution=1024
        ... )
    """
    return image.get_image_thumbnail_controller(image_id, resolution, additional_fields)


@auth()
def images_in_bbox(bbox: dict, **filters) -> str:
    """Gets a complete list of images with custom filter within a BBox.

    Args:
      bbox(dict): Bounding box coordinates
    Format::
    Example filters::
    - max_captured_at
    - min_captured_at
    - image_type: pano, flat, or all
    - compass_angle
    - sequence_id
    - organization_id
      bbox: dict:
      **filters:
      bbox: dict:
      bbox: dict:

    Returns:
      str: Output is a GeoJSON string that represents all the within a
      str: Output is a GeoJSON string that represents all the within a
      str: Output is a GeoJSON string that represents all the within a
      str: Output is a GeoJSON string that represents all the within a
      bbox after passing given filters
      Usage: :

    >>> {
                ...     'west': 'BOUNDARY_FROM_WEST',
                ...     'south': 'BOUNDARY_FROM_SOUTH',
                ...     'east': 'BOUNDARY_FROM_EAST',
                ...     'north': 'BOUNDARY_FROM_NORTH'
                ... }
        **filters (dict): Different filters that may be applied to the
            output

        >>> import mapillary as mly
        >>> mly.interface.set_access_token('MLY|XXX')
        >>> mly.interface.images_in_bbox(
        ...     bbox={
        ...         'west': 'BOUNDARY_FROM_WEST',
        ...         'south': 'BOUNDARY_FROM_SOUTH',
        ...         'east': 'BOUNDARY_FROM_EAST',
        ...         'north': 'BOUNDARY_FROM_NORTH'
        ...     },
        ...     max_captured_at='YYYY-MM-DD HH:MM:SS',
        ...     min_captured_at='YYYY-MM-DD HH:MM:SS',
        ...     image_type='pano',
        ...     compass_angle=(0, 360),
        ...     sequence_id='SEQUENCE_ID',
        ...     organization_id='ORG_ID'
        ... )
    """
    return image.get_images_in_bbox_controller(bounding_box=bbox, layer="image", zoom=14, filters=filters)


@auth()
def sequences_in_bbox(bbox: dict, **filters) -> str:
    """Gets a complete list of all sequences of images that satisfy given filters within
    a BBox.

    Args:
      bbox(dict): Bounding box coordinates
    Example::
    Example filters::
    - max_captured_at
    - min_captured_at
    - image_type: pano, flat, or all
    - org_id
      bbox: dict:
      **filters:
      bbox: dict:
      bbox: dict:

    Returns:
      str: Output is a GeoJSON string that contains all the filtered
      str: Output is a GeoJSON string that contains all the filtered
      str: Output is a GeoJSON string that contains all the filtered
      str: Output is a GeoJSON string that contains all the filtered
      sequences within a bbox. Sequences would NOT be cut at BBox
      str: Output is a GeoJSON string that contains all the filtered
      str: Output is a GeoJSON string that contains all the filtered
      str: Output is a GeoJSON string that contains all the filtered
      sequences within a bbox. Sequences would NOT be cut at BBox
      str: Output is a GeoJSON string that contains all the filtered
      str: Output is a GeoJSON string that contains all the filtered
      sequences within a bbox. Sequences would NOT be cut at BBox
      str: Output is a GeoJSON string that contains all the filtered
      sequences within a bbox. Sequences would NOT be cut at BBox
      boundary, would select all sequences which are partially or
      str: Output is a GeoJSON string that contains all the filtered
      str: Output is a GeoJSON string that contains all the filtered
      str: Output is a GeoJSON string that contains all the filtered
      sequences within a bbox. Sequences would NOT be cut at BBox
      str: Output is a GeoJSON string that contains all the filtered
      str: Output is a GeoJSON string that contains all the filtered
      sequences within a bbox. Sequences would NOT be cut at BBox
      str: Output is a GeoJSON string that contains all the filtered
      sequences within a bbox. Sequences would NOT be cut at BBox
      boundary, would select all sequences which are partially or
      str: Output is a GeoJSON string that contains all the filtered
      str: Output is a GeoJSON string that contains all the filtered
      sequences within a bbox. Sequences would NOT be cut at BBox
      str: Output is a GeoJSON string that contains all the filtered
      sequences within a bbox. Sequences would NOT be cut at BBox
      boundary, would select all sequences which are partially or
      str: Output is a GeoJSON string that contains all the filtered
      sequences within a bbox. Sequences would NOT be cut at BBox
      boundary, would select all sequences which are partially or
      entirely in BBox
      Usage: :

    >>> _ = {
                ...     'west': 'BOUNDARY_FROM_WEST',
                ...     'south': 'BOUNDARY_FROM_SOUTH',
                ...     'east': 'BOUNDARY_FROM_EAST',
                ...     'north': 'BOUNDARY_FROM_NORTH'
                ... }
        **filters (dict): Different filters that may be applied to the
            output

        >>> import mapillary as mly
        >>> mly.interface.set_access_token('MLY|XXX')
        >>> mly.interface.sequences_in_bbox(
        ...     bbox={
        ...         'west': 'BOUNDARY_FROM_WEST',
        ...         'south': 'BOUNDARY_FROM_SOUTH',
        ...         'east': 'BOUNDARY_FROM_EAST',
        ...         'north': 'BOUNDARY_FROM_NORTH'
        ...     },
        ...     max_captured_at='YYYY-MM-DD HH:MM:SS',
        ...     min_captured_at='YYYY-MM-DD HH:MM:SS',
        ...     image_type='pano',
        ...     org_id='ORG_ID'
        ... )
    """
    return image.get_images_in_bbox_controller(bounding_box=bbox, layer="sequence", zoom=14, filters=filters)


@auth()
def map_feature_points_in_bbox(bbox: dict, filter_values: list = None, **filters: dict) -> str:
    """Extracts map feature points within a bounding box (bbox)

    Args:
      bbox(dict): bbox coordinates as the argument
    Example::
    Example::
    Chronological filters,
    - *existed_at*: checks if a feature existed after a certain date depending on the time
    it was first seen at.
    - *existed_before*: filters out the features that existed after a given date
      bbox: dict:
      filter_values: list:  (Default value = None)
      **filters: dict:
      bbox: dict:
      filter_values: list:  (Default value = None)
      **filters: dict:
      bbox: dict:
      filter_values: list:  (Default value = None)
      **filters: dict:

    Returns:
      dict: GeoJSON Object
      Usage: :

    >>> _ = {
                ...     'west': 'BOUNDARY_FROM_WEST',
                ...     'south': 'BOUNDARY_FROM_SOUTH',
                ...     'east': 'BOUNDARY_FROM_EAST',
                ...     'north': 'BOUNDARY_FROM_NORTH'
                ... }
        filter_values (list): a list of filter values supported by the
            API

                >>> _ = ['object--support--utility-pole', 'object--street-light']
        **filters (dict): kwarg filters to be applied on the resulted
            GeoJSON

        >>> import mapillary as mly
        >>> mly.interface.set_access_token('MLY|XXX')
        >>> mly.interface.map_feature_points_in_bbox(
        ...     bbox={
        ...         'west': 'BOUNDARY_FROM_WEST',
        ...         'south': 'BOUNDARY_FROM_SOUTH',
        ...         'east': 'BOUNDARY_FROM_EAST',
        ...         'north': 'BOUNDARY_FROM_NORTH'
        ...     },
        ...     filter_values=['object--support--utility-pole', 'object--street-light'],
        ...     existed_at='YYYY-MM-DD HH:MM:SS',
        ...     existed_before='YYYY-MM-DD HH:MM:SS'
        ... )
    """
    return feature.get_map_features_in_bbox_controller(
        bbox=bbox, filters=filters, filter_values=filter_values, layer="points"
    )


@auth()
def traffic_signs_in_bbox(bbox: dict, filter_values: list = None, **filters: dict) -> str:
    """Extracts traffic signs within a bounding box (bbox)

    Args:
      bbox(dict): bbox coordinates as the argument
    Example::
    Example::
    Chronological filters,
    - *existed_at*: checks if a feature existed after a certain date depending on the time
    it was first seen at.
    - *existed_before*: filters out the features that existed after a given date
      bbox: dict:
      filter_values: list:  (Default value = None)
      **filters: dict:
      bbox: dict:
      filter_values: list:  (Default value = None)
      **filters: dict:
      bbox: dict:
      filter_values: list:  (Default value = None)
      **filters: dict:

    Returns:
      dict: GeoJSON Object
      Usage: :

    >>> {
                ...     'west': 'BOUNDARY_FROM_WEST',
                ...     'south': 'BOUNDARY_FROM_SOUTH',
                ...     'east': 'BOUNDARY_FROM_EAST',
                ...     'north': 'BOUNDARY_FROM_NORTH'
                ... }
        filter_values (list): a list of filter values supported by the
            API,

                >>> ['regulatory--advisory-maximum-speed-limit--g1', 'regulatory--atvs-permitted--g1']
        **filters (dict): kwarg filters to be applied on the resulted
            GeoJSON

        >>> import mapillary as mly
        >>> mly.interface.set_access_token('MLY|XXX')
        >>> mly.interface.traffic_signs_in_bbox(
        ...    bbox={
        ...         'west': 'BOUNDARY_FROM_WEST',
        ...         'south': 'BOUNDARY_FROM_SOUTH',
        ...         'east': 'BOUNDARY_FROM_EAST',
        ...         'north': 'BOUNDARY_FROM_NORTH'
        ...    },
        ...    filter_values=[
        ...        'regulatory--advisory-maximum-speed-limit--g1',
        ...        'regulatory--atvs-permitted--g1'
        ...    ],
        ...    existed_at='YYYY-MM-DD HH:MM:SS',
        ...    existed_before='YYYY-MM-DD HH:MM:SS'
        ... )
    """
    return feature.get_map_features_in_bbox_controller(
        bbox=bbox, filters=filters, filter_values=filter_values, layer="traffic_signs"
    )


@auth()
def images_in_geojson(
    geojson: dict,
    dir_cache: str = None,
    max_workers: int = 1,
    logger=None,
    **filters: dict,
):
    """Extracts all images within a shape.

    Args:
      geojson(dict): A geojson as the shape acting as the query
    extent
      **filters(dict (kwargs): Different filters that may be applied
    to the output, defaults to {}
      filters.max_captured_at(str): The max date. Format from 'YYYY',
    to 'YYYY-MM-DDTHH:MM:SS'
      filters.min_captured_at(str): The min date. Format from 'YYYY',
    to 'YYYY-MM-DDTHH:MM:SS'
      filters.image_type(str): The tile image_type to be obtained,
    either as 'flat', 'pano' (panoramic), or 'all'. See
    https://www.mapillary.com/developer/api-documentation/ under
    'image_type Tiles' for more information
      filters.compass_angle(int): The compass angle of the image
      filters.sequence_id(str): ID of the sequence this image belongs
    to
      filters.organization_id(str): ID of the organization this image
    belongs to. It can be absent
      geojson: dict:
      dir_cache: str:  (Default value = None)
      max_workers: int:  (Default value = 1)
      logger: (Default value = None)
      **filters: dict:
      geojson: dict:
      dir_cache: str:  (Default value = None)
      max_workers: int:  (Default value = 1)
      **filters: dict:
      geojson: dict:
      dir_cache: str:  (Default value = None)
      max_workers: int:  (Default value = 1)
      **filters: dict:

    Returns:
      mapillary.models.geojson.GeoJSON: A GeoJSON object
      Usage: :

    >>> import mapillary as mly
        >>> from mapillary.models.geojson import GeoJSON
        >>> import json
        >>> mly.interface.set_access_token('MLY|YYY')
        >>> data = mly.interface.images_in_geojson(json.load(open('my_geojson.geojson', mode='r')))
        >>> open('output_geojson.geojson', mode='w').write(data.encode())
    """
    return image.geojson_features_controller(
        geojson=geojson,
        is_image=True,
        filters=filters,
        dir_cache=dir_cache,
        max_workers=max_workers,
        logger=logger,
    )


@auth()
def images_in_shape(shape, **filters: dict):
    """Extracts all images within a shape or polygon.

    Format::

    Args:
      shape(dict): A shape that describes features, formatted as a
    geojson
      **filters(dict (kwargs): Different filters that may be applied
    to the output, defaults to {}
      filters.max_captured_at(str): The max date. Format from 'YYYY',
    to 'YYYY-MM-DDTHH:MM:SS'
      filters.min_captured_at(str): The min date. Format from 'YYYY',
    to 'YYYY-MM-DDTHH:MM:SS'
      filters.image_type(str): The tile image_type to be obtained,
    either as 'flat', 'pano' (panoramic), or 'all'. See
    https://www.mapillary.com/developer/api-documentation/ under
    'image_type Tiles' for more information
      filters.compass_angle(int): The compass angle of the image
      filters.sequence_id(str): ID of the sequence this image belongs
    to
      filters.organization_id(str): ID of the organization this image
    belongs to. It can be absent
      **filters: dict:
      **filters: dict:
      **filters: dict:

    Returns:
      mapillary.models.geojson.GeoJSON: A GeoJSON object
      Usage: :

    >>> {
        ...    "type": "FeatureCollection",
        ...     "features": [
        ...        {
        ...             "type": "Feature",
        ...             "properties": {},
        ...             "geometry": {
        ...                 "type": "Polygon",
        ...                 "coordinates": [
        ...                     [
        ...                         [
        ...                             7.2564697265625,
        ...                             43.69716905314008
        ...                         ],
        ...                         ...
        ...                     ]
        ...                 ]
        ...             }
        ...         }
        ...     ]
        ... }

        >>> import mapillary as mly
        >>> import json
        >>> mly.interface.set_access_token('MLY|XXX')
        >>> data = mly.interface.images_in_shape(json.load(open('polygon.geojson', mode='r')))
        >>> open('output_geojson.geojson', mode='w').write(data.encode())
    """
    return image.shape_features_controller(shape=shape, is_image=True, filters=filters)


@auth()
def map_features_in_geojson(geojson: dict, **filters: dict):
    """Extracts all map features within a geojson's boundaries.

    Args:
      geojson(dict): A geojson as the shape acting as the query
    extent
      **filters(dict (kwargs): Different filters that may be applied
    to the output, defaults to {}
      filters.zoom(int): The zoom level of the tiles to obtain,
    defaults to 14
      filters.max_captured_at(str): The max date. Format from 'YYYY',
    to 'YYYY-MM-DDTHH:MM:SS'
      filters.min_captured_at(str): The min date. Format from 'YYYY',
    to 'YYYY-MM-DDTHH:MM:SS'
      filters.image_type(str): The tile image_type to be obtained,
    either as 'flat', 'pano' (panoramic), or 'all'. See
    https://www.mapillary.com/developer/api-documentation/ under
    'image_type Tiles' for more information
      filters.compass_angle(int): The compass angle of the image
      filters.sequence_id(str): ID of the sequence this image belongs
    to
      filters.organization_id(str): ID of the organization this image
    belongs to. It can be absent
      geojson: dict:
      **filters: dict:
      geojson: dict:
      **filters: dict:
      geojson: dict:
      **filters: dict:

    Returns:
      mapillary.models.geojson.GeoJSON: A GeoJSON object
      Usage: :

    >>> import mapillary as mly
        >>> import json
        >>> mly.interface.set_access_token('MLY|YYY')
        >>> data = mly.interface.map_features_in_geojson(
        ...     json.load(
        ...         open('my_geojson.geojson', mode='r')
        ...     )
        ... )
        >>> open('output_geojson.geojson', mode='w').write(data.encode())
    """
    if isinstance(geojson, str):
        if "http" in geojson:
            geojson = json.loads(requests.get(geojson).content.decode("utf-8"))

    return image.geojson_features_controller(
        geojson=geojson,
        is_image=False,
        filters=filters,
    )


@auth()
def map_features_in_shape(shape: dict, **filters: dict):
    """Extracts all map features within a shape/polygon.

    Format::

    Args:
      shape(dict): A shape that describes features, formatted as a
    geojson
      **filters(dict (kwargs): Different filters that may be applied
    to the output, defaults to {}
      filters.zoom(int): The zoom level of the tiles to obtain,
    defaults to 14
      filters.max_captured_at(str): The max date. Format from 'YYYY',
    to 'YYYY-MM-DDTHH:MM:SS'
      filters.min_captured_at(str): The min date. Format from 'YYYY',
    to 'YYYY-MM-DDTHH:MM:SS'
      filters.image_type(str): The tile image_type to be obtained,
    either as 'flat', 'pano' (panoramic), or 'all'. See
    https://www.mapillary.com/developer/api-documentation/ under
    'image_type Tiles' for more information
      filters.compass_angle(int): The compass angle of the image
      filters.sequence_id(str): ID of the sequence this image belongs
    to
      filters.organization_id(str): ID of the organization this image
    belongs to. It can be absent
      shape: dict:
      **filters: dict:
      shape: dict:
      **filters: dict:
      shape: dict:
      **filters: dict:

    Returns:
      mapillary.models.geojson.GeoJSON: A GeoJSON object
      Usage: :

    >>> _ = {
        ...     "type": "FeatureCollection",
        ...     "features": [
        ...         {
        ...             "type": "Feature",
        ...             "properties": {},
        ...             "geometry": {
        ...                 "type": "Polygon",
        ...                 "coordinates": [
        ...                     [
        ...                         [
        ...                             7.2564697265625,
        ...                             43.69716905314008
        ...                         ],
        ...                         ...
        ...                     ]
        ...                 ]
        ...             }
        ...         }
        ...     ]
        ... }

        >>> import mapillary as mly
        >>> import json
        >>> mly.interface.set_access_token('MLY|XXX')
        >>> data = mly.interface.map_features_in_shape(json.load(open('polygon.geojson', mode='r')))
        >>> open('output_geojson.geojson', mode='w').write(data.encode())
    """
    if isinstance(shape, str):
        if "http" in shape:
            shape = json.loads(requests.get(shape).content.decode("utf-8"))

    return image.shape_features_controller(shape=shape, is_image=False, filters=filters)


@auth()
def feature_from_key(key: str, fields: list = []) -> str:
    """Gets a map feature for the given key argument.

    Args:
      key(int): The map feature ID to which will be used to get the
    feature
      fields(list): The fields to include. The field 'geometry' will
    always be included so you do not need to specify it, or if
    you leave it off, it will still be returned.
    Fields::
    1. first_seen_at - timestamp, timestamp of the least recent
    detection contributing to this feature
    2. last_seen_at - timestamp, timestamp of the most recent
    detection contributing to this feature
    3. object_value - string, what kind of map feature it is
    4. object_type - string, either a traffic_sign or point
    5. geometry - GeoJSON Point geometry
    6. images - list of IDs, which images this map feature was derived
    from
    Refer to https://www.mapillary.com/developer/api-documentation/#map-feature for more details
      key: str:
      fields: list:  (Default value = [])
      key: str:
      fields: list:  (Default value = [])
      key: str:
      fields: list:  (Default value = [])

    Returns:
      str: A GeoJSON string that represents the queried feature
      Usage: :

    >>> import mapillary as mly
        >>> mly.interface.set_access_token('MLY|XXX')
        >>> mly.interface.feature_from_key(
        ...     key='VALID_MAP_FEATURE_KEY',
        ...     fields=['object_value']
        ... )
    """
    return feature.get_feature_from_key_controller(key=int(key), fields=fields)


@auth()
def image_from_key(key: str, fields: list = None) -> str:
    """Gets an image for the given key argument.

    Args:
      key(int): The image unique key which will be used for image
    retrieval
      fields(list): The fields to include. The field 'geometry' will
    always be included so you do not need to specify it, or if
    you leave it off, it will still be returned.
    Fields,
    1. altitude - float, original altitude from Exif
    2. atomic_scale - float, scale of the SfM reconstruction around the image
    3. camera_parameters - array of float, intrinsic camera parameters
    4. camera_type - enum, type of camera projection (perspective, fisheye, or
    spherical)
    5. captured_at - timestamp, capture time
    6. compass_angle - float, original compass angle of the image
    7. computed_altitude - float, altitude after running image processing
    8. computed_compass_angle - float, compass angle after running image processing
    9. computed_geometry - GeoJSON Point, location after running image processing
    10. computed_rotation - enum, corrected orientation of the image
    11. exif_orientation - enum, orientation of the camera as given by the exif tag
    (see: https://sylvana.net/jpegcrop/exif_orientation.html)
    12. geometry - GeoJSON Point geometry
    13. height - int, height of the original image uploaded
    14. thumb_256_url - string, URL to the 256px wide thumbnail
    15. thumb_1024_url - string, URL to the 1024px wide thumbnail
    16. thumb_2048_url - string, URL to the 2048px wide thumbnail
    17. merge_cc - int, id of the connected component of images that were aligned
    together
    18. mesh - { id: string, url: string } - URL to the mesh
    19. quality_score - float, how good the image is (experimental)
    20. sequence - string, ID of the sequence
    21. sfm_cluster - { id: string, url: string } - URL to the point cloud
    22. width - int, width of the original image uploaded
    Refer to https://www.mapillary.com/developer/api-documentation/#image for more details
      key: str:
      fields: list:  (Default value = None)
      key: str:
      fields: list:  (Default value = None)
      key: str:
      fields: list:  (Default value = None)

    Returns:
      str: A GeoJSON string that represents the queried image
      Usage: :

    >>> import mapillary as mly
        >>> mly.interface.set_access_token('MLY|XXX')
        >>> mly.interface.image_from_key(
        ...     key='VALID_IMAGE_KEY',
        ...     fields=['captured_at', 'sfm_cluster', 'width']
        ... )
    """
    return image.get_image_from_key_controller(key=int(key), fields=fields)


@auth()
def save_locally(
    geojson_data: str,
    file_path: str = os.path.dirname(os.path.realpath(__file__)),
    file_name: str = None,
    extension: str = "geojson",
) -> None:
    """This function saves the geojson data locally as a file with the given file name,
    path, and format.

    Args:
      geojson_data(str): The GeoJSON data to be stored
      file_path(str): The path to save the data to. Defaults to the
    current directory path
      file_name(str): The name of the file to be saved. Defaults to
    'geojson'
      extension(str): The format to save the data as. Defaults to
    'geojson'
    Note::
      extension(str): The format to save the data as. Defaults to
    'geojson'
    Note::
    Allowed file format values at the moment are,
    - geojson
    - CSV
    *TODO*: More file format will be supported further in developemtn
    *TODO*: Suggestions and help needed at mapillary/mapillary-python-sdk!
      geojson_data: str:
      file_path: str:  (Default value = os.path.dirname(os.path.realpath(__file__)))
      file_name: str:  (Default value = None)
      extension: str:  (Default value = "geojson")
      geojson_data: str:
      file_path: str:  (Default value = os.path.dirname(os.path.realpath(__file__)))
      file_name: str:  (Default value = None)
      extension: str:  (Default value = "geojson")
      geojson_data: str:
      file_path: str:  (Default value = os.path.dirname(os.path.realpath(__file__)))
      file_name: str:  (Default value = None)
      extension: str:  (Default value = "geojson")

    Returns:
      None: None
      Usage: :

    >>> import mapillary as mly
        >>> mly.interface.set_access_token('MLY|XXX')
        >>> mly.interface.save_locally(
        ...     geojson_data=geojson_data,
        ...     file_path=os.path.dirname(os.path.realpath(__file__)),
        ...     file_name='test_geojson',
        ...     extension='geojson'
        ... )
        >>> mly.interface.save_locally(
        ...     geojson_data=geojson_data,
        ...     file_path=os.path.dirname(os.path.realpath(__file__)),
        ...     file_name='local_geometries',
        ...     extension='csv'
        ... )
    """
    # Check if a valid file format was provided
    if extension.lower() not in ["geojson", "csv"]:
        # If not, raise an error
        raise InvalidOptionError(
            param="format",
            value=extension,
            options=["geojson", "csv"],
        )

    return (
        save.save_as_geojson_controller(data=geojson_data, path=file_path, file_name=file_name)
        if extension.lower() == "geojson"
        else save.save_as_csv_controller(data=geojson_data, path=file_path, file_name=file_name)
    )
