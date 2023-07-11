# Copyright (c) Facebook, Inc. and its affiliates. (http://www.facebook.com)
# -*- coding: utf-8 -*-

"""
mapillary.controllers.image
~~~~~~~~~~~~~~~~~~~~~~~~~~~

This module implements the image filtering and analysis business logic functionalities of the
Mapillary Python SDK.

For more information, please check out https://www.mapillary.com/developer/api-documentation/

- Copyright: (c) 2021 Facebook
- License: MIT LICENSE
"""

# Library imports
import json

import mercantile
import shapely
from geojson import Polygon
from typing import Union

# # Configs
from zensvi.download.mapillary.config.api.entities import Entities
from zensvi.download.mapillary.config.api.vector_tiles import VectorTiles
from zensvi.download.mapillary.models.api.entities import EntityAdapter
from zensvi.download.mapillary.models.api.general import GeneralAdapter

# # Adapters
from zensvi.download.mapillary.models.api.vector_tiles import VectorTilesAdapter

# # Client
from zensvi.download.mapillary.models.client import Client

# # Exception Handling
from zensvi.download.mapillary.models.exceptions import InvalidImageKeyError

# # Class Representation
from zensvi.download.mapillary.models.geojson import GeoJSON, Coordinates

# # Utilities
from zensvi.download.mapillary.utils.filter import pipeline
from zensvi.download.mapillary.utils.format import (
    feature_to_geojson,
    merged_features_list_to_geojson,
    geojson_to_polygon,
    coord_or_list_to_dict,
)
from zensvi.download.mapillary.utils.verify import (
    image_check,
    image_bbox_check,
    sequence_bbox_check,
    resolution_check,
    valid_id,
)
from requests import HTTPError
from turfpy.measurement import bbox
from vt2geojson.tools import vt_bytes_to_geojson


def get_image_close_to_controller(
    longitude: float,
    latitude: float,
    kwargs: dict,
) -> GeoJSON:
    """
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
    """

    # Checking if a non valid key has been passed to the function If that is the case, throw an
    # exception
    image_check(kwargs=kwargs)

    unfiltered_data = VectorTilesAdapter().fetch_layer(
        layer="image",
        zoom=kwargs["zoom"] if "zoom" in kwargs else 14,
        longitude=longitude,
        latitude=latitude,
    )

    if kwargs == {}:
        return GeoJSON(geojson=unfiltered_data)

    # Filtering for the attributes obtained above
    if (
        len(unfiltered_data["features"]) != 0
        and unfiltered_data["features"][0]["properties"] != {}
    ):
        return GeoJSON(
            geojson=json.loads(
                merged_features_list_to_geojson(
                    pipeline(
                        data=unfiltered_data,
                        components=[
                            # Filter using kwargs.min_captured_at
                            {
                                "filter": "min_captured_at",
                                "min_timestamp": kwargs["min_captured_at"],
                            }
                            if "min_captured_at" in kwargs
                            else {},
                            # Filter using kwargs.max_captured_at
                            {
                                "filter": "max_captured_at",
                                "min_timestamp": kwargs["max_captured_at"],
                            }
                            if "max_captured_at" in kwargs
                            else {},
                            # Filter using kwargs.image_type
                            {"filter": "image_type", "tile": kwargs["image_type"]}
                            if "image_type" in kwargs
                            else {},
                            # Filter using kwargs.organization_id
                            {
                                "filter": "organization_id",
                                "organization_ids": kwargs["org_id"],
                            }
                            if "org_id" in kwargs
                            else {},
                            # Filter using kwargs.radius
                            {
                                "filter": "haversine_dist",
                                "radius": kwargs["radius"],
                                "coords": [longitude, latitude],
                            }
                            if "radius" in kwargs
                            else {},
                        ],
                    )
                )
            )
        )


def get_image_looking_at_controller(
    at: Union[dict, Coordinates, list],
    filters: dict,
) -> GeoJSON:
    """
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
    """

    # Converting 'at' of type Coordinates|List to dict
    at: dict = coord_or_list_to_dict(data=at)

    # Checking if a non valid key
    # has been passed to  the function
    # If that is the case, throw an exception
    image_check(kwargs=filters)

    at_image_data = GeneralAdapter().fetch_image_tiles(
        zoom=filters["zoom"] if "zoom" in filters else 14,
        longitude=at["lng"],
        latitude=at["lat"],
        layer="image",
    )

    if not at_image_data["features"]:
        return GeoJSON(geojson=at_image_data)

    # Filters are to be applied to the data retrieved from the database in the following logic
    # # 1. Filter by the filters provided by the filters parameter
    # # 2. Secondly, from the fetched tile, trim out data that falls outside the given radius, if
    # no radius is specified, assume 20m to be safe
    # # 2. Then, from the remaining feature points, extract only those that are qualified by the
    # "hits_by_look_at" function

    # Filter the unfiltered results by the given filters
    return GeoJSON(
        geojson=json.loads(
            merged_features_list_to_geojson(
                pipeline(
                    data=at_image_data,
                    components=[
                        # Filter by `max_captured_at`
                        {
                            "filter": "max_captured_at",
                            "max_timestamp": filters.get("max_captured_at"),
                        }
                        if "max_captured_at" in filters
                        else {},
                        # Filter by `min_captured_at`
                        {
                            "filter": "min_captured_at",
                            "min_timestamp": filters.get("min_captured_at"),
                        }
                        if "min_captured_at" in filters
                        else {},
                        # Filter by `image_type`
                        {"filter": "image_type", "type": filters.get("image_type")}
                        if "image_type" in filters and filters["image_type"] != "all"
                        else {},
                        # Filter by `organization_id`
                        {
                            "filter": "organization_id",
                            "organization_ids": filters.get("organization_id"),
                        }
                        if "organization_id" in filters
                        else {},
                        # Filter using kwargs.radius
                        {
                            "filter": "haversine_dist",
                            "radius": filters.get("radius"),
                            "coords": [at["lng"], at["lat"]],
                        }
                        if "radius" in filters
                        else {},
                        # Filter by `hits_by_look_at`
                        {"filter": "hits_by_look_at", "at": at},
                    ],
                )
            )
        )
    )


def is_image_being_looked_at_controller(
    at: Union[dict, Coordinates, list],
    filters: dict,
) -> bool:
    """
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
    """

    result: dict = get_image_looking_at_controller(at=at, filters=filters).to_dict()

    # If the result is empty, the image is not looked at, hence return False
    return len(result["features"]) != 0


def get_image_thumbnail_controller(image_id: str, resolution: int) -> str:
    """
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
    """

    # check if the entered resolution is one of the supported image sizes
    resolution_check(resolution)

    try:
        res = Client().get(Entities.get_image(image_id, [f"thumb_{resolution}_url"]))
    except HTTPError:
        # If given ID is an invalid image ID, let the user know
        raise InvalidImageKeyError(image_id)

    return json.loads(res.content.decode("utf-8"))[f"thumb_{resolution}_url"]


def get_images_in_bbox_controller(
    bounding_box: dict, layer: str, zoom: int, filters: dict
) -> str:
    """
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
    """

    # Check if the given filters are valid ones
    filters["zoom"] = filters.get("zoom", zoom)
    filters = (
        image_bbox_check(filters) if layer == "image" else sequence_bbox_check(filters)
    )

    # Instantiate the Client
    client = Client()

    # filtered images or sequence data will be appended to this list
    filtered_results = []

    # A list of tiles that are either confined within or intersect with the bbox
    tiles = list(
        mercantile.tiles(
            west=bounding_box["west"],
            south=bounding_box["south"],
            east=bounding_box["east"],
            north=bounding_box["north"],
            zooms=zoom,
        )
    )

    for tile in tiles:
        url = (
            VectorTiles.get_image_layer(x=tile.x, y=tile.y, z=tile.z)
            if layer == "image"
            else VectorTiles.get_sequence_layer(x=tile.x, y=tile.y, z=tile.z)
        )

        # Get the response from the API
        res = client.get(url)

        # Get the GeoJSON response by decoding the byte tile
        geojson = vt_bytes_to_geojson(
            b_content=res.content, layer=layer, z=tile.z, x=tile.x, y=tile.y
        )

        # Filter the unfiltered results by the given filters
        filtered_results.extend(
            pipeline(
                data=geojson,
                components=[
                    {"filter": "features_in_bounding_box", "bbox": bounding_box}
                    if layer == "image"
                    else {},
                    {
                        "filter": "max_captured_at",
                        "max_timestamp": filters.get("max_captured_at"),
                    }
                    if filters["max_captured_at"] is not None
                    else {},
                    {
                        "filter": "min_captured_at",
                        "min_timestamp": filters.get("min_captured_at"),
                    }
                    if filters["min_captured_at"] is not None
                    else {},
                    {"filter": "image_type", "type": filters.get("image_type")}
                    if filters["image_type"] is not None
                    or filters["image_type"] != "all"
                    else {},
                    {
                        "filter": "organization_id",
                        "organization_ids": filters.get("organization_id"),
                    }
                    if filters["organization_id"] is not None
                    else {},
                    {"filter": "sequence_id", "ids": filters.get("sequence_id")}
                    if layer == "image" and filters["sequence_id"] is not None
                    else {},
                    {"filter": "compass_angle", "angles": filters.get("compass_angle")}
                    if layer == "image" and filters["compass_angle"] is not None
                    else {},
                ],
            )
        )

    return merged_features_list_to_geojson(filtered_results)


def get_image_from_key_controller(key: int, fields: list) -> str:
    """
    A controller for getting properties of a certain image given the image key and
    the list of fields/properties to be returned

    :param key: The image key
    :type key: int

    :param fields: The list of fields to be returned
    :type fields: list

    :return: The requested image properties in GeoJSON format
    :rtype: str
    """

    valid_id(identity=key, image=True)

    # ? 'merged_features_list_to_geojson' takes list, 'feature_to_geojson' returns dict
    return merged_features_list_to_geojson(
        features_list=feature_to_geojson(
            json_data=EntityAdapter().fetch_image(image_id=key, fields=fields)
        )
    )


def geojson_features_controller(
    geojson: dict, is_image: bool = True, filters: dict = None
) -> GeoJSON:
    """
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
    """

    # Filter checking
    image_bbox_check(filters)

    # Extracting polygon from geojson, converting to dict
    polygon = geojson_to_polygon(geojson).to_dict()

    # Generating a coordinates list to extract from polygon
    coordinates_list = []

    # Going through each feature
    for feature in polygon["features"]:

        # Going through the coordinate's nested list
        for coordinates in feature["geometry"]["coordinates"][0]:
            # Appending a tuple of coordinates
            coordinates_list.append((coordinates[0], coordinates[1]))

    # Sending coordinates_list a input to form a Polygon
    polygon = Polygon(coordinates_list)

    # Getting the boundary parameters from polygon
    boundary = shapely.geometry.shape(polygon)

    if is_image:
        # Get a GeoJSON with features from tiles originating from coordinates
        # at specified zoom level
        layers: dict = (
            VectorTilesAdapter()
            .fetch_layers(
                # Sending coordinates for all the points within input geojson
                coordinates=bbox(polygon),
                # Fetching image layers for the geojson
                layer=filters["layer"] if "layer" in filters else "image",
                # Specifying zoom level, defaults to zoom if zoom not specified
                zoom=filters["zoom"] if "zoom" in filters else 14,
            )
            .to_dict()
        )
    else:
        # Get all the map features within the boundary box for the polygon
        layers: dict = (
            VectorTilesAdapter()
            .fetch_map_features(
                # Sending coordinates for all the points within input geojson
                coordinates=bbox(polygon),
                # Fetching image layers for the geojson
                feature_type=filters["feature_type"]
                if "feature_type" in filters
                else "point",
                # Specifying zoom level, defaults to zoom if zoom not specified
                zoom=filters["zoom"] if "zoom" in filters else 14,
            )
            .to_dict()
        )

    # Return as GeoJSON output
    return GeoJSON(
        # Load the geojson to convert to GeoJSON object
        geojson=json.loads(
            # Convert feature list to GeoJSON
            merged_features_list_to_geojson(
                # Execute pipeline for filters
                pipeline(
                    # Sending layers as input
                    data=layers,
                    # Specifying components for the filter
                    components=[
                        {"filter": "in_shape", "boundary": boundary},
                        # Filter using kwargs.min_captured_at
                        {
                            "filter": "min_captured_at",
                            "min_timestamp": filters["min_captured_at"],
                        }
                        if "min_captured_at" in filters
                        else {},
                        # Filter using filters.max_captured_at
                        {
                            "filter": "max_captured_at",
                            "min_timestamp": filters["max_captured_at"],
                        }
                        if "max_captured_at" in filters
                        else {},
                        # Filter using filters.image_type
                        {"filter": "image_type", "tile": filters["image_type"]}
                        if "image_type" in filters
                        else {},
                        # Filter using filters.organization_id
                        {
                            "filter": "organization_id",
                            "organization_ids": filters["org_id"],
                        }
                        if "organization_id" in filters
                        else {},
                        # Filter using filters.sequence_id
                        {"filter": "sequence_id", "ids": filters.get("sequence_id")}
                        if "sequence_id" in filters
                        else {},
                        # Filter using filters.compass_angle
                        {
                            "filter": "compass_angle",
                            "angles": filters.get("compass_angle"),
                        }
                        if "compass_angle" in filters
                        else {},
                    ],
                )
            )
        )
    )


def shape_features_controller(
    shape, is_image: bool = True, filters: dict = None
) -> GeoJSON:
    """
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
    """

    image_bbox_check(filters)

    # Generating a coordinates list to extract from polygon
    coordinates_list = []

    # Going through each feature
    for feature in shape["features"]:

        # Going through the coordinate's nested list
        for coordinates in feature["geometry"]["coordinates"][0]:
            # Appending a tuple of coordinates
            coordinates_list.append((coordinates[0], coordinates[1]))

    # Sending coordinates_list a input to form a Polygon
    polygon = Polygon(coordinates_list)

    # Getting the boundary parameters from polygon
    boundary = shapely.geometry.shape(polygon)

    if is_image:
        # Get all the map features within the boundary box for the polygon
        output: dict = (
            VectorTilesAdapter()
            .fetch_layers(
                # Sending coordinates for all the points within input geojson
                coordinates=bbox(polygon),
                # Fetching image layers for the geojson
                layer=filters["layer"] if "layer" in filters else "image",
                # Specifying zoom level, defaults to zoom if zoom not specified
                zoom=filters["zoom"] if "zoom" in filters else 14,
            )
            .to_dict()
        )
    else:
        # Get all the map features within the boundary box for the polygon
        output: dict = (
            VectorTilesAdapter()
            .fetch_map_features(
                # Sending coordinates for all the points within input geojson
                coordinates=bbox(polygon),
                # Fetching image layers for the geojson
                feature_type=filters["feature_type"]
                if "feature_type" in filters
                else "point",
                # Specifying zoom level, defaults to zoom if zoom not specified
                zoom=filters["zoom"] if "zoom" in filters else 14,
            )
            .to_dict()
        )

    # Return as GeoJSON output
    return GeoJSON(
        # Load the geojson to convert to GeoJSON object
        geojson=json.loads(
            # Convert feature list to GeoJSON
            merged_features_list_to_geojson(
                # Execute pipeline for filters
                pipeline(
                    # Sending layers as input
                    data=output,
                    # Specifying components for the filter
                    components=[
                        # Get only features within the given boundary
                        {"filter": "in_shape", "boundary": boundary},
                        # Filter using kwargs.min_captured_at
                        {
                            "filter": "min_captured_at",
                            "min_timestamp": filters["min_captured_at"],
                        }
                        if "min_captured_at" in filters
                        else {},
                        # Filter using filters.max_captured_at
                        {
                            "filter": "max_captured_at",
                            "min_timestamp": filters["max_captured_at"],
                        }
                        if "max_captured_at" in filters
                        else {},
                        # Filter using filters.image_type
                        {"filter": "image_type", "tile": filters["image_type"]}
                        if "image_type" in filters
                        else {},
                        # Filter using filters.organization_id
                        {
                            "filter": "organization_id",
                            "organization_ids": filters["org_id"],
                        }
                        if "organization_id" in filters
                        else {},
                        # Filter using filters.sequence_id
                        {"filter": "sequence_id", "ids": filters.get("sequence_id")}
                        if "sequence_id" in filters
                        else {},
                        # Filter using filters.compass_angle
                        {
                            "filter": "compass_angle",
                            "angles": filters.get("compass_angle"),
                        }
                        if "compass_angle" in filters
                        else {},
                    ],
                )
            )
        )
    )
