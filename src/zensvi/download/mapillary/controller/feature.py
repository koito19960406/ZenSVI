# Copyright (c) Facebook, Inc. and its affiliates. (http://www.facebook.com)
# -*- coding: utf-8 -*-

"""
mapillary.controllers.feature
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This module implements the feature extraction business logic functionalities of the Mapillary
Python SDK.

For more information, please check out https://www.mapillary.com/developer/api-documentation/

- Copyright: (c) 2021 Facebook
- License: MIT LICENSE
"""

# Configs
from mapillary.config.api.vector_tiles import VectorTiles

# Client
from mapillary.models.client import Client

# Utils
from mapillary.utils.verify import valid_id, points_traffic_signs_check
from mapillary.utils.filter import pipeline
from mapillary.utils.format import (
    merged_features_list_to_geojson,
    feature_to_geojson,
)

# Adapters
from mapillary.models.api.entities import EntityAdapter

# Package imports
import mercantile
from vt2geojson.tools import vt_bytes_to_geojson


def get_feature_from_key_controller(key: int, fields: list) -> str:
    """
    A controller for getting properties of a certain image given the image key and
    the list of fields/properties to be returned

    :param key: The image key
    :type key: int

    :param fields: List of possible fields
    :type fields: list

    :return: The requested feature properties in GeoJSON format
    :rtype: str
    """

    valid_id(identity=key, image=False)

    # ? feature_to_geojson returns dict, but merged_features_list_to_geojson takes list as input
    return merged_features_list_to_geojson(
        features_list=feature_to_geojson(
            json_data=EntityAdapter().fetch_map_feature(
                map_feature_id=key, fields=fields
            )
        )
    )


def get_map_features_in_bbox_controller(
    bbox: dict,
    filter_values: list,
    filters: dict,
    layer: str = "points",
) -> str:
    """
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
    """

    # Verifying the existence of the filter kwargs
    filters = points_traffic_signs_check(filters)

    # Instantiating Client for API requests
    client = Client()

    # Getting all tiles within or intersecting the bbox
    tiles = list(
        mercantile.tiles(
            west=bbox["west"],
            south=bbox["south"],
            east=bbox["east"],
            north=bbox["north"],
            zooms=14,
        )
    )

    # Filtered features lists from different tiles will be merged into
    # filtered_features
    filtered_features = []

    for tile in tiles:
        # Decide which endpoint to send a request to based on the layer
        url = (
            VectorTiles.get_map_feature_point(x=tile.x, y=tile.y, z=tile.z)
            if layer == "points"
            else VectorTiles.get_map_feature_traffic_sign(x=tile.x, y=tile.y, z=tile.z)
        )

        res = client.get(url)

        # Decoding byte tiles
        data = vt_bytes_to_geojson(res.content, tile.x, tile.y, tile.z)

        filtered_features.extend(
            pipeline(
                data=data,
                components=[
                    # Skip filtering based on filter_values if they're not specified by the user
                    {
                        "filter": "filter_values",
                        "values": filter_values,
                        "property": "value",
                    }
                    if filter_values is not None
                    else {},
                    # Check if the features actually lie within the bbox
                    {"filter": "features_in_bounding_box", "bbox": bbox},
                    # Checks if the feature existed after a given date
                    {
                        "filter": "existed_at",
                        "existed_at": filters["existed_at"],
                    }
                    if filters["existed_at"] is not None
                    else {},
                    # Filter out all the features after a given timestamp
                    {
                        "filter": "existed_before",
                        "existed_before": filters["existed_before"],
                    }
                    if filters["existed_before"] is not None
                    else {},
                ],
            )
        )

    return merged_features_list_to_geojson(filtered_features)
