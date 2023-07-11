# Copyright (c) Facebook, Inc. and its affiliates. (http://www.facebook.com)
# -*- coding: utf-8 -*-

"""
mapillary.utils.format
======================

This module deals with converting data to and from different file formats.

- Copyright: (c) 2021 Facebook
- License: MIT LICENSE
"""

# Package imports
import base64
import json
import typing
import mapbox_vector_tile
from collections.abc import MutableMapping
from typing import Union

# Local imports
# # Models
from zensvi.download.mapillary.models.geojson import Coordinates, GeoJSON


def feature_to_geojson(json_data: dict) -> dict:
    """
    Converts feature into a GeoJSON, returns output

    From::

        >>> {'geometry': {'type': 'Point', 'coordinates': [30.003755665554, 30.985948744314]},
        ... 'id':'506566177256016'}

    To::

        >>> {'type': 'FeatureCollection', 'features': [{'type': 'Feature', 'geometry': {'type':
        ... 'Point','coordinates': [30.98594605922699, 30.003757307208872]}, 'properties': {}}]}

    :param json_data: The feature as a JSON
    :type json_data: dict

    :return: The formatted GeoJSON
    :rtype: dict
    """

    # The geometry property will always be present
    keys = [key for key in json_data.keys() if key != "geometry"]

    feature = {"type": "Feature", "geometry": {}, "properties": {}}
    # Make sure that all keys exist and retrieve their values if specified

    for key in keys:
        feature["properties"][key] = json_data[key]

    return feature


def join_geojson_with_keys(
    geojson_src: dict,
    geojson_src_key: str,
    geojson_dest: dict,
    geojson_dest_key: str,
) -> dict:
    """
    Combines two GeoJSONS based on the similarity of their specified keys, similar to
    the SQL join functionality

    :param geojson_src: The starting GeoJSO source
    :type geojson_src: dict

    :param geojson_src_key: The key in properties specified for the GeoJSON source
    :type geojson_src_key: str

    :param geojson_dest: The GeoJSON to merge into
    :type geojson_dest: dict

    :param geojson_dest_key: The key in properties specified for the GeoJSON to merge into
    :type geojson_dest_key: dict

    :return: The merged GeoJSON
    :rtype: dict

    Usage::

        >>> join_geojson_with_keys(
        ...     geojson_src=geojson_src,
        ...     geojson_src_key='id',
        ...     geojson_dest=geojson_dest,
        ...     geojson_dest_key='id'
        ... )
    """

    # Go through the feature set in the src geojson
    for from_features in geojson_src["features"]:

        # Go through the feature set in the dest geojson
        for into_features in geojson_dest["features"]:

            # If either of the geojson features do not contain
            # their respective assumed keys, continue
            if (
                geojson_dest_key not in into_features["properties"]
                or geojson_src_key not in from_features["properties"]
            ):
                continue

            # Checking if two IDs match up
            if int(from_features["properties"][geojson_src_key]) == int(
                into_features["properties"][geojson_dest_key]
            ):

                # Firstly, extract the properties that exist in the
                # src_geojson for that feature
                old_properties = [key for key in from_features["properties"].keys()]

                # Secondly, extract the properties that exist in the
                # dest_json for that feature
                new_properties = [key for key in into_features["properties"].keys()]

                # Going through the old properties in the features of src_geojson
                for new_property in new_properties:

                    # Going through the new properties in the features of dest_geojson
                    if new_property not in old_properties:
                        # Put the new_feature
                        from_features["properties"][new_property] = old_properties[
                            "properties"
                        ][new_property]

    return geojson_src


def geojson_to_features_list(json_data: dict) -> list:
    """
    Converts a decoded output GeoJSON to a list of feature objects

    The purpose of this formatting utility is to obtain a list of individual features for
    decoded tiles that can be later extended to the output GeoJSON

    From::

        >>> {'type': 'FeatureCollection', 'features': [{'type': 'Feature', 'geometry':
        ... {'type': 'Point','coordinates': [30.98594605922699, 30.003757307208872]},
        ... 'properties': {}}]}

    To::

        >>> [{'type': 'Feature', 'geometry': {'type': 'Point',
        ... 'coordinates': [30.98594605922699, 30.003757307208872]}, 'properties': {}}]

    :param json_data: The given json data
    :type json_data: dict

    :return: The feature list
    :rtype: list
    """

    return json_data["features"]


def merged_features_list_to_geojson(features_list: list) -> str:
    """
    Converts a processed features list (i.e. a features list with all the needed features merged
    from multiple tiles) into a fully-featured GeoJSON

    From::

        >>> [{'type': 'Feature', 'geometry': {'type': 'Point',
        ... 'coordinates': [30.98594605922699, 30.003757307208872]}, 'properties': {}}, ...]

    To::

        >>> {'type': 'FeatureCollection', 'features': [{'type': 'Feature', 'geometry':
        ... {'type': 'Point','coordinates': [30.98594605922699, 30.003757307208872]},
        ... 'properties': {}}, ...]}

    :param features_list: a list of processed features merged from different tiles within a bbox
    :type features_list: list

    :return: GeoJSON string formatted with all the extra commas removed.
    :rtype: str
    """

    return json.dumps({"type": "FeatureCollection", "features": features_list})


def detection_features_to_geojson(feature_list: list) -> dict:
    """
    Converts a preprocessed list (i.e, features from the detections of either images or
    map_features from multiple segments) into a fully featured GeoJSON

    :param feature_list: A list of processed features merged from different segments within a
        detection
    :type feature_list: list

    :return: GeoJSON formatted as expected in a detection format
    :rtype: dict

    Example::

        >>> # From
        >>> [{'created_at': '2021-05-20T17:49:01+0000', 'geometry':
        ... 'GjUKBm1weS1vchIVEgIAABgDIg0JhiekKBoqAABKKQAPGgR0eXBlIgkKB3BvbHlnb24ogCB4AQ==',
        ... 'image': {'geometry': {'type': 'Point', 'coordinates': [-97.743279722222,
        ... 30.270651388889]}, 'id': '1933525276802129'}, 'value': 'regulatory--no-parking--g2',
        ... 'id': '1942105415944115'}, ... ]
        >>> # To
        >>> {'type': 'FeatureCollection', 'features': [{'type': 'Feature', 'geometry':
        ... {'type': 'Point', 'coordinates': [-97.743279722222, 30.270651388889]}, 'properties': {
        ... 'image_id': '1933525276802129', 'created_at': '2021-05-20T17:49:01+0000',
        ... 'pixel_geometry':
        ... 'GjUKBm1weS1vchIVEgIAABgDIg0JhiekKBoqAABKKQAPGgR0eXBlIgkKB3BvbHlnb24ogCB4AQ==',
        ... 'value': 'regulatory--no-parking--g2', 'id': '1942105415944115' } }, ... ]}
    """

    resulting_geojson = {
        # FeatureCollection type
        "type": "FeatureCollection",
        # List of features
        "features": [
            # Feature generation from feature_list
            {
                # Type is 'Feature'
                "type": "Feature",
                # Let 'geometry' be the `image` key, defaults to {} is `image` not in feature
                "geometry": {
                    "type": "Point",
                    "coordinates": feature["image"]["geometry"]["coordinates"],
                }
                if "image" in feature
                else {},
                # Property list
                "possible_none_properties": {
                    # Only if "image" was specified in the `fields` of the endpoint, else None
                    "image_id": feature["image"]["id"]
                    if "image" in "feature"
                    else None,
                    # Only if "created_at" was specified in the `fields` of the endpoint, else None
                    "created_at": feature["created_at"]
                    if "created_at" in feature
                    else None,
                    # Only if "geometry" was specified in the `fields` of the endpoint, else None
                    "pixel_geometry": feature["geometry"]
                    if "geometry" in feature
                    else None,
                    # Only if "value" was specified in the `fields` of the endpoint, else None
                    "value": feature["value"] if "value" in feature else None,
                    # "id" is always available in the response
                    "id": feature["id"],
                },
                "properties": {},
            }
            # Going through the given of features
            for feature in feature_list
        ],
    }

    # The next logic below removes features that defaulted to None

    # Through each feature in the resulting features
    for _feature in resulting_geojson["features"]:

        # Going through each property in the feature
        for key, value in _feature["possible_none_properties"].items():

            # If the _property is not None
            if value is not None:

                # Add that property to the _feature
                _feature["properties"][key] = value

    for _feature in resulting_geojson["features"]:

        del _feature["possible_none_properties"]

    # Finally return the output
    return resulting_geojson


def flatten_geojson(geojson: dict) -> list:
    """
    Flattens a GeoJSON dictionary to a dictionary with only the relevant keys.
    This is useful for writing to a CSV file.

    Output Structure::

        >>> {
        ...     "geometry": {
        ...         "type": "Point",
        ...         "coordinates": [71.45343, 12.523432]
        ...     },
        ...     "first_seen_at": "UNIX_TIMESTAMP",
        ...     "last_seen_at": "UNIX_TIMESTAMP",
        ...     "value": "regulatory--no-parking--g2",
        ...     "id": "FEATURE_ID",
        ...     "image_id": "IMAGE_ID"
        ... }

    :param geojson: The GeoJSON to flatten
    :type geojson: dict

    :return: A flattened GeoJSON
    :rtype: dict

    Note,
        1. The `geometry` key is always present in the output
        2. The properties are flattened to the following keys:
            - "first_seen_at"   (optional)
            - "last_seen_at"    (optional)
            - "value"           (optional)
            - "id"              (required)
            - "image_id"        (optional)
            - etc.
        3. If the 'geometry` type is `Point`, two more properties will be added:
            - "longitude"
            - "latitude"

    *TODO*: Further testing needed with different geometries, e.g., Polygon, etc.
    """

    for feature in geojson["features"]:
        # Check if the geometry is a Point
        if feature["geometry"]["type"] == "Point":
            # Add longitude and latitude properties to the feature
            feature["properties"]["longitude"] = feature["geometry"]["coordinates"][0]
            feature["properties"]["latitude"] = feature["geometry"]["coordinates"][1]

    # Return the flattened geojson
    return [
        {"geometry": _feature["geometry"], **_feature["properties"]}
        for _feature in geojson["features"]
    ]


def geojson_to_polygon(geojson: dict) -> GeoJSON:
    """
    Converts a GeoJSON into a collection of only geometry coordinates for the purpose of
    checking whether a given coordinate point exists within a shapely polygon

    From::

        >>> {
        ...     "type": "FeatureCollection",
        ...     "features": [
        ...         {
        ...             "geometry": {
        ...                 "coordinates": [
        ...                     -80.13069927692413,
        ...                     25.78523699486192
        ...                 ],
        ...                 "type": "Point"
        ...             },
        ...             "properties": {
        ...                 "first_seen_at": 1422984049000,
        ...                 "id": 481978503020355,
        ...                 "last_seen_at": 1422984049000,
        ...                 "value": "object--street-light"
        ...             },
        ...             "type": "Feature"
        ...         },
        ...         {
        ...             "geometry": {
        ...                 "coordinates": [
        ...                     -80.13210475444794,
        ...                     25.78362849816017
        ...                 ],
        ...                 "type": "Point"
        ...             },
        ...             "properties": {
        ...                 "first_seen_at": 1423228306666,
        ...                 "id": 252538103315239,
        ...                 "last_seen_at": 1423228306666,
        ...                 "value": "object--street-light"
        ...             },
        ...             "type": "Feature"
        ...         },
        ...         ...
        ...     ]
        ... }

    To::

        >>> {
        ... "type": "FeatureCollection",
        ... "features": [
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
        ...                         [
        ...                             7.27020263671875,
        ...                             43.69419030566581
        ...                         ],
        ...                         ...
        ...                     ]
        ...                 ]
        ...             }
        ...         }
        ...     ]
        ... }

    :param geojson: The input GeoJSON
    :type geojson: dict

    :return: A geojson of the format mentioned under 'To'
    :rtype: dict
    """

    return GeoJSON(
        geojson={
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "properties": {},
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [
                            [
                                # Double listed on purpose. See above example under 'To'
                                feature["geometry"]["coordinates"]
                                for feature in geojson["features"]
                            ]
                        ],
                    },
                }
            ],
        }
    )


def flatten_dictionary(
    data: typing.Union[dict, MutableMapping],
    parent_key: str = "",
    sep: str = "_",
) -> dict:
    """
    Flattens dictionaries

    From::

        >>> {'mpy-or': {'extent': 4096, 'version': 2, 'features': [{'geometry': {'type':
        ... 'Polygon', 'coordinates': [[[2402, 2776], [2408, 2776]]]}, 'properties': {}, 'id': 1,
        ... 'type': 3}]}}

    To::

        >>> {'mpy-or_extent': 4096, 'mpy-or_version': 2, 'mpy-or_features': [{'geometry':
        ... {'type':'Polygon', 'coordinates': [[[2402, 2776], [2408, 2776]]]}, 'properties':
        ... {}, 'id': 1,'type': 3}]}

    :param data: The dictionary itself
    :type data: dict

    :param parent_key: The root key to start from
    :type parent_key: str

    :param sep: The separator
    :type sep: str

    :return: A flattened dictionary
    :rtype: dict
    """

    # Final results
    items = []

    # Traversing dictionary items
    for key, value in data.items():

        # Getting the new key
        new_key = parent_key + sep + key if parent_key else key

        # Checking value instance
        # MutableMapping makes this Python 2.6+ compatible
        if isinstance(value, MutableMapping):

            # Extending items list
            items.extend(flatten_dictionary(value, new_key, sep=sep).items())

        # If not instance
        else:

            # Append to the list
            items.append((new_key, value))

    # Converting to dictionary, and returning
    return dict(items)


def normalize_list(coordinates: list, width: int = 4096, height: int = 4096) -> list:
    """
    Normalizes a list of coordinates with the respective width and the height

    From::

        >>> [[[2402, 2776], [2408, 2776]]]

    To::

        >>> normalize_list(coordinates)
        ... # [[[0.58642578125, 0.677734375], [0.587890625, 0.677734375]]]

    :param coordinates: The coordinate list to normalize
    :type coordinates: list

    :param width: The width of the coordinates to normalize with, defaults to 4096
    :type width: int

    :param height: The height of the coordinates to normalize with, defaults to 4096
    :type height: int

    :return: The normalized list
    :rtype: list
    """

    # Extracting the list from the coordinates
    coordinates = coordinates[0]

    # Initializing the coordinate list
    new_coordinates = []

    # Going through each coordinate pair
    for coordinate_pair in coordinates:

        # If it is already normalized ...
        if 0 <= coordinate_pair[0] <= 1 and 0 <= coordinate_pair[1] <= 1:
            # ... then append as is
            new_coordinates.append(coordinate_pair)

        # Appending the coordinates
        new_coordinates.append(
            # Appending a list pair of the width, height
            [coordinate_pair[0] / width, coordinate_pair[1] / height]
        )

    # Returning the results
    return [new_coordinates]


def decode_pixel_geometry(
    base64_string: str, normalized: bool = True, width: int = 4096, height: int = 4096
) -> dict:
    """
    Decodes the pixel geometry, and return the coordinates, which can be specified to be
    normalized

    :param base64_string: The pixel geometry encoded as a vector tile
    :type base64_string: str

    :param normalized: If normalization is required, defaults to True
    :type normalized: bool

    :param width: The width of the pixel geometry, defaults to 4096
    :type width: int

    :param height: The height of the pixel geometry, defaults to 4096
    :type height: int

    :return: A dictionary with coordinates as key, and value as the normalized list
    :rtype: list
    """

    # The data retrieved after being decoded as base64
    data = base64.decodebytes(base64_string.encode("utf-8"))

    # Getting the results from mapbox_vector_tile
    result = mapbox_vector_tile.decode(data)

    # Flattening the resultant dictionary
    flattened = flatten_dictionary(result)

    # Init geometry var
    geometry = None

    # For key in the flattened keys
    for key in flattened.keys():

        # If the key contains the word "features"
        if "features" in key:
            # Get the geometry
            geometry = flattened[key][0]

    # Extract the coordinate list
    coordinate_list = geometry["geometry"]["coordinates"]

    # Return output
    return (
        # Return coordinates as normalized values, if normalized is true
        {"coordinates": normalize_list(coordinate_list, width=width, height=height)}
        if normalized
        # Else return un-normalized coordinates
        else {"coordinates": coordinate_list}
    )


def decode_pixel_geometry_in_geojson(
    geojson: typing.Union[dict, GeoJSON],
    normalized: bool = True,
    width: int = 4096,
    height: int = 4096,
) -> GeoJSON:
    """
    Decodes all the pixel_geometry

    :param geojson: The GeoJSON representation to be decoded

    :param normalized: If normalization is required, defaults to True
    :type normalized: bool

    :param width: The width of the pixel geometry, defaults to 4096
    :type width: int

    :param height: The height of the pixel geometry, defaults to 4096
    :type height: int
    """

    # If geojson is of type GeoJSON, convert to dict
    if isinstance(geojson, GeoJSON):
        geojson: dict = geojson.to_dict()

    # Remove pass by reference for idempotent principle
    data = geojson.copy()

    # Iterate through the features
    for feature in data["features"]:
        # For the pixel_geometry feature, decode it
        feature["properties"]["pixel_geometry"] = decode_pixel_geometry(
            # Get the base64_string
            base64_string=feature["properties"]["pixel_geometry"],
            # Normalization variable
            normalized=normalized,
            # Width param
            width=width,
            # Height param
            height=height,
        )

    # Return output as GeoJSON
    return GeoJSON(geojson=data)


def coord_or_list_to_dict(data: Union[Coordinates, list, dict]) -> dict:
    """
    Converts a Coordinates object or a coordinates list to a dictionary

    :param data: The coordinates to convert
    :type data: Union[Coordinates, list]

    :return: The dictionary representation of the coordinates
    :rtype: dict
    """

    if isinstance(data, dict):
        return data

    data_copy = None

    # If data is a list, convert to dict
    if isinstance(data, list):
        data_copy = {"lng": data[0], "lat": data[1]}

    # if data is a Coordinate object, convert to dict
    if isinstance(data, Coordinates):
        data_copy = data.to_dict()

    # Return the dictionary
    return data_copy


def polygon_feature_to_bbox_list(
    polygon: dict, is_bbox_list_required: bool = True
) -> typing.Union[list, dict]:
    """
    Converts a polygon to a bounding box

    The polygon below has been obtained from https://geojson.io/. If you have a polygon,
    with only 4 array elements, then simply take the first element and append it to the
    coordinates to obtain the below example.

    Usage::

        >>> from mapillary.utils.format import polygon_feature_to_bbox_list
        >>> bbox = polygon_feature_to_bbox_list(polygon={
        ...     "type": "Feature",
        ...     "properties": {},
        ...     "geometry": {
        ...         "type": "Polygon",
        ...         "coordinates": [
        ...             [
        ...                 [
        ...                   48.1640625,
        ...                   38.41055825094609
        ...                 ],
        ...                 [
        ...                   62.22656249999999,
        ...                   38.41055825094609
        ...                 ],
        ...                 [
        ...                   62.22656249999999,
        ...                   45.336701909968134
        ...                 ],
        ...                 [
        ...                   48.1640625,
        ...                   45.336701909968134
        ...                 ],
        ...                 [
        ...                   48.1640625,
        ...                   38.41055825094609
        ...                 ]
        ...             ]
        ...        ]
        ... })
        >>> bbox
        ... [62.22656249999999, 48.1640625, 38.41055825094609, 45.336701909968134]

    :param polygon: The polygon to convert
    :type polygon: dict

    :param is_bbox_list_required: Flag if bbox is required as a list. If true, returns a list,
    else returns a dict
    :type is_bbox_list_required: bool
    :default is_bbox_list_required: True

    :return: The bounding box
    :rtype: typing.Union[list, dict]
    """

    west = polygon["geometry"]["coordinates"][0][1][0]
    south = polygon["geometry"]["coordinates"][0][0][0]
    east = polygon["geometry"]["coordinates"][0][0][1]
    north = polygon["geometry"]["coordinates"][0][2][1]

    if is_bbox_list_required:
        return [west, south, east, north]

    return {"west", west, "south", south, "east", east, "north", north}


def bbox_to_polygon(bbox: typing.Union[list, dict]) -> dict:
    """
    Converts a bounding box dictionary to a polygon

    Usage::

        >>> from mapillary.utils.format import bbox_to_polygon
        >>> bbox = [62.22656249999999, 48.1640625, 38.41055825094609, 45.336701909968134]
        >>> polygon = bbox_to_polygon(bbox=bbox)
        >>> polygon
        ... {
        ...     "type": "Feature",
        ...     "properties": {},
        ...     "geometry": {
        ...         "type": "Polygon",
        ...         "coordinates": [
        ...             [
        ...                 [
        ...                   48.1640625,
        ...                   38.41055825094609
        ...                 ],
        ...                 [
        ...                   62.22656249999999,
        ...                   38.41055825094609
        ...                 ],
        ...                 [
        ...                   62.22656249999999,
        ...                   45.336701909968134
        ...                 ],
        ...                 [
        ...                   48.1640625,
        ...                   45.336701909968134
        ...                 ],
        ...                 [
        ...                   48.1640625,
        ...                   38.41055825094609
        ...                 ]
        ...             ]
        ...        ]
        ... })

    :param bbox: The bounding box to convert
    :type bbox: dict

    :return: The polygon
    :rtype: dict
    """

    # Initializing the polygon
    polygon = {
        "type": "Feature",
        "properties": {},
        "geometry": {
            "type": "Polygon",
            "coordinates": [[[0, 0], [0, 0], [0, 0], [0, 0], [0, 0]]],
        },
    }

    # If bbox is a list, convert to dict
    if isinstance(bbox, list):
        bbox = {"west": bbox[0], "south": bbox[1], "east": bbox[2], "north": bbox[3]}

    # For the top left to the bottom left
    polygon["geometry"]["coordinates"][0][0] = [bbox["south"], bbox["east"]]

    # from the bottom left to the bottom right
    polygon["geometry"]["coordinates"][0][1] = [bbox["west"], bbox["east"]]

    # from the bottom right to the top right
    polygon["geometry"]["coordinates"][0][2] = [bbox["west"], bbox["north"]]

    # from the top right to the top left
    polygon["geometry"]["coordinates"][0][3] = [bbox["south"], bbox["north"]]

    # from the top left to the top left
    polygon["geometry"]["coordinates"][0][4] = [bbox["south"], bbox["east"]]

    # Returning the results
    return polygon
