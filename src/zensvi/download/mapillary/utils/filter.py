# Copyright (c) Facebook, Inc. and its affiliates. (http://www.facebook.com)
# -*- coding: utf-8 -*-

"""
mapillary.utils.filter
======================

This module contains the filter utilies for high level filtering logic

- Copyright: (c) 2021 Facebook
- License: MIT LICENSE
"""

import logging

import haversine
from geojson import Point, Feature

# Local imports
from mapillary.utils.time import date_to_unix_timestamp
from shapely.geometry import shape

# Package imports
from turfpy.measurement import bearing

logger = logging.getLogger("pipeline-logger")


def pipeline_component(func, data: list, exception_message: str, args: list) -> list:
    """
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
    """

    try:
        return func(data, *args)
    except TypeError as exception:
        logger.warning(f"{exception_message}, {exception}. Arguments passed, {args}")
        return []


def pipeline(data: dict, components: list) -> list:
    """
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
    """

    # Python treats dict objects as passed reference, thus
    # in order to not modify the previous state, we make a local copy
    __data = data.copy()["features"]

    # A mapping of different filters possible
    function_mappings = {
        "filter_values": filter_values,
        "max_captured_at": max_captured_at,
        "min_captured_at": min_captured_at,
        "haversine_dist": haversine_dist,
        "image_type": image_type,
        "organization_id": organization_id,
        "features_in_bounding_box": features_in_bounding_box,
        "existed_at": existed_at,
        "existed_before": existed_before,
        "sequence_id": sequence_id,
        "compass_angle": compass_angle,
        "hits_by_look_at": hits_by_look_at,
        "in_shape": in_shape,
        # Simply add the mapping of a new function,
        # nothing else will really need to changed
    }

    # Going through each of the components
    for component in components:

        # If component is simply empty, continue to next
        # iteration
        if component == {}:
            continue

        # Send to pipeline component, return data to `__data`
        __data = pipeline_component(
            # Map function respectively using the function_mappings dictionary
            func=function_mappings[f'{component["filter"]}'],
            # Send over the data
            data=__data,
            # Specify the message on the exception thrown
            exception_message=f'[pipeline - {component["filter"]}] Filter not applied, '
            "exception thrown",
            # Except the filter name, select the rest as args
            args=tuple(list(component.values())[1:]),
        )

    # Return the data
    return __data


def max_captured_at(data: list, max_timestamp: str) -> list:
    """
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
    """

    return [
        feature
        for feature in data
        if feature["properties"]["captured_at"] <= date_to_unix_timestamp(max_timestamp)
    ]


def min_captured_at(data: list, min_timestamp: str) -> list:
    """
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
    """

    return [
        feature
        for feature in data
        if feature["properties"]["captured_at"] >= date_to_unix_timestamp(min_timestamp)
    ]


def features_in_bounding_box(data: list, bbox: dict) -> list:
    """
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
    """

    # define an empty geojson as output
    output = []

    # For each feature in the filtered_data
    for feature in data:

        # If feature exists in bounding box
        if (
            feature["geometry"]["coordinates"][0] < bbox["east"]
            and feature["geometry"]["coordinates"][0] > bbox["west"]
        ) and (
            feature["geometry"]["coordinates"][1] > bbox["south"]
            and feature["geometry"]["coordinates"][1] < bbox["north"]
        ):
            # Append feature to the output
            output.append(feature)

    return output


def filter_values(data: list, values: list, property: str = "value") -> list:
    """
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
    """

    return [
        feature for feature in data if feature["properties"].get(property) in values
    ]


def existed_at(data: list, existed_at: str) -> list:
    """
    Whether the first_seen_at properly existed after a specified time period

    :param data: The feature list
    :type data: list

    :param existed_at: The UNIX timestamp
    :type existed_at: str

    :return: The feature list
    :rtype: list
    """

    return [
        feature
        for feature in data
        if feature["properties"]["first_seen_at"] > date_to_unix_timestamp(existed_at)
    ]


def existed_before(data: list, existed_before: str) -> list:
    """
    Whether the first_seen_at properly existed before a specified time period

    :param data: The feature list
    :type data: list

    :param existed_before: The UNIX timestamp
    :type existed_before: str

    :return: A feature list
    :rtype: list
    """

    return [
        feature
        for feature in data
        if feature["properties"]["first_seen_at"]
        <= date_to_unix_timestamp(existed_before)
    ]


def haversine_dist(data: dict, radius: float, coords: list, unit: str = "m") -> list:
    """
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
    """

    # Define an empty list
    output = []

    # Go through the features
    for feature in data:

        # If the calculated haversince distance is less than the radius ...
        if (
            haversine.haversine(coords, feature["geometry"]["coordinates"], unit=unit)
            < radius
        ):
            # ... append to the output
            output.append(feature)

    # Return the output
    return output


def image_type(data: list, image_type: str) -> list:
    """
    The parameter might be 'all' (both is_pano == true and false), 'pano' (is_pano == true only),
    or 'flat' (is_pano == false only)

    :param data: The data to be filtered
    :type data: list

    :param image_type: Either 'pano' (True), 'flat' (False), or 'all' (None)
    :type image_type: str

    :return: A feature list
    :rtype: list
    """

    # Checking what kind of parameter is passed
    bool_for_pano_filtering = (
        # Return true if type == 'pano'
        True
        if image_type == "pano"
        # Else false if type == 'flat'
        else False
    )

    # Return the images based on image type
    return [
        feature
        for feature in data
        if feature["properties"]["is_pano"] == bool_for_pano_filtering
    ]


def organization_id(data: list, organization_ids: list) -> list:
    """
    Select only features that contain the specific organization_id

    :param data: The data to be filtered
    :type data: dict

    :param organization_ids: The oragnization id(s) to filter through
    :type organization_ids: list

    :return: A feature list
    :rtype: dict
    """

    return [
        # Feature only if
        feature
        # through the feature in the data
        for feature in data
        # if the found org_id is in the list of organization_ids
        if "organization_id" in feature["properties"]
        and feature["properties"]["organization_id"] in organization_ids
    ]


def sequence_id(data: list, ids: list) -> list:
    """
    Filter out images that do not have the sequence_id in the list of ids

    :param data: The data to be filtered
    :type data: list

    :param ids: The sequence id(s) to filter through
    :type ids: list

    :return: A feature list
    :rtype: list
    """

    return [feature for feature in data if feature["properties"]["sequence_id"] in ids]


def compass_angle(data: list, angles: tuple = (0.0, 360.0)) -> list:
    """
    Filter out images that do not lie within compass angle range

    :param data: The data to be filtered
    :type data: list

    :param angles: The compass angle range to filter through
    :type angle: tuple of floats

    :return: A feature list
    :rtype: list
    """

    if len(angles) != 2:
        raise ValueError("Angles must be a tuple of length 2")
    if angles[0] > angles[1]:
        raise ValueError("First angle must be less than second angle")
    if angles[0] < 0.0 or angles[1] > 360.0:
        raise ValueError("Angles must be between 0 and 360")

    return [
        feature
        for feature in data
        if angles[0] <= feature["properties"]["compass_angle"] <= angles[1]
    ]


def is_looking_at(image_feature: Feature, look_at_feature: Feature) -> bool:
    """
    Return whether the image_feature is looking at the look_at_feature

    :param image_feature: The feature set of the image
    :type image_feature: dict

    :param look_at_feature: The feature that is being looked at
    :type look_at_feature: dict

    :return: Whether the diff is greater than 310, or less than 50
    :rtype: bool
    """

    # Pano accessible via the `get_image_layer`
    # in config/api/vector_tiles.py
    if image_feature["properties"]["is_pano"]:
        return True

    # Compass angle accessible via the `get_image_layer`
    # in config/api/vector_tiles.py
    if image_feature["properties"]["compass_angle"] < 0:
        return False

    # Getting the difference between the two provided GeoJSONs and the compass angle
    diff: int = (
        abs(
            bearing(start=image_feature, end=look_at_feature)
            - image_feature["properties"]["compass_angle"]
        )
        % 360
    )

    # If diff > 310 OR diff < 50
    return 310 < diff or diff < 50


def by_look_at_feature(image: dict, look_at_feature: Feature) -> bool:
    """
    Filter through the given image features and return only features with the look_at_feature

    :param image: The feature dictionary
    :type image: dict

    :param look_at_feature: Feature description
    :type look_at_feature: A WGS84 GIS feature, TurfPy

    :return: Whether the given feature is looking at the `look_at_features`
    :rtype: bool
    """

    # Converting the coordinates in coords
    coords = [image["geometry"]["coordinates"][0], image["geometry"]["coordinates"][1]]

    # Getting the feature using `Feature`, `Point` from TurfPy
    image_feature = Feature(
        geometry=Point(coords, {"compass_angle": image["properties"]["compass_angle"]})
    )

    image_feature["properties"] = image["properties"]

    # Does the `image_feature` look at the `look_at_feature`?
    return is_looking_at(image_feature, look_at_feature)


def hits_by_look_at(data: list, at: dict) -> list:
    """
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
    """

    # Converting the `at` into a Feature object from TurfPy
    at_feature = Feature(geometry=Point((at["lng"], at["lat"])))

    return list(filter(lambda image: by_look_at_feature(image, at_feature), data))


def in_shape(data: list, boundary) -> list:
    """
    Whether the given feature list lies within the shape

    :param data: A feature list to be filtered
    :type data: list

    :param boundary: Shapely helper for determining existence of point within a boundary
    :type boundary:

    :return: A feature list
    :rtype: list
    """

    # Generating output format
    output = []

    # Iterating over features
    for feature in data:

        # Extracting point from geometry feature
        point = shape(feature["geometry"])

        # Checking if point falls within the boundary using shapely.geometry.point.point
        if boundary.contains(point):
            # If true, append to output features
            output.append(feature)

    # Return output
    return output
