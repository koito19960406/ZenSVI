# Copyright (c) Facebook, Inc. and its affiliates. (http://www.facebook.com)
# -*- coding: utf-8 -*-
"""mapillary.utils.filter
======================

This module contains the filter utilies for high level filtering logic

- Copyright: (c) 2021 Facebook
- License: MIT LICENSE
"""

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

import haversine
from geojson import Feature, Point
from shapely.geometry import shape

# Package imports
from turfpy.measurement import bearing

# Local imports
from zensvi.download.mapillary.utils.time import date_to_unix_timestamp
from zensvi.utils.log import verbosity_tqdm

logger = logging.getLogger("pipeline-logger")


def pipeline_component(func, data: list, exception_message: str, args: list) -> list:
    """A pipeline component which is responsible for sending functional arguments over
    to the selected target function - throwing a warning in case of an exception.

    Args:
        func: The filter function to apply
        data: The list of features to filter
        exception_message: The exception message to print
        args: Arguments to pass to the filter function

    Returns:
        The filtered feature list

    Example:
        >>> # internally used in mapillary.utils.pipeline
    """
    try:
        return func(data, *args)
    except TypeError as exception:
        logger.warning(f"{exception_message}, {exception}. Arguments passed, {args}")
        return []


def max_captured_at(data: list, max_timestamp: str) -> list:
    """Selects only the feature items that are less than the max_timestamp.

    Args:
        data: The feature list
        max_timestamp: The UNIX timestamp as the max time

    Returns:
        Filtered feature list

    Example:
        >>> max_captured_at([{'type': 'Feature', 'geometry':
        ... {'type': 'Point', 'coordinates': [30.98594605922699, 30.003757307208872]}, 'properties':
        ... { ... }, ...}], '2020-05-23')
    """
    return [
        feature for feature in data if feature["properties"]["captured_at"] <= date_to_unix_timestamp(max_timestamp)
    ]


def min_captured_at(data: list, min_timestamp: str) -> list:
    """Selects only the feature items that are less than the min_timestamp.

    Args:
        data: The feature list
        min_timestamp: The UNIX timestamp as the max time

    Returns:
        Filtered feature list

    Example:
        >>> max_captured_at([{'type': 'Feature', 'geometry':
        ... {'type': 'Point', 'coordinates': [30.98594605922699, 30.003757307208872]}, 'properties':
        ... { ... }, ...}], '2020-05-23')
    """
    return [
        feature for feature in data if feature["properties"]["captured_at"] >= date_to_unix_timestamp(min_timestamp)
    ]


def features_in_bounding_box(data: list, bbox: dict) -> list:
    """Filter for extracting features only in a bounding box.

    Args:
        data: The features list to be checked
        bbox: Bounding box coordinates dictionary containing:
            west: Western boundary
            south: Southern boundary
            east: Eastern boundary
            north: Northern boundary

    Returns:
        Features that only exist within the specified bounding box

    Example:
        >>> bbox = {
        ...     'west': -122.4194,
        ...     'south': 37.7749,
        ...     'east': -122.4089,
        ...     'north': 37.7858
        ... }
        >>> features_in_bounding_box(data, bbox)
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
    """Filter the features based on the existence of a specified value in one of the
    properties.

    Args:
        data: The data to be filtered
        values: A list of values to filter by
        property: The specific parameter to look into

    Returns:
        A filtered feature list

    Note:
        *TODO*: Need documentation that lists the 'values', specifically, it refers to
        'value' *TODO*: under 'Detection', and 'Map feature', related to issue #65
    """
    return [feature for feature in data if feature["properties"].get(property) in values]


def existed_at(data: list, existed_at: str) -> list:
    """Filter features that existed after a specified time period.

    Args:
        data: The feature list
        existed_at: The UNIX timestamp

    Returns:
        The filtered feature list
    """
    return [feature for feature in data if feature["properties"]["first_seen_at"] > date_to_unix_timestamp(existed_at)]


def existed_before(data: list, existed_before: str) -> list:
    """Filter features that existed before a specified time period.

    Args:
        data: The feature list
        existed_before: The UNIX timestamp

    Returns:
        The filtered feature list
    """
    return [
        feature for feature in data if feature["properties"]["first_seen_at"] <= date_to_unix_timestamp(existed_before)
    ]


def haversine_dist(data: dict, radius: float, coords: list, unit: str = "m") -> list:
    """Returns features that are only in the radius specified using the Haversine
    distance.

    Args:
        data: The data to be filtered
        radius: Radius for coordinates to fall into
        coords: The input coordinates as [longitude, latitude]
        unit: Distance unit, one of 'ft', 'km', 'm', 'mi', 'nmi'

    Returns:
        A filtered feature list

    Note:
        Uses the haversine package: https://pypi.org/project/haversine/
    """
    # Define an empty list
    output = []

    # Go through the features
    for feature in data:

        # Reverse the order of the coords to (lon, lat) before feeding into the haversine function
        reversed_coords = coords[::-1]
        reversed_feature_coords = feature["geometry"]["coordinates"][::-1]

        # If the calculated haversine distance is less than the radius ...
        if haversine.haversine(reversed_coords, reversed_feature_coords, unit=unit) < radius:
            # ... append to the output
            output.append(feature)

    # Return the output
    return output


def image_type(data: list, image_type: str) -> list:
    """Filter images by their type (panoramic or flat).

    Args:
        data: The data to be filtered
        image_type: One of:
            'pano': Only panoramic images (is_pano == true)
            'flat': Only flat images (is_pano == false)
            'all': Both types

    Returns:
        A filtered feature list
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
    return [feature for feature in data if feature["properties"]["is_pano"] == bool_for_pano_filtering]


def organization_id(data: list, organization_ids: list) -> list:
    """Select only features from specific organizations.

    Args:
        data: The data to be filtered
        organization_ids: List of organization IDs to filter by

    Returns:
        A filtered feature list
    """
    return [
        # Feature only if
        feature
        # through the feature in the data
        for feature in data
        # if the found org_id is in the list of organization_ids
        if "organization_id" in feature["properties"] and feature["properties"]["organization_id"] in organization_ids
    ]


def sequence_id(data: list, ids: list) -> list:
    """Filter images by their sequence IDs.

    Args:
        data: The data to be filtered
        ids: List of sequence IDs to filter by

    Returns:
        A filtered feature list
    """
    return [feature for feature in data if feature["properties"]["sequence_id"] in ids]


def compass_angle(data: list, angles: tuple = (0.0, 360.0)) -> list:
    """Filter images by their compass angle range.

    Args:
        data: The data to be filtered
        angles: Tuple of (min_angle, max_angle) in degrees

    Returns:
        A filtered feature list

    Raises:
        ValueError: If angles tuple is invalid
    """
    if len(angles) != 2:
        raise ValueError("Angles must be a tuple of length 2")
    if angles[0] > angles[1]:
        raise ValueError("First angle must be less than second angle")
    if angles[0] < 0.0 or angles[1] > 360.0:
        raise ValueError("Angles must be between 0 and 360")

    return [feature for feature in data if angles[0] <= feature["properties"]["compass_angle"] <= angles[1]]


def is_looking_at(image_feature: Feature, look_at_feature: Feature) -> bool:
    """Check if an image is looking at a specific feature.

    Args:
        image_feature: The feature set of the image
        look_at_feature: The feature that is being looked at

    Returns:
        True if the image is looking at the feature, False otherwise
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
        abs(bearing(start=image_feature, end=look_at_feature) - image_feature["properties"]["compass_angle"]) % 360
    )

    # If diff > 310 OR diff < 50
    return 310 < diff or diff < 50


def by_look_at_feature(image: dict, look_at_feature: Feature) -> bool:
    """Check if an image is looking at a specific feature.

    Args:
        image: The feature dictionary
        look_at_feature: The feature to check if being looked at (WGS84 GIS feature)

    Returns:
        True if the image is looking at the feature, False otherwise
    """
    # Converting the coordinates in coords
    coords = [image["geometry"]["coordinates"][0], image["geometry"]["coordinates"][1]]

    # Getting the feature using `Feature`, `Point` from TurfPy
    image_feature = Feature(geometry=Point(coords, {"compass_angle": image["properties"]["compass_angle"]}))

    image_feature["properties"] = image["properties"]

    # Does the `image_feature` look at the `look_at_feature`?
    return is_looking_at(image_feature, look_at_feature)


def hits_by_look_at(data: list, at: dict) -> list:
    """Find features that look at specific coordinates.

    Args:
        data: List of features with an Image entity
        at: Dictionary containing coordinates:
            lng: longitude
            lat: latitude

    Returns:
        List of features looking at the specified coordinates

    Example:
        >>> coords = {'lng': -122.4194, 'lat': 37.7749}
        >>> hits_by_look_at(data, coords)
    """
    # Converting the `at` into a Feature object from TurfPy
    at_feature = Feature(geometry=Point((at["lng"], at["lat"])))

    return list(filter(lambda image: by_look_at_feature(image, at_feature), data))


def in_shape(data: list, boundary) -> list:
    """Filter features that lie within a shape boundary.

    Args:
        data: A feature list to be filtered
        boundary: Shapely geometry object defining the boundary

    Returns:
        List of features that fall within the boundary
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


def pipeline(data: dict, components: list, **kwargs) -> list:
    """Process features through multiple filters efficiently.

    Args:
        data: Dictionary containing features to filter
        components: List of filter components to apply
        **kwargs: Additional arguments including:
            max_workers: Number of workers for parallel processing
            verbosity: Level of verbosity for progress bars (0=none, 1=outer loops, 2=all loops). Defaults to 1.

    Returns:
        Filtered list of features

    Example:
        >>> pipeline(
        ...     data=data,
        ...     components=[
        ...         {"filter": "image_type", "image_type": "pano"},
        ...         {"filter": "organization_id", "organization_id": ["org1", "org2"]},
        ...     ],
        ...     max_workers=4,
        ...     verbosity=1
        ... )
    """
    __data = data.copy()["features"]

    # Initialize filter criteria with default values
    filter_criteria = {
        "max_captured_at": float("inf"),
        "min_captured_at": float("-inf"),
        "image_type": None,
        "organization_id": set(),
        "sequence_id": set(),
        "compass_angle": (0.0, 360.0),
        "in_shape": None,
    }

    # Update filter_criteria based on provided components
    for component in components:
        if component:
            filter_name = component["filter"]
            filter_value = component.get(filter_name)
            if filter_name in ["organization_id", "sequence_id"]:
                filter_criteria[filter_name].update(filter_value)
            else:
                filter_criteria[filter_name] = filter_value

    # Function to apply combined filters to a feature
    def apply_filters(feature):
        props = feature["properties"]
        if not (filter_criteria["min_captured_at"] <= props["captured_at"] <= filter_criteria["max_captured_at"]):
            return None
        if (
            (filter_criteria["image_type"] is not None)
            and (props["is_pano"] != (filter_criteria["image_type"] == "pano"))
            and (filter_criteria["image_type"] != "all")
        ):
            return None
        if (
            filter_criteria["organization_id"]
            and props.get("organization_id") not in filter_criteria["organization_id"]
        ):
            return None
        if filter_criteria["sequence_id"] and props.get("sequence_id") not in filter_criteria["sequence_id"]:
            return None
        if not (
            filter_criteria["compass_angle"][0] <= props.get("compass_angle", 0) <= filter_criteria["compass_angle"][1]
        ):
            return None

        # in_shape filter
        if filter_criteria["in_shape"]:
            polygon_list = filter_criteria["in_shape"]
            point = shape(feature["geometry"])
            # loop through each polygon in the list
            for polygon in polygon_list:
                if polygon.contains(point):
                    return feature
            return None

        return feature

    # Get verbosity level from kwargs or use default value of 1
    verbosity = kwargs.get("verbosity", 1)

    # Apply filters in parallel and display a progress bar based on verbosity
    with ThreadPoolExecutor(max_workers=kwargs.get("max_workers", None)) as executor:
        # Submit all tasks
        futures = [executor.submit(apply_filters, feature) for feature in __data]

        # Collect results as they are completed
        filtered_data = []
        for future in verbosity_tqdm(
            as_completed(futures), total=len(__data), desc="Filtering data", verbosity=verbosity, level=1
        ):
            result = future.result()
            if result:
                filtered_data.append(result)

    return filtered_data
