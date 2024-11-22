# Copyright (c) Facebook, Inc. and its affiliates. (http://www.facebook.com)
# -*- coding: utf-8 -*-
"""Mapillary GeoJSON model implementation.

This module contains classes for working with GeoJSON data from the Mapillary API.
It provides classes to represent GeoJSON features, properties, coordinates and geometry.

For more information about the API, please check out
https://www.mapillary.com/developer/api-documentation/.

Copyright:
    (c) 2021 Facebook
License:
    MIT LICENSE
"""

# Package
import json

# # Exceptions
from zensvi.download.mapillary.models.exceptions import InvalidOptionError

# Local


class Properties:
    """Class representing properties in a GeoJSON feature.

    Args:
        *properties: Variable length list of property dictionaries to add.
        **kwargs: Arbitrary keyword arguments to set as properties.

    Returns:
        Properties: A new Properties instance.

    Raises:
        InvalidOptionError: If properties argument is not a dictionary.
    """

    def __init__(self, *properties, **kwargs) -> None:
        """Initialize Properties with the given property values.

        Args:
            *properties: Variable length list of property dictionaries to add.
            **kwargs: Arbitrary keyword arguments to set as properties.
        """
        # Validate that the geojson passed is indeed a dictionary
        if not isinstance(properties, dict):
            # Raise InvalidOptionError
            InvalidOptionError(
                # The parameter that caused the exception
                param="Properties.__init__.properties",
                # The invalid value passed
                value=properties,
                # The keys that should be passed instead
                options=["dict"],
            )

        for item in properties:
            for key in item:
                setattr(self, key, item[key])
        for key in kwargs:
            setattr(self, key, kwargs[key])

    def to_dict(self):
        """Convert Properties to a dictionary.

        Returns:
            dict: Dictionary representation of the properties.
        """
        attr_representation = [key for key in dir(self) if not key.startswith("__") and key != "to_dict"]

        return {key: getattr(self, key) for key in attr_representation}

    def __str__(self):
        """Return the informal string representation of the Properties.

        Returns:
            str: String representation of properties as key-value pairs.
        """
        attr_representation = [key for key in dir(self) if not key.startswith("__") and key != "to_dict"]

        attr_key_value_pair = {key: getattr(self, key) for key in attr_representation}

        return f"{attr_key_value_pair}"

    def __repr__(self):
        """Return the formal string representation of the Properties.

        Returns:
            str: String representation of properties as key-value pairs.
        """
        attr_representation = [key for key in dir(self) if not key.startswith("__") and key != "to_dict"]

        attr_key_value_pair = {key: getattr(self, key) for key in attr_representation}

        return f"{attr_key_value_pair}"


class Coordinates:
    """Class representing coordinates in a GeoJSON geometry.

    Args:
        longitude (float): The longitude coordinate.
        latitude (float): The latitude coordinate.

    Returns:
        Coordinates: A new Coordinates instance.

    Raises:
        InvalidOptionError: If longitude or latitude are not floats.
    """

    def __init__(self, longitude: float, latitude: float) -> None:
        """Initialize Coordinates with longitude and latitude values.

        Args:
            longitude (float): The longitude coordinate.
            latitude (float): The latitude coordinate.

        Raises:
            InvalidOptionError: If longitude or latitude are not floats.
        """
        # Validate that the longitude passed is indeed a float
        if not isinstance(longitude, float):
            # Raise InvalidOptionError
            InvalidOptionError(
                # The parameter that caused the exception
                param="Coordinates.__init__.longitude",
                # The invalid value passed
                value=longitude,
                # The keys that should be passed instead
                options=["float"],
            )

        # Validate that the latitude passed is indeed a float
        if not isinstance(latitude, float):
            # Raise InvalidOptionError
            InvalidOptionError(
                # The parameter that caused the exception
                param="Coordinates.__init__.latitude",
                # The invalid value passed
                value=latitude,
                # The keys that should be passed instead
                options=["float"],
            )

        self.longitude = longitude
        self.latitude = latitude

    def to_list(self):
        """Convert coordinates to a list.

        Returns:
            list: Coordinates as [longitude, latitude].
        """
        return [self.longitude, self.latitude]

    def to_dict(self):
        """Convert coordinates to a dictionary.

        Returns:
            dict: Coordinates as {"lng": longitude, "lat": latitude}.
        """
        return {"lng": self.longitude, "lat": self.latitude}

    def __str__(self):
        """Return the informal string representation of the Coordinates.

        Returns:
            str: Comma-separated longitude and latitude values.
        """
        return f"{self.longitude}, {self.latitude}"

    def __repr__(self) -> str:
        """Return the formal string representation of the Coordinates.

        Returns:
            str: Comma-separated longitude and latitude values.
        """
        return f"{self.longitude}, {self.latitude}"


class Geometry:
    """Class representing geometry in a GeoJSON feature.

    Args:
        geometry (dict): Dictionary containing geometry type and coordinates.

    Returns:
        Geometry: A new Geometry instance.

    Raises:
        InvalidOptionError: If geometry is not a dictionary.
    """

    def __init__(self, geometry: dict) -> None:
        """Initialize Geometry with type and coordinates.

        Args:
            geometry (dict): Dictionary containing geometry type and coordinates.

        Raises:
            InvalidOptionError: If geometry is not a dictionary.
        """
        # Validate that the geojson passed is indeed a dictionary
        if not isinstance(geometry, dict):
            # Raise InvalidOptionError
            InvalidOptionError(
                # The parameter that caused the exception
                param="Geometry.__init__.geometry",
                # The invalid value passed
                value=geometry,
                # The keys that should be passed instead
                options=["dict"],
            )

        # Setting the type of the selected geometry
        self.type: str = geometry["type"]

        # Setting the coordinates of the geometry
        self.coordinates: Coordinates = Coordinates(geometry["coordinates"][0], geometry["coordinates"][1])

    def to_dict(self):
        """Convert geometry to a dictionary.

        Returns:
            dict: Dictionary with type and coordinates list.
        """
        return {"type": self.type, "coordinates": self.coordinates.to_list()}

    def __str__(self):
        """Return the informal string representation of the Geometry.

        Returns:
            str: String representation with type and coordinates.
        """
        return f"{{'type': {self.type}, 'coordinates': {self.coordinates.to_list()}}}"

    def __repr__(self):
        """Return the formal string representation of the Geometry.

        Returns:
            str: String representation with type and coordinates.
        """
        return f"{{'type': {self.type}, 'coordinates': {self.coordinates.to_list()}}}"


class Feature:
    """Class representing a feature in a GeoJSON FeatureCollection.

    Args:
        feature (dict): Dictionary containing feature properties and geometry.

    Returns:
        Feature: A new Feature instance.

    Raises:
        InvalidOptionError: If feature is not a dictionary.
    """

    def __init__(self, feature: dict) -> None:
        """Initialize Feature with geometry and properties.

        Args:
            feature (dict): Dictionary containing feature properties and geometry.

        Raises:
            InvalidOptionError: If feature is not a dictionary.
        """
        # Validate that the geojson passed is indeed a dictionary
        if not isinstance(feature, dict):
            # If not, raise `InvalidOptionError`
            InvalidOptionError(
                # The parameter that caused the exception
                param="Feature.__init__.feature",
                # The invalid value passed
                value=feature,
                # The type of value that should be passed instead
                options=["dict"],
            )

        # Setting the type of the selected FeatureList
        self.type = "Feature"

        # Setting the `geometry` property
        self.geometry = Geometry(feature["geometry"])

        # Setting the `properties` property
        self.properties = Properties(feature["properties"])

    def to_dict(self) -> dict:
        """Convert feature to a dictionary.

        Returns:
            dict: Dictionary containing type, geometry and properties.
        """
        return {
            "type": self.type,
            "geometry": self.geometry.to_dict(),
            "properties": self.properties.to_dict(),
        }

    def __str__(self) -> str:
        """Return the informal string representation of the Feature.

        Returns:
            str: String representation with type, geometry and properties.
        """
        return (
            f"{{" f"'type': '{self.type}', " f"'geometry': {self.geometry}, " f"'properties': {self.properties}" f"}}"
        )

    def __repr__(self) -> str:
        """Return the formal string representation of the Feature.

        Returns:
            str: String representation with type, geometry and properties.
        """
        return f"{{" f"'type': {self.type}, " f"'geometry': {self.geometry}, " f"'properties': {self.properties}" f"}}"

    def __hash__(self):
        """Generate hash value for Feature.

        Returns:
            int: Hash based on type, coordinates and capture time.
        """
        # Create a unique hash based on an immutable representation of the feature
        return hash(
            (
                self.type,
                (
                    self.geometry.coordinates.latitude,
                    self.geometry.coordinates.longitude,
                ),
                self.properties.captured_at,
            )
        )

    def __eq__(self, other):
        """Compare Feature equality.

        Args:
            other: Another Feature instance to compare with.

        Returns:
            bool: True if Features have same type, coordinates and capture time.
        """
        # Define equality based on type, coordinates, and other properties
        return (
            self.type == other.type
            and (
                self.geometry.coordinates.latitude,
                self.geometry.coordinates.longitude,
            )
            == (
                other.geometry.coordinates.latitude,
                other.geometry.coordinates.longitude,
            )
            and self.properties.captured_at == other.properties.captured_at
        )


class GeoJSON:
    """Class representing a complete GeoJSON object.

    Args:
        geojson (dict): Dictionary containing GeoJSON data.

    Returns:
        GeoJSON: A new GeoJSON instance.

    Raises:
        InvalidOptionError: If geojson is not a dictionary or has invalid keys.

    Example:
        >>> import mapillary as mly
        >>> from models.geojson import GeoJSON
        >>> mly.interface.set_access_token('MLY|XXX')
        >>> data = mly.interface.get_image_close_to(longitude=31, latitude=31)
        >>> geojson = GeoJSON(geojson=data)
        >>> type(geojson)
        ... <class 'mapillary.models.geojson.GeoJSON'>
    """

    def __init__(self, geojson: dict) -> None:
        """Initialize GeoJSON with type and features.

        Args:
            geojson (dict): Dictionary containing GeoJSON data.

        Raises:
            InvalidOptionError: If geojson is not a dictionary or has invalid keys.
        """
        # Validate that the geojson passed is indeed a dictionary
        if isinstance(geojson, dict):

            # The GeoJSON should only contain the keys of `type`, `features`, if not empty,
            # raise exception
            if [key for key in geojson.keys() if key not in ["type", "features"]]:
                # Raise InvalidOptionError
                InvalidOptionError(
                    # The parameter that caused the exception
                    param="GeoJSON.__init__.geojson",
                    # The invalid value passed
                    value=geojson,
                    # The keys that should be passed instead
                    options=["type", "features"],
                )

        # If the GeoJSON is not of type dictionary
        else:

            # Raise InvalidOptionError
            InvalidOptionError(
                # The parameter that caused the exception
                param="GeoJSON.__init__.geojson",
                # The invalid value passed
                value=geojson,
                # The keys that should be passed instead
                options=["type", "features"],
            )

        # Validate that the geojson passed is indeed a dictionary
        if not isinstance(geojson["features"], list):
            # If not, raise InvalidOptionError
            InvalidOptionError(
                # The parameter that caused the exception
                param="FeatureList.__init__.geojson['features']",
                # The invalid value passed
                value=geojson["features"],
                # The type of the value that should be passed
                options=["list"],
            )

        # Setting the type parameter
        self.type: str = geojson["type"]

        # Setting the list of features
        self.features: list = (
            [Feature(feature=feature) for feature in geojson["features"]]
            if (geojson["features"] != []) or (geojson["features"] is not None)
            else []
        )

        # Convert existing features to a set for faster lookup
        self.features_set = set(self.features)

    def append_features(self, features: list) -> None:
        """Append multiple features to the GeoJSON.

        Args:
            features (list): List of feature dictionaries to append.
        """
        # Iterating over features
        for feature in features:

            # Appending the feature to the GeoJSON
            self.append_feature(feature)

    def append_feature(self, feature_inputs: dict) -> None:
        """Append a single feature to the GeoJSON.

        Args:
            feature_inputs (dict): Feature dictionary to append.
        """
        # Converting to a feature object
        feature = Feature(feature=feature_inputs)

        if feature not in self.features_set:
            self.features.append(feature)
            self.features_set.add(feature)

    def encode(self) -> str:
        """Serialize the GeoJSON object to a JSON string.

        Returns:
            str: JSON string representation of the GeoJSON.
        """
        return json.dumps(self.__dict__)

    def to_dict(self):
        """Convert GeoJSON to a dictionary.

        Returns:
            dict: Dictionary containing type and features list.
        """
        return {
            "type": self.type,
            "features": ([feature.to_dict() for feature in self.features] if self.features != [] else []),
        }

    def __str__(self):
        """Return the informal string representation of the GeoJSON.

        Returns:
            str: String representation with type and features.
        """
        return f"{{'type': '{self.type}', 'features': {self.features}}}"

    def __repr__(self):
        """Return the formal string representation of the GeoJSON.

        Returns:
            str: String representation with type and features.
        """
        return f"{{'type': '{self.type}', 'features': {self.features}}}"
