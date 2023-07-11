# Copyright (c) Facebook, Inc. and its affiliates. (http://www.facebook.com)
# -*- coding: utf-8 -*-

"""
mapillary.models.api.entities
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This module contains the Adapter design for the Entities API of Mapillary API v4.

For more information, please check out https://www.mapillary.com/developer/api-documentation/.

- Copyright: (c) 2021 Facebook
- License: MIT LICENSE
"""

# Package imports
import mercantile
import typing
from vt2geojson.tools import vt_bytes_to_geojson

# Local imports

# # Models
from zensvi.download.mapillary.models.client import Client

# # Exception Handling
from zensvi.download.mapillary.models.exceptions import InvalidOptionError, LiteralEnforcementException

# # Config
from zensvi.download.mapillary.config.api.general import General

# Library imports
from requests import HTTPError


class GeneralAdapter(object):
    """
    General adaptor for using the API calls defined for the general module
    (mapillary.config.api.general)

    The GeneralAdaptor provides functions for getting preprocessed data from the API, through the
    API calls mentioned in the previously mentioned config.

    It performs parsing, property handling, easier logic for extracing information, focusing on
    adding a layer of abstraction by removing details of using `mercantile`, `ast`, et al, to
    focus on the inputs and outputs of functions

    Usage::

        >>> import mapillary
    """

    # layer types
    __LAYER_TYPES: typing.List[str] = [
        "overview",
        "sequence",
        "image",
        "map_feature",
        "traffic_sign",
    ]

    def __init__(self, *args: object) -> None:
        """Initializing GeneralAdaptor constuctor"""
        super().__init__(*args)

        # client object to deal with session and requests
        self.client = Client()

        # Setting the max zoom value
        self.__min_zoom = 0

        # Setting the min zoom value
        self.__max_zoom = 14

    def fetch_image_tiles(
        self,
        zoom: int,
        longitude: float,
        latitude: float,
        layer: str = "image",
    ) -> dict:
        """
        Get the tiles for a given image.

        :param zoom: Zoom level of the image.
        :type zoom: int

        :param longitude: Longitude of the image
        :type longitude: float

        :param latitude: Latitude of the image
        :type latitude: float

        :return: A dictionary containing the tiles for the image.
        :rtype: dict
        """

        # Perform validation checks
        self.__validation_checks(
            zoom=zoom,
            longitude=longitude,
            latitude=latitude,
            layer=layer,
        )

        # Get the tiles for the image
        return self.__preprocess_layer(
            longitude=longitude,
            latitude=latitude,
            zoom=zoom,
            feature_type="images",
            layer=layer,
        )

    def fetch_computed_image_tiles(
        self,
        zoom: int,
        longitude: float,
        latitude: float,
        layer: str = "image",
    ) -> dict:
        """
        Get the image type for a given image.

        :param zoom: The zoom to get the image type for.
        :type zoom: int

        :param longitude: The longitude of the image.
        :type longitude: float

        :param latitude: The latitude of the image.
        :type latitude: float

        :return: A dictionary containing the image type for the image.
        :rtype: dict
        """
        # Perform validation checks
        self.__validation_checks(
            zoom=zoom,
            longitude=longitude,
            latitude=latitude,
            layer=layer,
        )

        # Get the image type for the image
        return self.__preprocess_layer(
            longitude=longitude,
            latitude=latitude,
            zoom=zoom,
            feature_type="images",
            layer=layer,
            is_computed=True,
        )

    def fetch_map_features_tiles(
        self,
        zoom: int,
        longitude: float,
        latitude: float,
        layer: str = "image",
    ):
        """
        Get the map features for a given coordinate set

        :param zoom: The zoom value to get the map features for
        :type zoom: int

        :param longitude: The longitude of the image
        :type longitude: float

        :param latitude: The latitude of the image
        :type latitude: float

        :return: A dictionary containing the map features for the image.
        :rtype: dict
        """
        # Perform validation checks
        self.__validation_checks(
            zoom=zoom,
            longitude=longitude,
            latitude=latitude,
            layer=layer,
        )

        # Get the map features for the image
        return self.__preprocess_layer(
            longitude=longitude,
            latitude=latitude,
            zoom=zoom,
            feature_type="map_features__points",
            layer=layer,
        )

    def fetch_map_features_traffic_tiles(
        self, zoom: int, longitude: float, latitude: float, layer: str
    ):
        """
        Get the map feature traffic for a given coordinate set

        :param zoom: The zoom value to get the map features for
        :type zoom: int

        :param longitude: The longitude of the image
        :type longitude: float

        :param latitude: The latitude of the image
        :type latitude: float

        :return: A dictionary containing the map features for the image.
        :rtype: dict
        """
        # Perform validation checks
        self.__validation_checks(
            zoom=zoom,
            longitude=longitude,
            latitude=latitude,
            layer=layer,
        )

        # Get the map features for the image
        return self.__preprocess_layer(
            longitude=longitude,
            latitude=latitude,
            zoom=zoom,
            feature_type="map_features__traffic_signs",
            layer=layer,
        )

    def __preprocess_layer(
        self,
        longitude: float,
        latitude: float,
        zoom: int = 14,
        feature_type: str = "images",
        layer: str = "image",
        is_computed: bool = False,
    ) -> any:
        try:
            tile: mercantile.Tile = mercantile.tile(
                lng=longitude, lat=latitude, zoom=zoom
            )

            return vt_bytes_to_geojson(
                # Parameters appropriately
                b_content=self.client.get(
                    self.__preprocess_api_string(
                        # Turn coordinates into a tile
                        tile=mercantile.tile(lng=longitude, lat=latitude, zoom=zoom),
                        # the layer to retrieve from
                        layer=feature_type,
                        # is the layer computed
                        is_computed=is_computed,
                    )
                ).content,
                x=tile.x,
                y=tile.y,
                z=tile.z,
                layer=layer,
            )
        except HTTPError as e:
            raise HTTPError(e)

    def __preprocess_api_string(
        self,
        tile: mercantile.Tile,
        layer: str = "images",
        is_computed: bool = False,
    ) -> str:
        """
        Preprocess the API string for the given tile.

        :param tile: The tile to preprocess the API string for.
        :type tile: mercantile.Tile

        :param layer: The feature type to preprocess the API string for.
        :type layer: str

        :param is_computed: Whether the API string is for a computed image or not.
        :type is_computed: bool

        :return: The preprocessed API string.
        :rtype: str
        """

        if layer == "images":
            # Get the API string for the tile
            if is_computed:
                # If the API string is for a computed image
                return General.get_computed_image_type_tiles(
                    # The zoom value
                    z=tile.z,
                    # The longitude value
                    x=tile.x,
                    # The latitude value
                    y=tile.y,
                )
            else:
                # If the API string is for an image
                return General.get_image_type_tiles(
                    # The zoom value
                    z=tile.z,
                    # The longitude value
                    x=tile.x,
                    # The latitude value
                    y=tile.y,
                )

        elif layer == "map_features__points":
            # Get the API string for the tile
            return General.get_map_features_points_tiles(
                # The zoom value
                z=tile.z,
                # The longitude value
                x=tile.x,
                # The latitude value
                y=tile.y,
            )

        elif layer == "map_features__traffic_signs":
            # Get the API string for the tile
            return General.get_map_features_traffic_signs_tiles(
                # The zoom value
                z=tile.z,
                # The longitude value
                x=tile.x,
                # The latitude value
                y=tile.y,
            )

        else:
            raise InvalidOptionError(
                param="layer",
                value=layer,
                options=[
                    "images",
                    "map_features__points",
                    "map_features__traffic_signs",
                ],
            )

    @staticmethod
    def __check_parameters(
        longitude: float,
        latitude: float,
    ):
        """
        Range checking for the parameters of longitude, latitude, layer, zoom

        :param longitude: The longitude of the coordinates
        :type longitude: float

        :param latitude: The latitude of the coordinates
        :type latitude: float

        :return: A GeoJSON for that specific layer and the specified zoom level
        :rtype: dict
        """

        # Lng, Lat ranges, https://docs.mapbox.com/help/glossary/lat-lon/

        # If lng not in the range [-180, 180], inclusive
        if longitude <= -180 or longitude >= 180:
            # Raise exception
            raise InvalidOptionError(
                param="longitude", value=longitude, options=[-180, 180]
            )

        # If lat not in the range [-90, 90], inclusive
        if latitude <= -90 or latitude >= 90:
            # Raise exception
            raise InvalidOptionError(
                param="latitude", value=latitude, options=[-180, 180]
            )

    def __zoom_range_check(self, layer: str, zoom: int):

        """
        Checks for the correct zoom values for te specified layer

        :param layer: Either 'overview', 'sequence', 'image', or 'map'
        :type layer: str

        :param zoom: The zoom levels,
        :type zoom: int

        :raises InvalidOptionError: Invalid option passed

        :return: A GeoJSON for the return object
        :rtype: dict
        """

        # If zoom is not in the valid range of values
        if zoom < self.__min_zoom or zoom > self.__max_zoom:
            # Raise an exception for the invalid values passed
            raise InvalidOptionError(
                # Parameters accordingly
                param="zoom",
                value=zoom,
                options=[_ for _ in range(self.__min_zoom, self.__max_zoom + 1)],
            )

        # If layer was specified to be 'overview'
        if layer == "overview":

            # If zoom is not in the given range of values
            if zoom not in [_ for _ in range(self.__min_zoom, 5 + 1)]:
                # Raise an exception for the invalid zoom value
                raise InvalidOptionError(
                    # Parameters accordingly
                    param="zoom",
                    value=zoom,
                    options=[_ for _ in range(self.__min_zoom, 5 + 1)],
                )

        # If layer was specified to be 'sequence'
        elif layer == "sequence":

            # IF zoom not in the given range of values
            if zoom not in [_ for _ in range(6, 14 + 1)]:
                # Raise an exception for the invalid zoom value
                raise InvalidOptionError(
                    # Parameters accordingly
                    param="zoom",
                    value=zoom,
                    options=[_ for _ in range(6, 14 + 1)],
                )

        # If layer was specified to be 'image' or 'map'
        elif layer == "image" or layer == "map_feature" or layer == "traffic_sign":

            # If zoom was not 14
            if zoom != 14:
                # Raise an exception for the invalid zoom value
                raise InvalidOptionError(param="zoom", value=zoom, options=[14])

        # Else, an invalid layer string was passed
        else:

            # Raise a InvalidOptionError
            raise InvalidOptionError(
                # Parameters accordingly
                param="layer",
                value=layer,
                options=["overview", "sequence", "image"],
            )

    def __validation_checks(
        self,
        longitude: float,
        latitude: float,
        zoom: int,
        layer: str,
        layer_options: typing.List[str] = [],
    ) -> None:
        """
        Validation checks for the parameters of longitude, latitude, layer, zoom

        :param longitude: The longitude of the coordinates
        :type longitude: float

        :param latitude: The latitude of the coordinates
        :type latitude: float

        :param zoom: The zoom levels,
        :type zoom: int

        :param layer: Either 'overview', 'sequence', 'image', 'map_feature', or 'traffic_sign'
        :type layer: str

        :raises InvalidOptionError: Invalid option passed
        """
        # Check if the correct literals were passed to the function
        LiteralEnforcementException.enforce_literal(
            option_selected=layer,
            options=self.__LAYER_TYPES if layer_options == [] else layer_options,
            param="layer",
        )

        # Check the parameters for the correct range
        self.__check_parameters(longitude, latitude)

        # Check the zoom range for the specified layer
        self.__zoom_range_check(layer, zoom)
