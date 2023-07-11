# Copyright (c) Facebook, Inc. and its affiliates. (http://www.facebook.com)
# -*- coding: utf-8 -*-

"""
mapillary.models.api.vector_tiles
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This module contains the adapter design for the Vector Tiles API of Mapillary API v4, Vector tiles
provide an easy way to visualize vast amounts of data. Vector tiles support filtering and querying
rendered features. Mapillary vector tiles follow the Mapbox tile specification.

For more information, please check out https://www.mapillary.com/developer/api-documentation/.

- Copyright: (c) 2021 Facebook
- License: MIT LICENSE
"""

# Package imports
from vt2geojson.tools import vt_bytes_to_geojson
import mercantile

# Local imports
# # Config
from mapillary.config.api.vector_tiles import VectorTiles

# # Client import
from mapillary.models.client import Client

# # Exception handling
from mapillary.models.exceptions import InvalidOptionError

# # Models
from mapillary.models.geojson import GeoJSON


class VectorTilesAdapter(object):
    """
    Adapter model for dealing with the VectorTiles API, through the DRY principle. The
    VectorTilesAdapter class can be instantiated in the controller modules, providing an
    abstraction layer that uses the Client class, endpoints provided by the API v4 under
    `/config/api/vector_tiles.py`.

    It performs parsing, handling of layers, properties, and fields to make it easier to
    write higher level logic for extracing information, and lets developers to focus only
    on writing the high level business logic without having to repeat the process of parsing
    and using libraries such as `mercantile`, 'vt2geojson' and others, caring only about
    inputs/outputs

    Usage::

        >>> import mapillary
        >>> from mapillary.models.api.vector_tiles import VectorTilesAdapter
        >>> latitude, longitude = 30, 31
        >>> VectorTilesAdapter().fetch_layer(layer="image", zoom=14, longitude=longitude,
        ...     latitude=latitude,
        ... )
        >>> VectorTilesAdapter().fetch_layer(layer="sequence", zoom=10, longitude=longitude,
        ...     latitude=latitude,
        ... )
        >>> VectorTilesAdapter().fetch_layer(layer="overview", zoom=3, longitude=longitude,
        ...     latitude=latitude,
        ... )
    """

    # FOR DEVELOPERS
    # Future changes here will most likely work on the following aspects,
    # 1. Adjusting zoom levels
    # 1.1. Most likely changes in __zoom_range_check
    # 2. The zoom levels themselves
    # 2.1. Most likely changes in __init__
    # 3. Preprocessing steps depending on the layer targeted
    # 3.1. Most likely changes in __preprocess_layer, __preprocess_computed_layer,
    # __preprocess_features

    def __init__(self) -> None:
        """Initializing VectorTilesAdapter constructor"""

        # Initialize client object
        self.client = Client()

        # * See "FOR DEVELOPERS (2, 2.1)"

        # Setting the max zoom value
        self.__min_zoom = 0

        # Setting the min zoom value
        self.__max_zoom = 14

    def fetch_layer(
        self, layer: str, longitude: float, latitude: float, zoom: int = 14
    ) -> dict:
        """
        Fetches an image tile layer depending on the coordinates, and the layer selected
        along with the zoom level

        :param layer: Either 'overview', 'sequence', 'image'
        :type layer: str

        :param longitude: The longitude of the coordinates
        :type longitude: float

        :param latitude: The latitude of the coordinates
        :type latitude: float

        :param zoom: The zoom level, [0, 14], inclusive
        :type zoom: int

        :return: A GeoJSON for that specific layer and the specified zoom level
        :rtype: dict
        """

        # Checking if the correct parameters are passed
        VectorTilesAdapter.__check_parameters(longitude=longitude, latitude=latitude)

        # Check for the correct zoom values against the layer specified
        self.__zoom_range_check(layer=layer, zoom=zoom)

        # Return the results of the layer after preprocessing steps
        return self.__preprocess_layer(
            # The layer to retrieve from
            layer=layer,
            # Turn coordinates into a tile
            tile=mercantile.tile(lng=longitude, lat=latitude, zoom=zoom),
            # The zoom level
            zoom=zoom,
        )

    def fetch_computed_layer(
        self, layer: str, zoom: int, longitude: float, latitude: float
    ):
        """
        Same as `fetch_layer`, but gets in return computed tiles only.
        Depends on the layer, zoom level, longitude and the latitude specifications

        :param layer: Either 'overview', 'sequence', 'image'
        :type layer: str

        :param zoom: The zoom level, [0, 14], inclusive
        :type zoom: int

        :param longitude: The longitude of the coordinates
        :type longitude: float

        :param latitude: The latitude of the coordinates
        :type latitude: float

        :return: A GeoJSON for that specific layer and the specified zoom level
        :rtype: dict
        """

        # Checking if the correct parameters are passed
        VectorTilesAdapter.__check_parameters(longitude=longitude, latitude=latitude)

        # Check for the correct zoom values against the layer specified
        self.__zoom_range_check(layer=layer, zoom=zoom)

        # Return the results of the layer after preprocessing steps
        return self.__preprocess_computed_layer(
            # The layer to retrieve from
            layer=layer,
            # Turn coordinates into a tile
            tile=mercantile.tile(lng=longitude, lat=latitude, zoom=zoom),
            # The zoom level
            zoom=zoom,
        )

    def fetch_features(
        self, feature_type: str, zoom: int, longitude: float, latitude: float
    ):
        """
        Fetches specified features from the coordinates with the appropriate zoom level

        :param feature_type: Either `point`, or `tiles`
        :type feature_type: str

        :param zoom: The zoom level
        :type zoom: int

        :param longitude: The longitude of the coordinates
        :type longitude: float

        :param latitude: The latitude of the coordinates
        :type latitude: float

        :return: A GeoJSON for that specific layer and the specified zoom level
        :rtype: dict
        """

        # Checking if the correct parameters are passed
        VectorTilesAdapter.__check_parameters(longitude=longitude, latitude=latitude)

        # Check for the correct zoom values against the layer specified
        self.__zoom_range_check(layer="map", zoom=zoom)

        # Return the results of the layer after preprocessing steps
        return self.__preprocess_features(
            # The feature type to retrieve for
            feature_type=feature_type,
            # Turn coordinates into a tile
            tile=mercantile.tile(lng=longitude, lat=latitude, zoom=zoom),
            # The zoom level
            zoom=zoom,
        )

    def fetch_layers(
        self,
        coordinates: "list[list]",
        layer: str = "image",
        zoom: int = 14,
        is_computed: bool = False,
    ) -> GeoJSON:
        """
        Fetches multiple vector tiles based on a list of multiple coordinates in a listed format

        :param coordinates: A list of lists of coordinates to get the vector tiles for
        :type coordinates: "list[list]"

        :param layer: Either "overview", "sequence", "image", "traffic_sign", or "map_feature",
            defaults to "image"
        :type layer: str

        :param zoom: the zoom level [0, 14], inclusive. Defaults to 14
        :type zoom: int

        :param is_computed: Will to be fetched layers be computed? Defaults to False
        :type is_computed: bool

        :return: A geojson with merged features from all unique vector tiles
        :rtype: dict
        """

        # Check for the correct zoom values against the layer specified
        self.__zoom_range_check(layer=layer, zoom=zoom)

        # The output resultant geojson
        geojson: GeoJSON = GeoJSON(
            geojson={"type": "FeatureCollection", "features": []}
        )

        # A list of tiles that are either confined within or intersect with the bbox
        tiles = list(
            mercantile.tiles(
                west=coordinates[0],
                south=coordinates[1],
                east=coordinates[2],
                north=coordinates[3],
                zooms=zoom,
            )
        )

        print(
            f'[Vector Tiles API] Fetching {len(tiles)} {"tiles" if len(tiles) > 1 else "tile"}'
            "for images ..."
        )

        for tile in tiles:

            if is_computed:
                result = (
                    self.__preprocess_computed_layer(
                        # The layer to retrieve from
                        layer=layer,
                        # Turn coordinates into a tile
                        tile=tile,
                        # The zoom level
                        zoom=zoom,
                    )
                )["features"]
            else:
                result = (
                    self.__preprocess_layer(
                        # The layer to retrieve from
                        layer=layer,
                        # Turn coordinates into a tile
                        tile=tile,
                        # The zoom level
                        zoom=zoom,
                    )
                )["features"]

            geojson.append_features(result)

        return geojson

    def fetch_map_features(
        self,
        coordinates: "list[list]",
        feature_type: str,
        zoom: int = 14,
    ) -> GeoJSON:
        """
        Fetches map features based on a list Polygon object

        :param coordinates: A list of lists of coordinates to get the map features for
        :type coordinates: "list[list]"

        :param feature_type: Either "point", "traffic_signs", defaults to "point"
        :type feature_type: str

        :param zoom: the zoom level [0, 14], inclusive. Defaults to 14
        :type zoom: int

        :return: A geojson with merged features from all unique vector tiles
        :rtype: dict
        """

        # Check for the correct zoom values against the layer specified
        self.__zoom_range_check(layer="map_feature", zoom=zoom)

        # The output resultant geojson
        geojson: GeoJSON = GeoJSON(
            geojson={"type": "FeatureCollection", "features": []}
        )

        # A list of tiles that are either confined within or intersect with the bbox
        tiles = list(
            mercantile.tiles(
                west=coordinates[0],
                south=coordinates[1],
                east=coordinates[2],
                north=coordinates[3],
                zooms=zoom,
            )
        )

        print(
            f'[Vector Tiles API] Fetching {len(tiles)} {"tiles" if len(tiles) > 1 else "tile"}'
            "for map features ..."
        )

        for tile in tiles:
            geojson.append_features(
                self.__preprocess_features(
                    # The layer to retrieve from
                    feature_type=feature_type,
                    # Turn coordinates into a tile
                    tile=tile,
                    # The zoom level
                    zoom=zoom,
                )["features"]
            )

        return geojson

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

    def __preprocess_layer(self, layer: str, tile: mercantile.Tile, zoom: int):
        """
        Preprocessing un-computed layers

        :param layer: Either 'overview', 'sequence', 'image'
        :type layer: str

        :param tile: The specified tile
        :type tile: mercantile.Tile

        :param zoom: The zoom level
        :type zoom: int

        :return: A GeoJSON
        :rtype: dict
        """

        # * See "FOR DEVELOPERS (3, 3.1)"

        # Extract the url depending upon the layer
        url = ""

        # For overview
        if layer == "overview":
            url = VectorTiles.get_overview_layer(x=tile[0], y=tile[1], z=zoom)

        # For sequence
        elif layer == "sequence":
            url = VectorTiles.get_sequence_layer(x=tile[0], y=tile[1], z=zoom)

        # For image
        elif layer == "image":
            url = VectorTiles.get_image_layer(x=tile[0], y=tile[1], z=zoom)

        # No 'else' for InvalidOptionError, as checking done previously in __zoom_range_check

        # Convert bytes to GeoJSON
        return vt_bytes_to_geojson(
            # Parameters, appropriately
            b_content=self.client.get(url).content,
            x=tile.x,
            y=tile.y,
            z=tile.z,
            layer=layer,
        )

    def __preprocess_computed_layer(self, layer: str, tile: mercantile.Tile, zoom: int):
        """
        Preprocessing computed layers

        :param layer: Either 'overview', 'sequence', 'image'
        :type layer: str

        :param tile: The specified tile
        :type tile: mercantile.Tile

        :param zoom: The zoom level
        :type zoom: int

        :return: A GeoJSON
        :rtype: dict
        """

        # * See "FOR DEVELOPERS (3, 3.1)"

        # Extracting url from specified VectorTiles endpoint
        url = ""

        # For overview
        if layer == "overview":
            url = VectorTiles.get_computed_overview_layer(x=tile[0], y=tile[1], z=zoom)

        # For sequence
        elif layer == "sequence":
            url = VectorTiles.get_computed_sequence_layer(x=tile[0], y=tile[1], z=zoom)

        # For image
        elif layer == "image":
            url = VectorTiles.get_computed_image_layer(x=tile[0], y=tile[1], z=zoom)

        # No 'else' for InvalidOptionError, as checking done previously in __zoom_range_check

        # Convert bytes to geojson
        return vt_bytes_to_geojson(
            # Parameters appropriately
            b_content=self.client.get(url).content,
            x=tile.x,
            y=tile.y,
            z=tile.z,
            layer=layer,
        )

    def __preprocess_features(
        self, feature_type: str, tile: mercantile.Tile, zoom: int
    ) -> dict:
        """
        Preprocess features

        :param feature_type: Either 'point', 'traffic_signs'
        :type feature_type: str

        :param tile: The specified tile
        :type tile: mercantile.Tile

        :param zoom: The zoom level
        :type zoom: int

        :return: A GeoJSON
        :rtype: dict
        """

        # * See "FOR DEVELOPERS (3, 3.1)"

        # Extracting url from specified VectorTiles endpoint

        # For 'point'
        if feature_type == "point":

            url = VectorTiles.get_map_feature_point(x=tile[0], y=tile[1], z=zoom)

        # For 'traffic_signs'
        elif feature_type == "traffic_signs":
            url = VectorTiles.get_map_feature_traffic_sign(x=tile[0], y=tile[1], z=zoom)

        # If both are not present
        else:
            # Raise an exception for the invalid values passed
            raise InvalidOptionError(
                # Parameters accordingly
                param="feature_type",
                value=feature_type,
                options=["point", "traffic_sign"],
            )

        # Convert bytes to GeoJSON, and return
        return vt_bytes_to_geojson(
            # Parameters appropriately
            b_content=self.client.get(url).content,
            x=tile.x,
            y=tile.y,
            z=tile.z,
            layer=None,
        )
