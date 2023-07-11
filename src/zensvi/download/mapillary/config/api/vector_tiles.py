# Copyright (c) Facebook, Inc. and its affiliates. (http://www.facebook.com)
# -*- coding: utf-8 -*-

"""
mapillary.config.api.vector_tiles
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This module contains the class implementation of the VectorTile functionalities for the Vector Tile
aspect of the API v4 of Mapillary.

For more information, please check out https://www.mapillary.com/developer/api-documentation/.

- Copyright: (c) 2021 Facebook
- License: MIT LICENSE
"""


class VectorTiles:
    """
    Vector tiles provide an easy way to visualize vast amounts of data. Mapillary APIs are heavily
    based on vector tiles to provide the developers with flexibility to programmatically interact
    with the data they contain in custom ways. Vector tiles support filtering and querying rendered
    features. Mapillary vector tiles follow the Mapbox Tile Specification,
    https://docs.mapbox.com/vector-tiles/specification/
    """

    @staticmethod
    def get_overview_layer(
        x: float,
        y: float,
        z: float,
    ) -> str:
        """
        Contain positions of images and sequences with original
        geometries (not computed) for the layer 'overview'

        This layer offers,

        1. zoom: 0 - 5 (inclusive)
        2. geometry: Point
        3. data source: images

        With the following properties,

        1. captured_at, int, timestamp in ms since epoch
        2. id, int, ID of the image
        3. sequence_id, string, ID of the sequence this image belongs to
        4. is_pano, bool, if it is a panoramic image
        """

        return f"https://tiles.mapillary.com/maps/vtp/mly1_public/2/{z}/{x}/{y}/"

    @staticmethod
    def get_sequence_layer(
        x: float,
        y: float,
        z: float,
    ) -> str:
        """
        Contain positions of images and sequences with original geometries (not computed) for the
        layer 'sequence'

        This layer offers,

        1. zoom: 6 - 14 (inclusive)
        2. geometry: LineString
        3. data source: images captured in a single collection, sorted by captured_at

        With the following properties,

        1. captured_at, int, timestamp in ms since epoch
        2. id, string, ID  of the sequence (the legacy sequence key)
        3. image_id, int, ID of the 'best' (first) image representing the sequence
        4. organization_id, int, ID of the organization this image belongs to. It can be absent
        5. is_pano, bool, if it is a panoramic sequence
        """

        return f"https://tiles.mapillary.com/maps/vtp/mly1_public/2/{z}/{x}/{y}/"

    @staticmethod
    def get_image_layer(
        x: float,
        y: float,
        z: float,
    ) -> str:
        """
        Contain positions of images and sequences with original geometries (not computed) for the
        layer 'image'

        This layer offers,

        1. zoom: 14
        2. geometry: Point
        3. data source: images

        With the following properties,

        1. captured_at, int, timestamp in ms since epoch
        2. compass_angle, int, the compass angle of the image
        3. id, int, ID of the image
        4. sequence_id, string, ID of the sequence this image belongs to
        5. organization_id, int, ID of the organization this image belongs to. It can be absent
        6. is_pano, bool, if it is a panoramic image
        """

        return f"https://tiles.mapillary.com/maps/vtp/mly1_public/2/{z}/{x}/{y}/"

    @staticmethod
    def get_computed_overview_layer(
        x: float,
        y: float,
        z: float,
    ) -> str:
        """
        Contain positions of images and sequences with original geometries
        (computed) for the layer 'overview'

        This layer offers,

        1. zoom: 0 - 5 (inclusive)
        2. geometry: Point
        3. data source: images

        With the following properties,

        1. captured_at, int, timestamp in ms since epoch
        2. id, int, ID of the image
        3. sequence_id, string, ID of the sequence this image belongs to
        4. is_pano, bool, if it is a panoramic image
        """

        return (
            f"https://tiles.mapillary.com/maps/vtp/mly1_computed_public/2/{z}/{x}/{y}/"
        )

    @staticmethod
    def get_computed_sequence_layer(
        x: float,
        y: float,
        z: float,
    ) -> str:
        """
        Contain positions of images and sequences with original geometries (computed) for the
        layer 'sequence'

        This layer offers,

        1. zoom: 6 - 14 (inclusive)
        2. geometry: LineString
        3. data source: images captured in a single collection, sorted by captured_at

        With the following properties,

        1. captured_at, int, timestamp in ms since epoch
        2. id, string, ID  of the sequence (the legacy sequence key)
        3. image_id, int, ID of the 'best' (first) image representing the sequence
        4. organization_id, int, ID of the organization this image belongs to. It can be absent
        5. is_pano, bool, if it is a panoramic sequence
        """

        return (
            f"https://tiles.mapillary.com/maps/vtp/mly1_computed_public/2/{z}/{x}/{y}/"
        )

    @staticmethod
    def get_computed_image_layer(
        x: float,
        y: float,
        z: float,
    ) -> str:
        """
        Contain positions of images and sequences with original geometries (computed) for the
        layer 'image'

        This layer offers,

        1. zoom: 14
        2. geometry: Point
        3. data source: images

        With the following properties,

        1. captured_at, int, timestamp in ms since epoch
        2. compass_angle, int, the compass angle of the image
        3. id, int, ID of the image
        4. sequence_id, string, ID of the sequence this image belongs to
        5. organization_id, int, ID of the organization this image belongs to. It can be absent
        6. is_pano, bool, if it is a panoramic image
        """

        return (
            f"https://tiles.mapillary.com/maps/vtp/mly1_computed_public/2/{z}/{x}/{y}/"
        )

    @staticmethod
    def get_map_feature_point(
        x: float,
        y: float,
        z: float,
    ) -> str:
        """
        These tiles represent positions of map features which are detected on the Mapillary platform
        and are not traffic signs.

        This layer offers,

        1. zoom: 14
        2. geometry: Point
        3. data source: map features

        With the following resultant properties,

        1. id, int, ID of the image
        2. value, string, name of the class which this object represent
        3. first_seen_at, int, timestamp in ms since epoch, capture time of the earliest image on
            which the detection contribute to this map feature
        4. last_seen_at, int, timestamp in ms since epoch, capture time of the latest image on which
            the detection contribute to this map feature
        """

        return (
            f"https://tiles.mapillary.com/maps/vtp/mly_map_feature_point/2/{z}/{x}/{y}/"
        )

    @staticmethod
    def get_map_feature_traffic_sign(
        x: float,
        y: float,
        z: float,
    ) -> str:
        """
        These tiles represent positions of map features which are detected on the Mapillary
        platform and are traffic signs.

        The tile metadata is exactly the same as Map feature tiles, points, except that the
        layer name is traffic_sign.

        This layer offers,

        1. zoom: 14
        2. geometry: Point
        3. data source: map features

        With the following properties,

        1. id, int, ID of the image
        2. value, string, name of the class which this object represent
        3. first_seen_at, int, timestamp in ms since epoch, capture time of the earliest image on
            which the detection contribute to this map feature
        4. last_seen_at, int, timestamp in ms since epoch, capture time of the latest image on
            which the detection contribute to this map feature
        """

        return f"https://tiles.mapillary.com/maps/vtp/mly_map_feature_traffic_sign/2/{z}/{x}/{y}/"
