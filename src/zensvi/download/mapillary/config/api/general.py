# Copyright (c) Facebook, Inc. and its affiliates. (http://www.facebook.com)
# -*- coding: utf-8 -*-

"""
mapillary.config.api.general
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This module contains the class implementation of the
general metadata functionalities for the API v4 of Mapillary.

For more information, please check out https://www.mapillary.com/developer/api-documentation/.

- Copyright: (c) 2021 Facebook
- License: MIT LICENSE
"""


class General:
    """
    A general list of metadata API endpoints for API v4
    """

    @staticmethod
    def get_tile_metadata():
        """Root endpoint for metadata"""

        return "https://graph.mapillary.com/"

    @staticmethod
    def get_vector_tiles():
        """Root endpoint for vector tiles"""

        return "https://tiles.mapillary.com/"

    @staticmethod
    def get_image_type_tiles(x: float, y: float, z: float) -> str:
        """image_type tiles"""

        return f"https://tiles.mapillary.com/maps/vtp/mly1_public/2/{z}/{x}/{y}/"

    @staticmethod
    def get_computed_image_type_tiles(
        x: float,
        y: float,
        z: float,
    ) -> str:
        """Computed image_type tiles"""

        return (
            "https://tiles.mapillary.com/maps/vtp/mly1_computed_public/2"
            f"/{z}/{x}/{y}/"
        )

    @staticmethod
    def get_map_features_points_tiles(
        x: float,
        y: float,
        z: float,
    ) -> str:
        """Map features (points) tiles"""

        return (
            f"https://tiles.mapillary.com/maps/vtp/mly_map_feature_point/2"
            f"/{z}/{x}/{y}/"
        )

    @staticmethod
    def get_map_features_traffic_signs_tiles(
        x: float,
        y: float,
        z: float,
    ) -> str:
        """Map features (traffic signs) tiles"""

        return (
            "https://tiles.mapillary.com/maps/vtp/"
            f"mly_map_feature_traffic_sign/2/{z}/{x}/{y}"
        )
