# Copyright (c) Facebook, Inc. and its affiliates. (http://www.facebook.com)
# -*- coding: utf-8 -*-
"""mapillary.controllers.detection
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This module implements the detection based business logic functionalities of the Mapillary
Python SDK.

For more information, please check out https://www.mapillary.com/developer/api-documentation/

- Copyright: (c) 2021 Facebook
- License: MIT LICENSE
"""

# Package imports
import typing

# # Adapter Imports
from zensvi.download.mapillary.models.api.entities import EntityAdapter

# # Models
from zensvi.download.mapillary.models.geojson import GeoJSON

# # Rules
from zensvi.download.mapillary.utils.verify import valid_id

# Local imports


def get_image_detections_controller(image_id: typing.Union[str, int], fields: list = []) -> GeoJSON:
    """Get image detections with given (image) key.

    Args:
        image_id (str): The image id
        fields (list): The fields possible for the detection endpoint.
            Please see https://www.mapillary.com/developer/api-
            documentation for more information

    https://www.mapillary.com/developer/api-documentation

    Returns:
        dict: GeoJSON
    """
    # Checks if the Id given is indeed a valid image_id
    valid_id(identity=image_id, image=True)

    # Return results from the Adapter
    return GeoJSON(
        geojson=EntityAdapter().fetch_detections(
            identity=image_id,
            id_type=True,
            fields=fields,
        )
    )


def get_map_feature_detections_controller(map_feature_id: typing.Union[str, int], fields: list = []) -> GeoJSON:
    """Get image detections with given (map feature) key.

    Args:
        map_feature_id (str): The map feature id
        fields (list): The fields possible for the detection endpoint.
            Please see https://www.mapillary.com/developer/api-
            documentation for more information

    https://www.mapillary.com/developer/api-documentation

    Returns:
        dict: GeoJSON
    """
    # Checks if the Id given is indeed a valid image_id
    valid_id(identity=map_feature_id, image=False)

    # Return results from the Adapter
    return GeoJSON(
        geojson=EntityAdapter().fetch_detections(
            identity=map_feature_id,
            id_type=False,
            fields=fields,
        )
    )
