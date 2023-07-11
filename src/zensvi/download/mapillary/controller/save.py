# Copyright (c) Facebook, Inc. and its affiliates. (http://www.facebook.com)
# -*- coding: utf-8 -*-

"""
mapillary.controllers.save
~~~~~~~~~~~~~~~~~~~~~~~~~~

This module implements the saving business logic functionalities of the Mapillary Python SDK.

For more information, please check out https://www.mapillary.com/developer/api-documentation/

- Copyright: (c) 2021 Facebook
- License: MIT LICENSE
"""

# Package Imports
import os
import json
import csv
from shapely.geometry import shape

# Local Imports
from mapillary.utils.format import flatten_geojson
from mapillary.utils.time import date_to_unix_timestamp
from mapillary.utils.verify import check_file_name_validity


def save_as_csv_controller(data: str, path: str, file_name: str) -> None:
    """
    Save data as CSV to given file path

    :param data: The data to save as CSV
    :type data: str

    :param path: The path to save to
    :type path: str

    :param file_name: The file name to save as
    :type file_name: str

    :return: None
    :rtype: None
    """

    # Ensure that the geojson is a dictionary
    if isinstance(data, str):
        data = json.loads(data)

    # Flatten the geojson
    features = flatten_geojson(data)

    # Ensure that the file name is valid
    # Set the file name according to the given value. Default is
    # "mapillary_CURRENT_UNIX_TIMESTAMP_.csv"
    file_name = (
        file_name
        if (file_name is not None and check_file_name_validity(file_name))
        else f"mapillary_{date_to_unix_timestamp('*')}_"
    ) + ".csv"

    try:
        # Context manager for writing to file
        with open(os.path.join(path, file_name), "w", newline="") as file_path:
            # Enforce the header for field_names
            field_names = (
                ["ID", "WKT"]
                + [
                    key
                    for key in features[0].keys()
                    if key != "id" and key != "geometry"
                ]
                # organization_id may or may not exist in the flattened features
                # If it does exist, then its value will be reflected
                # If it does not exist, then it will be set to "NULL"
                + ["organization_id"]
            )

            # Create the csv writer
            writer = csv.DictWriter(file_path, fieldnames=field_names)

            # Write the header
            writer.writeheader()

            for feature in features:
                writer.writerow(
                    {
                        # ID is the id of the feature. It will always exist
                        # and will always be the first column
                        "ID": feature["id"],
                        # WKT is the geometry of the feature in well-known text format.
                        # It will always exist
                        "WKT": shape(feature["geometry"]).wkt,
                        # The rest of the columns are the features attributes
                        **{
                            key: feature[key]
                            for key in feature.keys()
                            if key != "id" and key != "geometry"
                        },
                        # organization_id is the id of the organization that the feature belongs to.
                        # If it does exist, then its value will be reflected
                        # If it does not exist, then it will be set to "NULL"
                        "organization_id": feature["organization_id"]
                        if "organization_id" in feature
                        else "NULL",
                    }
                )
    except Exception as e:
        # If there is an error, log it
        print(f"An error occurred: {e}")
    return None


def save_as_geojson_controller(data: str, path: str, file_name: str) -> None:
    """
    Save data as GeoJSON to given file path

    :param data: The data to save as GeoJSON
    :type data: str

    :param path: The path to save to
    :type path: str

    :param file_name: The file name to save as
    :type file_name: str

    :return: None
    :rtype: None
    """

    # Ensure that the geojson is a dictionary
    if isinstance(data, str):
        data = json.loads(data)

    # Ensure that the file name is valid
    # Set the file name according to the given value. Default is
    # "mapillary_CURRENT_UNIX_TIMESTAMP_.csv"
    file_name = (
        file_name
        if (file_name is not None and check_file_name_validity(file_name))
        else f"mapillary_{date_to_unix_timestamp('*')}_"
    ) + ".geojson"

    try:
        # Context manager for writing to file
        with open(os.path.join(path, file_name), "w") as file_path:
            json.dump(data, file_path, indent=4)
    except Exception as e:
        # If there is an error, log it
        print(e)

    return None
