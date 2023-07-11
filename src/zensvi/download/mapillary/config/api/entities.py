# Copyright (c) Facebook, Inc. and its affiliates. (http://www.facebook.com)
# -*- coding: utf-8 -*-

"""
mapillary.config.api.entities
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This module contains the class implementation of the Entities API endpoints
as string, for the entity API endpoint aspect of the API v4 of Mapillary.

For more information, please check out https://www.mapillary.com/developer/api-documentation/.

- Copyright: (c) 2021 Facebook
- License: MIT LICENSE
"""

# Local imports
import datetime
import logging
import typing

# # Utils
from zensvi.download.mapillary.utils.time import is_iso8601_datetime_format

# # Models
# # # Exception Handling
from zensvi.download.mapillary.models.exceptions import InvalidFieldError, InvalidNumberOfArguments
from zensvi.download.mapillary.models.config import Config
from zensvi.download.mapillary.models.logger import Logger

logger: logging.Logger = Logger.setup_logger(name="mapillary.config.api.entities")


class Entities:
    """
    Each API call requires specifying the fields of the Entity you're interested in explicitly.
    A sample image by ID request which returns the id and a computed geometry could look as
    below. For each entity available fields are listed in the relevant sections. All IDs are
    unique and the underlying metadata for each entity is accessible at
    https://graph.mapillary.com/:id?fields=A,B,C. The responses are uniform and always return
    a single object, unless otherwise stated (collection endpoints). All collection endpoint
    metadata are wrapped in a {"data": [ {...}, ...]} JSON object.

    Usage::

        $ GET 'https://graph.mapillary.com/$IMAGE_ID?access_token=TOKEN&fields=id,computed_geometry'
        ... {
        ...     "id": "$IMAGE_ID",
        ...     "computed_geometry": {
        ...         "type": "Point",
        ...         "coordinates": [0, 0]
        ...     }
        ... }
    """

    @staticmethod
    def get_image(image_id: str, fields: list) -> str:
        """
        Represents the metadata of the image on the Mapillary platform with
        the following properties.

        Usage::

            >>> 'https://graph.mapillary.com/:image_id' # endpoint

        Fields::

            1. altitude - float, original altitude from Exif
            2. atomic_scale - float, scale of the SfM reconstruction around the image
            3. camera_parameters - array of float, intrinsic camera parameters
            4. camera_type - enum, type of camera projection (perspective, fisheye, or spherical)
            5. captured_at - timestamp, capture time
            6. compass_angle - float, original compass angle of the image
            7. computed_altitude - float, altitude after running image processing
            8. computed_compass_angle - float, compass angle after running image processing
            9. computed_geometry - GeoJSON Point, location after running image processing
            10. computed_rotation - enum, corrected orientation of the image
            11. exif_orientation - enum, orientation of the camera as given by the exif tag
                (see: https://sylvana.net/jpegcrop/exif_orientation.html)
            12. geometry - GeoJSON Point geometry
            13. height - int, height of the original image uploaded
            14. thumb_256_url - string, URL to the 256px wide thumbnail
            15. thumb_1024_url - string, URL to the 1024px wide thumbnail
            16. thumb_2048_url - string, URL to the 2048px wide thumbnail
            17. merge_cc - int, id of the connected component of images that were aligned together
            18. mesh - { id: string, url: string } - URL to the mesh
            19. quality_score - float, how good the image is (experimental)
            20. sequence - string, ID of the sequence
            21. sfm_cluster - { id: string, url: string } - URL to the point cloud
            22. width - int, width of the original image uploaded
        """

        fields = Entities.__field_validity(
            given_fields=fields,
            actual_fields=Entities.get_image_fields(),
            endpoint="https://graph.mapillary.com/:image_id?fields=",
        )

        return f"https://graph.mapillary.com/{image_id}/?fields={','.join(fields)}"

    @staticmethod
    def get_images(
        image_ids: typing.Union[typing.List[str], typing.List[int]], fields: list
    ) -> str:
        """
        Represents the metadata of the image on the Mapillary platform with
        the following properties.

        Usage::

            >>> 'https://graph.mapillary.com/ids=ID1,ID2,ID3' # endpoint

        Parameters::

            A list of entity IDs separated by comma. The provided IDs must be in the same type
            (e.g. all image IDs, or all detection IDs)

        Fields::

            1. altitude - float, original altitude from Exif
            2. atomic_scale - float, scale of the SfM reconstruction around the image
            3. camera_parameters - array of float, intrinsic camera parameters
            4. camera_type - enum, type of camera projection (perspective, fisheye, or spherical)
            5. captured_at - timestamp, capture time
            6. compass_angle - float, original compass angle of the image
            7. computed_altitude - float, altitude after running image processing
            8. computed_compass_angle - float, compass angle after running image processing
            9. computed_geometry - GeoJSON Point, location after running image processing
            10. computed_rotation - enum, corrected orientation of the image
            11. exif_orientation - enum, orientation of the camera as given by the exif tag
                (see: https://sylvana.net/jpegcrop/exif_orientation.html)
            12. geometry - GeoJSON Point geometry
            13. height - int, height of the original image uploaded
            14. thumb_256_url - string, URL to the 256px wide thumbnail
            15. thumb_1024_url - string, URL to the 1024px wide thumbnail
            16. thumb_2048_url - string, URL to the 2048px wide thumbnail
            17. merge_cc - int, id of the connected component of images that were aligned together
            18. mesh - { id: string, url: string } - URL to the mesh
            19. quality_score - float, how good the image is (experimental)
            20. sequence - string, ID of the sequence
            21. sfm_cluster - { id: string, url: string } - URL to the point cloud
            22. width - int, width of the original image uploaded

        Raises::

            InvalidNumberOfArguments - if the number of ids passed is 0 or greater than 50
        """

        # TODO: while this function should not be responsible to check if
        # all of the IDs passed are image_ids or detections_ids, this comment
        # should be to remind that this logic should belong elsewhere or to integrated
        # in the future

        if len(image_ids) == 0 or len(image_ids) > 50:
            raise InvalidNumberOfArguments(
                number_of_params_passed=len(image_ids),
                actual_allowed_params=50,
                param="image_ids",
            )

        fields = Entities.__field_validity(
            given_fields=fields,
            actual_fields=Entities.get_image_fields(),
            endpoint="https://graph.mapillary.com/ids=",
        )

        return f"https://graph.mapillary.com/ids={','.join(image_ids)}?fields={','.join(fields)}"

    @staticmethod
    def search_for_images(  # noqa: C901, 'search_for_images is too complex'
        bbox: typing.List[float],
        start_captured_at: typing.Optional[datetime.datetime] = None,
        end_captured_at: typing.Optional[datetime.datetime] = None,
        limit: typing.Optional[int] = None,
        organization_id: typing.Union[
            typing.Optional[int], typing.Optional[str]
        ] = None,
        sequence_id: typing.Optional[typing.List[int]] = None,
        fields: typing.Optional[list] = [],
    ) -> str:
        """
        Represents the metadata of the image on the Mapillary platform with
        the following properties.

        Output Format::

            >>> 'https://graph.mapillary.com/search?bbox=LONG1,LAT1,LONG2,LAT2' # endpoint
            >>> 'https://graph.mapillary.com/search?bbox=LONG1,LAT1,LONG2,LAT2&start_time='
            'START_TIME' # endpoint
            >>> 'https://graph.mapillary.com/search?bbox=LONG1,LAT1,LONG2,LAT2&start_time='
            'START_TIME&end_time=END_TIME' # endpoint
            >>> 'https://graph.mapillary.com/search?bbox=LONG1,LAT1,LONG2,LAT2&start_time='
            'START_TIME&end_time=END_TIME&limit=LIMIT' # endpoint
            >>> 'https://graph.mapillary.com/search/images?bbox=LONG1,LAT1,LONG2,LAT2&start_time'
            '=START_TIME&end_time=END_TIME&limit=LIMIT&organization_id=ORGANIZATION_ID&'
            'sequence_id=SEQUENCE_ID1' # endpoint
            >>> 'https://graph.mapillary.com/search/images?bbox=LONG1,LAT1,LONG2,LAT2&start_time='
            'START_TIME&end_time=END_TIME&limit=LIMIT&organization_id=ORGANIZATION_ID&sequence_id'
            '=SEQUENCE_ID1,SEQUENCE_ID2,SEQUENCE_ID3' # endpoint

        Usage::

            >>> from mapillary.config.api.entities import Entities
            >>> bbox = [-180, -90, 180, 90]
            >>> start_captured_at = datetime.datetime(2020, 1, 1, 0, 0, 0)
            >>> end_captured_at = datetime.datetime(2022, 1, 1, 0, 0, 0)
            >>> organization_id = 123456789
            >>> sequence_ids = [123456789, 987654321]
            >>> Entities.search_for_images(bbox=bbox) # endpoint
            'https://graph.mapillary.com/search?bbox=-180,-90,180,90' # endpoint
            >>> Entities.search_for_images(bbox=bbox, start_captured_at=start_captured_at)
            'https://graph.mapillary.com/search?bbox=-180,-90,180,90&start_time=' # endpoint
            >>> Entities.search_for_images(bbox=bbox,
            ... start_captured_at=start_captured_at, end_captured_at=end_captured_at)
            'https://graph.mapillary.com/search?bbox=-180,-90,180,90&start_time=&'
            'end_time=' # endpoint
            >>> Entities.search_for_images(bbox=bbox,
            ... start_captured_at=start_captured_at, end_captured_at=end_captured_at,
            ... limit=100)
            'https://graph.mapillary.com/search?bbox=-180,-90,180,90&start_time=&end_time=&limit'
            '=100' # endpoint
            >>> Entities.search_for_images(bbox=bbox,
            ... start_captured_at=start_captured_at, end_captured_at=end_captured_at,
            ... limit=100, organization_id=organization_id, sequence_id=sequence_ids)
            'https://graph.mapillary.com/search/images?bbox=-180,-90,180,90&start_time=&end_time'
            '=&limit=100&organization_id=1234567890&sequence_id=1234567890' # endpoint

        :param bbox: float,float,float,float: filter images in the bounding box. Specify in this
        order: left, bottom, right, top (or minLon, minLat, maxLon, maxLat).
        :type bbox: typing.Union[typing.List[float], typing.Tuple[float, float, float, float],
        list, tuple]

        :param start_captured_at: filter images captured after. Specify in the ISO 8601 format.
        For example: "2022-08-16T16:42:46Z".
        :type start_time: typing.Union[typing.Optional[datetime.datetime], typing.Optional[str]]
        :default start_captured_at: None

        :param end_captured_at: filter images captured before. Same format as
        "start_captured_at".
        :type end_time: typing.Union[typing.Optional[datetime.datetime], typing.Optional[str]]
        :default end_captured_at: None

        :param limit: limit the number of images returned. Max and default is 2000. The 'default'
        here means the default value of `limit` assumed on the server's end if the limit param
        is not passed. In other words, if the `limit` parameter is set to `None`, the server will
        assume the `limit` parameter to be 2000, which is the same as setting the `limit`
        parameter to 2000 explicitly.
        :type limit: typing.Optional[int]
        :default limit: None

        :param organization_id: filter images contributed to the specified organization Id.
        :type organization_id: typing.Optional[int]
        :default organization_id: None

        :param sequence_id: filter images in the specified sequence Ids (separated by commas),
        For example, "[1234567890,1234567891,1234567892]".
        :type sequence_id: typing.Optional[typing.List[int], int]
        :default sequence_id: None

        :param fields: filter the fields returned. For example, "['atomic_scale', 'altitude',
        'camera_parameters']". For more information, see
        https://www.mapillary.com/developer/api-documentation/#image. To get list of all possible
        fields, please use Entities.get_image_fields()
        :type fields: typing.Optional[typing.List[str]]
        :default fields: []

        :return: endpoint for searching an image
        :rtype: str
        """

        fields = Entities.__field_validity(
            given_fields=fields,
            actual_fields=Entities.get_image_fields(),
            endpoint="https://graph.mapillary.com/images?bbox=,:parameters=,:fields=",
        )

        parameter_string: str = ""

        parameters = {
            "start_captured_at": start_captured_at,
            "end_captured_at": end_captured_at,
            "limit": limit,
            "organization_id": organization_id,
            "sequence_id": sequence_id,
        }

        # For each item in the given parameters ...
        for key, value in parameters.items():

            # ... if it is not None ...
            if value is not None:

                # ... if the key is about time ...
                if key in ["start_captured_at", "end_captured_at"]:

                    # ... if the datatype is a string ...
                    if isinstance(value, str):

                        # ... check if the string is a valid datetime in the ISO8601 format ...
                        if not is_iso8601_datetime_format(value):

                            # ... if not, raise an error ...
                            if Config.use_strict:

                                # Raising ValueError if strict mode is enabled
                                raise ValueError(
                                    f"""{key} must be in the ISO 8601 format. For example:
                                    '2022-08-16T16:42:46Z'."""
                                )

                            else:

                                logger.warning(
                                    f"{key} must be in the ISO 8601 format. For example:"
                                    f"'2022-08-16T16:42:46Z'. Disregarding {key} parameter."
                                    "Continuing without strict mode enabled."
                                )

                            # ... if not, just move on - no assumptions on the `date string` ...
                            continue

                    # ... if the value is a valid datetime object ...
                    elif isinstance(value, datetime.datetime):

                        # ... convert it to the ISO 8601 format required ...
                        value = value.strftime("%Y-%m-%dT%H:%M:%SZ")

                # ... if the key is limit ...
                if key == "limit":

                    # Check if it is within limits
                    if value > 2000:

                        # Log warning if not, waring mode is enabled - logger object declared
                        # globally is used
                        logger.warning(f"{key} is greater than 2000. Setting to 2000.")

                        # Set the value to 2000
                        value = 2000

                # ... if the key is sequence_id ...
                if key == "sequence_id":
                    # ... convert the list into string ...
                    value = ",".join(map(str, value))

                # ... add it to the parameter string
                parameter_string += f"&{key}={value}"

        # Return the endpoint
        return (
            f"https://graph.mapillary.com/images?bbox={','.join([str(val) for val in bbox])}"
            f"{parameter_string if parameter_string != '' else ''}"
            f"{'&fields=' + ','.join(fields) if fields != [] else ''}"
        )

    @staticmethod
    def get_image_fields() -> list:
        """
        Gets list of possible image fields

        :return: Image field list
        :rtype: list
        """

        return [
            "altitude",
            "atomic_scale",
            "camera_parameters",
            "camera_type",
            "captured_at",
            "compass_angle",
            "computed_altitude",
            "computed_compass_angle",
            "computed_geometry",
            "computed_rotation",
            "exif_orientation",
            "geometry",
            "height",
            "thumb_256_url",
            "thumb_1024_url",
            "thumb_2048_url",
            "merge_cc",
            "mesh",
            "quality_score",
            "sequence",
            "sfm_cluster",
            "width",
        ]

    @staticmethod
    def get_map_feature(map_feature_id: str, fields: list) -> str:
        """
        These are objects with a location which have been derived from
        multiple detections in multiple images.

        Usage::

            >>> 'https://graph.mapillary.com/:map_feature_id' # endpoint

        Fields::

            1. first_seen_at - timestamp, timestamp of the least recent detection
                contributing to this feature
            2. last_seen_at - timestamp, timestamp of the most recent detection
                contributing to this feature
            3. object_value - string, what kind of map feature it is
            4. object_type - string, either a traffic_sign or point
            5. geometry - GeoJSON Point geometry
            6. images - list of IDs, which images this map feature was derived from
        """

        fields = Entities.__field_validity(
            given_fields=fields,
            actual_fields=Entities.get_map_feature_fields(),
            endpoint="https://graph.mapillary.com/:map_feature_id?fields=",
        )

        return (
            f"https://graph.mapillary.com/{map_feature_id}/?fields={','.join(fields)}"
        )

    @staticmethod
    def get_map_feature_fields() -> list:
        """
        Gets map feature fields

        :return: Possible map feature fields
        :rtype: list
        """

        return [
            "first_seen_at",
            "last_seen_at",
            "object_value",
            "object_type",
            "geometry",
            "images",
        ]

    @staticmethod
    def get_detection_with_image_id(
        image_id: str,
        fields: list,
    ) -> str:
        """
        Represent an object detected in a single image. For convenience
        this version of the API serves detections as collections. They can be
        requested as a collection on the resource (e.g. image) they contribute
        or belong to.

        Usage::

            >>> 'https://graph.mapillary.com/:image_id/detections'
            >>> # detections in the image with ID image_id

        Fields::

            1. created_at - timestamp, when was this detection created
            2. geometry - string, base64 encoded polygon
            3. image - object, image the detection belongs to
            4. value - string, what kind of object the detection represents
        """

        fields = Entities.__field_validity(
            given_fields=fields,
            actual_fields=Entities.get_detection_with_image_id_fields(),
            endpoint="https://graph.mapillary.com/:image_id/detections/?fields=",
        )

        return f"https://graph.mapillary.com/{image_id}/detections/?fields={','.join(fields)}"

    @staticmethod
    def get_detection_with_image_id_fields() -> list:
        """
        Gets list of possible detections for image ids

        :return: Possible detection parameters
        :rtype: list
        """

        return ["created_at", "geometry", "image", "value"]

    @staticmethod
    def get_detection_with_map_feature_id(
        map_feature_id: str,
        fields: list,
    ) -> str:
        """
        Represent an object detected in a single image. For convenience
        this version of the API serves detections as collections. They can be
        requested as a collection on the resource (e.g. map feature) they
        contribute or belong to.

        Usage::

            >>> 'https://graph.mapillary.com/:map_feature_id/detections'
            >>> # detections in the image with ID map_feature_id

        Fields::

            1. created_at - timestamp, when was this detection created
            2. geometry - string, base64 encoded polygon
            3. image - object, image the detection belongs to
            4. value - string, what kind of object the detection represents
        """

        fields = Entities.__field_validity(
            given_fields=fields,
            actual_fields=Entities.get_detection_with_map_feature_id_fields(),
            endpoint="https://graph.mapillary.com/:map_feature_id/detections/?fields=",
        )

        return f"https://graph.mapillary.com/{map_feature_id}/detections/?fields={','.join(fields)}"

    @staticmethod
    def get_detection_with_map_feature_id_fields() -> list:
        """
        Gets list of possible field parameters for map features

        :return: Map feature detection fields
        :rtype: list
        """

        return ["created_at", "geometry", "image", "value"]

    @staticmethod
    def get_organization_id(
        organization_id: str,
        fields: list,
    ) -> str:
        """
        Represents an organization which can own the imagery if users
        upload to it

        Usage::

            >>> 'https://graph.mapillary.com/:organization_id' # endpoint

        Fields::

            1. slug - short name, used in URLs
            2. name - nice name
            3. description - public description of the organization
        """

        fields = Entities.__field_validity(
            given_fields=fields,
            actual_fields=Entities.get_organization_id_fields(),
            endpoint="https://graph.mapillary.com/:organization_id?fields=",
        )

        return (
            f"https://graph.mapillary.com/{organization_id}/?fields={','.join(fields)}"
        )

    @staticmethod
    def get_organization_id_fields() -> list:
        """
        Gets list of possible organization id fields

        :return: Possible organization fields
        :rtype: list
        """

        return ["slug", "name", "description"]

    @staticmethod
    def get_sequence(
        sequence_id: str,
    ) -> str:
        """
        Represents a sequence of Image IDs ordered by capture time

        Usage::

            >>> 'https://graph.mapillary.com/image_ids?sequence_id=XXX'
            >>> # endpoint

        Fields::

            1. id - ID of the image belonging to the sequence
        """

        return f"https://graph.mapillary.com/image_ids?sequence_id={sequence_id}"

    @staticmethod
    def __field_validity(
        given_fields: list, actual_fields: list, endpoint: str
    ) -> list:
        """
        Checks if the given_fields are the actual correct fields for the given endpoint
        Compares against the list provided in `actual_fields`

        :param given_fields: The fields given as argument to check in
        :type given_fields: list

        :param actual_fields: The fields to check against
        :type actual_fields: list

        :param endpoint: The endpoint that is being targeted
        :type endpoint: str

        :raises InvalidFieldError: Raised when an API endpoint is passed invalid field elements

        :return: The given_fields if everything is correct
        :rtype: list
        """

        if len(given_fields) == 0:
            return given_fields  # empty list: []

        # Converting the given_fields into lowercase
        given_fields = [field.lower() for field in given_fields]

        # Enforce the existence of 'geometry' in the given_fields
        if "geometry" not in given_fields:
            given_fields.append("geometry")

        # Going through all the given fields
        for field in given_fields:

            # If 'all' is encountered ...
            if field == "all":

                # ... simply return the actual_fields list
                return actual_fields

            # If a field does not exist in the actual_fields ...
            if field not in actual_fields:

                # Raise an InvalidFieldError error
                raise InvalidFieldError(
                    # Specifying what endpoint was specified
                    endpoint=endpoint,
                    # Specify what field triggered the exception
                    field=field,
                )

        # If no error occurred, and all the fields are correct, return the given_fields
        return given_fields
