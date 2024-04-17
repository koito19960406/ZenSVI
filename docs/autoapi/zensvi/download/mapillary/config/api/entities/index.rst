:py:mod:`zensvi.download.mapillary.config.api.entities`
=======================================================

.. py:module:: zensvi.download.mapillary.config.api.entities

.. autoapi-nested-parse::

   mapillary.config.api.entities
   ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

   This module contains the class implementation of the Entities API endpoints
   as string, for the entity API endpoint aspect of the API v4 of Mapillary.

   For more information, please check out https://www.mapillary.com/developer/api-documentation/.

   - Copyright: (c) 2021 Facebook
   - License: MIT LICENSE



Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   zensvi.download.mapillary.config.api.entities.Entities




Attributes
~~~~~~~~~~

.. autoapisummary::

   zensvi.download.mapillary.config.api.entities.logger


.. py:data:: logger
   :type: logging.Logger

   

.. py:class:: Entities


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

   .. py:method:: get_image(image_id: str, fields: list) -> str
      :staticmethod:

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


   .. py:method:: get_images(image_ids: Union[List[str], List[int]], fields: list) -> str
      :staticmethod:

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


   .. py:method:: search_for_images(bbox: List[float], start_captured_at: Optional[datetime.datetime] = None, end_captured_at: Optional[datetime.datetime] = None, limit: Optional[int] = None, organization_id: Union[Optional[int], Optional[str]] = None, sequence_id: Optional[List[int]] = None, fields: Optional[list] = []) -> str
      :staticmethod:

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


   .. py:method:: get_image_fields() -> list
      :staticmethod:

      Gets list of possible image fields

      :return: Image field list
      :rtype: list


   .. py:method:: get_map_feature(map_feature_id: str, fields: list) -> str
      :staticmethod:

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


   .. py:method:: get_map_feature_fields() -> list
      :staticmethod:

      Gets map feature fields

      :return: Possible map feature fields
      :rtype: list


   .. py:method:: get_detection_with_image_id(image_id: str, fields: list) -> str
      :staticmethod:

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


   .. py:method:: get_detection_with_image_id_fields() -> list
      :staticmethod:

      Gets list of possible detections for image ids

      :return: Possible detection parameters
      :rtype: list


   .. py:method:: get_detection_with_map_feature_id(map_feature_id: str, fields: list) -> str
      :staticmethod:

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


   .. py:method:: get_detection_with_map_feature_id_fields() -> list
      :staticmethod:

      Gets list of possible field parameters for map features

      :return: Map feature detection fields
      :rtype: list


   .. py:method:: get_organization_id(organization_id: str, fields: list) -> str
      :staticmethod:

      Represents an organization which can own the imagery if users
      upload to it

      Usage::

          >>> 'https://graph.mapillary.com/:organization_id' # endpoint

      Fields::

          1. slug - short name, used in URLs
          2. name - nice name
          3. description - public description of the organization


   .. py:method:: get_organization_id_fields() -> list
      :staticmethod:

      Gets list of possible organization id fields

      :return: Possible organization fields
      :rtype: list


   .. py:method:: get_sequence(sequence_id: str) -> str
      :staticmethod:

      Represents a sequence of Image IDs ordered by capture time

      Usage::

          >>> 'https://graph.mapillary.com/image_ids?sequence_id=XXX'
          >>> # endpoint

      Fields::

          1. id - ID of the image belonging to the sequence



