:py:mod:`zensvi.download.mapillary.utils.format`
================================================

.. py:module:: zensvi.download.mapillary.utils.format

.. autoapi-nested-parse::

   mapillary.utils.format
   ======================

   This module deals with converting data to and from different file formats.

   - Copyright: (c) 2021 Facebook
   - License: MIT LICENSE



Module Contents
---------------


Functions
~~~~~~~~~

.. autoapisummary::

   zensvi.download.mapillary.utils.format.feature_to_geojson
   zensvi.download.mapillary.utils.format.join_geojson_with_keys
   zensvi.download.mapillary.utils.format.geojson_to_features_list
   zensvi.download.mapillary.utils.format.merged_features_list_to_geojson
   zensvi.download.mapillary.utils.format.detection_features_to_geojson
   zensvi.download.mapillary.utils.format.flatten_geojson
   zensvi.download.mapillary.utils.format.geojson_to_polygon
   zensvi.download.mapillary.utils.format.flatten_dictionary
   zensvi.download.mapillary.utils.format.normalize_list
   zensvi.download.mapillary.utils.format.decode_pixel_geometry
   zensvi.download.mapillary.utils.format.decode_pixel_geometry_in_geojson
   zensvi.download.mapillary.utils.format.coord_or_list_to_dict
   zensvi.download.mapillary.utils.format.polygon_feature_to_bbox_list
   zensvi.download.mapillary.utils.format.bbox_to_polygon



.. py:function:: feature_to_geojson(json_data: dict) -> dict

   Converts feature into a GeoJSON, returns output

   From::

       >>> {'geometry': {'type': 'Point', 'coordinates': [30.003755665554, 30.985948744314]},
       ... 'id':'506566177256016'}

   To::

       >>> {'type': 'FeatureCollection', 'features': [{'type': 'Feature', 'geometry': {'type':
       ... 'Point','coordinates': [30.98594605922699, 30.003757307208872]}, 'properties': {}}]}

   :param json_data: The feature as a JSON
   :type json_data: dict

   :return: The formatted GeoJSON
   :rtype: dict


.. py:function:: join_geojson_with_keys(geojson_src: dict, geojson_src_key: str, geojson_dest: dict, geojson_dest_key: str) -> dict

   Combines two GeoJSONS based on the similarity of their specified keys, similar to
   the SQL join functionality

   :param geojson_src: The starting GeoJSO source
   :type geojson_src: dict

   :param geojson_src_key: The key in properties specified for the GeoJSON source
   :type geojson_src_key: str

   :param geojson_dest: The GeoJSON to merge into
   :type geojson_dest: dict

   :param geojson_dest_key: The key in properties specified for the GeoJSON to merge into
   :type geojson_dest_key: dict

   :return: The merged GeoJSON
   :rtype: dict

   Usage::

       >>> join_geojson_with_keys(
       ...     geojson_src=geojson_src,
       ...     geojson_src_key='id',
       ...     geojson_dest=geojson_dest,
       ...     geojson_dest_key='id'
       ... )


.. py:function:: geojson_to_features_list(json_data: dict) -> list

   Converts a decoded output GeoJSON to a list of feature objects

   The purpose of this formatting utility is to obtain a list of individual features for
   decoded tiles that can be later extended to the output GeoJSON

   From::

       >>> {'type': 'FeatureCollection', 'features': [{'type': 'Feature', 'geometry':
       ... {'type': 'Point','coordinates': [30.98594605922699, 30.003757307208872]},
       ... 'properties': {}}]}

   To::

       >>> [{'type': 'Feature', 'geometry': {'type': 'Point',
       ... 'coordinates': [30.98594605922699, 30.003757307208872]}, 'properties': {}}]

   :param json_data: The given json data
   :type json_data: dict

   :return: The feature list
   :rtype: list


.. py:function:: merged_features_list_to_geojson(features_list: list) -> str

   Converts a processed features list (i.e. a features list with all the needed features merged
   from multiple tiles) into a fully-featured GeoJSON

   From::

       >>> [{'type': 'Feature', 'geometry': {'type': 'Point',
       ... 'coordinates': [30.98594605922699, 30.003757307208872]}, 'properties': {}}, ...]

   To::

       >>> {'type': 'FeatureCollection', 'features': [{'type': 'Feature', 'geometry':
       ... {'type': 'Point','coordinates': [30.98594605922699, 30.003757307208872]},
       ... 'properties': {}}, ...]}

   :param features_list: a list of processed features merged from different tiles within a bbox
   :type features_list: list

   :return: GeoJSON string formatted with all the extra commas removed.
   :rtype: str


.. py:function:: detection_features_to_geojson(feature_list: list) -> dict

   Converts a preprocessed list (i.e, features from the detections of either images or
   map_features from multiple segments) into a fully featured GeoJSON

   :param feature_list: A list of processed features merged from different segments within a
       detection
   :type feature_list: list

   :return: GeoJSON formatted as expected in a detection format
   :rtype: dict

   Example::

       >>> # From
       >>> [{'created_at': '2021-05-20T17:49:01+0000', 'geometry':
       ... 'GjUKBm1weS1vchIVEgIAABgDIg0JhiekKBoqAABKKQAPGgR0eXBlIgkKB3BvbHlnb24ogCB4AQ==',
       ... 'image': {'geometry': {'type': 'Point', 'coordinates': [-97.743279722222,
       ... 30.270651388889]}, 'id': '1933525276802129'}, 'value': 'regulatory--no-parking--g2',
       ... 'id': '1942105415944115'}, ... ]
       >>> # To
       >>> {'type': 'FeatureCollection', 'features': [{'type': 'Feature', 'geometry':
       ... {'type': 'Point', 'coordinates': [-97.743279722222, 30.270651388889]}, 'properties': {
       ... 'image_id': '1933525276802129', 'created_at': '2021-05-20T17:49:01+0000',
       ... 'pixel_geometry':
       ... 'GjUKBm1weS1vchIVEgIAABgDIg0JhiekKBoqAABKKQAPGgR0eXBlIgkKB3BvbHlnb24ogCB4AQ==',
       ... 'value': 'regulatory--no-parking--g2', 'id': '1942105415944115' } }, ... ]}


.. py:function:: flatten_geojson(geojson: dict) -> list

   Flattens a GeoJSON dictionary to a dictionary with only the relevant keys.
   This is useful for writing to a CSV file.

   Output Structure::

       >>> {
       ...     "geometry": {
       ...         "type": "Point",
       ...         "coordinates": [71.45343, 12.523432]
       ...     },
       ...     "first_seen_at": "UNIX_TIMESTAMP",
       ...     "last_seen_at": "UNIX_TIMESTAMP",
       ...     "value": "regulatory--no-parking--g2",
       ...     "id": "FEATURE_ID",
       ...     "image_id": "IMAGE_ID"
       ... }

   :param geojson: The GeoJSON to flatten
   :type geojson: dict

   :return: A flattened GeoJSON
   :rtype: dict

   Note,
       1. The `geometry` key is always present in the output
       2. The properties are flattened to the following keys:
           - "first_seen_at"   (optional)
           - "last_seen_at"    (optional)
           - "value"           (optional)
           - "id"              (required)
           - "image_id"        (optional)
           - etc.
       3. If the 'geometry` type is `Point`, two more properties will be added:
           - "longitude"
           - "latitude"

   *TODO*: Further testing needed with different geometries, e.g., Polygon, etc.


.. py:function:: geojson_to_polygon(geojson: dict) -> zensvi.download.mapillary.models.geojson.GeoJSON

   Converts a GeoJSON into a collection of only geometry coordinates for the purpose of
   checking whether a given coordinate point exists within a shapely polygon

   From::

       >>> {
       ...     "type": "FeatureCollection",
       ...     "features": [
       ...         {
       ...             "geometry": {
       ...                 "coordinates": [
       ...                     -80.13069927692413,
       ...                     25.78523699486192
       ...                 ],
       ...                 "type": "Point"
       ...             },
       ...             "properties": {
       ...                 "first_seen_at": 1422984049000,
       ...                 "id": 481978503020355,
       ...                 "last_seen_at": 1422984049000,
       ...                 "value": "object--street-light"
       ...             },
       ...             "type": "Feature"
       ...         },
       ...         {
       ...             "geometry": {
       ...                 "coordinates": [
       ...                     -80.13210475444794,
       ...                     25.78362849816017
       ...                 ],
       ...                 "type": "Point"
       ...             },
       ...             "properties": {
       ...                 "first_seen_at": 1423228306666,
       ...                 "id": 252538103315239,
       ...                 "last_seen_at": 1423228306666,
       ...                 "value": "object--street-light"
       ...             },
       ...             "type": "Feature"
       ...         },
       ...         ...
       ...     ]
       ... }

   To::

       >>> {
       ... "type": "FeatureCollection",
       ... "features": [
       ...         {
       ...             "type": "Feature",
       ...             "properties": {},
       ...             "geometry": {
       ...                 "type": "Polygon",
       ...                 "coordinates": [
       ...                     [
       ...                         [
       ...                             7.2564697265625,
       ...                             43.69716905314008
       ...                         ],
       ...                         [
       ...                             7.27020263671875,
       ...                             43.69419030566581
       ...                         ],
       ...                         ...
       ...                     ]
       ...                 ]
       ...             }
       ...         }
       ...     ]
       ... }

   :param geojson: The input GeoJSON
   :type geojson: dict

   :return: A geojson of the format mentioned under 'To'
   :rtype: dict


.. py:function:: flatten_dictionary(data: Union[dict, collections.abc.MutableMapping], parent_key: str = '', sep: str = '_') -> dict

   Flattens dictionaries

   From::

       >>> {'mpy-or': {'extent': 4096, 'version': 2, 'features': [{'geometry': {'type':
       ... 'Polygon', 'coordinates': [[[2402, 2776], [2408, 2776]]]}, 'properties': {}, 'id': 1,
       ... 'type': 3}]}}

   To::

       >>> {'mpy-or_extent': 4096, 'mpy-or_version': 2, 'mpy-or_features': [{'geometry':
       ... {'type':'Polygon', 'coordinates': [[[2402, 2776], [2408, 2776]]]}, 'properties':
       ... {}, 'id': 1,'type': 3}]}

   :param data: The dictionary itself
   :type data: dict

   :param parent_key: The root key to start from
   :type parent_key: str

   :param sep: The separator
   :type sep: str

   :return: A flattened dictionary
   :rtype: dict


.. py:function:: normalize_list(coordinates: list, width: int = 4096, height: int = 4096) -> list

   Normalizes a list of coordinates with the respective width and the height

   From::

       >>> [[[2402, 2776], [2408, 2776]]]

   To::

       >>> normalize_list(coordinates)
       ... # [[[0.58642578125, 0.677734375], [0.587890625, 0.677734375]]]

   :param coordinates: The coordinate list to normalize
   :type coordinates: list

   :param width: The width of the coordinates to normalize with, defaults to 4096
   :type width: int

   :param height: The height of the coordinates to normalize with, defaults to 4096
   :type height: int

   :return: The normalized list
   :rtype: list


.. py:function:: decode_pixel_geometry(base64_string: str, normalized: bool = True, width: int = 4096, height: int = 4096) -> dict

   Decodes the pixel geometry, and return the coordinates, which can be specified to be
   normalized

   :param base64_string: The pixel geometry encoded as a vector tile
   :type base64_string: str

   :param normalized: If normalization is required, defaults to True
   :type normalized: bool

   :param width: The width of the pixel geometry, defaults to 4096
   :type width: int

   :param height: The height of the pixel geometry, defaults to 4096
   :type height: int

   :return: A dictionary with coordinates as key, and value as the normalized list
   :rtype: list


.. py:function:: decode_pixel_geometry_in_geojson(geojson: Union[dict, zensvi.download.mapillary.models.geojson.GeoJSON], normalized: bool = True, width: int = 4096, height: int = 4096) -> zensvi.download.mapillary.models.geojson.GeoJSON

   Decodes all the pixel_geometry

   :param geojson: The GeoJSON representation to be decoded

   :param normalized: If normalization is required, defaults to True
   :type normalized: bool

   :param width: The width of the pixel geometry, defaults to 4096
   :type width: int

   :param height: The height of the pixel geometry, defaults to 4096
   :type height: int


.. py:function:: coord_or_list_to_dict(data: Union[zensvi.download.mapillary.models.geojson.Coordinates, list, dict]) -> dict

   Converts a Coordinates object or a coordinates list to a dictionary

   :param data: The coordinates to convert
   :type data: Union[Coordinates, list]

   :return: The dictionary representation of the coordinates
   :rtype: dict


.. py:function:: polygon_feature_to_bbox_list(polygon: dict, is_bbox_list_required: bool = True) -> Union[list, dict]

   Converts a polygon to a bounding box

   The polygon below has been obtained from https://geojson.io/. If you have a polygon,
   with only 4 array elements, then simply take the first element and append it to the
   coordinates to obtain the below example.

   Usage::

       >>> from mapillary.utils.format import polygon_feature_to_bbox_list
       >>> bbox = polygon_feature_to_bbox_list(polygon={
       ...     "type": "Feature",
       ...     "properties": {},
       ...     "geometry": {
       ...         "type": "Polygon",
       ...         "coordinates": [
       ...             [
       ...                 [
       ...                   48.1640625,
       ...                   38.41055825094609
       ...                 ],
       ...                 [
       ...                   62.22656249999999,
       ...                   38.41055825094609
       ...                 ],
       ...                 [
       ...                   62.22656249999999,
       ...                   45.336701909968134
       ...                 ],
       ...                 [
       ...                   48.1640625,
       ...                   45.336701909968134
       ...                 ],
       ...                 [
       ...                   48.1640625,
       ...                   38.41055825094609
       ...                 ]
       ...             ]
       ...        ]
       ... })
       >>> bbox
       ... [62.22656249999999, 48.1640625, 38.41055825094609, 45.336701909968134]

   :param polygon: The polygon to convert
   :type polygon: dict

   :param is_bbox_list_required: Flag if bbox is required as a list. If true, returns a list,
   else returns a dict
   :type is_bbox_list_required: bool
   :default is_bbox_list_required: True

   :return: The bounding box
   :rtype: typing.Union[list, dict]


.. py:function:: bbox_to_polygon(bbox: Union[list, dict]) -> dict

   Converts a bounding box dictionary to a polygon

   Usage::

       >>> from mapillary.utils.format import bbox_to_polygon
       >>> bbox = [62.22656249999999, 48.1640625, 38.41055825094609, 45.336701909968134]
       >>> polygon = bbox_to_polygon(bbox=bbox)
       >>> polygon
       ... {
       ...     "type": "Feature",
       ...     "properties": {},
       ...     "geometry": {
       ...         "type": "Polygon",
       ...         "coordinates": [
       ...             [
       ...                 [
       ...                   48.1640625,
       ...                   38.41055825094609
       ...                 ],
       ...                 [
       ...                   62.22656249999999,
       ...                   38.41055825094609
       ...                 ],
       ...                 [
       ...                   62.22656249999999,
       ...                   45.336701909968134
       ...                 ],
       ...                 [
       ...                   48.1640625,
       ...                   45.336701909968134
       ...                 ],
       ...                 [
       ...                   48.1640625,
       ...                   38.41055825094609
       ...                 ]
       ...             ]
       ...        ]
       ... })

   :param bbox: The bounding box to convert
   :type bbox: dict

   :return: The polygon
   :rtype: dict


