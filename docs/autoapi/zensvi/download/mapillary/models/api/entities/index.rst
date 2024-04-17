:py:mod:`zensvi.download.mapillary.models.api.entities`
=======================================================

.. py:module:: zensvi.download.mapillary.models.api.entities

.. autoapi-nested-parse::

   mapillary.models.api.entities
   ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

   This module contains the Adapter design for the Entities API of Mapillary API v4.

   For more information, please check out https://www.mapillary.com/developer/api-documentation/.

   - Copyright: (c) 2021 Facebook
   - License: MIT LICENSE



Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   zensvi.download.mapillary.models.api.entities.EntityAdapter




.. py:class:: EntityAdapter


   Bases: :py:obj:`object`

   Adapter model for dealing with the Entity API, through the DRY principle. The EntityAdapter
   class can be instantiated in the controller modules, providing an abstraction layer that uses
   the Client class, endpoints provided by the API v4 under `/config/api/entities.py`.

   It performs parsing, handling of layers, properties, and fields to make it easier to
   write higher level logic for extracing information, and lets developers to focus only
   on writing the high level business logic without having to repeat the process of parsing
   and using libraries such as `mercantile`, `ast`, and others to only then care about the
   inputs and the outputs

   Usage::

       >>> import mapillary
       >>> from mapillary.models.api.entities import EntityAdapter
       >>> EntityAdapter().fetch_image(image_id='IMAGE_ID', fields=[
       ...     'altitude', 'atomic_scale', 'geometry', 'width'
       ... ])
       >>> EntityAdapter().fetch_map_feature(map_feature_id='MAP_FEATURE_ID', fields=[
       ...         'first_seen_at', 'last_seen_at', 'geometry'
       ...     ])

   .. py:method:: fetch_image(image_id: Union[int, str], fields: list = None) -> dict

      Fetches images depending on the image_id and the fields provided

      :param image_id: The image_id to extract for
      :type image_id: int

      :param fields: The fields to extract properties for, defaults to []
      :type fields: list

      :return: The fetched GeoJSON
      :rtype: dict


   .. py:method:: fetch_map_feature(map_feature_id: Union[int, str], fields: list = None)

      Fetches map features depending on the map_feature_id and the fields provided

      :param map_feature_id: The map_feature_id to extract for
      :type map_feature_id: int

      :param fields: The fields to extract properties for, defaults to []
      :type fields: list

      :return: The fetched GeoJSON
      :rtype: dict


   .. py:method:: fetch_detections(identity: int, id_type: bool = True, fields: list = [])

      Fetches detections depending on the id, detections for either map_features or
      images and the fields provided

      :param identity: The id to extract for
      :type identity: int

      :param id_type: Either True(id is for image), or False(id is for map_feature),
          defaults to True
      :type id_type: bool

      :param fields: The fields to extract properties for, defaults to []
      :type fields: list

      :return: The fetched GeoJSON
      :rtype: dict


   .. py:method:: is_image_id(identity: int, fields: list = None) -> bool

      Determines whether the given id is an image_id or a map_feature_id

      :param identity: The ID given to test
      :type identity: int

      :param fields: The fields to extract properties for, defaults to []
      :type fields: list

      :return: True if id is image, else False
      :rtype: bool



