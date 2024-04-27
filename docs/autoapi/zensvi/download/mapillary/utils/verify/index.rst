:py:mod:`zensvi.download.mapillary.utils.verify`
================================================

.. py:module:: zensvi.download.mapillary.utils.verify

.. autoapi-nested-parse::

   mapillary.controller.rules.verify
   =================================

   This module implements the verification of the filters or keys passed to each of the controllers
   under `./controllers` that implement the business logic functionalities of the Mapillary
   Python SDK.

   For more information, please check out https://www.mapillary.com/developer/api-documentation/

   - Copyright: (c) 2021 Facebook
   - License: MIT LICENSE



Module Contents
---------------


Functions
~~~~~~~~~

.. autoapisummary::

   zensvi.download.mapillary.utils.verify.international_dateline_check
   zensvi.download.mapillary.utils.verify.bbox_validity_check
   zensvi.download.mapillary.utils.verify.kwarg_check
   zensvi.download.mapillary.utils.verify.image_check
   zensvi.download.mapillary.utils.verify.resolution_check
   zensvi.download.mapillary.utils.verify.image_bbox_check
   zensvi.download.mapillary.utils.verify.sequence_bbox_check
   zensvi.download.mapillary.utils.verify.points_traffic_signs_check
   zensvi.download.mapillary.utils.verify.valid_id
   zensvi.download.mapillary.utils.verify.is_image_id
   zensvi.download.mapillary.utils.verify.check_file_name_validity



.. py:function:: international_dateline_check(bbox)


.. py:function:: bbox_validity_check(bbox)


.. py:function:: kwarg_check(kwargs: dict, options: list, callback: str) -> bool

   Checks for keyword arguments amongst the kwarg argument to fall into the options list

   :param kwargs: A dictionary that contains the keyword key-value pair arguments
   :type kwargs: dict

   :param options: A list of possible arguments in kwargs
   :type options: list

   :param callback: The function that called 'kwarg_check' in the case of an exception
   :type callback: str

   :raises InvalidOptionError: Invalid option exception

   :return: A boolean, whether the kwargs are appropriate or not
   :rtype: bool


.. py:function:: image_check(kwargs) -> bool

   For image entities, check if the arguments provided fall in the right category

   :param kwargs: A dictionary that contains the keyword key-value pair arguments
   :type kwargs: dict


.. py:function:: resolution_check(resolution: int) -> bool

   Checking for the proper thumbnail size of the argument

   :param resolution: The image size to fetch for
   :type resolution: int

   :raises InvalidOptionError: Invalid thumbnail size passed raises exception

   :return: A check if the size is correct
   :rtype: bool


.. py:function:: image_bbox_check(kwargs: dict) -> dict

   Check if the right arguments have been provided for the image bounding box

   :param kwargs: The dictionary parameters
   :type kwargs: dict

   :return: A final dictionary with the kwargs
   :rtype: dict


.. py:function:: sequence_bbox_check(kwargs: dict) -> dict

   Checking of the sequence bounding box

   :param kwargs: The final dictionary with the correct keys
   :type kwargs: dict

   :return: A dictionary with all the options available specifically
   :rtype: dict


.. py:function:: points_traffic_signs_check(kwargs: dict) -> dict

   Checks for traffic sign arguments

   :param kwargs: The parameters to be passed for filtering
   :type kwargs: dict

   :return: A dictionary with all the options available specifically
   :rtype: dict


.. py:function:: valid_id(identity: int, image=True) -> None

   Checks if a given id is valid as it is assumed. For example, is a given id expectedly an
   image_id or not? Is the id expectedly a map_feature_id or not?

   :param identity: The ID passed
   :type identity: int

   :param image: Is the passed id an image_id?
   :type image: bool

   :raises InvalidOptionError: Raised when invalid arguments are passed

   :return: None
   :rtype: None


.. py:function:: is_image_id(identity: int, fields: list = None) -> bool

   Checks if the id is an image_id

   :param identity: The id to be checked
   :type identity: int

   :param fields: The fields to be checked
   :type fields: list

   :return: True if the id is an image_id, else False
   :rtype: bool


.. py:function:: check_file_name_validity(file_name: str) -> bool

   Checks if the file name is valid

   Valid file names are,

   - Without extensions
   - Without special characters
   - A-Z, a-z, 0-9, _, -

   :param file_name: The file name to be checked
   :type file_name: str

   :return: True if the file name is valid, else False
   :rtype: bool


