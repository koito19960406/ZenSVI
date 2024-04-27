:py:mod:`zensvi.download.mapillary.utils.time`
==============================================

.. py:module:: zensvi.download.mapillary.utils.time

.. autoapi-nested-parse::

   mapillary.utils.time
   ====================

   This module contains the time utilies for the UNIX epoch seconds, the time and the date range, and
   the date filtering logic.

   - Copyright: (c) 2021 Facebook
   - License: MIT LICENSE



Module Contents
---------------


Functions
~~~~~~~~~

.. autoapisummary::

   zensvi.download.mapillary.utils.time.date_to_unix_timestamp
   zensvi.download.mapillary.utils.time.is_iso8601_datetime_format



.. py:function:: date_to_unix_timestamp(date: str) -> int

   A utility function that converts the given date
   into its UNIX epoch timestamp equivalent. It accepts the formats, ranging from
   YYYY-MM-DDTHH:MM:SS, to simply YYYY, and a permutation of the fields in between as well

   Has a special argument, '*', which returns current timestamp

   :param date: The date to get the UNIX timestamp epoch of
   :type date: str

   :return: The UNIX timestamp equivalent of the input date
   :rtype: int

   Usage::

       >>> from utils.time_utils import date_to_unix_timestamp
       >>> date_to_unix_timestamp('2020-10-23')
       ... "1603393200"


.. py:function:: is_iso8601_datetime_format(date_time: str) -> bool

   Checks if the date time is in ISO 8601 format

   :param date_time: The date time to be checked
   :type date_time: str

   :return: True if the date time is in ISO 8601 format, else False
   :rtype: bool


