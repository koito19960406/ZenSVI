# Copyright (c) Facebook, Inc. and its affiliates. (http://www.facebook.com)
# -*- coding: utf-8 -*-

"""
mapillary.utils.time
====================

This module contains the time utilies for the UNIX epoch seconds, the time and the date range, and
the date filtering logic.

- Copyright: (c) 2021 Facebook
- License: MIT LICENSE
"""

# Package imports
import datetime
import re


def date_to_unix_timestamp(date: str) -> int:
    """
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
    """

    # Returns the epoch current timestamp in milliseconds
    if date == "*":
        return int(datetime.datetime.now().timestamp()) * 1000

    # Return the epoch timestamp in miliseconds
    return int(datetime.datetime.fromisoformat(date).timestamp()) * 1000


def is_iso8601_datetime_format(date_time: str) -> bool:
    """
    Checks if the date time is in ISO 8601 format

    :param date_time: The date time to be checked
    :type date_time: str

    :return: True if the date time is in ISO 8601 format, else False
    :rtype: bool
    """

    return (
        re.match(r"(\d{4})\-(\d{2})\-(\d{2})T(\d{2})\:(\d{2})\:(\d{2})Z", date_time)
        is not None
    )
