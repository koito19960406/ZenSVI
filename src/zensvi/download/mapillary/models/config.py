# Copyright (c) Facebook, Inc. and its affiliates. (http://www.facebook.com)
# -*- coding: utf-8 -*-

"""
mapillary.models.config
~~~~~~~~~~~~~~~~~~~~~~~~

This module contains the Config class which sets up some global variables fo the duration of
the session that the SDK is in use for.

For more information, please check out https://www.mapillary.com/developer/api-documentation/.

- Copyright: (c) 2021 Facebook
- License: MIT License
"""

from .logger import Logger

Logger.setup_logger(name="mapillary.models.config")


class Config:
    """
    Config setup for the SDK

    Different parts of the SDK react differently depending on what is set

    Usage::

        >>> from mapillary.models.config import Config

    :param use_strict: If set to True, the SDK will raise an exception if an invalid arguments
    are sent to the functions in config.api calls. If set to False, the SDK will just log a warning.
    :type use_strict: bool
    :default use_strict: True
    """

    # Strict mode will raise exceptions when,
    # 1. Invalid arguments are passed to a function, such as those
    # files in the config.api directory. This can mean invalid date
    # formats, invalid IDs, invalid data types, etc.

    use_strict = True

    def __init__(self, use_strict: bool = True) -> None:
        """
        Initialize the Config class
        """

        self.use_strict = use_strict
