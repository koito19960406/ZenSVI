# Copyright (c) Facebook, Inc. and its affiliates. (https://www.facebook.com)
# Copyright (c) 2022 Mapillary, Inc. (https://www.mapillary.com)
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# -*- coding: utf-8 -*-
"""mapillary.utils.logger
~~~~~~~~~~~~~~~~~~~~~~

This module implements the logger for mapillary, which is a wrapper of the logger package and
the default configuration for each of the loggers per file.

"""

import logging
import os
import sys


class Logger:
    """A wrapper class for configuring and setting up logging.

    This class provides static methods to configure logging with consistent formatting
    and file output handling.

    Attributes:
        format_string (str): The format string used for log messages.
            Default: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        level (int): The default logging level. Default: logging.INFO
    """

    format_string: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    level: int = logging.INFO

    @staticmethod
    def setup_logger(name: str, level: int = logging.INFO) -> logging.Logger:
        """Function setup as many loggers as you want. To be used at the top of the
        file.

        Usage::

            >>> Logger.setup_logger(name='mapillary.xxxx.yyyy', level=logging.INFO)
            logger.Logger

        Args:
            name (str): The name of the logger
            level (int): The level of the logger

        Returns:
            logging.Logger: The logger object
        """
        # Basic logger setup
        logger: Logger = logging.getLogger(name)

        # stdout logger setup
        stream_handler: logging.StreamHandler = logging.StreamHandler(sys.stdout)
        logger.addHandler(stream_handler)

        try:
            logger.setLevel(level)
        except ValueError:
            logger.setLevel(logging.INFO)
            logger.warning("LOG_LEVEL: Invalid variable - defaulting to: INFO")

        # File logger setup
        formatter: logging.Formatter = logging.Formatter(Logger.format_string)
        handler: logging.FileHandler = logging.FileHandler(Logger.get_os_log_path(name.split(".")[1] + ".log"))
        handler.setFormatter(formatter)
        logger.addHandler(handler)

        return logger

    @staticmethod
    def get_os_log_path(log_file: str) -> str:
        """Get the path of the log file based on the OS.

        Args:
            log_file (str): The name of the log file

        Returns:
            str: The path where the logs will be stored
        """
        log_path: str = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")

        if not os.path.exists(log_path):
            os.makedirs(log_path)

        return os.path.join(log_path, log_file)
