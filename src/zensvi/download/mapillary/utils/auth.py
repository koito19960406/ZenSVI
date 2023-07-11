# Copyright (c) Facebook, Inc. and its affiliates. (http://www.facebook.com)
# -*- coding: utf-8 -*-

"""
mapillary.utils.auth
~~~~~~~~~~~~~~~~~~~~~

This module contains the authorization logic for the client class of Mapillary, responsible
for keeping track of the session token set

For more information, please check out https://www.mapillary.com/developer/api-documentation/.

- Copyright: (c) 2021 Facebook
- License: MIT LICENSE
"""

from functools import wraps

# Local imports
from mapillary.models.client import Client
from mapillary.models.exceptions import AuthError


def set_token(token: str) -> dict:
    """
    Allows the user to set access token to be able to interact with API v4

    :param token: Access token
    :return: Dictionary containing the access token
    """
    if len(token) < 1:
        raise AuthError("Token cannot be empty")
    try:
        Client.set_token(token)
    except AuthError:
        raise AuthError("Token is invalid")
    return {"token": "SUCCESS"}


def auth():
    """Wrap interface functions with logic for Client"""

    def auth_decorator(f):
        """Decorator function"""

        @wraps(f)
        def wrapper(*args, **kwargs):
            """
            :param args: Function arguments
            :param kwargs: Key word arguments
            :return: Return the specified function with args, kwargs
            """

            if Client.get_token() == "":
                # If empty, raise exception
                raise AuthError("Function called without setting the access token")

            # Return function called with arguments
            return f(*args, **kwargs)

        # Return wrapper
        return wrapper

    # Return decorator
    return auth_decorator
