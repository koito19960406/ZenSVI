# Copyright (c) Facebook, Inc. and its affiliates. (http://www.facebook.com)
# -*- coding: utf-8 -*-

"""
mapillary.config.__init__
~~~~~~~~~~~~~~~~~~~~~~~~~

This module contains the API v4 configuration and URL arguments.

Created to help in ease of use of the API v4.

# SOME NOTES FROM THE API

Over Authentication
!!!!!!!!!!!!!!!!!!!

1. All requests against https://graph.mapillary.com must be authorized. They require a client or
    user access tokens. Tokens can be sent in two ways,

    1. Using ?access_token=XXX query parameters. This is a preferred method for interacting with an
        vector tiles. Using this method is STRONGLY discouraged for sending user access tokens
    2. Using a header such as Authorization: OAuth XXX, where XXX is the token obtained either
        through the OAuth flow that your application implements or a client token from
        https://mapillary.com/dashboard/developers

For more information, please check out https://www.mapillary.com/developer/api-documentation/.

- Copyright: (c) 2021 Facebook
- License: MIT LICENSE
"""

from . import api  # noqa: F401
