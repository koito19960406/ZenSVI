# Copyright (c) Facebook, Inc. and its affiliates. (http://www.facebook.com)
# -*- coding: utf-8 -*-

"""
mapillary.models.client
~~~~~~~~~~~~~~~~~~~~~~~

This module contains aims to serve as a generalization for all API requests within the Mapillary
Python SDK.

Over Authentication
!!!!!!!!!!!!!!!!!!!

1. All requests against https://graph.mapillary.com must be authorized. They require a client or
    user access tokens. Tokens can be sent in two ways,

    1. Using ?access_token=XXX query parameters. This is a preferred method for interacting with
        vector tiles. Using this method is STRONGLY discouraged for sending user access tokens
    2. Using a header such as Authorization: OAuth XXX, where XXX is the token obtained either
        through the OAuth flow that your application implements or a client token from
        https://mapillary.com/dashboard/developers.

For more information, please check out https://www.mapillary.com/developer/api-documentation/.

- Copyright: (c) 2021 Facebook
- License: MIT LICENSE
"""

import json
import logging
import os
import sys
from math import floor

import requests

# Exception imports
from zensvi.download.mapillary.models.exceptions import InvalidTokenError

# Basic logger setup
logger = logging.getLogger("mapillary.utils.client")

# stdout logger setup
stream_handler = logging.StreamHandler(sys.stdout)
logger.addHandler(stream_handler)

# Setting log_level to INFO
log_level = os.environ.get("LOG_LEVEL", "INFO").upper()

# Check if in DEBUG mode to show debugging output
if os.environ.get("DEBUG") == "1":
    log_level = "DEBUG"
try:
    logger.setLevel(log_level)
except ValueError:
    logger.setLevel(logging.INFO)
    logger.warning("LOG_LEVEL: invalid variable - Defaulting to: INFO")


class Client:
    """
    Client setup for API communication.

    All requests for the Mapillary API v4 should go through this class

    Usage::

        >>> client = Client(access_token='MLY|XXX')
        >>> # for entities endpoints
        >>> client.get(endpoint='endpoint specific path', entity=True, params={
        ...     'fields': ['id', 'value']
        ... })
        >>> # for tiles endpoint
        >>> client.get(endpoint='endpoint specific path', entity=False)
    """

    # User Access token will be set once and used throughout all requests
    # within the same session
    __access_token = ""

    def __init__(self) -> None:

        # Session object setup to be referenced across future API calls.
        self.session = requests.Session()

    @staticmethod
    def __check_token_validity(token):
        res = requests.get(
            "https://graph.mapillary.com/1933525276802129?fields=id",
            headers={"Authorization": f"OAuth {token}"},
        )

        if res.status_code == 401:
            res_content = json.loads(res.content)
            raise InvalidTokenError(
                res_content["error"]["message"],
                res_content["error"]["type"],
                res_content["error"]["code"],
                res_content["error"]["fbtrace_id"],
            )

    @staticmethod
    def get_token() -> str:
        """
        Gets the access token

        :return: The access token
        """

        return Client.__access_token

    @staticmethod
    def set_token(access_token: str) -> None:
        """
        Sets the access token

        :param access_token: The access token to be set
        """

        Client.__check_token_validity(access_token)

        Client.__access_token = access_token

    def _initiate_request(self, url: str, method: str, params: dict = None):
        """
        Private method - For internal use only.
        This method is responsible for making tailored API requests to the mapillary API v4.
        It generalizes the requests and ties them to the same session.

        :param url: The request endpoint - required
        :type url: str

        :param method: HTTP method to be used - required
        :type method: str

        :param params: Query parameters to be attached to the request - optional
        :type params: dict
        """

        request = requests.Request(method, url, params=params)

        # create a prepared request with the request and the session info merged
        prepped_req = self.session.prepare_request(request)

        # Log the prepped request before sending it.
        Client._pprint_request(prepped_req)

        # Sending the request
        res = self.session.send(prepped_req)

        # Log the responses
        Client._pprint_response(res)

        # Handling the response status codes
        if res.status_code == requests.codes.ok:

            try:
                logger.debug(f"Response: {res.json()}")

            except ValueError:
                return res

        elif res.status_code >= 400:

            logger.error(f"Server responded with a {str(res.status_code)} error!")

            try:
                logger.debug(f"Error details: {str(res.json())}")

            except ValueError:
                logger.debug(
                    "[Client - _initiate_request, ValueError] res.json() not available,"
                    "empty response"
                )

            res.raise_for_status()

        return res

    def get(self, url: str = None, params: dict = {}):
        """
        Make GET requests to both mapillary main endpoints

        :param url: The specific path of the request URL
        :type url: str

        :param params: Query parameters to be attached to the URL (Dict)
        :type params: dict
        """
        # Check if an endpoint is specified.
        if url is None:
            logger.error("You need to specify an endpoint!")
            return

        # Determine Authentication method based on the requested endpoint
        if "https://graph.mapillary.com" in url:
            self.session.headers.update(
                {"Authorization": f"OAuth {self.__access_token}"}
            )
        else:
            params["access_token"] = params.get("access_token", self.__access_token)

        return self._initiate_request(url=url, method="GET", params=params)

    @staticmethod
    def _pprint_request(prepped_req):
        """
        Format::

            Method endpoint: HTTP/version
            Host: host
            Header_key: Header_value
            Body

        :param prepped_req: The prepped request object

        Reference::

            1. 'https://github.com/michaeldbianchi/Python-API-Client-Boilerplate/blob/fd1c82be9e98e'
                '24730c4631ffc30068272386669/exampleClient.py#L202'
        """

        method = prepped_req.method
        url = prepped_req.url

        headers = "\n".join(f"{k}: {v}" for k, v in prepped_req.headers.items())
        # Print body if present or empty string if not
        body = prepped_req.body or ""

        logger.info(f"Requesting {method} to {url}")

        logger.debug(
            "{}\n{} {} HTTP/1.1\n{}\n\n{}".format(
                "-----------REQUEST-----------", method, url, headers, body
            )
        )

    @staticmethod
    def _pprint_response(res):
        """
        Format::

            HTTP/version status_code status_text
            Header_key: Header_value
            Body

        :param res: Response object returned from the API request

        Reference::

            1. 'https://github.com/michaeldbianchi/Python-API-Client-Boilerplate/blob/fd1c82be9e98e'
                '24730c4631ffc30068272386669/exampleClient.py#L230'
        """

        http_v0, http_v1 = list(str(res.raw.version))
        http_v = f"HTTP/{http_v0}.{http_v1}"
        status_code = res.status_code
        status_text = res.reason
        headers = "\n".join(f"{k}: {v}" for k, v in res.headers.items())
        body = res.text or ""

        # Convert timedelta to milliseconds
        elapsed = floor(res.elapsed.total_seconds() * 1000)

        logger.info(f"Response {status_code} {status_text} received in {elapsed}ms")

        logger.debug(
            "{}\n{} {} {}\n{}\n\n{}".format(
                "-----------RESPONSE-----------",
                http_v,
                status_code,
                status_text,
                headers,
                body,
            )
        )
