:py:mod:`zensvi.download.mapillary.models.client`
=================================================

.. py:module:: zensvi.download.mapillary.models.client

.. autoapi-nested-parse::

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



Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   zensvi.download.mapillary.models.client.Client




Attributes
~~~~~~~~~~

.. autoapisummary::

   zensvi.download.mapillary.models.client.logger
   zensvi.download.mapillary.models.client.stream_handler
   zensvi.download.mapillary.models.client.log_level
   zensvi.download.mapillary.models.client.log_level


.. py:data:: logger

   

.. py:data:: stream_handler

   

.. py:data:: log_level

   

.. py:data:: log_level
   :value: 'DEBUG'

   

.. py:class:: Client


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

   .. py:method:: get_token() -> str
      :staticmethod:

      Gets the access token

      :return: The access token


   .. py:method:: set_token(access_token: str) -> None
      :staticmethod:

      Sets the access token

      :param access_token: The access token to be set


   .. py:method:: get(url: str = None, params: dict = {})

      Make GET requests to both mapillary main endpoints

      :param url: The specific path of the request URL
      :type url: str

      :param params: Query parameters to be attached to the URL (Dict)
      :type params: dict



