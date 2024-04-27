:py:mod:`zensvi.download.mapillary.utils.auth`
==============================================

.. py:module:: zensvi.download.mapillary.utils.auth

.. autoapi-nested-parse::

   mapillary.utils.auth
   ~~~~~~~~~~~~~~~~~~~~~

   This module contains the authorization logic for the client class of Mapillary, responsible
   for keeping track of the session token set

   For more information, please check out https://www.mapillary.com/developer/api-documentation/.

   - Copyright: (c) 2021 Facebook
   - License: MIT LICENSE



Module Contents
---------------


Functions
~~~~~~~~~

.. autoapisummary::

   zensvi.download.mapillary.utils.auth.set_token
   zensvi.download.mapillary.utils.auth.auth



.. py:function:: set_token(token: str) -> dict

   Allows the user to set access token to be able to interact with API v4

   :param token: Access token
   :return: Dictionary containing the access token


.. py:function:: auth()

   Wrap interface functions with logic for Client


