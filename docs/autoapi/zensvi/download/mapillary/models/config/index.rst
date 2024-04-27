:py:mod:`zensvi.download.mapillary.models.config`
=================================================

.. py:module:: zensvi.download.mapillary.models.config

.. autoapi-nested-parse::

   mapillary.models.config
   ~~~~~~~~~~~~~~~~~~~~~~~~

   This module contains the Config class which sets up some global variables fo the duration of
   the session that the SDK is in use for.

   For more information, please check out https://www.mapillary.com/developer/api-documentation/.

   - Copyright: (c) 2021 Facebook
   - License: MIT License



Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   zensvi.download.mapillary.models.config.Config




.. py:class:: Config(use_strict: bool = True)


   Config setup for the SDK

   Different parts of the SDK react differently depending on what is set

   Usage::

       >>> from mapillary.models.config import Config

   :param use_strict: If set to True, the SDK will raise an exception if an invalid arguments
   are sent to the functions in config.api calls. If set to False, the SDK will just log a warning.
   :type use_strict: bool
   :default use_strict: True

   .. py:attribute:: use_strict
      :value: True

      


