:py:mod:`zensvi.download.mapillary.models.geojson`
==================================================

.. py:module:: zensvi.download.mapillary.models.geojson

.. autoapi-nested-parse::

   mapillary.models.geojson
   ~~~~~~~~~~~~~~~~~~~~~~~~

   This module contains the class implementation for the geojson

   For more information about the API, please check out
   https://www.mapillary.com/developer/api-documentation/.

   - Copyright: (c) 2021 Facebook
   - License: MIT LICENSE



Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   zensvi.download.mapillary.models.geojson.Properties
   zensvi.download.mapillary.models.geojson.Coordinates
   zensvi.download.mapillary.models.geojson.Geometry
   zensvi.download.mapillary.models.geojson.Feature
   zensvi.download.mapillary.models.geojson.GeoJSON




.. py:class:: Properties(*properties, **kwargs)


   Representation for the properties in a GeoJSON

   :param properties: The properties as the input
   :type properties: dict

   :raises InvalidOptionError: Raised when the geojson passed is the invalid type - not a dict

   :return: A class representation of the model
   :rtype: mapillary.models.geojson.Properties

   .. py:method:: to_dict()

      Return the dictionary representation of the Properties


   .. py:method:: __str__()

      Return the informal string representation of the Properties


   .. py:method:: __repr__()

      Return the formal string representation of the Properties



.. py:class:: Coordinates(longitude: float, latitude: float)


   Representation for the coordinates in a geometry for a FeatureCollection

   :param longitude: The longitude of the coordinate set
   :type longitude: float

   :param latitude: The latitude of the coordinate set
   :type latitude: float

   :raises InvalidOptionError: Raised when invalid data types are passed to the coordinate set

   :return: A class representation of the Coordinates set
   :rtype: mapillary.models.geojson.Coordinates

   .. py:method:: to_list()

      Return the list representation of the Coordinates


   .. py:method:: to_dict()

      Return the dictionary representation of the Coordinates


   .. py:method:: __str__()

      Return the informal string representation of the Coordinates


   .. py:method:: __repr__() -> str

      Return the formal string representation of the Coordinates



.. py:class:: Geometry(geometry: dict)


   Representation for the geometry in a GeoJSON

   :param geometry: The geometry as the input
   :type geometry: dict

   :raises InvalidOptionError: Raised when the geometry passed is the invalid type - not a dict

   :return: A class representation of the model
   :rtype: mapillary.models.geojson.Geometry

   .. py:method:: to_dict()

      Return dictionary representation of the geometry


   .. py:method:: __str__()

      Return the informal string representation of the Geometry


   .. py:method:: __repr__()

      Return the formal string representation of the Geometry



.. py:class:: Feature(feature: dict)


   Representation for a feature in a feature list

   :param feature: The GeoJSON as the input
   :type feature: dict

   :raises InvalidOptionError: Raised when the geojson passed is the invalid type - not a dict

   :return: A class representation of the model
   :rtype: mapillary.models.geojson.Feature

   .. py:method:: to_dict() -> dict

      Return the dictionary representation of the Feature


   .. py:method:: __str__() -> str

      Return the informal string representation of the Feature


   .. py:method:: __repr__() -> str

      Return the formal string representation of the Feature


   .. py:method:: __hash__()

      Return hash(self).


   .. py:method:: __eq__(other)

      Return self==value.



.. py:class:: GeoJSON(geojson: dict)


   Representation for a geojson

   :param geojson: The GeoJSON as the input
   :type geojson: dict

   :raises InvalidOptionError: Raised when the geojson passed is the invalid type - not a dict

   :return: A class representation of the model
   :rtype: mapillary.models.geojson.GeoJSON

   Usage::

       >>> import mapillary as mly
       >>> from models.geojson import GeoJSON
       >>> mly.interface.set_access_token('MLY|XXX')
       >>> data = mly.interface.get_image_close_to(longitude=31, latitude=31)
       >>> geojson = GeoJSON(geojson=data)
       >>> type(geojson)
       ... <class 'mapillary.models.geojson.GeoJSON'>
       >>> type(geojson.type)
       ... <class 'str'>
       >>> type(geojson.features)
       ... <class 'list'>
       >>> type(geojson.features[0])
       ... <class 'mapillary.models.geojson.Feature'>
       >>> type(geojson.features[0].type)
       ... <class 'str'>
       >>> type(geojson.features[0].geometry)
       ... <class 'mapillary.models.geojson.Geometry'>
       >>> type(geojson.features[0].geometry.type)
       ... <class 'str'>
       >>> type(geojson.features[0].geometry.coordinates)
       ... <class 'list'>
       >>> type(geojson.features[0].properties)
       ... <class 'mapillary.models.geojson.Properties'>
       >>> type(geojson.features[0].properties.captured_at)
       ... <class 'int'>
       >>> type(geojson.features[0].properties.is_pano)
       ... <class 'str'>

   .. py:method:: append_features(features: list) -> None

      Given a feature list, append it to the GeoJSON object

      :param features: A feature list
      :type features: list

      :return: None


   .. py:method:: append_feature(feature_inputs: dict) -> None

      Given a feature dictionary, append it to the GeoJSON object

      :param feature_inputs: A feature as dict
      :type feature_inputs: dict

      :return: None


   .. py:method:: encode() -> str

      Serializes the GeoJSON object

      :return: Serialized GeoJSON


   .. py:method:: to_dict()

      Return the dict format representation of the GeoJSON


   .. py:method:: __str__()

      Return the informal string representation of the GeoJSON


   .. py:method:: __repr__()

      Return the formal string representation of the GeoJSON



