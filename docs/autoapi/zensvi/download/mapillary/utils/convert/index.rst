:py:mod:`zensvi.download.mapillary.utils.convert`
=================================================

.. py:module:: zensvi.download.mapillary.utils.convert


Module Contents
---------------


Functions
~~~~~~~~~

.. autoapisummary::

   zensvi.download.mapillary.utils.convert.convert_multi_feature_collection
   zensvi.download.mapillary.utils.convert.extract_coordinates_from_polygons



.. py:function:: convert_multi_feature_collection(feature_collection)

   Convert all MultiPolygon, MultiLineString, or MultiPoint features in a GeoJSON FeatureCollection
   to individual Polygon, LineString, or Point features, respectively.

   :param feature_collection: A GeoJSON FeatureCollection.
   :return: A new GeoJSON FeatureCollection with converted features.


.. py:function:: extract_coordinates_from_polygons(feature_collection)

   Check if all features in the GeoJSON are Polygons or MultiPolygons.
   If so, extract their coordinates as a list of lists (each list contains tuples of longitude, latitude).
   Raise an error if any feature is not a Polygon or MultiPolygon.

   :param feature_collection: A GeoJSON FeatureCollection.
   :return: A list of lists with coordinates.


