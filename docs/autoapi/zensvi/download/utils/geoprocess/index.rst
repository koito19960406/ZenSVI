:py:mod:`zensvi.download.utils.geoprocess`
==========================================

.. py:module:: zensvi.download.utils.geoprocess


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   zensvi.download.utils.geoprocess.GeoProcessor




.. py:class:: GeoProcessor(gdf, distance=1, grid=False, grid_size=1, id_columns=[], **kwargs)


   .. py:method:: get_lat_lon()


   .. py:method:: process_point(gdf)


   .. py:method:: process_multipoint(gdf)


   .. py:method:: process_linestring(gdf)


   .. py:method:: process_multilinestring(gdf)


   .. py:method:: get_street_points(polygon)


   .. py:method:: create_point_grid(polygon, grid_size, crs='EPSG:4326')

      Create a point grid within the bounding box of the input GeoDataFrame with the given grid size in meters.

      :param polygon: The input GeoDataFrame to get the bounding box from.
      :type polygon: geopandas.GeoDataFrame
      :param grid_size: The size of the grid in meters.
      :type grid_size: float
      :param crs: The coordinate reference system for the points. Defaults to "EPSG:4326".
      :type crs: str, optional

      :returns: A list of shapely Point objects in UTM coordinates.
      :rtype: list


   .. py:method:: utm_to_lat_lon(utm_points, utm_crs)


   .. py:method:: process_polygon(gdf)


   .. py:method:: process_multipolygon(gdf)



