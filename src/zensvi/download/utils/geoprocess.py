from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

import geopandas as gpd
import networkx
import numpy as np
import osmnx as ox
import pandas as pd
from pyproj import Transformer
from shapely.geometry import Point
from tqdm.auto import tqdm

from zensvi.utils.log import verbosity_tqdm

tqdm.pandas()


class GeoProcessor:
    """A class for processing geographic data and extracting points.

    This class handles various geometry types (Point, LineString, Polygon, etc.) and can
    extract points either by sampling along features or creating regular grids.

    Args:
        gdf (geopandas.GeoDataFrame): Input geodataframe containing geometries to process
        distance (float, optional): Sampling distance in meters. Defaults to 1.
        grid (bool, optional): Whether to use grid-based sampling. Defaults to False.
        grid_size (float, optional): Grid cell size in meters. Defaults to 1.
        id_columns (list, optional): List of columns to preserve in output. Defaults to [].
        verbosity (int, optional): Level of verbosity for progress bars (0=no progress, 1=outer loops only, 2=all loops).
                                  Defaults to 1.
        **kwargs: Additional keyword arguments:
            network_type (str): OSM network type to use. Defaults to "all".
            custom_filter (str): Custom OSM filter query.
    """

    def __init__(self, gdf, distance=1, grid=False, grid_size=1, id_columns=[], verbosity=1, **kwargs):
        self.gdf = gdf
        self.distance = distance
        self.processing_functions = {
            "Point": self.process_point,
            "MultiPoint": self.process_multipoint,
            "LineString": self.process_linestring,
            "MultiLineString": self.process_multilinestring,
            "Polygon": self.process_polygon,
            "MultiPolygon": self.process_multipolygon,
        }
        self.grid = grid
        self.grid_size = grid_size
        self.utm_crs = None
        self.id_columns = id_columns
        self.verbosity = verbosity
        if "network_type" in kwargs:
            self.network_type = kwargs["network_type"]
        else:
            self.network_type = "all"

        if "custom_filter" in kwargs:
            self.custom_filter = kwargs["custom_filter"]
        else:
            self.custom_filter = None

    def get_lat_lon(self):
        """Extract latitude and longitude points from input geometries.

        Processes each geometry type using the appropriate method and combines results.

        Returns:
            pandas.DataFrame: DataFrame containing extracted points with longitude and latitude columns.
        """
        self.gdf["feature_type"] = self.gdf["geometry"].apply(lambda x: x.geom_type)
        gdf_list = []

        for feature_type, func in self.processing_functions.items():
            sub_gdf = self.gdf[self.gdf["feature_type"] == feature_type]
            if not sub_gdf.empty:
                print(f"Getting longitude and latitude from {feature_type} feature")
                processed_gdf = func(sub_gdf)
                gdf_list.append(processed_gdf)

        result_gdf = pd.concat(gdf_list)
        return result_gdf

    def process_point(self, gdf):
        """Process Point geometries.

        Args:
            gdf (geopandas.GeoDataFrame): GeoDataFrame containing Point geometries.

        Returns:
            pandas.DataFrame: DataFrame with longitude and latitude columns.
        """
        gdf["longitude"] = gdf.geometry.x
        gdf["latitude"] = gdf.geometry.y
        return gdf[self.id_columns + ["longitude", "latitude"]]

    def process_multipoint(self, gdf):
        """Process MultiPoint geometries by exploding into individual points.

        Args:
            gdf (geopandas.GeoDataFrame): GeoDataFrame containing MultiPoint geometries.

        Returns:
            pandas.DataFrame: DataFrame with longitude and latitude columns.
        """
        gdf = gdf.explode("geometry").reset_index(drop=True)
        return self.process_point(gdf)

    def process_linestring(self, gdf):
        """Process LineString geometries by interpolating points along lines.

        Args:
            gdf (geopandas.GeoDataFrame): GeoDataFrame containing LineString geometries.

        Returns:
            pandas.DataFrame: DataFrame with longitude and latitude columns.
        """
        gdf_utm = ox.projection.project_gdf(gdf)
        self.utm_crs = gdf_utm.crs

        # Define a function to apply to each geometry
        def apply_interpolate(geometry):
            return list(ox.utils_geo.interpolate_points(geometry, dist=self.distance))

        # Apply the function to each geometry, respecting verbosity
        if self.verbosity >= 2:
            # Use progress_apply for visible progress bar
            gdf_utm["sample_points"] = gdf_utm["geometry"].progress_apply(apply_interpolate)
        else:
            # Use regular apply for no progress bar
            gdf_utm["sample_points"] = gdf_utm["geometry"].apply(apply_interpolate)

        gdf_utm = gdf_utm.explode("sample_points").reset_index(drop=True)

        gdf_utm["longitude"], gdf_utm["latitude"] = zip(*self.utm_to_lat_lon(gdf_utm["sample_points"], self.utm_crs))
        return gdf_utm[self.id_columns + ["longitude", "latitude"]]

    def process_multilinestring(self, gdf):
        """Process MultiLineString geometries by exploding into individual LineStrings.

        Args:
            gdf (geopandas.GeoDataFrame): GeoDataFrame containing MultiLineString geometries.

        Returns:
            pandas.DataFrame: DataFrame with longitude and latitude columns.
        """
        gdf = gdf.explode("geometry").reset_index(drop=True)
        return self.process_linestring(gdf)

    def get_street_points(self, polygon):
        """Extract points along street network within a polygon.

        Args:
            polygon (shapely.geometry.Polygon): Polygon to extract street network from.

        Returns:
            tuple: List of points and their UTM CRS.
        """
        graph = ox.graph_from_polygon(polygon, network_type=self.network_type, custom_filter=self.custom_filter)
        _, edges = ox.graph_to_gdfs(graph)
        edges_utm = ox.projection.project_gdf(edges)
        utm_crs = edges_utm.crs

        edges_utm["sample_points"] = edges_utm["geometry"].apply(
            lambda geom: list(ox.utils_geo.interpolate_points(geom, dist=self.distance))
        )
        edges_utm = edges_utm.explode("sample_points").reset_index(drop=True)
        return edges_utm["sample_points"].tolist(), utm_crs

    def create_point_grid(self, polygon, grid_size, crs="EPSG:4326"):
        """Create a regular grid of points within a polygon's bounding box.

        Args:
            polygon (geopandas.GeoDataFrame): Polygon to create grid within.
            grid_size (float): Grid cell size in meters.
            crs (str, optional): Coordinate reference system. Defaults to "EPSG:4326".

        Returns:
            tuple: List of points in UTM coordinates and their UTM CRS.
        """
        west, south, east, north = polygon.bounds

        polygon_geom = ox.utils_geo.bbox_to_poly(north, south, east, west)
        bounding_box = gpd.GeoDataFrame({"geometry": [polygon_geom]}, crs=crs)
        bounding_box_utm = ox.projection.project_gdf(bounding_box)
        utm_crs = bounding_box_utm.crs
        west_utm, south_utm, east_utm, north_utm = bounding_box_utm.total_bounds

        x_coords = np.arange(west_utm, east_utm, grid_size)
        y_coords = np.arange(south_utm, north_utm, grid_size)
        x_grid, y_grid = np.meshgrid(x_coords, y_coords)
        points = np.vstack((x_grid.ravel(), y_grid.ravel())).T
        points_utm = gpd.GeoSeries((Point(xy) for xy in points), crs=utm_crs)

        return [(point.x, point.y) for point in points_utm], utm_crs

    def utm_to_lat_lon(self, utm_points, utm_crs):
        """Convert UTM coordinates to latitude/longitude.

        Args:
            utm_points (list): List of points in UTM coordinates.
            utm_crs (str): UTM coordinate reference system.

        Returns:
            list: Points converted to latitude/longitude coordinates.
        """
        transformer = Transformer.from_crs(utm_crs, "EPSG:4326", always_xy=True)
        return [transformer.transform(*point) for point in utm_points]

    def process_polygon(self, gdf):
        """Process Polygon geometries by extracting points from street network or grid.

        Args:
            gdf (geopandas.GeoDataFrame): GeoDataFrame containing Polygon geometries.

        Returns:
            pandas.DataFrame: DataFrame with longitude and latitude columns.
        """
        batch_size = 100
        num_batches = (len(gdf) + batch_size - 1) // batch_size
        failed_geoms = []
        results = []

        for i in verbosity_tqdm(
            range(num_batches),
            desc=f"Processing polygon by batch size {min(batch_size, len(gdf))}",
            verbosity=self.verbosity,
            level=1,
        ):
            with ProcessPoolExecutor() as executor:
                batch_futures = {}
                for geom in gdf.geometry.iloc[i * batch_size : (i + 1) * batch_size]:
                    if not self.grid:
                        future = executor.submit(self.get_street_points, geom)
                    else:
                        future = executor.submit(self.create_point_grid, geom, self.grid_size)
                    batch_futures[future] = geom

                for future in verbosity_tqdm(
                    as_completed(batch_futures.keys()),
                    total=len(batch_futures),
                    desc=f"Processing polygon for batch #{i+1}",
                    verbosity=self.verbosity,
                    level=2,
                ):
                    geom = batch_futures[future]
                    try:
                        result = future.result()
                        results.append(result)
                    except (ValueError, networkx.exception.NetworkXPointlessConcept):
                        failed_geoms.append(geom)

        if len(failed_geoms) > 0:
            print("Retrying failed geoms by making grids")
            with ProcessPoolExecutor() as executor:
                retry_futures = {}
                for geom in verbosity_tqdm(
                    failed_geoms, desc="Preparing Failed Geoms", verbosity=self.verbosity, level=1
                ):
                    future = executor.submit(self.create_point_grid, geom, self.grid_size)
                    retry_futures[future] = geom

                for future in verbosity_tqdm(
                    as_completed(retry_futures.keys()),
                    total=len(retry_futures),
                    desc="Processing Failed Geoms",
                    verbosity=self.verbosity,
                    level=2,
                ):
                    result = future.result()
                    results.append(result)

        gdf["street_points"] = [result[0] for result in results]
        gdf["utm_crs"] = [result[1] for result in results]

        gdf_exploded = gdf.explode("street_points").reset_index(drop=True)
        gdf_exploded["utm_crs"] = gdf_exploded["utm_crs"].astype(str)

        with ThreadPoolExecutor() as executor:
            futures = {
                executor.submit(self.utm_to_lat_lon, group["street_points"], crs): crs
                for crs, group in gdf_exploded.groupby("utm_crs")
            }
            lat_lon_points = []
            for future in verbosity_tqdm(
                as_completed(futures),
                total=len(futures),
                desc="Converting UTM to Lat/Lon",
                verbosity=self.verbosity,
                level=1,
            ):
                lat_lon_points.extend(future.result())

        gdf_exploded[["longitude", "latitude"]] = pd.DataFrame(lat_lon_points, index=gdf_exploded.index)

        return gdf_exploded[self.id_columns + ["longitude", "latitude"]]

    def process_multipolygon(self, gdf):
        """Process MultiPolygon geometries by exploding into individual Polygons.

        Args:
            gdf (geopandas.GeoDataFrame): GeoDataFrame containing MultiPolygon geometries.

        Returns:
            pandas.DataFrame: DataFrame with longitude and latitude columns.
        """
        gdf = gdf.explode("geometry").reset_index(drop=True)
        return self.process_polygon(gdf)
