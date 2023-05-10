import geopandas as gpd
import geopy.distance
import osmnx as ox
import pandas as pd
from shapely.geometry import LineString, Point
from tqdm.auto import tqdm
tqdm.pandas()
import math
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import numpy as np
from pyproj import Transformer

class GeoProcessor:
    def __init__(self, gdf, distance=1, grid = False, grid_size = 20):
        self.gdf = gdf
        self.distance = distance
        self.processing_functions = {
            'Point': self.process_point,
            'MultiPoint': self.process_multipoint,
            'LineString': self.process_linestring,
            'MultiLineString': self.process_multilinestring,
            'Polygon': self.process_polygon,
            'MultiPolygon': self.process_multipolygon
        }
        self.grid = grid
        self.grid_size = grid_size
        self.utm_crs = None

    def get_lat_lon(self):
        self.gdf['feature_type'] = self.gdf['geometry'].apply(lambda x: x.geom_type)
        gdf_list = []

        for feature_type, func in self.processing_functions.items():
            sub_gdf = self.gdf[self.gdf['feature_type'] == feature_type]
            if not sub_gdf.empty:
                print(f"Getting longitude and latitude from {feature_type} feature")
                processed_gdf = func(sub_gdf)
                gdf_list.append(processed_gdf)

        result_gdf = pd.concat(gdf_list)
        return result_gdf

    # Define other processing functions (process_point, process_multipoint, etc.) as class methods here
    def process_point(self, gdf):
        gdf['longitude'] = gdf.geometry.x
        gdf['latitude'] = gdf.geometry.y
        return gdf[['longitude', 'latitude']]

    def process_multipoint(self, gdf):
        gdf = gdf.explode('geometry').reset_index(drop=True)
        gdf['longitude'] = gdf.geometry.x
        gdf['latitude'] = gdf.geometry.y
        return gdf[['longitude', 'latitude']]

    def process_linestring(self, gdf):
        # Project the LineStrings to UTM
        gdf_utm = ox.projection.project_gdf(gdf)
        self.utm_crs = gdf_utm.crs

        # Use osmnx.utils_geo.interpolate_points function to interpolate points along LineStrings
        gdf_utm['sample_points'] = gdf_utm['geometry'].progress_apply(lambda geom: list(ox.utils_geo.interpolate_points(geom, dist=self.distance)), desc="Interpolating Points")
        gdf_utm = gdf_utm.explode('sample_points').reset_index(drop=True)

        # Convert the UTM points to latitude and longitude
        gdf_utm['longitude'], gdf_utm['latitude'] = zip(*self.utm_to_lat_lon(gdf_utm['sample_points'].apply(lambda p: (p.x, p.y)).tolist(), self.utm_crs))
        return gdf_utm[['longitude', 'latitude']]

    def process_multilinestring(self, gdf):
        gdf = gdf.explode('geometry').reset_index(drop=True)

        # Project the MultiLineStrings to UTM
        gdf_utm = ox.projection.project_gdf(gdf)
        self.utm_crs = gdf_utm.crs

        # Use osmnx.utils_geo.interpolate_points function to interpolate points along LineStrings
        gdf_utm['sample_points'] = gdf_utm['geometry'].progress_apply(lambda geom: list(ox.utils_geo.interpolate_points(geom, dist=self.distance)), desc="Interpolating Points")
        gdf_utm = gdf_utm.explode('sample_points').reset_index(drop=True)

        # Convert the UTM points to latitude and longitude
        gdf_utm['longitude'], gdf_utm['latitude'] = zip(*self.utm_to_lat_lon(gdf_utm['sample_points'].apply(lambda p: (p.x, p.y)).tolist(), self.utm_crs))
        return gdf_utm[['longitude', 'latitude']]

    def get_street_points(self, polygon):
        graph = ox.graph_from_polygon(polygon, network_type='all')
        _, edges = ox.graph_to_gdfs(graph)
        # Project the street network to UTM
        edges_utm = ox.projection.project_gdf(edges)
        utm_crs = edges_utm.crs

        # Use osmnx.utils_geo.interpolate_points function to interpolate points along LineStrings
        edges_utm['sample_points'] = edges_utm['geometry'].apply(lambda geom: list(ox.utils_geo.interpolate_points(geom, dist=self.distance)))
        edges_utm = edges_utm.explode('sample_points').reset_index(drop=True)
        return edges_utm['sample_points'].tolist(), utm_crs

    def create_point_grid(self, polygon, grid_size, crs="EPSG:4326"):
        """
        Create a point grid within the bounding box of the input GeoDataFrame with the given grid size in meters.

        Args:
            polygon (geopandas.GeoDataFrame): The input GeoDataFrame to get the bounding box from.
            grid_size (float): The size of the grid in meters.
            crs (str, optional): The coordinate reference system for the points. Defaults to "EPSG:4326".

        Returns:
            list: A list of shapely Point objects in UTM coordinates.
        """
        # Compute the bounding box coordinates
        west, south, east, north = polygon.bounds

        # Convert the bounding box coordinates to UTM for accurate distance calculations
        polygon_geom = ox.utils_geo.bbox_to_poly(north, south, east, west)
        bounding_box = gpd.GeoDataFrame({"geometry": [polygon_geom]}, crs=crs)
        bounding_box_utm = ox.projection.project_gdf(bounding_box)
        utm_crs = bounding_box_utm.crs
        west_utm, south_utm, east_utm, north_utm = bounding_box_utm.total_bounds

        # Create the point grid using NumPy
        x_coords = np.arange(west_utm, east_utm, grid_size)
        y_coords = np.arange(south_utm, north_utm, grid_size)
        x_grid, y_grid = np.meshgrid(x_coords, y_coords)
        points = np.vstack((x_grid.ravel(), y_grid.ravel())).T
        # Create a GeoSeries of points in UTM coordinates
        points_utm = gpd.GeoSeries((Point(xy) for xy in points), crs=utm_crs)

        # Return the points as coordinates (x, y) in UTM coordinates
        return [(point.x, point.y) for point in points_utm], utm_crs
    
    def utm_to_lat_lon(self, utm_points, utm_crs):
        transformer = Transformer.from_crs(utm_crs, "EPSG:4326", always_xy=True)
        lat_lon_points = [transformer.transform(x, y) for x, y in tqdm(utm_points, desc="Converting UTM to Lat/Lon")]
        return lat_lon_points

    def process_polygon(self, gdf):
        with ProcessPoolExecutor() as executor:
            if self.grid == False:
                futures = [executor.submit(self.get_street_points, geom) for geom in tqdm(gdf.geometry, desc="Preparing Polygons")]
            else:
                futures = [executor.submit(self.create_point_grid, geom, self.grid_size) for geom in tqdm(gdf.geometry, desc="Preparing Polygons")]

            results = [future.result() for future in tqdm(as_completed(futures), total=len(gdf), desc="Processing Polygons")]
            gdf['street_points'] = [result[0] for result in results]
            self.utm_crs = results[0][1]

        gdf = gdf.explode('street_points').reset_index(drop=True)

        # Convert the UTM points to latitude and longitude
        gdf['longitude'], gdf['latitude'] = zip(*self.utm_to_lat_lon(gdf['street_points'].tolist(), self.utm_crs))

        return gdf[['longitude', 'latitude']]

    def process_multipolygon(self, gdf):
        gdf = gdf.explode('geometry').reset_index(drop=True)
        with ProcessPoolExecutor() as executor:
            if self.grid == False:
                futures = [executor.submit(self.get_street_points, geom) for geom in tqdm(gdf.geometry, desc="Preparing Polygons")]
            else:
                futures = [executor.submit(self.create_point_grid, geom, self.grid_size) for geom in tqdm(gdf.geometry, desc="Preparing Polygons")]
            gdf['street_points'] = [future.result() for future in tqdm(as_completed(futures), total=len(gdf), desc="Processing Polygons")]

        gdf = gdf.explode('street_points').reset_index(drop=True)
        
        # Project the UTM points back to the original CRS
        gdf['geometry'] = gpd.GeoSeries(gdf['street_points'], crs=self.utm_crs)
        gdf = gdf.to_crs("EPSG:4326")
        
        # Extract longitude and latitude
        gdf['longitude'] = gdf.geometry.x
        gdf['latitude'] = gdf.geometry.y
        
        return gdf[['longitude', 'latitude']]