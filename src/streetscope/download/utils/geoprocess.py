import geopandas as gpd
import geopy.distance
import osmnx as ox
import pandas as pd
from shapely.geometry import LineString
from tqdm.auto import tqdm
import math
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

class GeoProcessor:
    def __init__(self, gdf, distance=20):
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

    def get_sample_points(self, linestring, distance=20):
        length = linestring.length
        count = math.ceil(length / distance)
        return [linestring.interpolate((i / count) * length) for i in range(count + 1)]

    def process_linestring(self, gdf):
        with ProcessPoolExecutor() as executor:
            futures = [executor.submit(self.get_sample_points, geom, self.distance) for geom in gdf.geometry]
            gdf['sample_points'] = [future.result() for future in tqdm(as_completed(futures), total=len(gdf), desc="Processing LineStrings")]
        gdf = gdf.explode('sample_points').reset_index(drop=True)
        gdf['longitude'] = gdf.sample_points.x
        gdf['latitude'] = gdf.sample_points.y
        return gdf[['longitude', 'latitude']]

    def process_multilinestring(self, gdf):
        gdf = gdf.explode('geometry').reset_index(drop=True)
        with ProcessPoolExecutor() as executor:
            futures = [executor.submit(self.get_sample_points, geom, self.distance) for geom in gdf.geometry]
            gdf['sample_points'] = [future.result() for future in tqdm(as_completed(futures), total=len(gdf), desc="Processing LineStrings")]
        gdf = gdf.explode('sample_points').reset_index(drop=True)
        gdf['longitude'] = gdf.sample_points.x
        gdf['latitude'] = gdf.sample_points.y
        return gdf[['longitude', 'latitude']]

    def get_street_points(self, polygon):
        graph = ox.graph_from_polygon(polygon, network_type='all')
        _, edges = ox.graph_to_gdfs(graph)
        edges['sample_points'] = edges['geometry'].apply(self.get_sample_points, distance=self.distance)
        edges = edges.explode('sample_points').reset_index(drop=True)
        return edges['sample_points'].apply(lambda p: (p.x, p.y)).tolist()

    def process_polygon(self, gdf):
        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(self.get_street_points, geom) for geom in gdf.geometry]
            gdf['street_points'] = [future.result() for future in tqdm(as_completed(futures), total=len(gdf), desc="Processing Polygons")]
        gdf = gdf.explode('street_points').reset_index(drop=True)
        gdf['longitude'] = gdf.street_points.apply(lambda x: x[0])
        gdf['latitude'] = gdf.street_points.apply(lambda x: x[1])
        return gdf[['longitude', 'latitude']]

    def process_multipolygon(self, gdf):
        gdf = gdf.explode('geometry').reset_index(drop=True)
        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(self.get_street_points, geom) for geom in gdf.geometry]
            gdf['street_points'] = [future.result() for future in tqdm(as_completed(futures), total=len(gdf), desc="Processing Polygons")]
        gdf = gdf.explode('street_points').reset_index(drop=True)
        gdf['longitude'] = gdf.street_points.apply(lambda x: x[0])
        gdf['latitude'] = gdf.street_points.apply(lambda x: x[1])
        return gdf[['longitude', 'latitude']]
    
if __name__ == "__main__":
    # Example usage:
    gdf = gpd.read_file('/Users/koichiito/Downloads/Delft/Delft.shp')
    processor = GeoProcessor(gdf)
    result_gdf = processor.get_lat_lon()
    # save as csv
    result_gdf.to_csv('/Users/koichiito/Downloads/Delft/Delft.csv', index=False)