import os
import random
import time
import datetime
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings
import requests
import cv2
import pkg_resources
from pathlib import Path
import geopandas as gpd
from tqdm import tqdm
from shapely.geometry import Point
import warnings
from shapely.errors import ShapelyDeprecationWarning
warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning)

from streetscope.download.utils.imtool import ImageTool
from streetscope.download.utils.get_pids import panoids
from streetscope.download.utils.geoprocess import GeoProcessor


class StreetViewDownloader:
    def __init__(self, gsv_api_key = None, log_path = None, nthreads = 5, distance = 1, grid = False, grid_size = 20):
        if gsv_api_key == None:
            warnings.warn("Please provide your Google Street View API key to augment metadata.")
        self._gsv_api_key = gsv_api_key
        self._log_path = log_path
        self._nthreads = nthreads
        self._distance = distance
        self._user_agent = self._get_ua()
        self._grid = grid
        self._grid_size = grid_size

    @property
    def gsv_api_key(self):
        return self._gsv_api_key    
    @gsv_api_key.setter
    def gsv_api_key(self,gsv_api_key):
        self._gsv_api_key = gsv_api_key

    @property
    def log_path(self):
        return self._log_path    
    @log_path.setter
    def log_path(self,log_path):
        self._log_path = log_path
        
    @property
    def nthreads(self):
        return self._nthreads    
    @nthreads.setter
    def nthreads(self,nthreads):
        self._nthreads = nthreads
    
    @property
    def distance(self):
        return self._distance    
    @distance.setter
    def distance(self,distance):
        self._distance = distance
    
    @property
    def grid(self):
        return self._grid    
    @grid.setter
    def grid(self,grid):
        self._grid = grid
        
    @property
    def grid_size(self):
        return self._grid_size    
    @grid_size.setter
    def grid_size(self,grid_size):
        self._grid_size = grid_size
    
    @property
    def user_agent(self):
        return self._user_agent  
    
    def _get_ua(self):
        user_agent_file = pkg_resources.resource_filename('streetscope.download.utils', 'UserAgent.csv')
        UA = []
        with open(user_agent_file, 'r') as f:
            for line in f:
                ua = {"user_agent": line.strip()}
                UA.append(ua)
        return UA

    def _read_pids(self, path_pid):
        pid_df = pd.read_csv(path_pid)
        # get unique pids as a list
        pids = pid_df.iloc[:,0].unique().tolist()
        return pids

    def _check_already(self, all_panoids):
        name_r, all_panoids_f = set(), []
        for name in os.listdir(self.dir_output):
            name_r.add(name.split(".")[0])

        for pid in all_panoids:
            if pid not in name_r:
                all_panoids_f.append(pid)
        return all_panoids_f

    def _get_nthreads_pid(self, panoids):
        # Output path for the images
        all_pid, panos = [], []
        for i in range(len(panoids)):
            if i % self.nthreads != 0 or i == 0:
                panos.append(panoids[i])
            else:
                all_pid.append(panos)
                panos = []
        return all_pid

    def _log_write(self, pids):
        with open(self.log_path, 'a+') as fw:
            for pid in pids:
                fw.write(pid+'\n')
    
    def _augment_metadata(self, df):
        def get_year_month(pid):
            url = "https://maps.googleapis.com/maps/api/streetview/metadata?pano={}&key={}".format(pid, self.gsv_api_key)
            response = requests.get(url)
            response = response.json()
            if response['status'] == 'OK':
                # get year and month from date
                try:
                    date = response['date']
                    year = date.split("-")[0]
                    month = date.split("-")[1]
                except Exception:
                    year = None
                    month = None
                return {"year": year, "month": month}
            return {"year": None, "month": None}    

        def worker(index, row):
            panoid = row['panoid']
            year_month = get_year_month(panoid)
            return index, year_month
        
        with ThreadPoolExecutor() as executor:
            futures = {executor.submit(worker, i, row): i for i, row in tqdm(df.iterrows(), total = len(df), desc="Preparing to augment metadata")}
            for future in tqdm(as_completed(futures), total=len(futures), desc="Augmenting timestamp metadata"):
                row_index, year_month = future.result()
                df.at[row_index, 'year'] = year_month['year']
                df.at[row_index, 'month'] = year_month['month']
        return df
                    
    def _get_pids_from_csv(self, input_csv_file, closest=False, disp=False):
        def get_street_view_info(longitude, latitude):
            results = panoids(latitude, longitude, closest=closest, disp=disp)
            return results

        def worker(row):
            input_longitude = row['longitude']
            input_latitude = row['latitude']
            return (input_longitude, input_latitude), get_street_view_info(input_longitude, input_latitude)

        df = pd.read_csv(input_csv_file)
        def standardize_column_names(df):
            longitude_variants = ['longitude', 'long', 'lon', 'lng', "x"]
            latitude_variants = ['latitude', 'lat', 'lt', "y"]
            # convert all column names to lowercase
            df.columns = [col.lower() for col in df.columns]      
            for col in df.columns:
                if col in longitude_variants:
                    df.rename(columns={col: 'longitude'}, inplace=True)
                elif col in latitude_variants:
                    df.rename(columns={col: 'latitude'}, inplace=True)

            return df
        df = standardize_column_names(df)
        results = []

        with ThreadPoolExecutor() as executor:
            futures = {executor.submit(worker, row): (row['longitude'], row['latitude']) for _, row in tqdm(df.iterrows(), total = len(df), desc="Preparing to get pids")}
            for future in tqdm(as_completed(futures), total=len(futures), desc="Getting pids"):
                (input_longitude, input_latitude), row_results = future.result()
                for result in row_results:
                    result['input_longitude'] = input_longitude
                    result['input_latitude'] = input_latitude
                    results.append(result)

        results_df = pd.DataFrame(results)
        return results_df
    
    def _get_pids_from_shp(self, input_shp_file, closest=False, disp=False):  
        def get_street_view_info(longitude, latitude):
            results = panoids(latitude, longitude, closest=closest, disp=disp)
            return results

        def worker(row):
            input_longitude = row['longitude']
            input_latitude = row['latitude']
            results = get_street_view_info(input_longitude, input_latitude)
            return (input_longitude, input_latitude), results

        # read shapefile
        gdf = gpd.read_file(input_shp_file)
        gp = GeoProcessor(gdf, distance=self.distance, grid=self.grid, grid_size=self.grid_size)
        df = gp.get_lat_lon()
        results = []

        with ThreadPoolExecutor() as executor:
            futures = {executor.submit(worker, row): (row['longitude'], row['latitude']) for _, row in tqdm(df.iterrows(), total=len(df), desc="Preparing to get pids")}
            for future in tqdm(as_completed(futures), total=len(futures), desc="Getting pids"):
                (input_longitude, input_latitude), row_results = future.result()
                for result in row_results:
                    result['input_longitude'] = input_longitude
                    result['input_latitude'] = input_latitude
                    results.append(result)

        results_df = pd.DataFrame(results)

        # Check if lat and lon are within input polygons
        polygons = gpd.GeoSeries([geom for geom in gdf['geometry'] if geom.type in ['Polygon', 'MultiPolygon']])

        # Convert lat, lon to Points and create a GeoSeries
        points = gpd.GeoSeries([Point(lon, lat) for lon, lat in zip(results_df['lon'], results_df['lat'])])

        # Add progress bar for within_polygon calculation
        tqdm.pandas(total=len(points), desc="Checking points within polygons")
        within_polygon = points.progress_apply(lambda point: polygons.contains(point).any())

        results_df['within_polygon'] = within_polygon

        # Return only those points within polygons
        results_within_polygons_df = results_df[results_df['within_polygon']]
        return results_within_polygons_df

    
    def get_pids(self, path_pid, lat = None, lng = None, input_csv_file = "", input_shp_file = "", closest=False, disp=False, augment_metadata=False):
        if lat != None and lng != None:
            pid = panoids(lat, lng, closest=closest, disp=disp)
        elif input_csv_file != "":
            pid = self._get_pids_from_csv(input_csv_file, closest=closest, disp=disp)
        elif input_shp_file != "":
            pid = self._get_pids_from_shp(input_shp_file, closest=closest, disp=disp)
        else:
            raise ValueError("Please input the lat and lng, csv file, or shapefile.")
        # save the pids
        pid_df = pd.DataFrame(pid)
        pid_df = pid_df.drop_duplicates(subset='panoid')
        if augment_metadata & (self.gsv_api_key != None):
            pid_df = self._augment_metadata(pid_df)
        elif augment_metadata & (self.gsv_api_key == None):
            raise ValueError("Please set the gsv api key by calling the gsv_api_key method.")
        pid_df.to_csv(path_pid, index=False)
        print("The panorama IDs have been saved to {}".format(path_pid))
    
    def download_gsv(self, dir_output, path_pid = None, zoom=2, h_tiles=4, v_tiles=2, cropped=False, full=True, 
                lat=None, lng=None, input_csv_file="", input_shp_file = "", closest=False, disp=False, augment_metadata=False):
        # set dir_output as attribute and create the directory
        self.dir_output = dir_output
        Path(dir_output).mkdir(parents=True, exist_ok=True)
        
        # If path_pid is None, call get_pids function first
        if path_pid is None:
            print("Getting pids...")
            path_pid = os.path.join(self.dir_output, "pids.csv")
            self.get_pids(path_pid, lat=lat, lng=lng,
                        input_csv_file=input_csv_file, input_shp_file = input_shp_file, closest=closest, disp=disp, augment_metadata=augment_metadata)

        # Horizontal Google Street View tiles
        # zoom 3: (8, 4); zoom 5: (26, 13) zoom 2: (4, 2) zoom 1: (2, 1);4:(8,16)
        # zoom = 2
        # h_tiles = 4  # 26
        # v_tiles = 2  # 13
        # cropped = False
        # full = True
        # create a folder within self.dir_output
        panorama_output = os.path.join(self.dir_output, "panorama")
        os.makedirs(panorama_output, exist_ok=True)
        
        panoids = self._read_pids(path_pid)
        panoids_rest = self._check_already(panoids)

        panoids = self._read_pids(path_pid)
        panoids_rest = self._check_already(panoids)

        UAs = random.choices(self.user_agent, k = len(panoids_rest))
        ImageTool.dwl_multiple(panoids_rest, zoom, v_tiles, h_tiles, panorama_output, UAs, cropped, full, log_path=self.log_path)