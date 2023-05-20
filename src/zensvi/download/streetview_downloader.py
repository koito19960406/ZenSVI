import os
import random
import time
import datetime
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed, ProcessPoolExecutor
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
import csv
import glob
import shutil
import numpy as np

from zensvi.download.utils.imtool import ImageTool
from zensvi.download.utils.get_pids import panoids
from zensvi.download.utils.geoprocess import GeoProcessor
from zensvi.download.utils.helpers import standardize_column_names, create_buffer_gdf

class StreetViewDownloader:
    def __init__(self, gsv_api_key = None, log_path = None, nthreads = 0, distance = 1, grid = False, grid_size = 1):
        if gsv_api_key == None:
            warnings.warn("Please provide your Google Street View API key to augment metadata.")
        self._gsv_api_key = gsv_api_key
        self._log_path = log_path
        self._nthreads = nthreads
        self._distance = distance
        self._user_agent = self._get_ua()
        self._proxies = self._get_proxies()
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
        user_agent_file = pkg_resources.resource_filename('zensvi.download.utils', 'UserAgent.csv')
        UA = []
        with open(user_agent_file, 'r') as f:
            for line in f:
                ua = {"user_agent": line.strip()}
                UA.append(ua)
        return UA
    
    @property
    def proxies(self):
        return self._proxies
    
    def _get_proxies(self):
        proxies_file = pkg_resources.resource_filename('zensvi.download.utils', 'proxies.csv')
        proxies = []
        # open with "utf-8" encoding to avoid UnicodeDecodeError
        with open(proxies_file, 'r', encoding = "utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                ip = row['ip']
                port = row['port']
                protocols = row['protocols']
                proxy_dict = {protocols: f"{ip}:{port}"}
                proxies.append(proxy_dict)
        return proxies
    
    def _read_pids(self, path_pid):
        pid_df = pd.read_csv(path_pid)
        # get unique pids as a list
        pids = pid_df.iloc[:,0].unique().tolist()
        return pids

    def _check_already(self, all_panoids):
        # Get the set of already downloaded images
        name_r = set(name.split(".")[0] for name in tqdm(os.listdir(self.panorama_output), desc="Checking already downloaded images"))

        # Filter the list of all panoids to only include those not already downloaded
        all_panoids = list(set(all_panoids) - name_r)

        return all_panoids

    def _log_write(self, pids):
        with open(self.log_path, 'a+') as fw:
            for pid in pids:
                fw.write(pid+'\n')
    
    def _augment_metadata(self, df):
        if self.cache_pids_augmented.exists():
            df = pd.read_csv(self.cache_pids_augmented)
            print("The augmented panorama IDs have been read from the cache")
            return df
        
        # Create a new directory called "augmented_metadata_checkpoints"
        dir_cache_augmented_metadata = self.dir_cache / 'augmented_pids'
        dir_cache_augmented_metadata.mkdir(parents=True, exist_ok=True)

        # Load all the checkpoint csv files
        checkpoints = glob.glob(str(dir_cache_augmented_metadata / '*.csv'))
        checkpoint_start_index = len(checkpoints)

        if checkpoint_start_index > 0:
            completed_rows = pd.concat([pd.read_csv(checkpoint) for checkpoint in checkpoints], ignore_index=True)
            completed_indices = completed_rows.index.unique()

            # Filter df to get remaining indices to augment metadata for
            df = df.loc[~df.index.isin(completed_indices)]

        def get_year_month(pid, proxies):
            url = "https://maps.googleapis.com/maps/api/streetview/metadata?pano={}&key={}".format(pid, self.gsv_api_key)
            while True:
                proxy = random.choice(proxies)
                try:
                    response = requests.get(url, proxies=proxy, timeout=5)
                    break
                except Exception as e:
                    print(f"Proxy {proxy} is not working. Exception: {e}")
                    continue

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

        def worker(row, proxies):
            panoid = row.panoid
            year_month = get_year_month(panoid, proxies)
            return row.Index, year_month

        batch_size = 1000  # Modify this to a suitable value
        num_batches = (len(df) + batch_size - 1) // batch_size

        for i in tqdm(range(num_batches), desc=f"Augmenting metadata by batch size {min(batch_size, len(df))}"):
            batch_df = df.iloc[i*batch_size : (i+1)*batch_size].copy()  # Copy the batch data to a new dataframe
            with ThreadPoolExecutor() as executor:
                batch_futures = {executor.submit(worker, row, self.proxies): row.Index for row in batch_df.itertuples()}
                for future in tqdm(as_completed(batch_futures), total=len(batch_futures), desc=f"Augmenting metadata for batch #{i+1}"):
                    row_index, year_month = future.result()
                    batch_df.at[row_index, 'year'] = year_month['year']
                    batch_df.at[row_index, 'month'] = year_month['month']

            # Save checkpoint for each batch
            batch_df.to_csv(f'{dir_cache_augmented_metadata}/checkpoint_batch_{checkpoint_start_index+i+1}.csv', index=False)
        
        # Merge all checkpoints into a single dataframe
        df = pd.concat([pd.read_csv(checkpoint) for checkpoint in glob.glob(str(dir_cache_augmented_metadata / '*.csv'))], ignore_index=True)

        # save the augmented metadata
        df.to_csv(self.cache_pids_augmented, index=False)
        # delete cache_lat_lon
        self.cache_lat_lon.unlink() 
        # delete cache_pids_raw
        self.cache_pids_raw.unlink()
        # delete the cache directory
        if dir_cache_augmented_metadata.exists():
            shutil.rmtree(dir_cache_augmented_metadata)
        return df

    def _get_pids_from_csv(self, df, id_columns=None, closest=False, disp=False):
        # 1. Create a new directory called "pids" to store each batch pids
        dir_cache_pids = self.dir_cache / 'raw_pids'
        dir_cache_pids.mkdir(parents=True, exist_ok=True)

        # 2. Load all the checkpoint csv files
        checkpoints = glob.glob(str(dir_cache_pids / '*.csv'))
        checkpoint_start_index = len(checkpoints)
        if checkpoint_start_index > 0:
            dataframes = []
            for checkpoint in checkpoints:
                try:
                    df_checkpoint = pd.read_csv(checkpoint)
                    dataframes.append(df_checkpoint)
                except pd.errors.EmptyDataError:
                    print(f"Warning: {checkpoint} is empty and has been skipped.")
                    continue
            completed_rows = pd.concat(dataframes, ignore_index=True)

            completed_ids = completed_rows['lat_lon_id'].drop_duplicates()

            # Merge on the ID column, keeping track of where each row originates
            merged = df.merge(completed_ids, on='lat_lon_id', how='outer', indicator=True)

            # Filter out rows that come from the 'completed_ids' DataFrame
            df = merged[merged['_merge'] == 'left_only'].drop(columns='_merge')

        def get_street_view_info(longitude, latitude, proxies):
            results = panoids(latitude, longitude, proxies, closest=closest, disp=disp)
            return results

        def worker(row):
            input_longitude = row.longitude
            input_latitude = row.latitude
            lat_lon_id = row.lat_lon_id
            id_dict = {column: getattr(row, column) for column in id_columns} if id_columns else {}
            return lat_lon_id, (input_longitude, input_latitude), get_street_view_info(input_longitude, input_latitude, self.proxies), id_dict

        results = []
        batch_size = 1000  # Modify this to a suitable value
        num_batches = (len(df) + batch_size - 1) // batch_size
        failed_rows = []
        
        # if there's no rows to process, return completed_ids
        if len(df) == 0:
            return completed_ids
        
        # if not, process the rows
        for i in tqdm(range(num_batches), desc=f"Getting pids by batch size {min(batch_size, len(df))}"):
            with ThreadPoolExecutor() as executor:
                batch_futures = {executor.submit(worker, row): row for row in df.iloc[i*batch_size : (i+1)*batch_size].itertuples()}
                for future in tqdm(as_completed(batch_futures), total=len(batch_futures), desc=f"Getting pids for batch #{i+1}"):
                    try:
                        lat_lon_id, (input_longitude, input_latitude), row_results, id_dict = future.result()
                        for result in row_results:
                            result['input_longitude'] = input_longitude
                            result['input_latitude'] = input_latitude
                            result['lat_lon_id'] = lat_lon_id
                            result.update(id_dict)
                            results.append(result)
                    except Exception as e:
                        print(f"Error: {e}")
                        failed_rows.append(batch_futures[future])  # Store the failed row

                # Save checkpoint for each batch
                if len(results) > 0:
                    pd.DataFrame(results).to_csv(f'{dir_cache_pids}/checkpoint_batch_{checkpoint_start_index+i+1}.csv', index=False)
                results = []  # Clear the results list for the next batch

        # Merge all checkpoints into a single dataframe
        results_df = pd.concat([pd.read_csv(checkpoint) for checkpoint in glob.glob(str(dir_cache_pids / '*.csv'))], ignore_index=True)

        # Retry failed rows
        if failed_rows:
            print("Retrying failed rows...")
            with ThreadPoolExecutor() as executor:
                retry_futures = {executor.submit(worker, row): row for row in failed_rows}
                for future in tqdm(as_completed(retry_futures), total=len(retry_futures), desc="Retrying failed rows"):
                    try:
                        lat_lon_id, (input_longitude, input_latitude), row_results, id_dict = future.result()
                        for result in row_results:
                            result['input_longitude'] = input_longitude
                            result['input_latitude'] = input_latitude
                            result['lat_lon_id'] = lat_lon_id
                            result.update(id_dict)
                            results.append(result)
                    except Exception as e:
                        print(f"Failed again: {e}")

            # Save the results of retried rows as another checkpoint
            if len(results) > 0:
                pd.DataFrame(results).to_csv(f'{dir_cache_pids}/checkpoint_retry.csv', index=False)
                # Merge the retry checkpoint into the final dataframe
                retry_df = pd.read_csv(f'{dir_cache_pids}/checkpoint_retry.csv')
                results_df = pd.concat([results_df, retry_df], ignore_index=True)

        # now save results_df as a new cache after dropping lat_lon_id and drop duplicates in panoid
        results_df = results_df.drop(columns='lat_lon_id')
        results_df = results_df.drop_duplicates(subset='panoid')
        results_df.to_csv(self.cache_pids_raw, index=False)

        # delete the cache directory
        if dir_cache_pids.exists():
            shutil.rmtree(dir_cache_pids)
        return results_df


    def _get_pids_from_gdf(self, gdf, id_columns, closest=False, disp=False):  
        if self.cache_lat_lon.exists():
            df = pd.read_csv(self.cache_lat_lon)
            print("The lat and lon have been read from the cache")
        else:
            # read shapefile
            gp = GeoProcessor(gdf, distance=self.distance, grid=self.grid, grid_size=self.grid_size, id_columns = id_columns)
            df = gp.get_lat_lon()
            df['lat_lon_id'] = np.arange(1, len(df) + 1)
            # save df to cache
            df.to_csv(self.cache_lat_lon, index=False)

        if self.cache_pids_raw.exists():
            print("The raw panorama IDs have been read from the cache")
            results_df = pd.read_csv(self.cache_pids_raw)
        else:
            # Use _get_pids_from_csv to get pids from df
            results_df = self._get_pids_from_csv(df, id_columns, closest=closest, disp=disp)

        # Check if lat and lon are within input polygons
        polygons = gpd.GeoSeries([geom for geom in gdf['geometry'] if geom.type in ['Polygon', 'MultiPolygon']])

        # Convert lat, lon to Points and create a GeoSeries
        points = gpd.GeoSeries([Point(lon, lat) for lon, lat in zip(results_df['lon'], results_df['lat'])])

        # Create a GeoDataFrame with the points and an index column
        points_gdf = gpd.GeoDataFrame(geometry=points, crs=gdf.crs)
        points_gdf['index'] = range(len(points_gdf))

        # Create a spatial index on the polygons GeoSeries
        polygons_sindex = polygons.sindex

        # Function to check whether a point is within any polygon
        def is_within_polygon(point):
            possible_matches_index = list(polygons_sindex.intersection(point.bounds))
            possible_matches = polygons.iloc[possible_matches_index]
            precise_matches = possible_matches.contains(point)
            return precise_matches.any()

        # Add progress bar for within_polygon calculation
        with tqdm(total=len(points), desc="Checking points within polygons") as pbar:
            within_polygon = []
            for point in points_gdf['geometry']:
                within_polygon.append(is_within_polygon(point))
                pbar.update()

        results_df['within_polygon'] = within_polygon

        # Return only those points within polygons
        results_within_polygons_df = results_df[results_df['within_polygon']]
        # Drop the 'within_polygon' column
        results_within_polygons_df = results_within_polygons_df.drop(columns='within_polygon')
        return results_within_polygons_df


    def get_pids(self, path_pid, lat = None, lon = None, input_csv_file = "", input_shp_file = "", id_columns=None, buffer = 0, closest=False, disp=False, augment_metadata=False):
        if id_columns is not None:
            if isinstance(id_columns, str):
                id_columns = [id_columns.lower()]
            elif isinstance(id_columns, list):
                id_columns = [column.lower() for column in id_columns]
        else:
            id_columns = []
        if lat != None and lon != None:
            pid = panoids(lat, lon, closest=closest, disp=disp)
        elif input_csv_file != "":
            df = pd.read_csv(input_csv_file)
            df = standardize_column_names(df)
            if buffer > 0:
                gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.longitude, df.latitude), crs='EPSG:4326')
                gdf = create_buffer_gdf(gdf, buffer)
                pid = self._get_pids_from_gdf(gdf, id_columns, closest=False, disp=False)
            else:
                if self.cache_pids_raw.exists():
                    pid = pd.read_csv(self.cache_pids_raw)
                    print("The raw panorama IDs have been read from the cache")
                else:
                    pid = self._get_pids_from_csv(df, id_columns, closest=False, disp=False)
        elif input_shp_file != "":
            gdf = gpd.read_file(input_shp_file)
            if buffer > 0:
                gdf = create_buffer_gdf(gdf, buffer)
            pid = self._get_pids_from_gdf(gdf, id_columns, closest=False, disp=False)
        else:
            raise ValueError("Please input the lat and lon, csv file, or shapefile.")
        # convert pid to dataframe
        pid_df = pd.DataFrame(pid)
        
        if augment_metadata & (self.gsv_api_key != None):
            pid_df = self._augment_metadata(pid_df)
        elif augment_metadata & (self.gsv_api_key == None):
            raise ValueError("Please set the gsv api key by calling the gsv_api_key method.")
        pid_df.to_csv(path_pid, index=False)
        print("The panorama IDs have been saved to {}".format(path_pid)) 

    def _set_dirs(self, dir_output):
        # set dir_output as attribute and create the directory
        self.dir_output = Path(dir_output)
        self.dir_output.mkdir(parents=True, exist_ok=True)
        # set dir_cache as attribute and create the directory
        self.dir_cache = self.dir_output / "cache_zensvi"
        self.dir_cache.mkdir(parents=True, exist_ok=True)
        # set other cache directories
        self.cache_lat_lon = self.dir_cache / "lat_lon.csv"
        self.cache_pids_raw = self.dir_cache / "pids_raw.csv"
        self.cache_pids_augmented = self.dir_cache / "pids_augemented.csv"
        
    def download_gsv(self, dir_output, path_pid = None, zoom=2, h_tiles=4, v_tiles=2, cropped=False, full=True, 
                lat=None, lon=None, input_csv_file="", input_shp_file = "", id_columns=None, buffer = 0, closest=False, 
                disp=False, augment_metadata=False, update_pids = False):
        # set necessary directories
        self._set_dirs(dir_output)
        
        # call get_pids function first if path_pid is None
        if (path_pid is None) & (self.cache_pids_augmented.exists() == False):
            print("Getting pids...")
            path_pid = self.dir_output / "pids.csv"
            if path_pid.exists() & (update_pids == False):
                print("update_pids is set to False. So the following csv file will be used: {}".format(path_pid))
            else:
                self.get_pids(path_pid, lat=lat, lon=lon,
                            input_csv_file=input_csv_file, input_shp_file = input_shp_file, id_columns=id_columns, buffer = buffer, closest=closest, disp=disp, augment_metadata=augment_metadata)
        elif self.cache_pids_augmented.exists():
            # copy the cache pids_augmented to path_pid
            path_pid = self.dir_output / "pids.csv"
            shutil.copy2(self.cache_pids_augmented, path_pid)
            print("The augmented panorama IDs have been saved to {}".format(path_pid))
        # Horizontal Google Street View tiles
        # zoom 3: (8, 4); zoom 5: (26, 13) zoom 2: (4, 2) zoom 1: (2, 1);4:(8,16)
        # zoom = 2
        # h_tiles = 4  # 26
        # v_tiles = 2  # 13
        # cropped = False
        # full = True
        # create a folder within self.dir_output
        self.panorama_output = self.dir_output / "panorama"
        self.panorama_output.mkdir(parents=True, exist_ok=True)
        
        panoids = self._read_pids(path_pid)
        
        if len(panoids) == 0:
            print("There is no panorama ID to download")
            return
        else:
            panoids_rest = self._check_already(panoids)

        if len(panoids_rest) > 0:
            UAs = random.choices(self.user_agent, k = len(panoids_rest))
            ImageTool.dwl_multiple(panoids_rest, zoom, v_tiles, h_tiles, self.panorama_output, UAs, self.proxies, cropped, full, log_path=self.log_path)
        else:
            print("All images have been downloaded")
        
        # delete the cache directory
        if self.dir_cache.exists():
            shutil.rmtree(self.dir_cache)
            print("The cache directory has been deleted")