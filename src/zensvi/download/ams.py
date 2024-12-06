import os
import warnings
import pandas as pd
import geopandas as gpd
from pathlib import Path
from shapely.geometry import Point
import requests
from PIL import Image
from tqdm import tqdm
from io import BytesIO
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
import random
import osmnx as ox

from zensvi.download.base import BaseDownloader
from zensvi.download.utils.helpers import standardize_column_names
from zensvi.utils.log import Logger
from zensvi.download.utils.geoprocess import GeoProcessor

warnings.filterwarnings("ignore", category=UserWarning)

class AMSDownloader(BaseDownloader):
    """
    Amsterdam Street View Downloader class.

    :param log_path: Path to the log file. Defaults to None.
    :type log_path: str, optional
    """

    def __init__(self, log_path=None):
        super().__init__(log_path)
        if log_path is not None:
            self.logger = Logger(log_path)
        else:
            self.logger = None

    def _set_dirs(self, dir_output):
        self.dir_output = Path(dir_output)
        self.dir_output.mkdir(parents=True, exist_ok=True)
        self.dir_cache = self.dir_output / "cache_zensvi"
        self.dir_cache.mkdir(parents=True, exist_ok=True)
        self.cache_lat_lon = self.dir_cache / "lat_lon.csv"
        self.cache_pids_raw = self.dir_cache / "pids_raw.csv"

    def _get_pids_from_df(self, df, id_columns=None):
        if self.cache_lat_lon.exists():
            df = pd.read_csv(self.cache_lat_lon)
            print("The lat and lon have been read from the cache")
        else:
            if isinstance(df, gpd.GeoDataFrame):
                gp = GeoProcessor(
                    df,
                    distance=self.distance,
                    grid=self.grid,
                    grid_size=self.grid_size,
                )
                df = gp.get_lat_lon()
            df["lat_lon_id"] = range(1, len(df) + 1)
            df.to_csv(self.cache_lat_lon, index=False)

        if self.cache_pids_raw.exists():
            print("The raw panorama IDs have been read from the cache")
            results_df = pd.read_csv(self.cache_pids_raw)
        else:
            results = []

            def worker(row):
                pids = self._get_raw_pids(row.latitude, row.longitude, self.buffer)
                result = []
                for pid in pids:
                    result.append({
                        "lat_lon_id": row.lat_lon_id,
                        "input_latitude": row.latitude,
                        "input_longitude": row.longitude,
                        "pano_id": pid
                    })
                return result

            with ThreadPoolExecutor() as executor:
                futures = {executor.submit(worker, row): row for _, row in df.iterrows()}
                for future in tqdm(as_completed(futures), total=len(futures), desc="Getting pids"):
                    try:
                        results.extend(future.result())
                    except Exception as e:
                        print(f"Error: {e}")

            results_df = pd.DataFrame(results)
            # drop duplicates in pano_id
            results_df = results_df.drop_duplicates(subset=["pano_id"])
            results_df.to_csv(self.cache_pids_raw, index=False)

        return results_df

    def _get_raw_pids(self, lat, lon, buffer):
        """Get raw panorama IDs from the Amsterdam Street View API."""
        url = f'https://api.data.amsterdam.nl/panorama/panoramas/?tags=mission-2022&near={lon},{lat}&radius={buffer}&srid=4326'
        proxy = random.choice(self.proxies)
        user_agent = random.choice(self.user_agents)
        headers = {'User-Agent': user_agent['user_agent']}  # Extract the string from the dictionary
        response = requests.get(url, proxies=proxy, headers=headers)
        data = json.loads(response.content)
        panoramas = data['_embedded']['panoramas']
        return [item['pano_id'] for item in panoramas]

    def _filter_pids_date(self, pid_df, start_date, end_date):
        """Filter panorama IDs by date."""
        pid_df['date'] = pd.to_datetime(pid_df['timestamp'], unit='ms')
        return pid_df[(pid_df['date'] >= start_date) & (pid_df['date'] <= end_date)]

    def _save_image(self, pid, data, cropped):
        """Save an image to disk."""
        img_path = os.path.join(self.dir_output, f"{pid}.jpg")
        try:
            proxy = random.choice(self.proxies)
            headers = {'User-Agent': random.choice(self.user_agents)['user_agent']}
            image = Image.open(BytesIO(requests.get(data['_links']['equirectangular_medium']['href'], proxies=proxy, headers=headers).content))
            if cropped:
                image = image.crop((0, 0, image.width, image.height // 2))
            image.save(img_path)
        except Exception as e:
            self.logger.log_failed_pids(pid)
            
    def download_svi(self, 
                     dir_output: str,
                     path_pid: str = None,
                     cropped: bool = False,
                     lat: float = None,
                     lon: float = None,
                     input_csv_file: str = "",
                     input_shp_file: str = "",
                     input_place_name: str = "",
                     buffer: int = 0,
                     distance: int = 10,
                     start_date: str = None,
                     end_date: str = None,
                     metadata_only: bool = False,
                     grid: bool = False,
                     grid_size: int = 100
                     ):
        """
        Download street view images from Amsterdam Street View API using specified parameters.

        :param dir_output: The output directory.
        :type dir_output: str
        :param path_pid: The path to the panorama ID file. Defaults to None.
        :type path_pid: str, optional
        :param cropped: Whether to crop the images. Defaults to False.
        :type cropped: bool, optional
        :param lat: The latitude for the images. Defaults to None.
        :type lat: float, optional
        :param lon: The longitude for the images. Defaults to None.
        :type lon: float, optional
        :param input_csv_file: The input CSV file. Defaults to "".
        :type input_csv_file: str, optional
        :param input_shp_file: The input shapefile. Defaults to "".
        :type input_shp_file: str, optional
        :param input_place_name: The input place name. Defaults to "".
        :type input_place_name: str, optional
        :param buffer: The buffer size. Defaults to 0.
        :type buffer: int, optional
        :param distance: The sampling distance for lines. Defaults to 10.
        :type distance: int, optional
        :param start_date: The start date for the panorama IDs. Format is isoformat (YYYY-MM-DD). Defaults to None.
        :type start_date: str, optional
        :param end_date: The end date for the panorama IDs. Format is isoformat (YYYY-MM-DD). Defaults to None.
        :type end_date: str, optional
        :param metadata_only: Whether to download metadata only. Defaults to False.
        :type metadata_only: bool, optional
        :param grid: Grid parameter for the GeoProcessor. Defaults to False.
        :type grid: bool, optional
        :param grid_size: Grid size parameter for the GeoProcessor. Defaults to 1.
        :type grid_size: int, optional
        :returns: None

        :raises ValueError: If neither lat and lon, csv file, shapefile, nor place name is provided.
        """
        
        if self.logger is not None:
            self.logger.log_args(
                "AMSDownloader download_svi",
                dir_output=dir_output,
                path_pid=path_pid,
                cropped=cropped,
                lat=lat,
                lon=lon,
                input_csv_file=input_csv_file,
                input_shp_file=input_shp_file,
                input_place_name=input_place_name,
                buffer=buffer,
                distance=distance,
                start_date=start_date,
                end_date=end_date,
                metadata_only=metadata_only,
                grid=grid,
                grid_size=grid_size
            )
            
        self.grid = grid
        self.grid_size = grid_size
        self._set_dirs(dir_output)
        self.buffer = buffer
        self.distance = distance

        if input_csv_file:
            df = pd.read_csv(input_csv_file)
            df = standardize_column_names(df)
            pid_df = self._get_pids_from_df(df)
        elif input_shp_file:
            gdf = gpd.read_file(input_shp_file)
            pid_df = self._get_pids_from_df(gdf)
        elif input_place_name:
            gdf = ox.geocode_to_gdf(input_place_name)
            pid_df = self._get_pids_from_df(gdf)
        elif lat is not None and lon is not None:
            df = pd.DataFrame({'latitude': [lat], 'longitude': [lon]})
            pid_df = self._get_pids_from_df(df)
        else:
            raise ValueError("Please provide either lat and lon, input_csv_file, input_shp_file, or input_place_name.")

        if start_date and end_date:
            pid_df = self._filter_pids_date(pid_df, start_date, end_date)

        def process_pid(pid):
            img_url = f"https://api.data.amsterdam.nl/panorama/panoramas/{pid}/"
            proxy = random.choice(self.proxies)
            headers = {'User-Agent': random.choice(self.user_agents)['user_agent']}
            response = requests.get(img_url, proxies=proxy, headers=headers)
            data = json.loads(response.content)
            if response.status_code == 200:
                if not metadata_only:
                    self._save_image(pid, data, cropped)

                return {
                    'geometry': Point(data['geometry']['coordinates'][:2]),
                    'lat': data['geometry']['coordinates'][1],
                    'lon': data['geometry']['coordinates'][0],
                    'pano_id': data['pano_id'],
                    'timestamp': data['timestamp'],
                    'mission_year': data['mission_year'],
                    'roll': data['roll'],
                    'pitch': data['pitch'],
                    'heading': data['heading']
                }
            else:
                print(f"Failed to download data for PID {pid}")
                return None

        results = []
        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(process_pid, row.pano_id) for _, row in pid_df.iterrows()]
            for future in tqdm(as_completed(futures), total=len(futures), desc="Downloading images and metadata"):
                result = future.result()
                if result is not None:
                    results.append(result)

        final_df = gpd.GeoDataFrame(results, crs="EPSG:4326")
        final_df.to_csv(os.path.join(self.dir_output, "ams_pids.csv"), index=False)
        print(f"Metadata saved to {os.path.join(self.dir_output, 'ams_pids.csv')}")