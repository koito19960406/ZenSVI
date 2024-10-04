import random
import datetime
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings
import requests
from pathlib import Path
import geopandas as gpd
from tqdm import tqdm
import warnings
from shapely.errors import ShapelyDeprecationWarning
import json
import os

warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning)

import glob
import shutil
import numpy as np
import osmnx as ox
from PIL import Image

from zensvi.download.base import BaseDownloader
import zensvi.download.mapillary.interface as mly
from zensvi.download.utils.geoprocess import GeoProcessor
from zensvi.download.utils.helpers import standardize_column_names, check_and_buffer
from zensvi.utils.log import Logger

# set logging level to warning
import logging

logging.getLogger("mapillary.utils.client").setLevel(logging.WARNING)


class MLYDownloader(BaseDownloader):
    """
    Mapillary Downloader class.

    Args:
        mly_api_key (str, optional): Mapillary API key. Defaults to None.
        log_path (str, optional): Path to the log file. Defaults to None.
        max_workers (int, optional): Number of workers for parallel processing. Defaults to None.
    """

    def __init__(self, mly_api_key, log_path=None, max_workers=None):
        super().__init__(log_path)
        self._mly_api_key = mly_api_key
        self._max_workers = max_workers
        mly.set_access_token(self.mly_api_key)
        # initialize the logger
        if log_path is not None:
            self.logger = Logger(log_path)
        else:
            self.logger = None

    @property
    def mly_api_key(self):
        """Property for Mapillary API key.

        :return: mly_api_key
        :rtype: str
        """
        return self._mly_api_key

    @mly_api_key.setter
    def mly_api_key(self, mly_api_key):
        self._mly_api_key = mly_api_key

    @property
    def max_workers(self):
        """Property for the number of workers for parallel processing.

        :return: max_workers
        :rtype: int
        """
        return self._max_workers

    @max_workers.setter
    def max_workers(self, max_workers):
        if max_workers is None:
            self._max_workers = min(32, os.cpu_count() + 4)
        else:
            self._max_workers = max_workers

    def _read_pids(self, path_pid):
        pid_df = pd.read_csv(path_pid)
        # drop NA values in id columns
        pid_df = pid_df.dropna(subset=["id"])
        # get unique pids (ie "id" columns) as a list
        pids = pid_df["id"].astype("int64").unique().tolist()
        return pids

    def _set_dirs(self, dir_output):
        # set dir_output as attribute and create the directory
        self.dir_output = Path(dir_output)
        self.dir_output.mkdir(parents=True, exist_ok=True)
        self.pids_url = self.dir_output / "pids_urls.csv"
        # set dir_cache as attribute and create the directory
        self.dir_cache = self.dir_output / "cache_zensvi"
        self.dir_cache.mkdir(parents=True, exist_ok=True)
        # set other cache directories
        self.cache_lat_lon = self.dir_cache / "lat_lon.csv"
        self.cache_pids_raw = self.dir_cache / "pids_raw.csv"

    def _get_pids_from_gdf(self, gdf, mly_kwargs, **kwargs):
        # set crs to EPSG:4326 if it's None
        if gdf.crs is None:
            gdf = gdf.set_crs("EPSG:4326")
        elif gdf.crs != "EPSG:4326":
            # convert to EPSG:4326
            gdf = gdf.to_crs("EPSG:4326")

        # convert to FeatureCollection
        geojson = json.loads(gdf.to_json())

        # use get pids with mly.interface.images_in_geojson
        if kwargs["use_cache"]:
            result_json = mly.images_in_geojson(
                geojson,
                dir_cache=self.dir_cache,
                max_workers=self.max_workers,
                logger=self.logger,
                **mly_kwargs,
            ).to_dict()["features"]
        else:
            result_json = mly.images_in_geojson(
                geojson, max_workers=self.max_workers, logger=self.logger, **mly_kwargs
            ).to_dict()["features"]

        # convert to geodataframe
        result_gdf = gpd.GeoDataFrame.from_features(result_json)

        # return None if there is no result
        if len(result_gdf) == 0:
            return None

        # add lon and lat columns
        result_gdf["lon"] = result_gdf["geometry"].apply(lambda geom: geom.x)
        result_gdf["lat"] = result_gdf["geometry"].apply(lambda geom: geom.y)

        # drop geometry column
        result_df = result_gdf.drop(columns="geometry")

        return result_df

    def _get_raw_pids(self, **kwargs):
        mly_allowed_keys = {
            "compass_angle",
            "image_type",
            "min_captured_at",
            "max_captured_at",
            "organization_id",
        }
        mly_kwargs = {k: v for k, v in kwargs.items() if k in mly_allowed_keys}
        if self.cache_pids_raw.exists():
            pid = pd.read_csv(self.cache_pids_raw)
            print("The raw panorama IDs have been read from the cache")
            return pid

        # input: lat and lon
        if kwargs["lat"] is not None and kwargs["lon"] is not None:
            # create a geodataframe with lat and lon
            df = pd.DataFrame({"lat": [kwargs["lat"]], "lon": [kwargs["lon"]]})
            gdf = gpd.GeoDataFrame(
                df, geometry=gpd.points_from_xy(df.lon, df.lat), crs="EPSG:4326"
            )

        # input: csv file
        elif kwargs["input_csv_file"] != "":
            df = pd.read_csv(kwargs["input_csv_file"])
            df = standardize_column_names(df)
            gdf = gpd.GeoDataFrame(
                df,
                geometry=gpd.points_from_xy(df.longitude, df.latitude),
                crs="EPSG:4326",
            )

        # input: shapefile
        elif kwargs["input_shp_file"] != "":
            gdf = gpd.read_file(kwargs["input_shp_file"])

        # input: place name
        elif kwargs["input_place_name"] != "":
            print("Geocoding the input place name")
            gdf = ox.geocoder.geocode_to_gdf(kwargs["input_place_name"])
            # raise error if the input_place_name is not found
            if len(gdf) == 0:
                raise ValueError(
                    "The input_place_name is not found. Please try another place name."
                )
        else:
            raise ValueError("Please input the lat and lon, csv file, or shapefile.")

        # check geometry type and buffer
        gdf = check_and_buffer(gdf, kwargs["buffer"])
        # get pid
        pid = self._get_pids_from_gdf(gdf, mly_kwargs, **kwargs)

        return pid

    def _filter_pids_date(self, pid_df, start_date, end_date):
        # create a temporary column date from captured_at (milliseconds from Unix epoch)
        pid_df["date"] = pd.to_datetime(pid_df["captured_at"], unit="ms")
        # check if start_date and end_date are in the correct format with regex. If not, raise error
        if start_date is not None:
            try:
                start_date = datetime.datetime.strptime(start_date, "%Y-%m-%d")
            except ValueError:
                raise ValueError("Incorrect start_date format, should be YYYY-MM-DD")
        if end_date is not None:
            try:
                end_date = datetime.datetime.strptime(end_date, "%Y-%m-%d")
            except ValueError:
                raise ValueError("Incorrect end_date format, should be YYYY-MM-DD")
        # if start_date is not None, filter out the rows with date < start_date
        pid_df = (
            pid_df[pid_df["date"] >= start_date] if start_date is not None else pid_df
        )
        # if end_date is not None, filter out the rows with date > end_date
        pid_df = pid_df[pid_df["date"] <= end_date] if end_date is not None else pid_df
        # drop the temporary column date
        pid_df = pid_df.drop(columns="date")
        return pid_df

    def _get_pids(self, path_pid, **kwargs):
        # get raw pid
        pid = self._get_raw_pids(**kwargs)

        if pid is None or len(pid) == 0:
            print("There is no panorama ID to download")
            return

        # # Assuming that df is your DataFrame and that 'geometry' column contains Point objects
        # # convert the geometry column into shapely geometry objects
        # pid['geometry'] = pid['geometry'].apply(wkt.loads)
        # pid['lon'] = pid['geometry'].apply(lambda geom: geom.x)
        # pid['lat'] = pid['geometry'].apply(lambda geom: geom.y)
        # # drop geometry column
        # pid = pid.drop(columns='geometry')
        # move the "id" column to the first column
        pid = pid[["id"] + [col for col in pid.columns if col != "id"]]

        # keep id,captured_at,compass_angle,is_pano,organization_id,sequence_id,input_latitude,input_longitude,lon,lat drop other columns
        pid = pid[
            [
                "id",
                "captured_at",
                "compass_angle",
                "creator_id",
                "is_pano",
                "organization_id",
                "sequence_id",
                "lon",
                "lat",
            ]
        ]

        pid.to_csv(path_pid, index=False)
        print("The panorama IDs have been saved to {}".format(path_pid))

        # if use_cache, delete self.dir_cache / "tiles_results" directory
        if kwargs["use_cache"]:
            dir_cache_tiles = self.dir_cache / "tiles_results"
            if dir_cache_tiles.exists():
                shutil.rmtree(dir_cache_tiles)
                print("The cache directory for tiles has been deleted")

    def _get_urls_mly(self, path_pid, resolution=1024):
        # check if seld.cache_pids_urls exists
        if self.pids_url.exists():
            print("The panorama URLs have been read from the cache")
            return

        dir_cache_urls = self.dir_cache / "urls"
        dir_cache_urls.mkdir(parents=True, exist_ok=True)

        checkpoints = glob.glob(str(dir_cache_urls / "*.csv"))
        checkpoint_start_index = len(checkpoints)

        panoids = set(self._read_pids(path_pid))  # Convert to set for faster operations
        if len(panoids) == 0:
            print("There is no panorama ID to download")
            return

        # Read all panoids from the checkpoint files
        completed_panoids = set()  # Use set for faster operations
        for checkpoint in checkpoints:
            try:
                df_checkpoint = pd.read_csv(checkpoint)
                completed_panoids.update(df_checkpoint["id"].tolist())
            except pd.errors.EmptyDataError:
                print(f"Warning: {checkpoint} is empty and has been skipped.")
                continue

        # Filter out the panoids that have already been processed
        panoids = list(
            panoids - completed_panoids
        )  # Subtract sets and convert back to list
        if len(panoids) == 0:
            print("All images have been downloaded")
            return

        def worker(panoid, resolution):
            url = mly.image_thumbnail(panoid, resolution=resolution)
            return panoid, url

        results = {}
        batch_size = 1000  # Modify this to a suitable value
        num_batches = (len(panoids) + batch_size - 1) // batch_size

        for i in tqdm(
            range(num_batches),
            desc=f"Getting urls by batch size {min(batch_size, len(panoids))}",
        ):
            with ThreadPoolExecutor() as executor:
                batch_futures = {
                    executor.submit(worker, panoid, resolution): panoid
                    for panoid in panoids[i * batch_size : (i + 1) * batch_size]
                }

                for future in tqdm(
                    as_completed(batch_futures),
                    total=len(batch_futures),
                    desc=f"Getting urls for batch #{i+1}",
                ):
                    current_panoid = batch_futures[future]
                    try:
                        panoid, url = future.result()
                        results[panoid] = url
                    except Exception as e:
                        print(f"Error: {e}")
                        if self.logger is not None:
                            self.logger.log_failed_pids(current_panoid)
                        continue

            if len(results) > 0:
                pd.DataFrame.from_dict(results, orient="index").reset_index().rename(
                    columns={"index": "id", 0: "url"}
                ).to_csv(
                    f"{dir_cache_urls}/checkpoint_batch_{checkpoint_start_index+i+1}.csv",
                    index=False,
                )
            results = {}

        # Merge all checkpoints into a single dataframe
        results_df = pd.concat(
            [
                pd.read_csv(checkpoint)
                for checkpoint in glob.glob(str(dir_cache_urls / "*.csv"))
            ],
            ignore_index=True,
        )
        results_df.to_csv(self.pids_url, index=False)

        if dir_cache_urls.exists():
            shutil.rmtree(dir_cache_urls)

    def _download_images_mly(self, path_pid, cropped, batch_size, start_date, end_date):
        checkpoints = glob.glob(str(self.panorama_output / "**/*.png"), recursive=True)

        # Read already downloaded images and convert to ids
        downloaded_ids = set(
            [Path(file_path).stem for file_path in checkpoints]
        )  # Use set for faster operations

        pid_df = pd.read_csv(path_pid).dropna(subset=["id"])
        pid_df["id"] = pid_df["id"].astype("int64")
        urls_df = pd.read_csv(self.pids_url)
        urls_df["id"] = urls_df["id"].astype("int64")
        # merge pid_df and urls_df
        urls_df = urls_df.merge(pid_df, on="id", how="left")
        # filter out the rows by date
        urls_df = self._filter_pids_date(urls_df, start_date, end_date)

        # Filter out the ids that have already been processed
        urls_df = urls_df[
            ~urls_df["id"].isin(downloaded_ids)
        ]  # Use isin for efficient operation

        def worker(row, output_dir, cropped):
            url, panoid = row.url, row.id
            user_agent = random.choice(self.user_agents)
            proxy = random.choice(self.proxies)

            image_name = f"{panoid}.png"  # Use id for file name
            image_path = output_dir / image_name
            try:
                response = requests.get(
                    url, headers=user_agent, proxies=proxy, timeout=10
                )
                if response.status_code == 200:
                    with open(image_path, "wb") as f:
                        f.write(response.content)

                    if cropped:
                        img = Image.open(image_path)
                        w, h = img.size
                        img_cropped = img.crop((0, 0, w, h // 2))
                        img_cropped.save(image_path)

                else:
                    if self.logger is not None:
                        self.logger.log_failed_pids(panoid)
            except Exception as e:
                if self.logger is not None:
                    self.logger.log_failed_pids(panoid)
                print(f"Error: {e}")

        num_batches = (len(urls_df) + batch_size - 1) // batch_size

        # Calculate current highest batch number
        existing_batches = glob.glob(str(self.panorama_output / "batch_*"))
        existing_batch_numbers = [
            int(Path(batch).name.split("_")[-1]) for batch in existing_batches
        ]
        start_batch_number = max(existing_batch_numbers, default=0)

        for i in tqdm(
            range(start_batch_number, start_batch_number + num_batches),
            desc=f"Downloading images by batch size {min(batch_size, len(urls_df))}",
        ):
            # Create a new sub-folder for each batch
            batch_out_path = self.panorama_output / f"batch_{i+1}"
            batch_out_path.mkdir(exist_ok=True)

            with ThreadPoolExecutor() as executor:
                batch_futures = {
                    executor.submit(worker, row, batch_out_path, cropped): row.id
                    for row in urls_df.iloc[
                        i * batch_size : (i + 1) * batch_size
                    ].itertuples()
                }
                for future in tqdm(
                    as_completed(batch_futures),
                    total=len(batch_futures),
                    desc=f"Downloading images for batch #{i+1}",
                ):
                    try:
                        future.result()
                    except Exception as e:
                        print(f"Error: {e}")

    def download_svi(
        self,
        dir_output,
        path_pid=None,
        lat=None,
        lon=None,
        input_csv_file="",
        input_shp_file="",
        input_place_name="",
        buffer=0,
        update_pids=False,
        resolution=1024,
        cropped=False,
        batch_size=1000,
        start_date=None,
        end_date=None,
        metadata_only=False,
        use_cache=True,
        **kwargs,
    ):
        """
        Downloads street view images from Mapillary using specified parameters.

        Args:
            dir_output (str): Directory where output files and images will be stored.
            path_pid (str, optional): Path to a file containing panorama IDs. If not provided, IDs will be fetched based on other parameters.
            lat (float, optional): Latitude to fetch panorama IDs around this point. Must be used with `lon`.
            lon (float, optional): Longitude to fetch panorama IDs around this point. Must be used with `lat`.
            input_csv_file (str, optional): Path to a CSV file containing locations for which to fetch panorama IDs.
            input_shp_file (str, optional): Path to a shapefile containing geographic locations for fetching panorama IDs.
            input_place_name (str, optional): A place name for geocoding to fetch panorama IDs.
            buffer (int, optional): Buffer size in meters to expand the geographic area for panorama ID fetching.
            update_pids (bool, optional): If True, will update panorama IDs even if a valid `path_pid` is provided. Defaults to False.
            resolution (int, optional): The resolution of the images to download. Defaults to 1024.
            cropped (bool, optional): If True, images will be cropped to the upper half. Defaults to False.
            batch_size (int, optional): Number of images to process in each batch. Defaults to 1000.
            start_date (str, optional): Start date (YYYY-MM-DD) to filter images by capture date.
            end_date (str, optional): End date (YYYY-MM-DD) to filter images by capture date.
            metadata_only (bool, optional): If True, skips downloading images and only fetches metadata. Defaults to False.
            use_cache (bool, optional): If True, uses cached data to speed up the operation. Defaults to True.
            **kwargs: Additional keyword arguments that are passed to the API.

        Returns:
            None: This method does not return a value but will save files directly to the specified output directory.

        Raises:
            ValueError: If required parameters for fetching panorama IDs are not adequately specified.
            FileNotFoundError: If `path_pid` is specified but the file does not exist.

        Notes:
            This method logs significant events and errors, making it suitable for both interactive usage and automated workflows.
        """
        if self.logger is not None:
            # record the arguments
            self.logger.log_args(
                "MLYDownloader download_svi",
                dir_output,
                path_pid,
                lat,
                lon,
                input_csv_file,
                input_shp_file,
                input_place_name,
                buffer,
                update_pids,
                resolution,
                cropped,
                batch_size,
                start_date,
                end_date,
                metadata_only,
                use_cache,
                **kwargs,
            )
        # set necessary directories
        self._set_dirs(dir_output)

        # call _get_pids function first if path_pid is None
        if path_pid is None:
            print("Getting pids...")
            path_pid = self.dir_output / "mly_pids.csv"
            if path_pid.exists() & (update_pids == False):
                print(
                    "update_pids is set to False. So the following csv file will be used: {}".format(
                        path_pid
                    )
                )
            else:
                self._get_pids(
                    path_pid,
                    lat=lat,
                    lon=lon,
                    input_csv_file=input_csv_file,
                    input_shp_file=input_shp_file,
                    input_place_name=input_place_name,
                    buffer=buffer,
                    start_date=start_date,
                    end_date=end_date,
                    use_cache=use_cache,
                    **kwargs,
                )
        else:
            # check if the path_pid exists
            if Path(path_pid).exists():
                print("The following csv file will be used: {}".format(path_pid))
            else:
                self._get_pids(
                    path_pid,
                    lat=lat,
                    lon=lon,
                    input_csv_file=input_csv_file,
                    input_shp_file=input_shp_file,
                    input_place_name=input_place_name,
                    buffer=buffer,
                    **kwargs,
                )
        # stop if metadata_only is True
        if metadata_only:
            print("The metadata has been downloaded")
            return

        # create a folder within self.dir_output
        self.panorama_output = self.dir_output / "mly_svi"
        self.panorama_output.mkdir(parents=True, exist_ok=True)

        # get urls
        if path_pid.exists():
            self._get_urls_mly(path_pid, resolution=resolution)
            # download images
            self._download_images_mly(
                path_pid, cropped, batch_size, start_date, end_date
            )
        else:
            print(
                "There is no panorama ID to download within the given input parameters"
            )

        # delete the cache directory
        if self.dir_cache.exists():
            shutil.rmtree(self.dir_cache)
            print("The cache directory has been deleted")
