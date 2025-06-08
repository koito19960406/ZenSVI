import datetime
import glob
import logging
import os
import random
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import geopandas as gpd
import osmnx as ox
import pandas as pd
import requests
from PIL import Image
from shapely.errors import ShapelyDeprecationWarning

import zensvi.download.kartaview.download_functions as kv
from zensvi.download.base import BaseDownloader
from zensvi.download.utils.helpers import check_and_buffer, standardize_column_names
from zensvi.utils.log import Logger, verbosity_tqdm

warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning)
logging.getLogger("mapillary.utils.client").setLevel(logging.WARNING)


class KVDownloader(BaseDownloader):
    """KartaView Downloader class.

    Args:
        log_path (str, optional): Path to the log file. Defaults to None.
        max_workers (int, optional): Number of workers for parallel processing. Defaults to None.
        verbosity (int, optional): Level of verbosity for progress bars. Defaults to 1.
                                  0 = no progress bars, 1 = outer loops only, 2 = all loops.
    """

    def __init__(self, log_path=None, max_workers=None, verbosity=1):
        super().__init__(log_path)
        self._max_workers = max_workers
        self._verbosity = verbosity
        # initialize the logger
        if log_path is not None:
            # make log directory
            Path(log_path).parent.mkdir(parents=True, exist_ok=True)
            self.logger = Logger(log_path)
        else:
            self.logger = None

    @property
    def max_workers(self):
        """Property for the number of workers for parallel processing.

        Returns:
            int: max_workers
        """
        return self._max_workers

    @max_workers.setter
    def max_workers(self, max_workers):
        if max_workers is None:
            self._max_workers = min(32, os.cpu_count() + 4)
        else:
            self._max_workers = max_workers

    @property
    def verbosity(self):
        """Property for the verbosity level of progress bars.

        Returns:
            int: verbosity level
        """
        return self._verbosity

    @verbosity.setter
    def verbosity(self, verbosity):
        self._verbosity = verbosity

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
        self.pids_url = self.dir_output / "pids_urls_kv.csv"
        # # set dir_cache as attribute and create the directory
        # self.dir_cache = self.dir_output / "cache_zensvi_kv"
        # self.dir_cache.mkdir(parents=True, exist_ok=True)
        # # set other cache directories
        # self.cache_lat_lon = self.dir_cache / "lat_lon_kv.csv"
        # self.cache_pids_raw = self.dir_cache / "pids_raw_kv.csv"

    def _get_pids_from_gdf(self, gdf):
        # set crs to EPSG:4326 if it's None
        if gdf.crs is None:
            gdf = gdf.set_crs("EPSG:4326")
        elif gdf.crs != "EPSG:4326":
            # convert to EPSG:4326
            gdf = gdf.to_crs("EPSG:4326")
        # use get pids with kv.get_points_in_shape
        result_df = kv.get_points_in_shape(gdf)
        return result_df

    def _get_raw_pids(self, **kwargs):
        # if self.cache_pids_raw.exists():
        #     pid = pd.read_csv(self.cache_pids_raw)
        #     print("The raw image IDs have been read from the cache")
        #     return pid

        # input: lat and lon
        if kwargs["lat"] is not None and kwargs["lon"] is not None:
            # create a geodataframe with lat and lon
            df = pd.DataFrame({"lat": [kwargs["lat"]], "lon": [kwargs["lon"]]})
            gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.lon, df.lat), crs="EPSG:4326")

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
                raise ValueError("The input_place_name is not found. Please try another place name.")
        else:
            raise ValueError("Please input the lat and lon, csv file, or shapefile.")

        # check geometry type and buffer
        gdf = check_and_buffer(gdf, kwargs["buffer"])
        # get pid
        pid = self._get_pids_from_gdf(gdf)
        pid = self._filter_pids_date(pid, kwargs["start_date"], kwargs["end_date"])

        return pid

    def _filter_pids_date(self, pid_df, start_date, end_date):
        # create a temporary column date from captured_at (milliseconds from Unix epoch)
        pid_df["date"] = pd.to_datetime(pid_df["shotDate"], format="%Y-%m-%d %H:%M:%S.%f")
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
        pid_df = pid_df[pid_df["date"] >= start_date] if start_date is not None else pid_df
        # if end_date is not None, filter out the rows with date > end_date
        pid_df = pid_df[pid_df["date"] <= end_date] if end_date is not None else pid_df
        # drop the temporary column date
        pid_df = pid_df.drop(columns="date")
        return pid_df

    def _get_pids(self, path_pid, **kwargs):
        # get raw pid
        pid = self._get_raw_pids(**kwargs)

        if pid is None:
            print("There is no image ID to download")
            return

        # move the "id" column to the first column
        pid = pid[["id"] + [col for col in pid.columns if col != "id"]]

        pid.to_csv(path_pid, index=False)
        print("The image IDs have been saved to {}".format(path_pid))

    def _get_urls_kv(self, path_pid):

        pid = pd.read_csv(path_pid)
        if len(pid) == 0:
            print("There is no image ID to download")
            return
        urls = pid[["id", "fileurlProc"]].rename(columns={"fileurlProc": "url"})
        urls.to_csv(self.pids_url, index=False)

    def _download_images_kv(self, path_pid, cropped, batch_size, start_date, end_date):
        checkpoints = glob.glob(str(self.panorama_output / "**/*.png"), recursive=True)

        # Read already downloaded images and convert to ids
        downloaded_ids = set([int(Path(file_path).stem) for file_path in checkpoints])  # Use set for faster operations

        pid_df = pd.read_csv(path_pid).dropna(subset=["id"])
        pid_df["id"] = pid_df["id"].astype("int64")
        urls_df = pd.read_csv(self.pids_url)
        urls_df["id"] = urls_df["id"].astype("int64")
        # merge pid_df and urls_df
        urls_df = urls_df.merge(pid_df, on="id", how="left")
        # filter out the rows by date
        urls_df = self._filter_pids_date(urls_df, start_date, end_date)

        # Filter out the ids that have already been processed
        urls_df = urls_df[~urls_df["id"].isin(downloaded_ids)]  # Use isin for efficient operation

        def worker(row, output_dir, cropped, max_retries=5):
            url, panoid = row.url, row.id
            user_agent = random.choice(self._user_agents)
            proxy = random.choice(self._proxies)

            image_name = f"{panoid}.png"  # Use id for file name
            image_path = output_dir / image_name

            for retry in range(max_retries):
                try:
                    response = requests.get(url, headers=user_agent, proxies=proxy, timeout=10)
                    if response.status_code == 200:
                        with open(image_path, "wb") as f:
                            f.write(response.content)

                        if cropped:
                            img = Image.open(image_path)
                            w, h = img.size
                            img_cropped = img.crop((0, 0, w, h // 2))
                            img_cropped.save(image_path)
                        break
                    else:
                        if retry == max_retries - 1:  # Last retry
                            if self.logger is not None:
                                self.logger.log_failed_pid(panoid)
                except Exception as e:
                    if retry == max_retries - 1:  # Last retry
                        if self.logger is not None:
                            self.logger.log_failed_pid(panoid)
                        print(f"Error: {e}")
                    else:
                        print(f"Retry {retry + 1}/{max_retries} failed: {e}")

        num_batches = (len(urls_df) + batch_size - 1) // batch_size

        # Calculate current highest batch number
        existing_batches = glob.glob(str(self.panorama_output / "batch_*"))
        existing_batch_numbers = [int(Path(batch).name.split("_")[-1]) for batch in existing_batches]
        start_batch_number = max(existing_batch_numbers, default=0)

        for i in verbosity_tqdm(
            range(start_batch_number, start_batch_number + num_batches),
            desc=f"Downloading images by batch size {min(batch_size, len(urls_df))}",
            level=1,
            verbosity=self.verbosity,
        ):
            # Create a new sub-folder for each batch
            batch_out_path = self.panorama_output / f"batch_{i+1}"
            batch_out_path.mkdir(exist_ok=True)

            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                batch_futures = {
                    executor.submit(worker, row, batch_out_path, cropped): row.id
                    for row in urls_df.iloc[
                        (i - start_batch_number) * batch_size : (i + 1 - start_batch_number) * batch_size
                    ].itertuples()
                }
                for future in verbosity_tqdm(
                    as_completed(batch_futures),
                    total=len(batch_futures),
                    desc=f"Downloading images for batch #{i+1}",
                    level=2,
                    verbosity=self.verbosity,
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
        buffer=10,
        update_pids=False,
        cropped=False,
        batch_size=1000,
        start_date=None,
        end_date=None,
        metadata_only=False,
        max_workers=None,
    ):
        """Downloads street view images from KartaView using specified parameters.

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
            cropped (bool, optional): If True, images will be cropped to the upper half. Defaults to False.
            batch_size (int, optional): Number of images to process in each batch. Defaults to 1000.
            start_date (str, optional): Start date (YYYY-MM-DD) to filter images by capture date.
            end_date (str, optional): End date (YYYY-MM-DD) to filter images by capture date.
            metadata_only (bool, optional): If True, skips downloading images and only fetches metadata. Defaults to False.
            max_workers (int, optional): Number of workers for parallel processing. If None, it will be set to min(32, os.cpu_count() + 4).

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
                "KVDownloader download_svi",
                dir_output=dir_output,
                path_pid=path_pid,
                lat=lat,
                lon=lon,
                input_csv_file=input_csv_file,
                input_shp_file=input_shp_file,
                input_place_name=input_place_name,
                buffer=buffer,
                update_pids=update_pids,
                cropped=cropped,
                batch_size=batch_size,
                start_date=start_date,
                end_date=end_date,
                metadata_only=metadata_only,
                max_workers=max_workers,
            )
        # Set max_workers
        self.max_workers = max_workers

        # set necessary directories
        self._set_dirs(dir_output)

        # call _get_pids function first if path_pid is None
        if path_pid is None:
            print("Getting pids...")
            path_pid = self.dir_output / "kv_pids.csv"
            if path_pid.exists() & (not update_pids):
                print("update_pids is set to False. So the following csv file will be used: {}".format(path_pid))
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
                    start_date=start_date,
                    end_date=end_date,
                )
        # stop if metadata_only is True
        if metadata_only:
            print("The metadata has been downloaded")
            return

        # create a folder within self.dir_output
        self.panorama_output = self.dir_output / "kv_svi"
        self.panorama_output.mkdir(parents=True, exist_ok=True)

        # get urls
        if path_pid.exists():
            self._get_urls_kv(path_pid)
            # download images
            self._download_images_kv(path_pid, cropped, batch_size, start_date, end_date)
        else:
            print("There is no imagae ID to download within the given input parameters")

        # # delete the cache directory
        # if self.dir_cache.exists():
        #     shutil.rmtree(self.dir_cache)
        #     print("The cache directory has been deleted")
