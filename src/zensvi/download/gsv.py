import datetime
import glob
import math
import os
import random
import shutil
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Union

import geopandas as gpd
import numpy as np
import osmnx as ox
import pandas as pd
import requests
from PIL import Image
from shapely.errors import ShapelyDeprecationWarning
from shapely.geometry import Point
from streetlevel import streetview

from zensvi.download.base import BaseDownloader
from zensvi.download.utils.geoprocess import GeoProcessor
from zensvi.download.utils.get_pids import panoids
from zensvi.download.utils.helpers import create_buffer_gdf, standardize_column_names
from zensvi.download.utils.imtool import ImageTool
from zensvi.utils.log import Logger, verbosity_tqdm

warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning)


class GSVDownloader(BaseDownloader):
    """Google Street View Downloader class.

    Args:
        gsv_api_key (str, optional): Google Street View API key. Defaults to None.
        log_path (str, optional): Path to the log file. Defaults to None.
        distance (int, optional): Distance parameter for the GeoProcessor. Defaults to 1.
        grid (bool, optional): Grid parameter for the GeoProcessor. Defaults to False.
        grid_size (int, optional): Grid size parameter for the GeoProcessor. Defaults to 1.
        max_workers (int, optional): Number of workers for parallel processing. Defaults to None.
        verbosity (int, optional): Level of verbosity for progress bars. Defaults to 1.
                                  0 = no progress bars, 1 = outer loops only, 2 = all loops.

    Raises:
        Warning: If gsv_api_key is not provided.
    """

    def __init__(
        self,
        gsv_api_key: str = None,
        log_path: str = None,
        distance: int = 1,
        grid: bool = False,
        grid_size: int = 1,
        max_workers: int = None,
        verbosity: int = 1,
    ):
        super().__init__(log_path)
        if gsv_api_key is None:
            warnings.warn("Please provide your Google Street View API key to augment metadata.")
        self._gsv_api_key = gsv_api_key
        # initialize the logger
        if self.log_path is not None:
            self.logger = Logger(log_path)
        else:
            self.logger = None
        self._distance = distance
        self._grid = grid
        self._grid_size = grid_size
        self._max_workers = max_workers
        self._verbosity = verbosity

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
            int: verbosity level (0=no progress, 1=outer loops only, 2=all loops)
        """
        return self._verbosity

    @verbosity.setter
    def verbosity(self, verbosity):
        self._verbosity = verbosity

    def _augment_metadata(self, df):
        if self.cache_pids_augmented.exists():
            df = pd.read_csv(self.cache_pids_augmented)
            print("The augmented panorama IDs have been read from the cache")
            return df

        # Create a new directory called "augmented_metadata_checkpoints"
        dir_cache_augmented_metadata = self.dir_cache / "augmented_pids"
        dir_cache_augmented_metadata.mkdir(parents=True, exist_ok=True)

        # Load all the checkpoint csv files
        checkpoints = glob.glob(str(dir_cache_augmented_metadata / "*.csv"))
        checkpoint_start_index = len(checkpoints)

        if checkpoint_start_index > 0:
            completed_rows = pd.concat(
                [pd.read_csv(checkpoint) for checkpoint in checkpoints],
                ignore_index=True,
            )
            completed_indices = completed_rows.index.unique()

            # Filter df to get remaining indices to augment metadata for
            df = df.loc[~df.index.isin(completed_indices)]

        def get_year_month(pid, proxies):
            url = "https://maps.googleapis.com/maps/api/streetview/metadata?pano={}&key={}".format(
                pid, self._gsv_api_key
            )
            while True:
                proxy = random.choice(proxies)
                try:
                    response = requests.get(url, proxies=proxy, timeout=5)
                    break
                except Exception as e:
                    print(f"Proxy {proxy} is not working. Exception: {e}")
                    continue

            response = response.json()
            if response["status"] == "OK":
                # get year and month from date
                try:
                    date = response["date"]
                    year = date.split("-")[0]
                    month = date.split("-")[1]
                except Exception as e:
                    print(f"Error while getting year and month: {e}")
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

        for i in verbosity_tqdm(
            range(num_batches),
            desc=f"Augmenting metadata by batch size {min(batch_size, len(df))}",
            verbosity=self.verbosity,
            level=1,
        ):
            batch_df = df.iloc[i * batch_size : (i + 1) * batch_size].copy()  # Copy the batch data to a new dataframe
            with ThreadPoolExecutor() as executor:
                batch_futures = {
                    executor.submit(worker, row, self._proxies): row.Index for row in batch_df.itertuples()
                }
                for future in verbosity_tqdm(
                    as_completed(batch_futures),
                    total=len(batch_futures),
                    desc=f"Augmenting metadata for batch #{i+1}",
                    verbosity=self.verbosity,
                    level=2,
                ):
                    row_index, year_month = future.result()
                    if year_month["year"] is not None:
                        batch_df.at[row_index, "year"] = year_month["year"]
                        batch_df.at[row_index, "month"] = year_month["month"]

            # Save checkpoint for each batch
            batch_df.to_csv(
                f"{dir_cache_augmented_metadata}/checkpoint_batch_{checkpoint_start_index+i+1}.csv",
                index=False,
            )

        # Merge all checkpoints into a single dataframe
        df = pd.concat(
            [pd.read_csv(checkpoint) for checkpoint in glob.glob(str(dir_cache_augmented_metadata / "*.csv"))],
            ignore_index=True,
        )

        # save the augmented metadata
        df.to_csv(self.cache_pids_augmented, index=False)
        # delete cache_lat_lon
        if self.cache_lat_lon.exists():
            self.cache_lat_lon.unlink()
        # delete cache_pids_raw
        if self.cache_pids_raw.exists():
            self.cache_pids_raw.unlink()
        # delete the cache directory
        if dir_cache_augmented_metadata.exists():
            shutil.rmtree(dir_cache_augmented_metadata)
        return df

    def _get_pids_from_df(self, df, id_columns=None):
        # 1. Create a new directory called "pids" to store each batch pids
        dir_cache_pids = self.dir_cache / "raw_pids"
        dir_cache_pids.mkdir(parents=True, exist_ok=True)

        # 2. Load all the checkpoint csv files
        checkpoints = glob.glob(str(dir_cache_pids / "*.csv"))
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

            completed_ids = completed_rows["lat_lon_id"].drop_duplicates()

            # Merge on the ID column, keeping track of where each row originates
            merged = df.merge(completed_ids, on="lat_lon_id", how="outer", indicator=True)

            # Filter out rows that come from the 'completed_ids' DataFrame
            df = merged[merged["_merge"] == "left_only"].drop(columns="_merge")

        def get_street_view_info(longitude, latitude, proxies):
            results = panoids(
                latitude,
                longitude,
                proxies,
            )
            return results

        def worker(row):
            input_longitude = row.longitude
            input_latitude = row.latitude
            lat_lon_id = row.lat_lon_id
            id_dict = {column: getattr(row, column) for column in id_columns} if id_columns else {}
            return (
                lat_lon_id,
                (input_longitude, input_latitude),
                get_street_view_info(input_longitude, input_latitude, self._proxies),
                id_dict,
            )

        # set lat_lon_id if it doesn't exist
        if "lat_lon_id" not in df.columns:
            df["lat_lon_id"] = np.arange(1, len(df) + 1)
        results = []
        batch_size = 1000  # Modify this to a suitable value
        num_batches = (len(df) + batch_size - 1) // batch_size
        failed_rows = []

        # if there's no rows to process, return completed_ids
        if len(df) == 0:
            try:
                return completed_ids
            except NameError:
                # If completed_ids is not defined (no checkpoints), return empty DataFrame
                return pd.DataFrame()

        # if not, process the rows
        for i in verbosity_tqdm(
            range(num_batches),
            desc=f"Getting pids by batch size {min(batch_size, len(df))}",
            verbosity=self.verbosity,
            level=1,
        ):
            with ThreadPoolExecutor() as executor:
                batch_futures = {
                    executor.submit(worker, row): row
                    for row in df.iloc[i * batch_size : (i + 1) * batch_size].itertuples()
                }
                for future in verbosity_tqdm(
                    as_completed(batch_futures),
                    total=len(batch_futures),
                    desc=f"Getting pids for batch #{i+1}",
                    verbosity=self.verbosity,
                    level=2,
                ):
                    try:
                        (
                            lat_lon_id,
                            (input_longitude, input_latitude),
                            row_results,
                            id_dict,
                        ) = future.result()
                        for result in row_results:
                            result["input_latitude"] = input_latitude
                            result["input_longitude"] = input_longitude
                            result["lat_lon_id"] = lat_lon_id
                            result.update(id_dict)
                            results.append(result)
                    except Exception as e:
                        print(f"Error: {e}")
                        failed_rows.append(batch_futures[future])  # Store the failed row

                # Save checkpoint for each batch
                if len(results) > 0:
                    pd.DataFrame(results).to_csv(
                        f"{dir_cache_pids}/checkpoint_batch_{checkpoint_start_index+i+1}.csv",
                        index=False,
                    )
                results = []  # Clear the results list for the next batch

        # Merge all checkpoints into a single dataframe
        results_df = pd.concat(
            [pd.read_csv(checkpoint) for checkpoint in glob.glob(str(dir_cache_pids / "*.csv"))],
            ignore_index=True,
        )

        # Retry failed rows
        if failed_rows:
            print("Retrying failed rows...")
            with ThreadPoolExecutor() as executor:
                retry_futures = {executor.submit(worker, row): row for row in failed_rows}
                for future in verbosity_tqdm(
                    as_completed(retry_futures),
                    total=len(retry_futures),
                    desc="Retrying failed rows",
                    verbosity=self.verbosity,
                    level=2,
                ):
                    try:
                        (
                            lat_lon_id,
                            (input_longitude, input_latitude),
                            row_results,
                            id_dict,
                        ) = future.result()
                        for result in row_results:
                            result["input_latitude"] = input_latitude
                            result["input_longitude"] = input_longitude
                            result["lat_lon_id"] = lat_lon_id
                            result.update(id_dict)
                            results.append(result)
                    except Exception as e:
                        print(f"Failed again: {e}")

            # Save the results of retried rows as another checkpoint
            if len(results) > 0:
                pd.DataFrame(results).to_csv(f"{dir_cache_pids}/checkpoint_retry.csv", index=False)
                # Merge the retry checkpoint into the final dataframe
                retry_df = pd.read_csv(f"{dir_cache_pids}/checkpoint_retry.csv")
                results_df = pd.concat([results_df, retry_df], ignore_index=True)

        # now save results_df as a new cache after dropping lat_lon_id
        results_df = results_df.drop(columns="lat_lon_id")
        # drop duplicates in panoid and id_columns
        results_df = results_df.drop_duplicates(subset=["panoid"] + id_columns)
        results_df.to_csv(self.cache_pids_raw, index=False)

        # delete the cache directory
        if dir_cache_pids.exists():
            shutil.rmtree(dir_cache_pids)
        return results_df

    def _get_pids_from_gdf(self, gdf, **kwargs):
        """Get panorama IDs from a GeoDataFrame of polygons.

        Args:
          gdf: GeoDataFrame of polygons
          **kwargs: Additional keyword arguments

        Returns:
          pandas DataFrame containing panorama IDs

        """
        # Create processor params, preferring kwargs values over self values
        processor_params = {
            "distance": kwargs.get("distance", self._distance),
            "grid": kwargs.get("grid", self._grid),
            "grid_size": kwargs.get("grid_size", self._grid_size),
            "verbosity": kwargs.get("verbosity", self.verbosity),
        }

        # Remove these keys from kwargs to avoid duplication
        for key in processor_params:
            if key in kwargs:
                del kwargs[key]

        geo_processor = GeoProcessor(
            gdf,
            **processor_params,
            **kwargs,
        )
        points_df = geo_processor.get_lat_lon()
        points_df["lat_lon_id"] = np.arange(1, len(points_df) + 1)

        # add a column to the points dataframe to indicate whether each point is within one of the polygons
        points_df["within_polygon"] = False

        # Check if points is within polygons with progress bar
        for idx, row in verbosity_tqdm(
            points_df.iterrows(),
            total=len(points_df),
            desc="Checking points within polygons",
            verbosity=self.verbosity,
            level=1,
        ):
            point = Point(row["longitude"], row["latitude"])
            # Check if the point is within any of the polygons
            within = False
            for _, poly_row in gdf.iterrows():
                if point.within(poly_row.geometry):
                    within = True
                    break
            points_df.at[idx, "within_polygon"] = within

        # Return only those points within polygons
        points_within_polygons_df = points_df[points_df["within_polygon"]]
        # Drop the 'within_polygon' column
        points_within_polygons_df = points_within_polygons_df.drop(columns="within_polygon")

        # if there are no points within polygons, use all points
        if len(points_within_polygons_df) == 0:
            print("Warning: No points were found within polygons. Using all generated points.")
            points_within_polygons_df = points_df.drop(columns="within_polygon")

        # Get PIDs for these points
        results_df = self._get_pids_from_df(
            points_within_polygons_df, kwargs["id_columns"] if "id_columns" in kwargs else None
        )
        return results_df

    def _get_raw_pids(self, **kwargs):
        if self.cache_pids_raw.exists():
            pid = pd.read_csv(self.cache_pids_raw)
            print("The raw panorama IDs have been read from the cache")
            return pid

        if kwargs["lat"] is not None and kwargs["lon"] is not None:
            pid = panoids(kwargs["lat"], kwargs["lon"], self._proxies)
            pid = pd.DataFrame(pid)
            # add input_lat and input_lon
            pid["input_latitude"] = kwargs["lat"]
            pid["input_longitude"] = kwargs["lon"]
        elif kwargs["input_csv_file"] != "":
            df = pd.read_csv(kwargs["input_csv_file"])
            df = standardize_column_names(df)
            if kwargs["buffer"] > 0:
                gdf = gpd.GeoDataFrame(
                    df,
                    geometry=gpd.points_from_xy(df.longitude, df.latitude),
                    crs="EPSG:4326",
                )
                gdf = create_buffer_gdf(gdf, kwargs["buffer"])
                pid = self._get_pids_from_gdf(gdf, **kwargs)
            else:
                pid = self._get_pids_from_df(df, kwargs["id_columns"])
        elif kwargs["input_shp_file"] != "":
            gdf = gpd.read_file(kwargs["input_shp_file"])
            if kwargs["buffer"] > 0:
                gdf = create_buffer_gdf(gdf, kwargs["buffer"])
            pid = self._get_pids_from_gdf(gdf, **kwargs)
        elif kwargs["input_place_name"] != "":
            print("Geocoding the input place name")
            gdf = ox.geocoder.geocode_to_gdf(kwargs["input_place_name"])
            # raise error if the input_place_name is not found
            if len(gdf) == 0:
                raise ValueError("The input_place_name is not found. Please try another place name.")
            if kwargs["buffer"] > 0:
                gdf = create_buffer_gdf(gdf, kwargs["buffer"])
            pid = self._get_pids_from_gdf(gdf, **kwargs)
        else:
            raise ValueError("Please input the lat and lon, csv file, or shapefile.")

        return pid

    def _get_pids(self, path_pid, **kwargs):
        id_columns = kwargs["id_columns"]
        if id_columns is not None:
            if isinstance(id_columns, str):
                id_columns = [id_columns.lower()]
            elif isinstance(id_columns, list):
                id_columns = [column.lower() for column in id_columns]
        else:
            id_columns = []
        # update id_columns
        kwargs["id_columns"] = id_columns

        # get raw pid
        pid = self._get_raw_pids(**kwargs)

        if kwargs["augment_metadata"] & (self._gsv_api_key is not None):
            pid = self._augment_metadata(pid)
        elif kwargs["augment_metadata"] & (self._gsv_api_key is None):
            raise ValueError("Please set the gsv api key by calling the gsv_api_key method.")
        pid.to_csv(path_pid, index=False)
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

    def _filter_pids_date(self, pid_df, start_date, end_date):
        # create a temporary column date from year and month
        # Fill NA values with a placeholder value
        pid_df["year"] = pid_df["year"].fillna(-1)
        pid_df["month"] = pid_df["month"].fillna(-1)

        # Replace Inf values with a placeholder value
        pid_df["year"] = pid_df["year"].replace([np.inf, -np.inf], -1)
        pid_df["month"] = pid_df["month"].replace([np.inf, -np.inf], -1)

        # Convert to int and then to string, and create the 'date' column
        pid_df["date"] = pid_df["year"].astype(int).astype(str) + "-" + pid_df["month"].astype(int).astype(str)

        # Optionally, you can replace the placeholder values in 'date' column back to NaN
        pid_df["date"] = pid_df["date"].replace("-1--1", np.nan)

        # convert to datetime
        pid_df["date"] = pd.to_datetime(pid_df["date"], format="%Y-%m")
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

    # function to download depth images (ndarray) save to the output directory as .npy files
    def _download_depth(self, path_pid: str, dir_output: str, **kwargs) -> None:
        # read the panorama IDs
        panoids = pd.read_csv(path_pid)["panoid"].tolist()
        # create a folder within dir_output
        panorama_output = Path(dir_output) / "gsv_depth"
        panorama_output.mkdir(parents=True, exist_ok=True)

        # define a function to download depth images
        def download_depth_image(pid):
            pano = streetview.find_panorama_by_id(pid, download_depth=True)
            data = pano.depth.data.copy()
            # pano.depth.data is 256 x 512 ndarray
            # adjust the dimension of the depth image based on the zoom level
            # convert to pillow image
            img = Image.fromarray(data)
            img = img.convert("L")  # Convert to grayscale
            # zoom 0: 512 x 256, zoom 1: 1024 x 512, zoom 2: 2048 x 1024, zoom 3: 4096 x 2048, zoom 4: 8192 x 4096, zoom 5: 16384 x 8192
            if kwargs["zoom"] == 1:
                img = img.resize((1024, 512))
            elif kwargs["zoom"] == 2:
                img = img.resize((2048, 1024))
            elif kwargs["zoom"] == 3:
                img = img.resize((4096, 2048))
            elif kwargs["zoom"] == 4:
                img = img.resize((8192, 4096))
            elif kwargs["zoom"] == 5:
                img = img.resize((16384, 8192))
            img.save(panorama_output / f"{pid}.png")

        # use ThreadPoolExecutor and as_completed to download the depth images
        with ThreadPoolExecutor() as executor:
            futures = {executor.submit(download_depth_image, pid): pid for pid in panoids}
            for future in verbosity_tqdm(
                as_completed(futures),
                desc="Downloading depth images",
                total=len(futures),
                verbosity=self.verbosity,
                level=1,
            ):
                pid = futures[future]
                try:
                    future.result()
                except Exception as e:
                    self.logger.log_error(f"Error downloading depth image for panorama ID {pid}: {e}")
                    continue

    def download_svi(
        self,
        dir_output: str,
        path_pid: str = None,
        zoom: int = 2,
        cropped: bool = False,
        full: bool = True,
        lat: float = None,
        lon: float = None,
        input_csv_file: str = "",
        input_shp_file: str = "",
        input_place_name: str = "",
        id_columns: Union[str, List[str]] = None,
        buffer: int = 0,
        augment_metadata: bool = False,
        batch_size: int = 1000,
        update_pids: bool = False,
        start_date: str = None,
        end_date: str = None,
        metadata_only: bool = False,
        download_depth=False,
        max_workers: int = None,
        verbosity: int = None,
        **kwargs,
    ) -> None:
        """Downloads street view images from Google Street View API using specified
        parameters.

        Args:
            dir_output (str): The output directory.
            path_pid (str, optional): The path to the panorama ID file. Defaults to None.
            zoom (int, optional): The zoom level for the images. Defaults to 2.
            cropped (bool, optional): Whether to crop the images. Defaults to False.
            full (bool, optional): Whether to download full images. Defaults to True.
            lat (float, optional): The latitude for the images. Defaults to None.
            lon (float, optional): The longitude for the images. Defaults to None.
            input_csv_file (str, optional): The input CSV file. Defaults to "".
            input_shp_file (str, optional): The input shapefile. Defaults to "".
            input_place_name (str, optional): The input place name. Defaults to "".
            id_columns (Union[str, List[str]], optional): The ID columns. Defaults to None.
            buffer (int, optional): The buffer size. Defaults to 0.
            augment_metadata (bool, optional): Whether to augment the metadata. Defaults to False.
            batch_size (int, optional): The batch size for downloading. Defaults to 1000.
            update_pids (bool, optional): Whether to update the panorama IDs. Defaults to False.
            start_date (str, optional): The start date for the panorama IDs. Format is isoformat (YYYY-MM-DD). Defaults to None.
            end_date (str, optional): The end date for the panorama IDs. Format is isoformat (YYYY-MM-DD). Defaults to None.
            metadata_only (bool, optional): Whether to download metadata only. Defaults to False.
            download_depth (bool, optional): Whether to download depth images. Defaults to False.
            max_workers (int, optional): Number of workers for parallel processing.
                If not specified, uses the value set during initialization.
            verbosity (int, optional): Level of verbosity for progress bars (0=no progress, 1=outer loops only, 2=all loops).
                If not specified, uses the value set during initialization.
            **kwargs: Additional keyword arguments.

        Returns:
            None

        Raises:
            ValueError: If the zoom level is not between 0 and 5.
            ValueError: If the start_date or end_date format is incorrect.
            ValueError: If the input_place_name is not found.
            ValueError: If neither lat and lon, csv file, or shapefile is provided.
            ValueError: If the gsv_api_key is not set when augment_metadata is True.

        Notes:
            This method logs significant events and errors, making it suitable for both interactive usage and automated workflows.
        """
        if self.logger is not None:
            # record the arguments
            self.logger.log_args(
                "GSVDownloader download_svi",
                dir_output=dir_output,
                path_pid=path_pid,
                zoom=zoom,
                cropped=cropped,
                full=full,
                lat=lat,
                lon=lon,
                input_csv_file=input_csv_file,
                input_shp_file=input_shp_file,
                input_place_name=input_place_name,
                id_columns=id_columns,
                buffer=buffer,
                augment_metadata=augment_metadata,
                batch_size=batch_size,
                update_pids=update_pids,
                start_date=start_date,
                end_date=end_date,
                metadata_only=metadata_only,
                max_workers=max_workers,
                **kwargs,
            )

        # Set max_workers and verbosity
        self.max_workers = max_workers
        if verbosity is not None:
            self.verbosity = verbosity

        # set necessary directories
        self._set_dirs(dir_output)

        # call _get_pids function first if path_pid is None
        if (path_pid is None) & (not self.cache_pids_augmented.exists()):
            print("Getting pids...")
            path_pid = self.dir_output / "gsv_pids.csv"
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
                    id_columns=id_columns,
                    buffer=buffer,
                    augment_metadata=augment_metadata,
                    **kwargs,
                )
        elif self.cache_pids_augmented.exists():
            # copy the cache pids_augmented to path_pid
            path_pid = self.dir_output / "gsv_pids.csv"
            shutil.copy2(self.cache_pids_augmented, path_pid)
            print("The augmented panorama IDs have been saved to {}".format(path_pid))
        # Horizontal Google Street View tiles
        # zoom 3: (8, 4); zoom 5: (26, 13) zoom 2: (4, 2) zoom 1: (2, 1);4:(8,16)
        # zoom = 2
        # h_tiles = 4  # 26
        # v_tiles = 2  # 13
        # cropped = False
        # full = True
        # stop if metadata_only is True
        if metadata_only:
            print("The metadata has been downloaded")
            return
        # create a folder within self.dir_output
        self.panorama_output = self.dir_output / "gsv_panorama"
        self.panorama_output.mkdir(parents=True, exist_ok=True)

        panoids = self._read_pids(path_pid, start_date, end_date)

        if len(panoids) == 0:
            print("There is no panorama ID to download")
            return
        else:
            panoids_rest = self._check_already(panoids)

        if len(panoids_rest) > 0:
            UAs = random.choices(self._user_agents, k=len(panoids_rest))
            # check zoom level is 0<=zoom<=5
            if zoom < 0 or zoom > 5:
                raise ValueError("zoom level should be between 0 and 6")
            h_tiles = 2**zoom
            v_tiles = math.ceil(h_tiles / 2)
            ImageTool.dwl_multiple(
                panoids_rest,
                zoom,
                v_tiles,
                h_tiles,
                self.panorama_output,
                UAs,
                self._proxies,
                cropped,
                full,
                batch_size=batch_size,
                logger=self.logger,
                max_workers=self.max_workers,
                verbosity=self.verbosity,
            )
        else:
            print("All images have been downloaded")

        if download_depth:
            self._download_depth(path_pid, dir_output, zoom=zoom)

        # delete the cache directory
        if self.dir_cache.exists():
            shutil.rmtree(self.dir_cache)
            print("The cache directory has been deleted")

    def update_metadata(
        self,
        input_pid_file: str,
        output_pid_file: str = None,
        max_workers: int = None,
        verbosity: int = None,
    ) -> None:
        """Updates metadata (year and month) for existing panorama IDs without redoing the entire download process.

        Args:
            input_pid_file (str): Path to the existing panorama ID CSV file.
            output_pid_file (str, optional): Path where the updated CSV will be saved.
                                            If None, will overwrite the input file. Defaults to None.
            max_workers (int, optional): Number of workers for parallel processing.
                                        If not specified, uses the value set during initialization.
            verbosity (int, optional): Level of verbosity for progress bars.
                                      If not specified, uses the value set during initialization.

        Returns:
            None

        Raises:
            ValueError: If gsv_api_key is not provided.
            FileNotFoundError: If the input_pid_file does not exist.
        """
        if self._gsv_api_key is None:
            raise ValueError("Please provide your Google Street View API key to augment metadata.")

        input_path = Path(input_pid_file)
        if not input_path.exists():
            raise FileNotFoundError(f"Input file {input_pid_file} does not exist.")

        # Set output path
        output_path = Path(output_pid_file) if output_pid_file else input_path

        # Set max_workers and verbosity
        self.max_workers = max_workers
        if verbosity is not None:
            self.verbosity = verbosity

        # Create a temporary directory for cache
        temp_dir = Path(output_path.parent) / "temp_update_metadata"
        temp_dir.mkdir(parents=True, exist_ok=True)

        # Set up directory structure using _set_dirs method
        self._set_dirs(temp_dir)

        # Read the input PID file
        df = pd.read_csv(input_path)
        print(f"Read {len(df)} panorama IDs from {input_path}")

        # Augment metadata
        print("Augmenting metadata with year and month information...")
        updated_df = self._augment_metadata(df)

        # Save the augmented metadata
        updated_df.to_csv(output_path, index=False)
        print(f"Updated panorama IDs have been saved to {output_path}")

        # Clean up temp directory
        if self.dir_cache.exists():
            shutil.rmtree(self.dir_cache)
            print("Temporary cache directory has been deleted")
