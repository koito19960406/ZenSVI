import asyncio
import json
import os
import random
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
from pathlib import Path

import aiohttp
import geopandas as gpd
import osmnx as ox
import pandas as pd
import requests
from PIL import Image
from shapely.geometry import Point

from zensvi.download.base import BaseDownloader
from zensvi.download.utils.geoprocess import GeoProcessor
from zensvi.download.utils.helpers import standardize_column_names
from zensvi.utils.log import Logger, verbosity_tqdm

warnings.filterwarnings("ignore", category=UserWarning)


class AMSDownloader(BaseDownloader):
    """Amsterdam Street View Downloader class.

    Args:
        log_path (str, optional): Path to the log file. Defaults to
            None.
        max_workers (int, optional): Number of workers for parallel processing. Defaults to None.
        verbosity (int, optional): Level of verbosity for progress bars. Defaults to 1.
                                  0 = no progress bars, 1 = outer loops only, 2 = all loops.
    """

    def __init__(self, log_path=None, max_workers=None, verbosity=1):
        super().__init__(log_path)
        self._max_workers = max_workers
        self._verbosity = verbosity
        self._max_concurrency = None
        if log_path is not None:
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
            int: verbosity level (0=no progress, 1=outer loops only, 2=all loops)
        """
        return self._verbosity

    @verbosity.setter
    def verbosity(self, verbosity):
        self._verbosity = verbosity

    @property
    def max_concurrency(self):
        """Property for the maximum concurrency for async operations.

        Returns:
            int: max_concurrency level
        """
        return self._max_concurrency

    @max_concurrency.setter
    def max_concurrency(self, max_concurrency):
        if max_concurrency is None:
            self._max_concurrency = min(100, os.cpu_count() * 4)
        else:
            self._max_concurrency = max_concurrency

    def _set_dirs(self, dir_output):
        self.dir_output = Path(dir_output)
        self.dir_output.mkdir(parents=True, exist_ok=True)
        self.dir_cache = self.dir_output / "cache_zensvi"
        self.dir_cache.mkdir(parents=True, exist_ok=True)
        self.cache_lat_lon = self.dir_cache / "lat_lon.csv"
        self.cache_pids_raw = self.dir_cache / "pids_raw.csv"

    def _get_pids_from_df(self, df, id_columns=None, use_async=False):
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
                    verbosity=self.verbosity,
                )
                df = gp.get_lat_lon()
            df["lat_lon_id"] = range(1, len(df) + 1)
            df.to_csv(self.cache_lat_lon, index=False)

        if self.cache_pids_raw.exists():
            print("The raw panorama IDs have been read from the cache")
            results_df = pd.read_csv(self.cache_pids_raw)
        else:
            if use_async:
                results_df = asyncio.run(self._get_pids_from_df_async(df))
            else:
                results_df = self._get_pids_from_df_sync(df)

        return results_df

    async def _get_pids_from_df_async(self, df):
        """Async version of PID fetching using async/await for concurrency control."""

        async def get_raw_pids_async(session, semaphore, lat, lon, buffer):
            """Async version of _get_raw_pids."""
            async with semaphore:
                url = f"https://api.data.amsterdam.nl/panorama/panoramas/?near={lon},{lat}&radius={buffer}&srid=4326"

                for attempt in range(10):
                    try:
                        proxy = random.choice(self._proxies)
                        user_agent = random.choice(self._user_agents)
                        headers = {"User-Agent": user_agent["user_agent"]}
                        proxy_url = f"http://{proxy['http']}" if proxy and "http" in proxy else None

                        async with session.get(
                            url, headers=headers, proxy=proxy_url, timeout=aiohttp.ClientTimeout(total=10)
                        ) as response:
                            if response.status == 200:
                                data = await response.json()
                                panoramas = data["_embedded"]["panoramas"]
                                return [item["pano_id"] for item in panoramas]
                    except Exception as e:
                        if attempt == 9:  # Last attempt
                            warnings.warn(f"Failed to get panorama IDs after 10 attempts: {e}")
                            return []
                        continue
                return []

        async def worker_async(session, semaphore, row):
            pids = await get_raw_pids_async(session, semaphore, row.latitude, row.longitude, self.buffer)
            result = []
            for pid in pids:
                result.append(
                    {
                        "lat_lon_id": row.lat_lon_id,
                        "input_latitude": row.latitude,
                        "input_longitude": row.longitude,
                        "pano_id": pid,
                    }
                )
            return result

        results = []
        semaphore = asyncio.Semaphore(self.max_concurrency)

        async with aiohttp.ClientSession() as session:
            tasks = [worker_async(session, semaphore, row) for _, row in df.iterrows()]

            completed = 0
            total_tasks = len(tasks)

            with verbosity_tqdm(
                tasks,
                total=total_tasks,
                desc="Getting pids",
                verbosity=self.verbosity,
                level=1,
            ) as pbar:
                for task in asyncio.as_completed(tasks):
                    try:
                        task_results = await task
                        results.extend(task_results)
                    except Exception as e:
                        print(f"Error: {e}")
                    finally:
                        completed += 1
                        if self.verbosity >= 1:
                            pbar.update(1)

        results_df = pd.DataFrame(results)
        # drop duplicates in pano_id
        results_df = results_df.drop_duplicates(subset=["pano_id"])
        results_df.to_csv(self.cache_pids_raw, index=False)
        return results_df

    def _get_pids_from_df_sync(self, df):
        """Synchronous version of PID fetching using ThreadPoolExecutor."""
        results = []

        def worker(row):
            pids = self._get_raw_pids(row.latitude, row.longitude, self.buffer)
            result = []
            for pid in pids:
                result.append(
                    {
                        "lat_lon_id": row.lat_lon_id,
                        "input_latitude": row.latitude,
                        "input_longitude": row.longitude,
                        "pano_id": pid,
                    }
                )
            return result

        with ThreadPoolExecutor() as executor:
            futures = {executor.submit(worker, row): row for _, row in df.iterrows()}
            for future in verbosity_tqdm(
                as_completed(futures), total=len(futures), desc="Getting pids", verbosity=self.verbosity, level=1
            ):
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
        url = f"https://api.data.amsterdam.nl/panorama/panoramas/?near={lon},{lat}&radius={buffer}&srid=4326"

        for attempt in range(10):
            try:
                proxy = random.choice(self._proxies)
                user_agent = random.choice(self._user_agents)
                headers = {"User-Agent": user_agent["user_agent"]}  # Extract the string from the dictionary
                response = requests.get(url, proxies=proxy, headers=headers)
                response.raise_for_status()
                data = json.loads(response.content)
                panoramas = data["_embedded"]["panoramas"]
                return [item["pano_id"] for item in panoramas]
            except Exception as e:
                if attempt == 9:  # Last attempt
                    warnings.warn(f"Failed to get panorama IDs after 5 attempts: {e}")
                    return []
                print(f"Attempt {attempt + 1} failed: {e}")
                continue

    def _filter_pids_date(self, pid_df, start_date, end_date):
        """Filter panorama IDs by date."""
        pid_df["date"] = pd.to_datetime(pid_df["timestamp"], unit="ms")
        return pid_df[(pid_df["date"] >= start_date) & (pid_df["date"] <= end_date)]

    def _save_image(self, pid, data, cropped):
        """Save an image to disk."""
        img_path = os.path.join(self.dir_output, f"{pid}.jpg")
        try:
            proxy = random.choice(self._proxies)
            headers = {"User-Agent": random.choice(self._user_agents)["user_agent"]}
            image = Image.open(
                BytesIO(
                    requests.get(
                        data["_links"]["equirectangular_medium"]["href"],
                        proxies=proxy,
                        headers=headers,
                    ).content
                )
            )
            if cropped:
                image = image.crop((0, 0, image.width, image.height // 2))
            image.save(img_path)
        except Exception:
            self.logger.log_failed_pid(pid)

    def _save_image_sync(self, pid, img_content, cropped):
        """Synchronous helper for image saving and processing."""
        try:
            img_path = os.path.join(self.dir_output, f"{pid}.jpg")
            image = Image.open(BytesIO(img_content))
            if cropped:
                image = image.crop((0, 0, image.width, image.height // 2))
            image.save(img_path)
        except Exception:
            if self.logger:
                self.logger.log_failed_pid(pid)

    def download_svi(
        self,
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
        grid_size: int = 100,
        max_workers: int = None,
        verbosity: int = None,
        use_async: bool = False,
        max_concurrency: int = None,
    ):
        """Download street view images from Amsterdam Street View API using specified
        parameters.

        Args:
            dir_output (str): The output directory.
            path_pid (str, optional): The path to the panorama ID file.
                Defaults to None.
            cropped (bool, optional): Whether to crop the images.
                Defaults to False.
            lat (float, optional): The latitude for the images. Defaults
                to None.
            lon (float, optional): The longitude for the images.
                Defaults to None.
            input_csv_file (str, optional): The input CSV file. Defaults
                to "".
            input_shp_file (str, optional): The input shapefile.
                Defaults to "".
            input_place_name (str, optional): The input place name.
                Defaults to "".
            buffer (int, optional): The buffer size. Defaults to 0.
            distance (int, optional): The sampling distance for lines.
                Defaults to 10.
            start_date (str, optional): The start date for the panorama
                IDs. Format is isoformat (YYYY-MM-DD). Defaults to None.
            end_date (str, optional): The end date for the panorama IDs.
                Format is isoformat (YYYY- MM-DD). Defaults to None.
            metadata_only (bool, optional): Whether to download metadata
                only. Defaults to False.
            grid (bool, optional): Grid parameter for the GeoProcessor.
                Defaults to False.
            grid_size (int, optional): Grid size parameter for the
                GeoProcessor. Defaults to 1.
            max_workers (int, optional): Number of workers for parallel processing.
                If not specified, uses the value set during initialization.
            verbosity (int, optional): Level of verbosity for progress bars.
                If not specified, uses the value set during initialization.
            use_async (bool, optional): If True, uses async/await for concurrent downloads. Defaults to False.
            max_concurrency (int, optional): Maximum number of concurrent async downloads. If None, defaults to min(100, os.cpu_count() * 4).

        Returns:
            None

        Raises:
            ValueError: If neither lat and lon, csv file, shapefile, nor
                place name is provided.
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
                grid_size=grid_size,
                max_workers=max_workers,
                verbosity=verbosity,
                use_async=use_async,
                max_concurrency=max_concurrency,
            )

        # Set max_workers if provided
        if max_workers is not None:
            self.max_workers = max_workers

        # Set verbosity if provided
        if verbosity is not None:
            self.verbosity = verbosity

        # Set max_concurrency
        self.max_concurrency = max_concurrency

        self.grid = grid
        self.grid_size = grid_size
        self._set_dirs(dir_output)
        self.buffer = buffer
        self.distance = distance

        if input_csv_file:
            df = pd.read_csv(input_csv_file)
            df = standardize_column_names(df)
            pid_df = self._get_pids_from_df(df, use_async=use_async)
        elif input_shp_file:
            gdf = gpd.read_file(input_shp_file)
            pid_df = self._get_pids_from_df(gdf, use_async=use_async)
        elif input_place_name:
            gdf = ox.geocode_to_gdf(input_place_name)
            pid_df = self._get_pids_from_df(gdf, use_async=use_async)
        elif lat is not None and lon is not None:
            df = pd.DataFrame({"latitude": [lat], "longitude": [lon]})
            pid_df = self._get_pids_from_df(df, use_async=use_async)
        else:
            raise ValueError("Please provide either lat and lon, input_csv_file, input_shp_file, or input_place_name.")

        if start_date and end_date:
            pid_df = self._filter_pids_date(pid_df, start_date, end_date)

        async def process_pid_async(session, semaphore, pid, max_retries=5):
            """Async version of process_pid with concurrency control."""
            async with semaphore:
                img_url = f"https://api.data.amsterdam.nl/panorama/panoramas/{pid}/"

                for attempt in range(max_retries):
                    try:
                        proxy = random.choice(self._proxies)
                        headers = {"User-Agent": random.choice(self._user_agents)["user_agent"]}
                        proxy_url = f"http://{proxy['http']}" if proxy and "http" in proxy else None

                        async with session.get(
                            img_url, headers=headers, proxy=proxy_url, timeout=aiohttp.ClientTimeout(total=10)
                        ) as response:
                            if response.status == 200:
                                data = await response.json()

                                if not metadata_only:
                                    # Download image asynchronously
                                    image_url = data["_links"]["equirectangular_medium"]["href"]
                                    async with session.get(image_url, headers=headers, proxy=proxy_url) as img_response:
                                        if img_response.status == 200:
                                            img_content = await img_response.read()

                                            # Use asyncio to run file I/O and image processing in executor
                                            loop = asyncio.get_event_loop()
                                            await loop.run_in_executor(
                                                None, self._save_image_sync, pid, img_content, cropped
                                            )

                                return {
                                    "geometry": Point(data["geometry"]["coordinates"][:2]),
                                    "lat": data["geometry"]["coordinates"][1],
                                    "lon": data["geometry"]["coordinates"][0],
                                    "pano_id": data["pano_id"],
                                    "timestamp": data["timestamp"],
                                    "mission_year": data["mission_year"],
                                    "roll": data["roll"],
                                    "pitch": data["pitch"],
                                    "heading": data["heading"],
                                }
                    except Exception as e:
                        if attempt == max_retries - 1:  # Last attempt
                            print(f"Failed to download data for PID {pid} after {max_retries} attempts: {str(e)}")
                            return None
                        continue
                return None

        def process_pid(pid, max_retries=5):
            img_url = f"https://api.data.amsterdam.nl/panorama/panoramas/{pid}/"

            for attempt in range(max_retries):
                try:
                    proxy = random.choice(self._proxies)
                    headers = {"User-Agent": random.choice(self._user_agents)["user_agent"]}
                    response = requests.get(img_url, proxies=proxy, headers=headers)
                    data = json.loads(response.content)

                    if response.status_code == 200:
                        if not metadata_only:
                            self._save_image(pid, data, cropped)

                        return {
                            "geometry": Point(data["geometry"]["coordinates"][:2]),
                            "lat": data["geometry"]["coordinates"][1],
                            "lon": data["geometry"]["coordinates"][0],
                            "pano_id": data["pano_id"],
                            "timestamp": data["timestamp"],
                            "mission_year": data["mission_year"],
                            "roll": data["roll"],
                            "pitch": data["pitch"],
                            "heading": data["heading"],
                        }
                except Exception as e:
                    if attempt == max_retries - 1:  # Last attempt
                        print(f"Failed to download data for PID {pid} after {max_retries} attempts: {str(e)}")
                        return None
                    continue
            return None

        results = []

        if use_async:
            # Use async processing
            semaphore = asyncio.Semaphore(self.max_concurrency)

            async def run_async_downloads():
                async with aiohttp.ClientSession() as session:
                    tasks = [process_pid_async(session, semaphore, row.pano_id) for _, row in pid_df.iterrows()]

                    completed = 0
                    total_tasks = len(tasks)
                    results_async = []

                    # Use tqdm directly for context manager, respecting verbosity
                    from tqdm import tqdm

                    should_disable = self.verbosity == 0 or 1 > self.verbosity
                    with tqdm(
                        total=total_tasks,
                        desc="Downloading images and metadata",
                        disable=should_disable,
                    ) as pbar:
                        for coro in asyncio.as_completed(tasks):
                            try:
                                result = await coro
                                if result is not None:
                                    results_async.append(result)
                            except Exception as e:
                                print(f"Error in async download: {e}")
                            finally:
                                completed += 1
                                if self.verbosity >= 1:
                                    pbar.update(1)

                    return results_async

            # Run async downloads in event loop
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # If we're already in an async context, create a new thread
                    import concurrent.futures

                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        future = executor.submit(asyncio.run, run_async_downloads())
                        results = future.result()
                else:
                    results = loop.run_until_complete(run_async_downloads())
            except RuntimeError:
                # No event loop exists, create one
                results = asyncio.run(run_async_downloads())
        else:
            # Use synchronous processing
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = [executor.submit(process_pid, row.pano_id) for _, row in pid_df.iterrows()]
                for future in verbosity_tqdm(
                    as_completed(futures),
                    total=len(futures),
                    desc="Downloading images and metadata",
                    verbosity=self.verbosity,
                    level=1,
                ):
                    result = future.result()
                    if result is not None:
                        results.append(result)
        if len(results) == 0:
            print("No data downloaded. Please check the input parameters.")
            return

        final_df = gpd.GeoDataFrame(results, crs="EPSG:4326", geometry="geometry")
        final_df.to_csv(os.path.join(self.dir_output, "ams_pids.csv"), index=False)
        print(f"Metadata saved to {os.path.join(self.dir_output, 'ams_pids.csv')}")
