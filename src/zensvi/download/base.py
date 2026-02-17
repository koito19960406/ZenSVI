import csv
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Union

import pandas as pd

# Use importlib.resources for Python 3.9+ compatibility
try:
    from importlib.resources import files
except ImportError:
    # Fallback for Python 3.8 and earlier
    from importlib_resources import files


class BaseDownloader(ABC):
    """Abstract base class for downloading street view images.

    This class provides common functionality for downloading street view images from
    different providers. It handles user agents, proxies, logging and checking for
    already downloaded images.

    Attributes:
        _log_path (str): Path to log file for recording downloaded image IDs
        _user_agents (list): List of user agent strings to use for requests
        _proxies (list): List of proxy servers to use for requests
    """

    @abstractmethod
    def __init__(self, log_path: Optional[str] = None) -> None:
        """Initialize the downloader.

        Args:
            log_path (str, optional): Path to log file. Defaults to None.
        """
        self._log_path = log_path
        self._user_agents = self._get_ua()
        self._proxies = self._get_proxies()

    @property
    def log_path(self) -> Optional[str]:
        """Property for log_path.

        Returns:
            str: Path to the log file
        """
        return self._log_path

    @log_path.setter
    def log_path(self, log_path: Optional[str]) -> None:
        """Setter for log_path.

        Args:
            log_path (str): Path to the log file
        """
        self._log_path = log_path

    def _get_proxies(self) -> List[Dict[str, str]]:
        """Get list of proxy servers from CSV file.

        Returns:
            list: List of dictionaries containing proxy information
        """
        # Use importlib.resources to access package data files
        try:
            utils_files = files("zensvi.download.utils")
            proxies_file = utils_files / "proxies.csv"

            # Read the file content
            with proxies_file.open("r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                proxies = []
                for row in reader:
                    ip = row["ip"]
                    port = row["port"]
                    protocols = row["protocols"]
                    proxy_dict = {protocols: f"{ip}:{port}"}
                    proxies.append(proxy_dict)
            return proxies
        except Exception:
            # Fallback: return empty list if file not found
            # This ensures the package works even without proxy file
            return []

    def _get_ua(self) -> List[Dict[str, str]]:
        """Get list of user agents from CSV file.

        Returns:
            list: List of dictionaries containing user agent strings
        """
        # Use importlib.resources to access package data files
        try:
            utils_files = files("zensvi.download.utils")
            user_agent_file = utils_files / "UserAgent.csv"

            # Read the file content
            UA = []
            with user_agent_file.open("r", encoding="utf-8") as f:
                for line in f:
                    ua = {"user_agent": line.strip()}
                    UA.append(ua)
            return UA
        except Exception:
            # Fallback: return a default user agent if file not found
            return [{"user_agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}]

    def _log_write(self, pids: List[str]) -> None:
        """Write panorama IDs to log file.

        Args:
            pids (list): List of panorama IDs to log
        """
        if self.log_path is None:
            return
        with open(self.log_path, "a+") as fw:
            for pid in pids:
                fw.write(pid + "\n")

    def _check_already(self, all_panoids: List[str]) -> List[str]:
        """Check which panorama IDs have already been downloaded.

        Args:
            all_panoids (list): List of all panorama IDs to check

        Returns:
            list: List of panorama IDs that have not been downloaded yet
        """
        # Get the set of already downloaded images
        name_r = set()
        for dirpath, dirnames, filenames in os.walk(self.panorama_output):
            for name in filenames:
                name_r.add(name.split(".")[0])

        # Filter the list of all panoids to only include those not already downloaded
        all_panoids = list(set(all_panoids) - name_r)
        return all_panoids

    def _read_pids(
        self, path_pid: Union[str, Path], start_date: Optional[str], end_date: Optional[str]
    ) -> List[str]:
        """Read and filter panorama IDs from CSV file.

        Args:
            path_pid (str): Path to CSV file containing panorama IDs
            start_date (str): Start date for filtering
            end_date (str): End date for filtering

        Returns:
            list: List of filtered panorama IDs
        """
        pid_df = pd.read_csv(path_pid)
        # filter pids by date
        pid_df = self._filter_pids_date(pid_df, start_date, end_date)
        # get unique pids as a list
        pids = pid_df.iloc[:, 0].unique().tolist()
        return pids

    @abstractmethod
    def _filter_pids_date(
        self, pid_df: pd.DataFrame, start_date: Optional[str], end_date: Optional[str]
    ) -> pd.DataFrame:
        """Filter panorama IDs by date range.

        Args:
            pid_df (pandas.DataFrame): DataFrame containing panorama IDs and dates
            start_date (str): Start date for filtering
            end_date (str): End date for filtering

        Returns:
            pandas.DataFrame: Filtered DataFrame
        """
        pass

    @abstractmethod
    def download_svi(
        self,
        dir_output: str,
        lat: Optional[float] = None,
        lon: Optional[float] = None,
        input_csv_file: str = "",
        input_shp_file: str = "",
        input_place_name: str = "",
        id_columns: Optional[List[str]] = None,
        buffer: float = 0,
        update_pids: bool = False,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        metadata_only: bool = False,
    ) -> None:
        """Download street view images.

        Args:
            dir_output (str): Output directory for downloaded images
            lat (float, optional): Latitude. Defaults to None.
            lon (float, optional): Longitude. Defaults to None.
            input_csv_file (str, optional): Input CSV file path. Defaults to "".
            input_shp_file (str, optional): Input shapefile path. Defaults to "".
            input_place_name (str, optional): Input place name. Defaults to "".
            id_columns (list, optional): ID columns. Defaults to None.
            buffer (float, optional): Buffer distance. Defaults to 0.
            update_pids (bool, optional): Whether to update PIDs. Defaults to False.
            start_date (str, optional): Start date for filtering. Defaults to None.
            end_date (str, optional): End date for filtering. Defaults to None.
            metadata_only (bool, optional): Whether to download metadata only. Defaults to False.
        """
        pass
