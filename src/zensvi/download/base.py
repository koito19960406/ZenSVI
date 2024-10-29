import csv
import os
from abc import ABC, abstractmethod

import pandas as pd
import pkg_resources


class BaseDownloader(ABC):
    @abstractmethod
    def __init__(self, log_path=None):
        self._log_path = log_path
        self._user_agents = self._get_ua()
        self._proxies = self._get_proxies()

    @property
    def log_path(self):
        """Property for log_path.

        Returns:
            str: log_path
        """
        return self._log_path

    @log_path.setter
    def log_path(self, log_path):
        self._log_path = log_path

    def _get_proxies(self):
        proxies_file = pkg_resources.resource_filename("zensvi.download.utils", "proxies.csv")
        proxies = []
        # open with "utf-8" encoding to avoid UnicodeDecodeError
        with open(proxies_file, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                ip = row["ip"]
                port = row["port"]
                protocols = row["protocols"]
                proxy_dict = {protocols: f"{ip}:{port}"}
                proxies.append(proxy_dict)
        return proxies

    def _get_ua(self):
        user_agent_file = pkg_resources.resource_filename("zensvi.download.utils", "UserAgent.csv")
        UA = []
        with open(user_agent_file, "r") as f:
            for line in f:
                ua = {"user_agent": line.strip()}
                UA.append(ua)
        return UA

    def _log_write(self, pids):
        if self.log_path is None:
            return
        with open(self.log_path, "a+") as fw:
            for pid in pids:
                fw.write(pid + "\n")

    def _check_already(self, all_panoids):
        # Get the set of already downloaded images
        name_r = set()
        for dirpath, dirnames, filenames in os.walk(self.panorama_output):
            for name in filenames:
                name_r.add(name.split(".")[0])

        # Filter the list of all panoids to only include those not already downloaded
        all_panoids = list(set(all_panoids) - name_r)
        return all_panoids

    def _read_pids(self, path_pid, start_date, end_date):
        pid_df = pd.read_csv(path_pid)
        # filter pids by date
        pid_df = self._filter_pids_date(pid_df, start_date, end_date)
        # get unique pids as a list
        pids = pid_df.iloc[:, 0].unique().tolist()
        return pids

    @abstractmethod
    def _filter_pids_date(self, pid_df, start_date, end_date):
        pass

    @abstractmethod
    def download_svi(
        self,
        dir_output,
        lat=None,
        lon=None,
        input_csv_file="",
        input_shp_file="",
        input_place_name="",
        id_columns=None,
        buffer=0,
        update_pids=False,
        start_date=None,
        end_date=None,
        metadata_only=False,
    ):
        """Download street view images.

        Args:
            dir_output: Output directory for downloaded images
            lat: Latitude
            lon: Longitude
            input_csv_file: Input CSV file path
            input_shp_file: Input shapefile path
            input_place_name: Input place name
            id_columns: ID columns
            buffer: Buffer distance
            update_pids: Whether to update PIDs
            start_date: Start date for filtering
            end_date: End date for filtering
            metadata_only: Whether to download metadata only
        """
        pass
