import os
import warnings
import pandas as pd
import geopandas as gpd
from pathlib import Path
from shapely.geometry import Point
import requests
from PIL import Image
from tqdm import tqdm
import json
from pyproj import Transformer
from shapely.errors import ShapelyDeprecationWarning
from zensvi.download.base import BaseDownloader

warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning)


class AMSDownloader(BaseDownloader):
    def __init__(self, ams_api_key=None, log_path=None):
        super().__init__(log_path)
        self.ams_api_key = ams_api_key

    def _set_dirs(self, dir_output):
        self.dir_output = Path(dir_output)
        self.dir_output.mkdir(parents=True, exist_ok=True)
        self.cache_dir = self.dir_output / "cache_ams"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        # set other cache directories
        self.cache_lat_lon = self.dir_cache / "lat_lon.csv"
        self.cache_pids_raw = self.dir_cache / "pids_raw.csv"

    def _get_pids(self, path_pid, **kwargs):
        # get raw pid
        pid = self._get_raw_pids(**kwargs)

        if pid is None:
            print("There is no panorama ID to download")
            return


    def _read_pids(self, path_pid):
        pid_df = pd.read_csv(path_pid)
        return pid_df['id'].unique().tolist()

    def _get_pids_from_gdf(self, gdf):
        """Extract pids from a GeoDataFrame."""
        # Dummy implementation, adjust as needed
        return gdf['id'].tolist()

    def _get_raw_pids(self, **kwargs):
        """Retrieve raw panorama IDs based on location."""
        # Default mission year: 2022
        url = f'https://api.data.amsterdam.nl/panorama/panoramas/?tags=mission-2022&near={kwargs["lon"]},{kwargs["lat"]}&radius={kwargs["buffer"]}&srid=4326'
        response = requests.get(url)
        data = json.loads(response.content)
        panoramas = data['_embedded']['panoramas']
        pids = []
        pids.extend([item['pano_id'] for item in panoramas])    
        return pids

    def _filter_pids_date(self, pid_df, start_date, end_date):
        """Filter PIDs by date."""
        pid_df['date'] = pd.to_datetime(pid_df['captured_at'], unit='ms')
        return pid_df[(pid_df['date'] >= start_date) & (pid_df['date'] <= end_date)]

    def download_svi(self, dir_output, start_date=None, end_date=None, **kwargs):
        """
        Download SVI from https://api.data.amsterdam.nl
        """
        self._set_dirs(dir_output)
        # path_pid = kwargs.get('path_pid', None)
        # if not path_pid:
        #     raise ValueError("Path to PID file must be provided")
        # TODO
        # pids = self._read_pids(path_pid)
        # filtered_pids = self._filter_pids_date(pd.DataFrame(pids, columns=['id', 'captured_at']), start_date, end_date)
        pids = self._get_raw_pids(kwargs)
        for pid in pids:
            img_url = f"https://api.data.amsterdam.nl/panorama/panoramas/{pid}/"
            response = requests.get(img_url)
            if response.status_code == 200:
                img_path = self.dir_output + "\\"+ f"{pid}.jpg"
                with open(img_path, 'wb') as f:
                    f.write(response.content)
            else:
                print(f"Failed to download image for PID {pid}")


