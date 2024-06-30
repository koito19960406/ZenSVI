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
from typing import List, Union
from pyproj import Transformer
from shapely.errors import ShapelyDeprecationWarning
from zensvi.download.base import BaseDownloader
from zensvi.utils.log import Logger

warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning)


class AMSDownloader(BaseDownloader):
    def __init__(self, ams_api_key=None, log_path=None):
        super().__init__(log_path)
        self.ams_api_key = ams_api_key
        if log_path is not None:
            self.logger = Logger(log_path)
        else:
            self.logger = None

    def _set_dirs(self, dir_output):
        self.dir_output = Path(dir_output)
        # self.dir_output.mkdir(parents=True, exist_ok=True)
        # self.cache_dir = self.dir_output / "cache_ams"
        # self.cache_dir.mkdir(parents=True, exist_ok=True)
        # set other cache directories
        # self.cache_lat_lon = self.dir_cache / "lat_lon.csv"
        # self.cache_pids_raw = self.dir_cache / "pids_raw.csv"

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

    def _get_raw_pids(self, lat, lon, buffer):
        """Retrieve raw panorama IDs based on location."""
        # Default mission year: 2022
        url = f'https://api.data.amsterdam.nl/panorama/panoramas/?tags=mission-2022&near={lon},{lat}&radius={buffer}&srid=4326'
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

    def download_svi(self, 
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
        **kwargs):
        """
        Download SVI from https://api.data.amsterdam.nl

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
        **kwargs: Additional keyword arguments.
        """
        if self.logger is not None:
            self.logger.log_args(
                "AMSDownloader download_svi",
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
                **kwargs
            )
        self._set_dirs(dir_output)
        # path_pid = kwargs.get('path_pid', None)
        # if not path_pid:
        #     raise ValueError("Path to PID file must be provided")
        # TODO
        # pids = self._read_pids(path_pid)
        # filtered_pids = self._filter_pids_date(pd.DataFrame(pids, columns=['id', 'captured_at']), start_date, end_date)

        pids = self._get_raw_pids(lat, lon, buffer)

        df = pd.DataFrame()

        for pid in pids:
            img_url = f"https://api.data.amsterdam.nl/panorama/panoramas/{pid}/"
            response = requests.get(img_url)
            data = json.loads(response.content)
            if response.status_code == 200:
                img_path = os.path.join(self.dir_output, f"{pid}.jpg")
                image = Image.open(BytesIO(requests.get(data['_links']['equirectangular_medium']['href']).content))
                image.save(img_path)
            else:
                print(f"Failed to download image for PID {pid}")

            if not metadata_only:

                data_dict = {}
                data_dict['geometry'] = [Point(data['geometry']['coordinates'][:2])]
                data_dict['lat'] = [data['geometry']['coordinates'][1]]
                data_dict['lon'] = [data['geometry']['coordinates'][0]]
                items = ["pano_id", "timestamp",'mission_year','roll','pitch', 'heading']
                for item in items:
                    data_dict[item] = [data[item]]

                _df = pd.DataFrame.from_dict(data_dict)
                df = pd.concat([df,_df])  
        df.to_csv(os.path.join(self.dir_output, "asv_pids.csv"))  

