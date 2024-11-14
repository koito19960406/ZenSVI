import os

from huggingface_hub import HfApi, hf_hub_download

from zensvi.download.base import BaseDownloader
from zensvi.utils.log import Logger


class GSDownloader(BaseDownloader):
    """Global Streetscapes class to download the original NUS Global Streetscapes
    dataset.

    To download the models used in the dataset, please see
    `cv/classification/utils/global_streetscapes.py`
    """

    def __init__(self, log_path=None):
        super().__init__(log_path)
        self._repo_id = "NUS-UAL/global-streetscapes"
        self._repo_type = "dataset"

        # initialize the logger
        if log_path is not None:
            self.logger = Logger(log_path)
        else:
            self.logger = None

    def _filter_pids_date(self):
        """Required abstract methods from parent class."""
        pass

    def download_svi(self):
        """Required abstract methods from parent class."""
        pass

    @property
    def repo_id(self):
        """Property for Huggingface download.

        :return: repo_id
        :rtype: str
        """
        return self._repo_id

    @property
    def repo_type(self):
        """Property for Huggingface download.

        :return: repo_type
        :rtype: str
        """
        return self._repo_type

    def _download_folder(self, folder_path, local_dir):
        """Download an entire folder from a huggingface dataset repository.

        :param folder_path: Folder path within the repository
        :type folder_path: str
        :param local_dir: Local folder path to download the data into
        :type local_dir: str
        """
        api = HfApi()

        os.makedirs(local_dir, exist_ok=True)

        # list all files in the repo, keep the ones within folder_path
        all_files = api.list_repo_files(self.repo_id, repo_type=self.repo_type)
        files_list = [f for f in all_files if f.startswith(folder_path)]

        # download each of those files
        for file_path in files_list:
            hf_hub_download(repo_id=self.repo_id, repo_type=self.repo_type, filename=file_path, local_dir=local_dir)

    def download_all_data(self, local_dir="data/"):
        """Download all folders and files, recursively, from `data/` This folder
        contains all metadata (csv) for all images of the Global Streetscapes dataset.

        :param local_dir: Local folder to download the data
        :type local_dir: str
        """
        self._download_folder("data/", local_dir)

    def download_manual_labels(self, local_dir="manual_labels/"):
        """Download all folders and files, recursively, from `manual_labels/` This
        folder contains all the manual labels (csv) for train and test images as well as
        the raw images compressed in in tar.gz.

        :param local_dir: Local folder to download the data
        :type local_dir: str
        """
        self._download_folder("manual_labels/", local_dir)

    def download_train(self, local_dir="manual_labels/train/"):
        """Download all folders and files, recursively, from `manual_labels/train/` This
        folder contains all the manual labels (csv) for only the train images.

        :param local_dir: Local folder to download the data
        :type local_dir: str
        """
        self._download_folder("manual_labels/train/", local_dir)

    def download_test(self, local_dir="manual_labels/test/"):
        """Download all folders and files, recursively, from `manual_labels/test/` This
        folder contains all the manual labels (csv) for only the test images.

        :param local_dir: Local folder to download the data
        :type local_dir: str
        """
        self._download_folder("manual_labels/test/", local_dir)

    def download_img_tar(self, local_dir="manual_labels/img/"):
        """Download all folders and files, recursively, from `manual_labels/img/` This
        folder contains all the raw images, compressed in tar.gz.

        :param local_dir: Local folder to download the data
        :type local_dir: str
        """
        self._download_folder("manual_labels/img/", local_dir)
