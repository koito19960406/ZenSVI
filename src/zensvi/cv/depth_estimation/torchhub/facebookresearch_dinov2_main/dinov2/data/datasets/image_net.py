# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import csv
import logging
import os
from enum import Enum
from typing import Callable, List, Optional, Tuple, Union

import numpy as np

from .extended import ExtendedVisionDataset

logger = logging.getLogger("dinov2")
_Target = int


class _Split(Enum):
    """Enumeration for dataset splits."""

    TRAIN = "train"
    VAL = "val"
    TEST = "test"  # NOTE: torchvision does not support the test split

    @property
    def length(self) -> int:
        """Returns the number of samples in the split.

        Returns:
            int: The number of samples in the split.
        """
        split_lengths = {
            _Split.TRAIN: 1_281_167,
            _Split.VAL: 50_000,
            _Split.TEST: 100_000,
        }
        return split_lengths[self]

    def get_dirname(self, class_id: Optional[str] = None) -> str:
        """Gets the directory name for the split.

        Args:
            class_id (Optional[str]): The class ID to include in the path. Defaults to None.

        Returns:
            str: The directory name for the split.
        """
        return self.value if class_id is None else os.path.join(self.value, class_id)

    def get_image_relpath(self, actual_index: int, class_id: Optional[str] = None) -> str:
        """Generates the relative path for an image.

        Args:
            actual_index (int): The actual index of the image.
            class_id (Optional[str]): The class ID. Defaults to None.

        Returns:
            str: The relative path to the image.
        """
        dirname = self.get_dirname(class_id)
        if self == _Split.TRAIN:
            basename = f"{class_id}_{actual_index}"
        else:  # self in (_Split.VAL, _Split.TEST):
            basename = f"ILSVRC2012_{self.value}_{actual_index:08d}"
        return os.path.join(dirname, basename + ".JPEG")

    def parse_image_relpath(self, image_relpath: str) -> Tuple[str, int]:
        """Parses the image relative path to extract class ID and actual index.

        Args:
            image_relpath (str): The relative path of the image.

        Returns:
            Tuple[str, int]: A tuple containing the class ID and actual index.
        """
        assert self != _Split.TEST
        dirname, filename = os.path.split(image_relpath)
        class_id = os.path.split(dirname)[-1]
        basename, _ = os.path.splitext(filename)
        actual_index = int(basename.split("_")[-1])
        return class_id, actual_index


class ImageNet(ExtendedVisionDataset):
    """ImageNet dataset class extending the ExtendedVisionDataset.

    Attributes:
        Target (Union[_Target]): Type for target values.
        Split (Union[_Split]): Type for dataset splits.
    """

    Target = Union[_Target]
    Split = Union[_Split]

    def __init__(
        self,
        *,
        split: "ImageNet.Split",
        root: str,
        extra: str,
        transforms: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> None:
        """Initializes the ImageNet dataset.

        Args:
            split (ImageNet.Split): The dataset split (train, val, test).
            root (str): The root directory of the dataset.
            extra (str): The directory for extra data.
            transforms (Optional[Callable]): Transformations to apply to the images. Defaults to None.
            transform (Optional[Callable]): Transformations to apply to the images. Defaults to None.
            target_transform (Optional[Callable]): Transformations to apply to the targets. Defaults to None.
        """
        super().__init__(root, transforms, transform, target_transform)
        self._extra_root = extra
        self._split = split

        self._entries = None
        self._class_ids = None
        self._class_names = None

    @property
    def split(self) -> "ImageNet.Split":
        """Returns the dataset split.

        Returns:
            ImageNet.Split: The dataset split.
        """
        return self._split

    def _get_extra_full_path(self, extra_path: str) -> str:
        """Constructs the full path for extra data.

        Args:
            extra_path (str): The extra path to be appended.

        Returns:
            str: The full path to the extra data.
        """
        return os.path.join(self._extra_root, extra_path)

    def _load_extra(self, extra_path: str) -> np.ndarray:
        """Loads extra data from the specified path.

        Args:
            extra_path (str): The path to the extra data.

        Returns:
            np.ndarray: The loaded extra data.
        """
        extra_full_path = self._get_extra_full_path(extra_path)
        return np.load(extra_full_path, mmap_mode="r")

    def _save_extra(self, extra_array: np.ndarray, extra_path: str) -> None:
        """Saves extra data to the specified path.

        Args:
            extra_array (np.ndarray): The extra data to save.
            extra_path (str): The path where the data will be saved.
        """
        extra_full_path = self._get_extra_full_path(extra_path)
        os.makedirs(self._extra_root, exist_ok=True)
        np.save(extra_full_path, extra_array)

    @property
    def _entries_path(self) -> str:
        """Returns the path for the entries file.

        Returns:
            str: The path for the entries file.
        """
        return f"entries-{self._split.value.upper()}.npy"

    @property
    def _class_ids_path(self) -> str:
        """Returns the path for the class IDs file.

        Returns:
            str: The path for the class IDs file.
        """
        return f"class-ids-{self._split.value.upper()}.npy"

    @property
    def _class_names_path(self) -> str:
        """Returns the path for the class names file.

        Returns:
            str: The path for the class names file.
        """
        return f"class-names-{self._split.value.upper()}.npy"

    def _get_entries(self) -> np.ndarray:
        """Retrieves the entries for the dataset.

        Returns:
            np.ndarray: The entries for the dataset.
        """
        if self._entries is None:
            self._entries = self._load_extra(self._entries_path)
        assert self._entries is not None
        return self._entries

    def _get_class_ids(self) -> np.ndarray:
        """Retrieves the class IDs for the dataset.

        Returns:
            np.ndarray: The class IDs for the dataset.

        Raises:
            AssertionError: If the split is TEST, as class IDs are not available.
        """
        if self._split == _Split.TEST:
            assert False, "Class IDs are not available in TEST split"
        if self._class_ids is None:
            self._class_ids = self._load_extra(self._class_ids_path)
        assert self._class_ids is not None
        return self._class_ids

    def _get_class_names(self) -> np.ndarray:
        """Retrieves the class names for the dataset.

        Returns:
            np.ndarray: The class names for the dataset.

        Raises:
            AssertionError: If the split is TEST, as class names are not available.
        """
        if self._split == _Split.TEST:
            assert False, "Class names are not available in TEST split"
        if self._class_names is None:
            self._class_names = self._load_extra(self._class_names_path)
        assert self._class_names is not None
        return self._class_names

    def find_class_id(self, class_index: int) -> str:
        """Finds the class ID for a given class index.

        Args:
            class_index (int): The index of the class.

        Returns:
            str: The class ID corresponding to the class index.
        """
        class_ids = self._get_class_ids()
        return str(class_ids[class_index])

    def find_class_name(self, class_index: int) -> str:
        """Finds the class name for a given class index.

        Args:
            class_index (int): The index of the class.

        Returns:
            str: The class name corresponding to the class index.
        """
        class_names = self._get_class_names()
        return str(class_names[class_index])

    def get_image_data(self, index: int) -> bytes:
        """Retrieves the image data for a given index.

        Args:
            index (int): The index of the image.

        Returns:
            bytes: The image data as bytes.
        """
        entries = self._get_entries()
        actual_index = entries[index]["actual_index"]

        class_id = self.get_class_id(index)

        image_relpath = self.split.get_image_relpath(actual_index, class_id)
        image_full_path = os.path.join(self.root, image_relpath)
        with open(image_full_path, mode="rb") as f:
            image_data = f.read()
        return image_data

    def get_target(self, index: int) -> Optional[Target]:
        """Retrieves the target for a given index.

        Args:
            index (int): The index of the target.

        Returns:
            Optional[Target]: The target value, or None if the split is TEST.
        """
        entries = self._get_entries()
        class_index = entries[index]["class_index"]
        return None if self.split == _Split.TEST else int(class_index)

    def get_targets(self) -> Optional[np.ndarray]:
        """Retrieves all targets for the dataset.

        Returns:
            Optional[np.ndarray]: The targets for the dataset, or None if the split is TEST.
        """
        entries = self._get_entries()
        return None if self.split == _Split.TEST else entries["class_index"]

    def get_class_id(self, index: int) -> Optional[str]:
        """Retrieves the class ID for a given index.

        Args:
            index (int): The index of the class.

        Returns:
            Optional[str]: The class ID, or None if the split is TEST.
        """
        entries = self._get_entries()
        class_id = entries[index]["class_id"]
        return None if self.split == _Split.TEST else str(class_id)

    def get_class_name(self, index: int) -> Optional[str]:
        """Retrieves the class name for a given index.

        Args:
            index (int): The index of the class.

        Returns:
            Optional[str]: The class name, or None if the split is TEST.
        """
        entries = self._get_entries()
        class_name = entries[index]["class_name"]
        return None if self.split == _Split.TEST else str(class_name)

    def __len__(self) -> int:
        """Returns the number of entries in the dataset.

        Returns:
            int: The number of entries in the dataset.
        """
        entries = self._get_entries()
        assert len(entries) == self.split.length
        return len(entries)

    def _load_labels(self, labels_path: str) -> List[Tuple[str, str]]:
        """Loads labels from a specified file.

        Args:
            labels_path (str): The path to the labels file.

        Returns:
            List[Tuple[str, str]]: A list of tuples containing class IDs and class names.

        Raises:
            RuntimeError: If the labels file cannot be read.
        """
        labels_full_path = os.path.join(self.root, labels_path)
        labels = []

        try:
            with open(labels_full_path, "r") as f:
                reader = csv.reader(f)
                for row in reader:
                    class_id, class_name = row
                    labels.append((class_id, class_name))
        except OSError as e:
            raise RuntimeError(f'can not read labels file "{labels_full_path}"') from e

        return labels

    def _dump_entries(self) -> None:
        """Dumps the entries to a file based on the dataset split."""
        split = self.split
        if split == ImageNet.Split.TEST:
            dataset = None
            sample_count = split.length
            max_class_id_length, max_class_name_length = 0, 0
        else:
            labels_path = "labels.txt"
            logger.info(f'loading labels from "{labels_path}"')
            labels = self._load_labels(labels_path)

            # NOTE: Using torchvision ImageFolder for consistency
            from torchvision.datasets import ImageFolder

            dataset_root = os.path.join(self.root, split.get_dirname())
            dataset = ImageFolder(dataset_root)
            sample_count = len(dataset)
            max_class_id_length, max_class_name_length = -1, -1
            for sample in dataset.samples:
                _, class_index = sample
                class_id, class_name = labels[class_index]
                max_class_id_length = max(len(class_id), max_class_id_length)
                max_class_name_length = max(len(class_name), max_class_name_length)

        dtype = np.dtype(
            [
                ("actual_index", "<u4"),
                ("class_index", "<u4"),
                ("class_id", f"U{max_class_id_length}"),
                ("class_name", f"U{max_class_name_length}"),
            ]
        )
        entries_array = np.empty(sample_count, dtype=dtype)

        if split == ImageNet.Split.TEST:
            old_percent = -1
            for index in range(sample_count):
                percent = 100 * (index + 1) // sample_count
                if percent > old_percent:
                    logger.info(f"creating entries: {percent}%")
                    old_percent = percent

                actual_index = index + 1
                class_index = np.uint32(-1)
                class_id, class_name = "", ""
                entries_array[index] = (actual_index, class_index, class_id, class_name)
        else:
            class_names = {class_id: class_name for class_id, class_name in labels}

            assert dataset
            old_percent = -1
            for index in range(sample_count):
                percent = 100 * (index + 1) // sample_count
                if percent > old_percent:
                    logger.info(f"creating entries: {percent}%")
                    old_percent = percent

                image_full_path, class_index = dataset.samples[index]
                image_relpath = os.path.relpath(image_full_path, self.root)
                class_id, actual_index = split.parse_image_relpath(image_relpath)
                class_name = class_names[class_id]
                entries_array[index] = (actual_index, class_index, class_id, class_name)

        logger.info(f'saving entries to "{self._entries_path}"')
        self._save_extra(entries_array, self._entries_path)

    def _dump_class_ids_and_names(self) -> None:
        """Dumps class IDs and names to files based on the dataset split."""
        split = self.split
        if split == ImageNet.Split.TEST:
            return

        entries_array = self._load_extra(self._entries_path)

        max_class_id_length, max_class_name_length, max_class_index = -1, -1, -1
        for entry in entries_array:
            class_index, class_id, class_name = (
                entry["class_index"],
                entry["class_id"],
                entry["class_name"],
            )
            max_class_index = max(int(class_index), max_class_index)
            max_class_id_length = max(len(str(class_id)), max_class_id_length)
            max_class_name_length = max(len(str(class_name)), max_class_name_length)

        class_count = max_class_index + 1
        class_ids_array = np.empty(class_count, dtype=f"U{max_class_id_length}")
        class_names_array = np.empty(class_count, dtype=f"U{max_class_name_length}")
        for entry in entries_array:
            class_index, class_id, class_name = (
                entry["class_index"],
                entry["class_id"],
                entry["class_name"],
            )
            class_ids_array[class_index] = class_id
            class_names_array[class_index] = class_name

        logger.info(f'saving class IDs to "{self._class_ids_path}"')
        self._save_extra(class_ids_array, self._class_ids_path)

        logger.info(f'saving class names to "{self._class_names_path}"')
        self._save_extra(class_names_array, self._class_names_path)

    def dump_extra(self) -> None:
        """Dumps extra data (entries, class IDs, and class names) to files."""
        self._dump_entries()
        self._dump_class_ids_and_names()
