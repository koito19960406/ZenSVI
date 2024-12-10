# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Tuple

from torch.utils.data import Dataset


class DatasetWithEnumeratedTargets(Dataset):
    """Dataset wrapper that provides enumerated targets.

    This class extends the PyTorch Dataset to include the index of the sample
    as part of the target, allowing for easier tracking of samples.

    Args:
        dataset (Dataset): The dataset to wrap.

    """

    def __init__(self, dataset: Dataset) -> None:
        """Initializes the DatasetWithEnumeratedTargets.

        Args:
            dataset (Dataset): The dataset to wrap.
        """
        self._dataset = dataset

    def get_image_data(self, index: int) -> bytes:
        """Retrieves the image data for a given index.

        Args:
            index (int): The index of the image to retrieve.

        Returns:
            bytes: The raw image data.
        """
        return self._dataset.get_image_data(index)

    def get_target(self, index: int) -> Tuple[Any, int]:
        """Retrieves the target data for a given index, including the index.

        Args:
            index (int): The index of the target to retrieve.

        Returns:
            Tuple[Any, int]: A tuple containing the index and the target data.
        """
        target = self._dataset.get_target(index)
        return (index, target)

    def __getitem__(self, index: int) -> Tuple[Any, Tuple[Any, int]]:
        """Fetches the image and target data for a given index.

        Args:
            index (int): The index of the data to retrieve.

        Returns:
            Tuple[Any, Tuple[Any, int]]: A tuple containing the image and a tuple
            of the index and target data.
        """
        image, target = self._dataset[index]
        target = index if target is None else target
        return image, (index, target)

    def __len__(self) -> int:
        """Returns the length of the dataset.

        Returns:
            int: The number of samples in the dataset.
        """
        return len(self._dataset)
