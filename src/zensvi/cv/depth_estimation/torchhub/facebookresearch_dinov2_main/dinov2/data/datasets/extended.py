# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Tuple

from torchvision.datasets import VisionDataset

from .decoders import ImageDataDecoder, TargetDecoder


class ExtendedVisionDataset(VisionDataset):
    """Extended Vision Dataset for loading images and targets.

    This dataset serves as a base class for loading image data and corresponding targets.
    It requires subclasses to implement the methods for retrieving image data and targets.

    Attributes:
        transforms (callable, optional): A function/transform to apply to the images and targets.
    """

    def __init__(self, *args, **kwargs) -> None:
        """Initializes the ExtendedVisionDataset.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        super().__init__(*args, **kwargs)  # type: ignore

    def get_image_data(self, index: int) -> bytes:
        """Retrieves the image data for a given index.

        Args:
            index (int): The index of the image to retrieve.

        Returns:
            bytes: The raw image data.

        Raises:
            NotImplementedError: If the method is not implemented in a subclass.
        """
        raise NotImplementedError

    def get_target(self, index: int) -> Any:
        """Retrieves the target data for a given index.

        Args:
            index (int): The index of the target to retrieve.

        Returns:
            Any: The target data.

        Raises:
            NotImplementedError: If the method is not implemented in a subclass.
        """
        raise NotImplementedError

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """Fetches the image and target data for a given index.

        Args:
            index (int): The index of the data to retrieve.

        Returns:
            Tuple[Any, Any]: A tuple containing the image and target data.

        Raises:
            RuntimeError: If the image cannot be read for the given sample index.
        """
        try:
            image_data = self.get_image_data(index)
            image = ImageDataDecoder(image_data).decode()
        except Exception as e:
            raise RuntimeError(f"can not read image for sample {index}") from e
        target = self.get_target(index)
        target = TargetDecoder(target).decode()

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target

    def __len__(self) -> int:
        """Returns the total number of samples in the dataset.

        Returns:
            int: The number of samples in the dataset.

        Raises:
            NotImplementedError: If the method is not implemented in a subclass.
        """
        raise NotImplementedError
