# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from io import BytesIO
from typing import Any

from PIL import Image


class Decoder:
    """Abstract base class for decoders.

    This class defines the interface for decoding data.
    """

    def decode(self) -> Any:
        """Decode the data.

        Raises:
            NotImplementedError: If the decode method is not implemented.
        """
        raise NotImplementedError


class ImageDataDecoder(Decoder):
    """Decoder for image data.

    This class is responsible for decoding image data from bytes.

    Args:
        image_data (bytes): The image data in bytes format.
    """

    def __init__(self, image_data: bytes) -> None:
        self._image_data = image_data

    def decode(self) -> Image:
        """Decode the image data to a PIL Image.

        Returns:
            Image: The decoded image in RGB format.
        """
        f = BytesIO(self._image_data)
        return Image.open(f).convert(mode="RGB")


class TargetDecoder(Decoder):
    """Decoder for target data.

    This class is responsible for returning the target data.

    Args:
        target (Any): The target data to be decoded.
    """

    def __init__(self, target: Any):
        self._target = target

    def decode(self) -> Any:
        """Return the target data.

        Returns:
            Any: The target data.
        """
        return self._target
