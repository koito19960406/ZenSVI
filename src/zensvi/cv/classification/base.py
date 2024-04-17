from typing import Tuple, Union
from pathlib import Path
import torch

# abstract class
from abc import ABC, abstractmethod


class BaseClassifier(ABC):
    """A base class for image classification.

    :param device: The device that the model should be loaded onto. Options are "cpu", "cuda", or "mps".
        If `None`, the model tries to use a GPU if available; otherwise, falls back to CPU.
    :type device: str, optional
    """

    def __init__(self, device=None):
        self.device = self._get_device(device)

    def _get_device(self, device) -> torch.device:
        """
        Get the appropriate device for running the model.

        Returns:
            torch.device: The device to use for running the model.
        """
        if device is not None:
            if device not in ["cpu", "cuda", "mps"]:
                raise ValueError(f"Unknown device: {device}")
            else:
                print(f"Using {device.upper()}")
                return torch.device(device)
        if torch.cuda.is_available():
            print("Using GPU")
            return torch.device("cuda")
        else:
            print("Using CPU")
            return torch.device("cpu")

    @abstractmethod
    def classify(
        self,
        dir_input: Union[str, Path],
        dir_image_output: Union[str, Path, None] = None,
        dir_summary_output: Union[str, Path, None] = None,
        batch_size: int = 1,
        save_image_options: str = "cam_image blend_image",
        save_format: str = "json csv",
        csv_format: str = "long",  # "long" or "wide"
    ) -> None:
        """A method to classify images.

        :param dir_input: directory containing input images.
        :type dir_input: Union[str, Path]
        :param dir_image_output: directory to save output images, defaults to None
        :type dir_image_output: Union[str, Path, None], optional
        :param dir_summary_output: directory to save summary output, defaults to None
        :type dir_summary_output: Union[str, Path, None], optional
        :param batch_size: batch size for inference, defaults to 1
        :type batch_size: int, optional
        :param save_image_options: save options for images, defaults to "cam_image blend_image". Options are "cam_image" and "blend_image". Please add a space between options.
        :type save_image_options: str, optional
        :param save_format: save format for the output, defaults to "json csv". Options are "json" and "csv". Please add a space between options.
        :type save_format: str, optional
        :param csv_format: csv format for the output, defaults to "long". Options are "long" and "wide".
        :type csv_format: str, optional
        """
        pass
