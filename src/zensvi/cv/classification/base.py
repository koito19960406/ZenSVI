# abstract class
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Union

import torch


class BaseClassifier(ABC):
    """A base class for image classification.

    Args:
        device (str, optional): The device that the model should be loaded onto.
            Options are "cpu", "cuda", or "mps". If `None`, the model tries to use
            a GPU if available; otherwise, falls back to CPU.
    """

    def __init__(self, device=None):
        self.device = self._get_device(device)

    def _get_device(self, device) -> torch.device:
        """Get the appropriate device for running the model.

        Args:
            device (str, optional): Device to use. Options are "cpu", "cuda", or "mps".

        Returns:
            torch.device: The device to use for running the model.

        Raises:
            ValueError: If an unknown device type is specified.
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
        """Classify images in a directory.

        Args:
            dir_input (Union[str, Path]): Directory containing input images.
            dir_image_output (Union[str, Path, None], optional): Directory to save output images.
                Defaults to None.
            dir_summary_output (Union[str, Path, None], optional): Directory to save summary output.
                Defaults to None.
            batch_size (int, optional): Batch size for inference. Defaults to 1.
            save_image_options (str, optional): Space-separated options for image output.
                Options are "cam_image" and "blend_image". Defaults to "cam_image blend_image".
            save_format (str, optional): Space-separated output formats.
                Options are "json" and "csv". Defaults to "json csv".
            csv_format (str, optional): Format for CSV output.
                Options are "long" and "wide". Defaults to "long".
        """
        pass
