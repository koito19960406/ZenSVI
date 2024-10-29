# abstract class
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Union

import torch


class BaseClassifier(ABC):
    """A base class for image classification.

    Args:
      device(str): The device that the model should be
    loaded onto. Options are "cpu", "cuda", or "mps". If `None`,
    the model tries to use a GPU if available; otherwise, falls
    back to CPU.

    Returns:

    """

    def __init__(self, device=None):
        self.device = self._get_device(device)

    def _get_device(self, device) -> torch.device:
        """Get the appropriate device for running the model.

        Args:
          device:

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

        Args:
          dir_input(Union[str): directory containing input
        images.
          dir_image_output(Union[str): directory to save output images, defaults to None
          dir_summary_output(Union[str): directory to save summary output, defaults to None
          batch_size(int): batch size for inference,
        defaults to 1
          save_image_options(str): save options for images,
        defaults to "cam_image blend_image". Options are
        "cam_image" and "blend_image". Please add a space
        between options.
          save_format(str): save format for the output,
        defaults to "json csv". Options are "json" and "csv".
        Please add a space between options.
          csv_format(str): csv format for the output,
        defaults to "long". Options are "long" and "wide".
          dir_input: Union[str:
          Path]:
          dir_image_output: Union[str:
          Path:
          None]: (Default value = None)
          dir_summary_output: Union[str:
          batch_size: int:  (Default value = 1)
          save_image_options: str:  (Default value = "cam_image blend_image")
          save_format: str:  (Default value = "json csv")
          csv_format: str:  (Default value = "long")
          # "long" or "wide":
          dir_input: Union[str:
          dir_image_output: Union[str:
          dir_summary_output: Union[str:
          batch_size: int:  (Default value = 1)
          save_image_options: str:  (Default value = "cam_image blend_image")
          save_format: str:  (Default value = "json csv")
          csv_format: str:  (Default value = "long")
          dir_input: Union[str:
          dir_image_output: Union[str:
          dir_summary_output: Union[str:
          batch_size: int:  (Default value = 1)
          save_image_options: str:  (Default value = "cam_image blend_image")
          save_format: str:  (Default value = "json csv")
          csv_format: str:  (Default value = "long")

        Returns:

        """
        pass
