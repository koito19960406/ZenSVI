from typing import Tuple, Union
from pathlib import Path
import torch

# abstract class
from abc import ABC, abstractmethod


class BaseClassifier(ABC):
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
    ) -> Tuple[Path, Path]:
        pass
