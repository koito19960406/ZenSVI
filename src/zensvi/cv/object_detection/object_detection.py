import json
import os
import threading
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Union

import cv2
import pandas as pd
import torch
from groundingdino.util.inference import annotate, load_image, load_model, predict
from torch.utils.data import Dataset

from zensvi.utils.log import verbosity_tqdm

current_dir = os.path.dirname(__file__)
models_dir = Path(__file__).parent.parent.parent.parent.parent / "models"


def _download_weights():
    weights_dir = models_dir / "groundingdino"
    weights_file = weights_dir / "groundingdino_swint_ogc.pth"
    if not weights_file.exists():
        weights_dir.mkdir(parents=True, exist_ok=True)
        url_weights = (
            "https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth"
        )
        print("Downloading GroundingDINO weights...")
        urllib.request.urlretrieve(url_weights, str(weights_file))
        print(f"Weights downloaded to {weights_file}")
    else:
        print("Weights file already exists.")


_download_weights()


class ImageDataset(Dataset):
    """A dataset class for loading image files.

    This class handles filtering and loading of image files, keeping only .jpg, .jpeg and .png files
    that don't start with a dot.

    Args:
        image_files (List[Path]): A list of paths to image files.

    Attributes:
        image_files (List[Path]): The filtered list of valid image file paths.
    """

    def __init__(self, image_files: List[Path]):
        self.image_files = [
            image_file
            for image_file in image_files
            if image_file.suffix.lower() in [".jpg", ".jpeg", ".png"] and not image_file.name.startswith(".")
        ]

    def __len__(self) -> int:
        """Gets the number of images in the dataset.

        Returns:
            int: The number of images in the dataset.
        """
        return len(self.image_files)

    def __getitem__(self, idx: int) -> Path:
        """Gets the path to an image at the specified index.

        Args:
            idx (int): The index of the image to retrieve.

        Returns:
            Path: The path to the image file.
        """
        return self.image_files[idx]


class ObjectDetector:
    """Class for detecting objects in images using GroundingDINO model.

    This class provides functionality to detect objects in images using the GroundingDINO model.
    It can process single images or directories of images, annotate them with bounding boxes and labels,
    and save detection summaries in various formats.

    Args:
        config_path (str, optional): Path to GroundingDINO config file. Defaults to included config.
        weights_path (str, optional): Path to model weights file. Defaults to included weights.
        text_prompt (str, optional): Text prompt for object detection. Defaults to "tree . building .".
        box_threshold (float, optional): Confidence threshold for box detection. Defaults to 0.35.
        text_threshold (float, optional): Confidence threshold for text. Defaults to 0.25.
        verbosity (int, optional): Level of verbosity for progress bars. Defaults to 1.
                                  0 = no progress bars, 1 = outer loops only, 2 = all loops.

    Attributes:
        model: The loaded GroundingDINO model.
        text_prompt (str): Text prompt used for detection.
        box_threshold (float): Box confidence threshold.
        text_threshold (float): Text confidence threshold.
        model_lock (threading.Lock): Lock for thread-safe model inference.
        verbosity (int): Level of verbosity for progress reporting.
        device: The device used for inference. Options: "cpu", "cuda", or "mps".
    """

    def __init__(
        self,
        config_path: str = os.path.join(current_dir, "config/GroundingDINO_SwinT_OGC.py"),
        weights_path: str = str(models_dir / "groundingdino" / "groundingdino_swint_ogc.pth"),
        text_prompt: str = "tree . building .",
        box_threshold: float = 0.35,
        text_threshold: float = 0.25,
        verbosity: int = 1,
        device=None,  # Options: "cpu", "cuda", or "mps"
    ):
        self.model = load_model(config_path, weights_path)
        self.text_prompt = text_prompt
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self.verbosity = verbosity
        # Create a lock to serialize access to the model inference
        self.model_lock = threading.Lock()
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
            print(f"Using {device.upper()}")
            return torch.device(device)
        if torch.cuda.is_available():
            print("Using GPU")
            return torch.device("cuda")
        else:
            print("Using CPU")
            return torch.device("cpu")

    def _process_image(self, image_file: Path, dir_image_output: Union[str, Path, None]) -> dict:
        """Process a single image for object detection.

        Loads an image, runs object detection, annotates with bounding boxes and labels,
        saves the annotated image, and returns detection results.

        Args:
            image_file (Path): Path to the input image file.
            dir_image_output (Union[str, Path, None]): Directory to save annotated image.
                If None, no image is saved.

        Returns:
            dict: Detection results containing:
                - filename_key (str): Image file stem name
                - boxes (list): Detected bounding boxes
                - logits (list): Detection confidence scores
                - phrases (list): Detected object labels
        """
        # Load image (returns the original image and a processed version)
        image_source, image = load_image(str(image_file))

        # Use the lock to ensure that only one thread runs inference at a time
        with self.model_lock:
            boxes, logits, phrases = predict(
                model=self.model,
                image=image,
                caption=self.text_prompt,
                box_threshold=self.box_threshold,
                text_threshold=self.text_threshold,
                device=self.device,
            )

        # Convert boxes and logits to a format that can be JSON serialized
        boxes_serializable = boxes.cpu().tolist() if boxes is not None else []
        logits_serializable = logits.cpu().tolist() if logits is not None else []

        # Only save the annotated image if dir_image_output is provided
        if dir_image_output is not None:
            # Annotate the image with bounding boxes and labels
            annotated_frame = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)

            # Ensure the directory exists within the output directory
            output_dir = Path(dir_image_output)
            output_dir.mkdir(parents=True, exist_ok=True)

            # Save the annotated image
            output_path = output_dir / image_file.name
            cv2.imwrite(str(output_path), annotated_frame)

        return {
            "filename_key": image_file.stem,
            "boxes": boxes_serializable,
            "logits": logits_serializable,
            "phrases": phrases,
        }

    def detect_objects(
        self,
        dir_input: Union[str, Path],
        dir_image_output: Union[str, Path, None] = None,
        dir_summary_output: Union[str, Path, None] = None,
        save_format: str = "json",  # Options: "json", "csv", or "json csv"
        max_workers: int = 4,
        verbosity: int = None,
        group_by_object: bool = False,  # Group detections by object type per image
    ):
        """Detect objects in images and save results.

        Processes images from input directory/file, saves annotated images and detection summaries.
        Only processes unprocessed images (those without existing annotated versions).

        Args:
            dir_input (Union[str, Path]): Input image file or directory path.
            dir_image_output (Union[str, Path, None], optional): Directory to save annotated images.
                If None, no images are saved, only summary data (dir_summary_output must be provided).
            dir_summary_output (Union[str, Path, None], optional): Directory to save detection summaries.
                If None, no summary data is saved, only annotated images (dir_image_output must be provided).
            save_format (str, optional): Format for saving summaries ("json", "csv", or "json csv").
                Defaults to "json".
            max_workers (int, optional): Maximum number of parallel workers. Defaults to 4.
            verbosity (int, optional): Level of verbosity for progress bars.
                If None, uses the instance's verbosity level.
                0 = no progress bars, 1 = outer loops only, 2 = all loops.
            group_by_object (bool, optional): If True, groups detections by object type per image and
                counts occurrences. If False, returns detailed detection data. Defaults to False.

        Raises:
            ValueError: If dir_input is neither a file nor directory.
            ValueError: If neither dir_image_output nor dir_summary_output is provided.
        """
        # Use instance verbosity if not specified
        if verbosity is None:
            verbosity = self.verbosity

        # Validate that at least one output directory is provided
        if dir_image_output is None and dir_summary_output is None:
            raise ValueError("At least one of dir_image_output or dir_summary_output must be provided.")

        # Collect image files from input (handle both file and directory)
        dir_input = Path(dir_input)
        if dir_input.is_file():
            image_files = [dir_input]
        elif dir_input.is_dir():
            image_extensions = [".jpg", ".jpeg", ".png"]
            image_files = [f for f in dir_input.iterdir() if f.suffix.lower() in image_extensions]
        else:
            raise ValueError("dir_input must be either a file or a directory.")

        if not image_files:
            print("No image files found. Skipping object detection.")
            return

        # Check if we need to skip already processed images
        if dir_image_output is not None:
            # Ensure the output directory exists
            output_dir = Path(dir_image_output)
            output_dir.mkdir(parents=True, exist_ok=True)

            # Check which images have already been processed
            remaining_files = []
            for image_file in image_files:
                annotated_file = output_dir / f"{image_file.name}"
                if not annotated_file.exists():
                    remaining_files.append(image_file)

            if not remaining_files:
                print("All images have been processed. Skipping detection.")
                return
            else:
                print(f"Found {len(remaining_files)} unprocessed images out of {len(image_files)} total.")
        else:
            # If not saving images, process all files
            remaining_files = image_files
            print(f"Processing {len(remaining_files)} images for detection summary only.")

        # Dictionary to collect detection summaries (per image)
        summary_dict = {}

        # Process images in parallel using ThreadPoolExecutor with a progress bar
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(self._process_image, image_file, dir_image_output): image_file
                for image_file in remaining_files
            }
            for future in verbosity_tqdm(
                as_completed(futures), total=len(futures), desc="Processing images", verbosity=verbosity, level=1
            ):
                try:
                    result = future.result()
                    summary_dict[result["filename_key"]] = {
                        "boxes": result["boxes"],
                        "logits": result["logits"],
                        "phrases": result["phrases"],
                    }
                except Exception as e:
                    image_file = futures[future]
                    print(f"Error processing {image_file}: {e}")

        # Flatten the results so that each detection (object) is a row
        object_rows = []
        for filename_key, data in summary_dict.items():
            boxes = data["boxes"]
            logits = data["logits"]
            phrases = data["phrases"]
            # Assume each detection is aligned by index
            for box, logit, phrase in zip(boxes, logits, phrases):
                object_rows.append({"filename_key": filename_key, "box": box, "logit": logit, "phrase": phrase})

        # If grouping by object is requested, create a grouped summary
        grouped_summary = None
        if group_by_object:
            grouped_summary = {}
            for row in object_rows:
                filename = row["filename_key"]
                phrase = row["phrase"]

                if filename not in grouped_summary:
                    grouped_summary[filename] = {}

                if phrase not in grouped_summary[filename]:
                    grouped_summary[filename][phrase] = 1
                else:
                    grouped_summary[filename][phrase] += 1

        # If a summary output directory is provided, save the summary in the specified format(s)
        if dir_summary_output is not None:
            summary_dir = Path(dir_summary_output)
            summary_dir.mkdir(parents=True, exist_ok=True)

            if "json" in save_format:
                # Save detailed results
                json_path = summary_dir / "detection_summary.json"
                with open(json_path, "w") as f:
                    json.dump(object_rows, f, indent=2)
                print(f"Saved detection summary to {json_path}")

                # Save grouped results if requested
                if group_by_object and grouped_summary:
                    grouped_json_path = summary_dir / "detection_summary_grouped.json"
                    with open(grouped_json_path, "w") as f:
                        json.dump(grouped_summary, f, indent=2)
                    print(f"Saved grouped detection summary to {grouped_json_path}")

            if "csv" in save_format:
                # Save detailed results
                df = pd.DataFrame(object_rows)
                csv_path = summary_dir / "detection_summary.csv"
                df.to_csv(csv_path, index=False)
                print(f"Saved detection summary CSV to {csv_path}")

                # Save grouped results if requested
                if group_by_object and grouped_summary:
                    # Convert nested dict to dataframe
                    rows = []
                    for filename, objects in grouped_summary.items():
                        for phrase, count in objects.items():
                            rows.append({"filename_key": filename, "object": phrase, "count": count})

                    grouped_df = pd.DataFrame(rows)
                    grouped_csv_path = summary_dir / "detection_summary_grouped.csv"
                    grouped_df.to_csv(grouped_csv_path, index=False)
                    print(f"Saved grouped detection summary CSV to {grouped_csv_path}")
