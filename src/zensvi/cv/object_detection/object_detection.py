from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Union
import cv2
import os
import sys
import urllib.request
import threading

current_dir = os.path.dirname(__file__)

from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from groundingdino.util.inference import load_model, load_image, predict, annotate


def download_weights():
    weights_dir = os.path.join(current_dir, "weights")
    weights_file = os.path.join(weights_dir, "groundingdino_swint_ogc.pth")
    if not os.path.exists(weights_file):
        os.makedirs(weights_dir, exist_ok=True)
        url_weights = "https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth"
        print("Downloading GroundingDINO weights...")
        urllib.request.urlretrieve(url_weights, weights_file)
        print(f"Weights downloaded to {weights_file}")
    else:
        print("Weights file already exists.")


download_weights()


class ImageDataset(Dataset):
    def __init__(self, image_files: List[Path]):
        self.image_files = [
            image_file for image_file in image_files
            if image_file.suffix.lower() in [".jpg", ".jpeg", ".png"]
            and not image_file.name.startswith(".")
        ]

    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(self, idx: int) -> Path:
        return self.image_files[idx]


class ObjectDetector:
    def __init__(
        self,
        config_path: str = os.path.join(current_dir, "config/GroundingDINO_SwinT_OGC.py"),
        weights_path: str = os.path.join(current_dir, "weights/groundingdino_swint_ogc.pth"),
        text_prompt: str = "tree . building .",
        box_threshold: float = 0.35,
        text_threshold: float = 0.25,
    ):
        self.model = load_model(config_path, weights_path)
        self.text_prompt = text_prompt
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        # Create a lock to serialize access to the model inference
        self.model_lock = threading.Lock()

    def _process_image(self, image_file: Path, dir_output: Union[str, Path]) -> dict:
        """
        Processes a single image: loads the image, runs object detection,
        annotates the image with bounding boxes and labels, saves the result,
        and returns a summary dictionary with lists of detections.
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
                text_threshold=self.text_threshold
            )

        # Annotate the original image with detection results
        annotated_frame = annotate(
            image_source=image_source,
            boxes=boxes,
            logits=logits,
            phrases=phrases
        )

        # Save the annotated image (prefixing filename with 'annotated_')
        output_path = Path(os.path.join(dir_output, 'annotated_img')) / f"{image_file.name}"
        cv2.imwrite(str(output_path), annotated_frame)

        # Convert arrays to lists (if necessary) for JSON serialization
        boxes_serializable = boxes.tolist() if hasattr(boxes, "tolist") else boxes
        logits_serializable = logits.tolist() if hasattr(logits, "tolist") else logits

        return {
            "filename_key": image_file.stem,
            "boxes": boxes_serializable,
            "logits": logits_serializable,
            "phrases": phrases
        }

    def detect_objects(
        self,
        dir_input: Union[str, Path],
        dir_output: Union[str, Path],
        dir_summary_output: Union[str, Path, None] = None,
        save_format: str = "json",  # Options: "json", "csv", or "json csv"
        max_workers: int = 4
    ):
        """
        Detects objects in all images within dir_input, saves annotated images to dir_output,
        and (if provided) saves a summary of detection results (one row per detection)
        in the specified formats in dir_summary_output.
        Before running, it checks whether an annotated image already exists.
        """
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

        # Ensure the output directory exists
        output_dir = Path(os.path.join(dir_output, 'annotated_img'))
        output_dir.mkdir(parents=True, exist_ok=True)

        # Check which images have already been processed by verifying the existence
        # of the corresponding annotated image file.
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

        # Dictionary to collect detection summaries (per image)
        summary_dict = {}

        # Process images in parallel using ThreadPoolExecutor with a progress bar
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(self._process_image, image_file, dir_output): image_file
                for image_file in remaining_files
            }
            for future in tqdm(as_completed(futures), total=len(futures), desc="Processing images"):
                try:
                    result = future.result()
                    summary_dict[result["filename_key"]] = {
                        "boxes": result["boxes"],
                        "logits": result["logits"],
                        "phrases": result["phrases"]
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
                object_rows.append({
                    "filename_key": filename_key,
                    "box": box,
                    "logit": logit,
                    "phrase": phrase
                })

        # If a summary output directory is provided, save the summary in the specified format(s)
        if dir_summary_output is not None:
            summary_dir = Path(dir_summary_output)
            summary_dir.mkdir(parents=True, exist_ok=True)

            if "json" in save_format:
                json_path = summary_dir / "detection_summary.json"
                with open(json_path, "w") as f:
                    import json
                    json.dump(object_rows, f, indent=2)
                print(f"Saved detection summary to {json_path}")

            if "csv" in save_format:
                import pandas as pd
                df = pd.DataFrame(object_rows)
                csv_path = summary_dir / "detection_summary.csv"
                df.to_csv(csv_path, index=False)
                print(f"Saved detection summary CSV to {csv_path}")