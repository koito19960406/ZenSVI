import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Union

import cv2
import pandas as pd

from zensvi.utils.log import verbosity_tqdm


def _detect_edges_single_image(image_path: Path, dir_image_output: Union[str, Path]) -> dict:
    """Detect edges in a single image using various techniques.

    Args:
        image_path (Path): The path to the image file.
        dir_image_output (Union[str, Path]): Directory to save the processed images.

    Returns:
        dict: A dictionary containing the edge ratios for each technique used.
    """
    # Load image in grayscale
    image = cv2.imread(str(image_path), 0)
    # get total number of pixels
    total_pixels = image.size

    techniques = {
        "Canny": cv2.Canny(image, 100, 200),
        "SobelX": cv2.convertScaleAbs(cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)),
        "SobelY": cv2.convertScaleAbs(cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)),
        "Laplacian": cv2.convertScaleAbs(cv2.Laplacian(image, cv2.CV_64F)),
    }

    edge_ratios = {}
    for technique, edges in techniques.items():
        if dir_image_output:
            # Save processed image
            output_path = os.path.join(dir_image_output, f"{image_path.stem}_{technique}.png")
            cv2.imwrite(output_path, edges)

        # Count edges
        edge_ratios[technique] = cv2.countNonZero(edges) / total_pixels

    return edge_ratios


def _detect_blob_single_image(image_path: Path, dir_image_output: Union[str, Path] = None) -> dict:
    """Detect blobs in a single image.

    Args:
        image_path (Path): The path to the image file.
        dir_image_output (Union[str, Path], optional): Directory to save the processed image. Defaults to None.

    Returns:
        dict: A dictionary containing the count of detected blobs.
    """
    # Load image
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    # Setup SimpleBlobDetector parameters.
    params = cv2.SimpleBlobDetector_Params()
    # Set parameters as needed. For demonstration, default parameters are used.

    # Create a detector with the parameters
    detector = cv2.SimpleBlobDetector_create(params)

    # Detect blobs
    keypoints = detector.detect(image)

    # Draw detected blobs as red circles. (cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob)
    image_with_blobs = cv2.drawKeypoints(
        image, keypoints, None, (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )

    # Save processed image if dir_image_output is provided
    if dir_image_output:
        output_path = os.path.join(dir_image_output, f"{image_path.stem}_blobs.png")
        cv2.imwrite(output_path, image_with_blobs)

    # Return the number of blobs detected
    return {"blob_count": len(keypoints)}


def _detect_blur_single_image(image_path: Path, dir_image_output: Union[str, Path] = None) -> dict:
    """Detect blur in a single image.

    Args:
        image_path (Path): The path to the image file.
        dir_image_output (Union[str, Path], optional): Directory to save the processed image. Defaults to None.

    Returns:
        dict: A dictionary containing the blur measure and a boolean indicating if the image is blurry.
    """
    # Load image in grayscale
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    # Compute the Laplacian of the image and then the focus
    # measure is the variance of the Laplacian
    variance_of_laplacian = cv2.Laplacian(image, cv2.CV_64F).var()

    # Determine if the image is blurry
    is_blurry = int(variance_of_laplacian < 100)  # threshold value is arbitrary; adjust based on needs

    # Return the blur measure and a boolean indicating if it's blurry
    return {"blur_measure": variance_of_laplacian, "is_blurry": is_blurry}


def _calculate_hsl_single_image(image_path: Path, dir_image_output: Union[str, Path] = None) -> dict:
    """Calculate the average HSL values of a single image.

    Args:
        image_path (Path): The path to the image file.
        dir_image_output (Union[str, Path], optional): Directory to save the processed image. Defaults to None.

    Returns:
        dict: A dictionary containing the average hue, saturation, and lightness.
    """
    # Load image
    image = cv2.imread(str(image_path))
    # Convert to HSL color space
    hsl_image = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
    # Calculate averages for each channel
    h, l, s = cv2.split(hsl_image)
    avg_hue = h.mean()
    avg_saturation = s.mean()
    avg_lightness = l.mean()

    # Return the HSL averages
    return {
        "avg_hue": avg_hue,
        "avg_saturation": avg_saturation,
        "avg_lightness": avg_lightness,
    }


def _detect_all_features_single_image(image_path: Path, dir_image_output: Union[str, Path] = None) -> dict:
    """Aggregate all feature detection results for a single image.

    Args:
        image_path (Path): The path to the image file.
        dir_image_output (Union[str, Path], optional): Directory to save the processed images. Defaults to None.

    Returns:
        dict: A dictionary containing all detected features for the image.
    """
    # Aggregate results from all individual feature detection functions
    features = {"filename_key": image_path.stem}
    features.update(_detect_edges_single_image(image_path, dir_image_output))
    features.update(_detect_blob_single_image(image_path, dir_image_output))
    features.update(_detect_blur_single_image(image_path, dir_image_output))
    features.update(_calculate_hsl_single_image(image_path, dir_image_output))

    return features


def get_low_level_features(
    dir_input: Union[str, Path],
    dir_image_output: Union[str, Path] = None,
    dir_summary_output: Union[str, Path] = None,
    save_format: str = "json csv",
    csv_format: str = "long",
    verbosity: int = 1,
) -> None:
    """Processes images from the specified input directory or single image file to
    detect various low-level features, which include edge detection, blob detection,
    blur detection, and HSL color space analysis. It optionally saves the processed
    images and a summary of the features detected.

    Args:
        dir_input (Union[str, Path]): The input directory or image file path.
        dir_image_output (Union[str, Path], optional): Directory to save processed images. Defaults to None.
        dir_summary_output (Union[str, Path], optional): Directory to save summary results. Defaults to None.
        save_format (str, optional): Format to save the summary results. Defaults to "json csv".
        csv_format (str, optional): Format for CSV output. Defaults to "long".
        verbosity (int, optional): Level of verbosity for progress bars. Defaults to 1.
                                  0 = no progress bars, 1 = outer loops only, 2 = all loops.

    Returns:
        None: The function does not return any value but outputs results to the specified directories.

    Raises:
        ValueError: If neither dir_image_output nor dir_summary_output is provided, indicating that at least one output directory must be specified.
    """
    if not dir_image_output and not dir_summary_output:
        raise ValueError("At least one of dir_image_output and dir_summary_output must be provided")

    if dir_image_output:
        Path(dir_image_output).mkdir(parents=True, exist_ok=True)
    if dir_summary_output:
        Path(dir_summary_output).mkdir(parents=True, exist_ok=True)

    image_extensions = [
        ".jpg",
        ".jpeg",
        ".png",
        ".tif",
        ".tiff",
        ".bmp",
        ".dib",
        ".pbm",
        ".pgm",
        ".ppm",
        ".sr",
        ".ras",
        ".exr",
        ".jp2",
    ]
    images = (
        [file for ext in image_extensions for file in Path(dir_input).rglob(f"*{ext}")]
        if Path(dir_input).is_dir()
        else [Path(dir_input)]
    )

    summary_data = []
    with ThreadPoolExecutor() as executor:
        future_to_image = {
            executor.submit(_detect_all_features_single_image, image, dir_image_output): image for image in images
        }
        for future in verbosity_tqdm(
            as_completed(future_to_image), desc="Processing images", total=len(images), verbosity=verbosity, level=1
        ):
            image = future_to_image[future]
            try:
                result = future.result()
                summary_data.append(result)
            except Exception as exc:
                print(f"{image} generated an exception: {exc}")

    # Save the aggregated results
    if dir_summary_output:
        if "json" in save_format:
            summary_path = os.path.join(dir_summary_output, "low_level_features.json")
            with open(summary_path, "w") as f:
                json.dump(summary_data, f)
        if "csv" in save_format:
            summary_path = os.path.join(dir_summary_output, "low_level_features.csv")
            summary_df = pd.DataFrame(summary_data)
            if csv_format == "long":
                summary_df = summary_df.melt(id_vars="filename_key", var_name="feature", value_name="value")
            summary_df.to_csv(summary_path, index=False)
