import pandas as pd
import matplotlib.pyplot as plt
import glob
from PIL import Image
import numpy as np
from typing import Union, Tuple
from pathlib import Path

from .font_property import _get_font_properties


def _clean_pattern(pattern, image_extensions):
    # find the image file extensions used in the pattern
    pattern_extensions = [ext for ext in image_extensions if ext in pattern][0]
    pattern = pattern.replace(pattern_extensions, "")
    # Remove common regex characters (you might need to extend this list)
    regex_chars = ["*", ".", "?", "+", "^", "$", "(", ")", "[", "]", "{", "}", "|"]
    for char in regex_chars:
        pattern = pattern.replace(char, "")
    return pattern


def plot_image(
    dir_image_input: Union[str, Path],
    n_row: int,
    n_col: int,
    subplot_width: int = 3,
    subplot_height: int = 3,
    dir_csv_input: Union[str, Path] = None,
    csv_file_pattern: str = "*.csv",
    image_file_pattern: str = None,
    sort_by: str = "random",
    ascending: bool = True,
    use_all: bool = False,
    title: str = None,
    path_output: Union[str, Path] = None,
    random_seed: int = 42,
    font_size: int = 30,
    dark_mode: bool = False,
    dpi: int = 300,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Generates a grid of images based on specified parameters and optionally annotates them using data from a CSV file.
    Images can be displayed in a random or sorted order according to metadata provided in a CSV file.

    Args:
        dir_image_input (Union[str, Path]): Directory path containing image files.
        n_row (int): Number of rows in the image grid.
        n_col (int): Number of columns in the image grid.
        subplot_width (int, optional): Width of each subplot. Defaults to 3.
        subplot_height (int, optional): Height of each subplot. Defaults to 3.
        dir_csv_input (Union[str, Path], optional): Directory path containing CSV files with metadata. Defaults to None.
        csv_file_pattern (str, optional): Pattern to match CSV files in the directory. Defaults to None.
        image_file_pattern (str, optional): Pattern to match image files in the directory. Defaults to None.
        sort_by (str, optional): Column name to sort the images by; set to "random" for random order. Defaults to "random".
        ascending (bool, optional): Sort order. True for ascending, False for descending. Defaults to True.
        use_all (bool, optional): If True, use all available images, otherwise use only a subset to fit the grid. Defaults to False.
        title (str, optional): Title of the plot. Defaults to None.
        path_output (Union[str, Path], optional): Path to save the output plot. Defaults to None.
        random_seed (int, optional): Seed for random operations to ensure reproducibility. Defaults to 42.
        font_size (int, optional): Font size for the plot title. Defaults to 30.
        dark_mode (bool, optional): Set to True to use a dark theme for the plot. Defaults to False.
        dpi (int, optional): Resolution in dots per inch for saving the image. Defaults to 300.

    Returns:
        Tuple[plt.Figure, plt.Axes]: A tuple containing the matplotlib figure and axes objects.

    Raises:
        ValueError: If the specified number of rows and columns does not match the available number of images.
        KeyError: If the 'sort_by' column is not found in the provided CSV files.
    """
    # Function implementation remains the same.
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

    if dark_mode:
        plt.style.use("dark_background")
    # get a list of image files recursively
    # List of possible image file extensions
    if image_file_pattern is not None:
        image_files = list(Path(dir_image_input).rglob(image_file_pattern))
    else:
        dir_image_input = Path(dir_image_input)
        # Collect all image files matching the extensions
        image_files = []
        for ext in image_extensions:
            for file in dir_image_input.rglob(f"*{ext}"):
                image_files.append(file)

    # Find CSV files matching the pattern
    if dir_csv_input is not None and csv_file_pattern is not None:
        csv_files = glob.glob(
            str(Path(dir_csv_input) / "**" / csv_file_pattern), recursive=True
        )
        # combine all CSV files into a single DataFrame
        df = pd.concat([pd.read_csv(file) for file in csv_files], ignore_index=True)
        # Pre-filter image file names without extensions
        if image_file_pattern is not None:
            image_file_pattern_no_regex = _clean_pattern(
                image_file_pattern, image_extensions
            )
            image_file_names = {
                str(file.stem).replace(image_file_pattern_no_regex, ""): file
                for file in image_files
            }
        else:
            image_file_names = {file.stem: file for file in image_files}
        # map image file names to the DataFrame by using "filename_key" column and image_file_names keys
        df["filename_key"] = df["filename_key"].astype(str)
        df["image_full_path"] = df["filename_key"].map(image_file_names)
        # Remove rows with missing image file paths
        df_filtered = df.dropna(subset=["image_full_path"])

        # Randomly shuffle the DataFrame
        if not use_all:
            # only get the random n_row * n_col rows
            if n_row * n_col > len(df_filtered):
                raise ValueError(
                    f"n_row * n_col ({n_row * n_col}) is greater than the number of images ({len(df_filtered)})"
                )
            rows = np.random.choice(df_filtered.index, n_row * n_col, replace=False)
            df_filtered = df_filtered.loc[rows]

        # Filter and sort the DataFrame
        if sort_by.lower() != "random":
            try:
                df_filtered = df_filtered.sort_values(
                    by=sort_by, ascending=ascending
                ).reset_index(drop=True)
            except KeyError:
                raise KeyError(f"Column '{sort_by}' not found in the CSV file")
        else:
            df_filtered = df_filtered.sample(
                frac=1, random_state=random_seed
            ).reset_index(
                drop=True
            )  # Randomly shuffle the DataFrame

    else:
        # Randomly shuffle the image files
        np.random.seed(random_seed)
        np.random.shuffle(image_files)
        # Create a DataFrame with the image files
        df_filtered = pd.DataFrame({"image_full_path": image_files})

    # Prepare the subplot
    fig, axes = plt.subplots(
        n_row, n_col, figsize=(n_col * subplot_width, n_row * subplot_height)
    )
    fig.suptitle(title)

    # Flatten the axes array for easy indexing
    axes = axes.flatten()

    for idx, ax in enumerate(axes):
        if idx < len(df_filtered):
            image_path = df_filtered.iloc[idx]["image_full_path"]
            img = Image.open(image_path)
            ax.imshow(img)
            ax.axis("off")
        else:
            ax.axis("off")

    # set title
    prop_title, _, _ = _get_font_properties(font_size)
    fig.suptitle(
        title, fontproperties=prop_title, color="#2b2b2b" if not dark_mode else "white"
    )

    if path_output:
        plt.savefig(path_output, bbox_inches="tight", dpi=dpi)
    return fig, ax
