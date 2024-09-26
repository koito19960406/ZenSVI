from typing import Union, List, Tuple
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob

from .font_property import _get_font_properties


def plot_hist(
    dir_input: Union[str, Path],
    columns: List[str],
    csv_file_pattern: str = "*.csv",
    path_output: Union[str, Path] = None,
    legend: bool = True,
    title: str = None,
    legend_title: str = None,
    fig_size: Tuple[int, int] = (10, 10),
    dpi: int = 300,
    font_size: int = 30,
    dark_mode: bool = False,
    **kwargs
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plots hist (Kernel Density Estimate) plots for specified columns from a CSV file using Seaborn.

    Args:
        path_input (Union[str, Path]): Path to the CSV file.
        columns (List[str]): List of column names to plot hists for.
        path_output (Union[str, Path], optional): Path where the plotted figure will be saved. Defaults to None.
        legend (bool): Whether to add a legend to the plot. Defaults to True.
        title (str, optional): Title of the plot. Defaults to None.
        legend_title (str, optional): Title for the legend. Defaults to None.
        dpi (int): Dots per inch (resolution) of the output image. Defaults to 300.
        font_size (int): Font size for titles and legend. Defaults to 30.
        dark_mode (bool): Whether to use a dark theme for the plot. Defaults to False.
        **kwargs: Additional keyword arguments passed to seaborn.histplot.

    Returns:
        Tuple[plt.Figure, plt.Axes]: A tuple containing the Matplotlib figure and axes objects.
    """
    prop_title, prop, prop_legend = _get_font_properties(font_size)
    sns.set_theme(context="notebook", style="whitegrid", font=prop.get_family())

    # list of csv files
    if Path(dir_input).is_file():
        csv_files = [dir_input]
    else:
        dir_input = Path(dir_input)
        csv_files = glob.glob(str(dir_input / "**" / csv_file_pattern), recursive=True)
    df_list = [pd.read_csv(file) for file in csv_files]
    df = pd.concat(df_list, ignore_index=True)

    # make sure the df is wide format by checking duplicates in filename_key
    if df["filename_key"].duplicated().any():
        # convert to wide format by assuming the second column is the label and the third column is the value
        # rename the columns to filename_key, label, value
        df = df.rename(columns={df.columns[-2]: "label", df.columns[-1]: "value"})
        df = df.pivot(
            index="filename_key", columns="label", values="value"
        ).reset_index()
    else:
        pass

    # filter out columns in df with columns
    df = df[columns]

    # Create plot
    fig, ax = plt.subplots(figsize=fig_size)

    if dark_mode:
        plt.style.use("dark_background")
        font_color = "white"
    else:
        font_color = "black"

    sns.histplot(data=df, ax=ax, **kwargs)
    sns.despine()

    if legend:
        # use prop_legend for legend font properties
        ax.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.1),
            ncol=3,
            title=legend_title,
            labels=columns,
            prop=prop_legend,
            title_fontproperties=prop,
            frameon=False,
        )

    ax.set_xlabel("Value")
    ax.set_ylabel("Density")

    # Set overall figure title
    if title:
        ax.set_title(title, fontproperties=prop_title, color=font_color)

    plt.tight_layout()

    if path_output:
        plt.savefig(path_output, bbox_inches="tight", dpi=dpi)

    return fig, ax
