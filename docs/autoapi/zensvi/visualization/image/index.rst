:py:mod:`zensvi.visualization.image`
====================================

.. py:module:: zensvi.visualization.image


Module Contents
---------------


Functions
~~~~~~~~~

.. autoapisummary::

   zensvi.visualization.image.plot_image



.. py:function:: plot_image(dir_image_input: Union[str, pathlib.Path], n_row: int, n_col: int, subplot_width: int = 3, subplot_height: int = 3, dir_csv_input: Union[str, pathlib.Path] = None, csv_file_pattern: str = '*.csv', image_file_pattern: str = None, sort_by: str = 'random', ascending: bool = True, use_all: bool = False, title: str = None, path_output: Union[str, pathlib.Path] = None, random_seed: int = 42, font_size: int = 30, dark_mode: bool = False, dpi: int = 300) -> Tuple[matplotlib.pyplot.Figure, matplotlib.pyplot.Axes]

   Generates a grid of images based on specified parameters and optionally annotates them using data from a CSV file.
   Images can be displayed in a random or sorted order according to metadata provided in a CSV file.

   :param dir_image_input: Directory path containing image files.
   :type dir_image_input: Union[str, Path]
   :param n_row: Number of rows in the image grid.
   :type n_row: int
   :param n_col: Number of columns in the image grid.
   :type n_col: int
   :param subplot_width: Width of each subplot. Defaults to 3.
   :type subplot_width: int, optional
   :param subplot_height: Height of each subplot. Defaults to 3.
   :type subplot_height: int, optional
   :param dir_csv_input: Directory path containing CSV files with metadata. Defaults to None.
   :type dir_csv_input: Union[str, Path], optional
   :param csv_file_pattern: Pattern to match CSV files in the directory. Defaults to None.
   :type csv_file_pattern: str, optional
   :param image_file_pattern: Pattern to match image files in the directory. Defaults to None.
   :type image_file_pattern: str, optional
   :param sort_by: Column name to sort the images by; set to "random" for random order. Defaults to "random".
   :type sort_by: str, optional
   :param ascending: Sort order. True for ascending, False for descending. Defaults to True.
   :type ascending: bool, optional
   :param use_all: If True, use all available images, otherwise use only a subset to fit the grid. Defaults to False.
   :type use_all: bool, optional
   :param title: Title of the plot. Defaults to None.
   :type title: str, optional
   :param path_output: Path to save the output plot. Defaults to None.
   :type path_output: Union[str, Path], optional
   :param random_seed: Seed for random operations to ensure reproducibility. Defaults to 42.
   :type random_seed: int, optional
   :param font_size: Font size for the plot title. Defaults to 30.
   :type font_size: int, optional
   :param dark_mode: Set to True to use a dark theme for the plot. Defaults to False.
   :type dark_mode: bool, optional
   :param dpi: Resolution in dots per inch for saving the image. Defaults to 300.
   :type dpi: int, optional

   :returns: A tuple containing the matplotlib figure and axes objects.
   :rtype: Tuple[plt.Figure, plt.Axes]

   :raises ValueError: If the specified number of rows and columns does not match the available number of images.
   :raises KeyError: If the 'sort_by' column is not found in the provided CSV files.


