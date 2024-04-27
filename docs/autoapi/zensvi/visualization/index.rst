:py:mod:`zensvi.visualization`
==============================

.. py:module:: zensvi.visualization


Submodules
----------
.. toctree::
   :titlesonly:
   :maxdepth: 1

   hist/index.rst
   image/index.rst
   kde/index.rst
   map/index.rst


Package Contents
----------------


Functions
~~~~~~~~~

.. autoapisummary::

   zensvi.visualization.plot_map
   zensvi.visualization.plot_image
   zensvi.visualization.plot_kde
   zensvi.visualization.plot_hist



.. py:function:: plot_map(path_pid: Union[str, pathlib.Path], pid_column: str = 'panoid', dir_input: Union[str, pathlib.Path] = None, csv_file_pattern: str = '*.csv', variable_name: str = None, plot_type: str = 'point', path_output: Union[str, pathlib.Path] = None, resolution: int = 7, cmap: str = 'viridis', legend: bool = True, title: str = None, legend_title: str = None, basemap_source: Any = ctx.providers.CartoDB.PositronNoLabels, figure_size: Tuple[int, int] = (10, 10), dpi: int = 300, font_size: int = 30, dark_mode: bool = False, **kwargs) -> Tuple[matplotlib.pyplot.Figure, matplotlib.pyplot.Axes]

   Plots a geographic map from data points, with options for line or hexagonal aggregations, coloring by variables,
   and using different base maps.

   :param path_pid: Path to the CSV file containing longitude and latitude and other metadata.
   :type path_pid: Union[str, Path]
   :param pid_column: Column name in CSV that acts as a primary key or identifier. Defaults to "panoid".
   :type pid_column: str, optional
   :param dir_input: Directory path where additional CSV data files are stored, matched by pattern. Defaults to None.
   :type dir_input: Union[str, Path], optional
   :param csv_file_pattern: Pattern to match CSV files in the directory. Defaults to None.
   :type csv_file_pattern: str, optional
   :param variable_name: Name of the variable in CSV to use for coloring and aggregation. Defaults to None.
   :type variable_name: str, optional
   :param plot_type: Type of plot to generate: 'point', 'line', or 'hexagon'. Defaults to "point".
   :type plot_type: str, optional
   :param path_output: Path where the plotted figure will be saved. Defaults to None.
   :type path_output: Union[str, Path], optional
   :param resolution: Resolution level for H3 hexagonal tiling. Defaults to 7.
   :type resolution: int, optional
   :param cmap: Colormap for the plot. Defaults to "viridis".
   :type cmap: str, optional
   :param legend: Whether to add a color legend to the plot. Defaults to True.
   :type legend: bool, optional
   :param title: Title of the plot. Defaults to None.
   :type title: str, optional
   :param legend_title: Title for the legend. Defaults to None.
   :type legend_title: str, optional
   :param basemap_source: Contextily basemap source. Defaults to ctx.providers.CartoDB.PositronNoLabels.
   :type basemap_source: Any, optional
   :param dpi: Dots per inch (resolution) of the output image file. Defaults to 300.
   :type dpi: int, optional
   :param font_size: Font size for titles and legend. Defaults to 30.
   :type font_size: int, optional
   :param dark_mode: Whether to use a dark theme for the plot. Defaults to False.
   :type dark_mode: bool, optional
   :param \*\*kwargs: Additional keyword arguments passed to GeoPandas plot function.

   :returns: A tuple containing the Matplotlib figure and axes objects.
   :rtype: Tuple[plt.Figure, plt.Axes]

   :raises ValueError: If an invalid `plot_type` is provided.


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


.. py:function:: plot_kde(dir_input: Union[str, pathlib.Path], columns: List[str], csv_file_pattern: str = '*.csv', path_output: Union[str, pathlib.Path] = None, legend: bool = True, title: str = None, legend_title: str = None, fig_size: Tuple[int, int] = (10, 10), dpi: int = 300, font_size: int = 30, dark_mode: bool = False, **kwargs) -> Tuple[matplotlib.pyplot.Figure, matplotlib.pyplot.Axes]

   Plots KDE (Kernel Density Estimate) plots for specified columns from a CSV file using Seaborn.

   :param path_input: Path to the CSV file.
   :type path_input: Union[str, Path]
   :param columns: List of column names to plot KDEs for.
   :type columns: List[str]
   :param path_output: Path where the plotted figure will be saved. Defaults to None.
   :type path_output: Union[str, Path], optional
   :param legend: Whether to add a legend to the plot. Defaults to True.
   :type legend: bool
   :param title: Title of the plot. Defaults to None.
   :type title: str, optional
   :param legend_title: Title for the legend. Defaults to None.
   :type legend_title: str, optional
   :param dpi: Dots per inch (resolution) of the output image. Defaults to 300.
   :type dpi: int
   :param font_size: Font size for titles and legend. Defaults to 30.
   :type font_size: int
   :param dark_mode: Whether to use a dark theme for the plot. Defaults to False.
   :type dark_mode: bool
   :param \*\*kwargs: Additional keyword arguments passed to seaborn.kdeplot.

   :returns: A tuple containing the Matplotlib figure and axes objects.
   :rtype: Tuple[plt.Figure, plt.Axes]


.. py:function:: plot_hist(dir_input: Union[str, pathlib.Path], columns: List[str], csv_file_pattern: str = '*.csv', path_output: Union[str, pathlib.Path] = None, legend: bool = True, title: str = None, legend_title: str = None, fig_size: Tuple[int, int] = (10, 10), dpi: int = 300, font_size: int = 30, dark_mode: bool = False, **kwargs) -> Tuple[matplotlib.pyplot.Figure, matplotlib.pyplot.Axes]

   Plots hist (Kernel Density Estimate) plots for specified columns from a CSV file using Seaborn.

   :param path_input: Path to the CSV file.
   :type path_input: Union[str, Path]
   :param columns: List of column names to plot hists for.
   :type columns: List[str]
   :param path_output: Path where the plotted figure will be saved. Defaults to None.
   :type path_output: Union[str, Path], optional
   :param legend: Whether to add a legend to the plot. Defaults to True.
   :type legend: bool
   :param title: Title of the plot. Defaults to None.
   :type title: str, optional
   :param legend_title: Title for the legend. Defaults to None.
   :type legend_title: str, optional
   :param dpi: Dots per inch (resolution) of the output image. Defaults to 300.
   :type dpi: int
   :param font_size: Font size for titles and legend. Defaults to 30.
   :type font_size: int
   :param dark_mode: Whether to use a dark theme for the plot. Defaults to False.
   :type dark_mode: bool
   :param \*\*kwargs: Additional keyword arguments passed to seaborn.histplot.

   :returns: A tuple containing the Matplotlib figure and axes objects.
   :rtype: Tuple[plt.Figure, plt.Axes]


