:py:mod:`zensvi.visualization.map`
==================================

.. py:module:: zensvi.visualization.map


Module Contents
---------------


Functions
~~~~~~~~~

.. autoapisummary::

   zensvi.visualization.map.plot_map



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


