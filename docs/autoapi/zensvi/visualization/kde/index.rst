:py:mod:`zensvi.visualization.kde`
==================================

.. py:module:: zensvi.visualization.kde


Module Contents
---------------


Functions
~~~~~~~~~

.. autoapisummary::

   zensvi.visualization.kde.plot_kde



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


