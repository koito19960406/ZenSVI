:py:mod:`zensvi.metadata`
=========================

.. py:module:: zensvi.metadata


Submodules
----------
.. toctree::
   :titlesonly:
   :maxdepth: 1

   mly_metadata/index.rst


Package Contents
----------------

Classes
~~~~~~~

.. autoapisummary::

   zensvi.metadata.MLYMetadata




.. py:class:: MLYMetadata(path_input: Union[str, pathlib.Path])


   A class to compute metadata for the MLY dataset.

   :param path_input: path to the input CSV file (e.g., "mly_pids.csv"). The CSV file should contain the following columns: "id", "lat", "lon", "captured_at", "compass_angle", "creator_id", "sequence_id", "organization_id", "is_pano".
   :type path_input: Union[str, Path]

   .. py:method:: compute_metadata(unit: str = 'image', grid_resolution: int = 7, coverage_buffer: int = 50, indicator_list: str = 'all', path_output: Union[str, pathlib.Path] = None)

      Compute metadata for the dataset.

      :param unit: The unit of analysis. Defaults to "image".
      :type unit: str
      :param grid_resolution: The resolution of the H3 grid. Defaults to 7.
      :type grid_resolution: int
      :param indicator_list: List of indicators to compute metadata for. Defaults to "all". Use space-separated string of indicators or "all". Options for image-level metadata: "year", "month", "day", "hour", "day_of_week", "relative_angle". Options for grid-level metadata: "coverage", "count", "days_elapsed", "most_recent_date", "oldest_date", "number_of_years", "number_of_months", "number_of_days", "number_of_hours", "number_of_days_of_week", "number_of_daytime", "number_of_nighttime", "average_compass_angle", "average_relative_angle", "average_is_pano", "number_of_users", "number_of_sequences", "number_of_organizations". Defaults to "all".
      :type indicator_list: str
      :param path_output: Path to save the output metadata. Defaults to None.
      :type path_output: Union[str, Path]

      :return: A DataFrame containing the computed metadata.
      :rtype: pd.DataFrame



