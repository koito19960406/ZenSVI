:py:mod:`zensvi.cv.low_level.low_level`
=======================================

.. py:module:: zensvi.cv.low_level.low_level


Module Contents
---------------


Functions
~~~~~~~~~

.. autoapisummary::

   zensvi.cv.low_level.low_level.get_low_level_features



.. py:function:: get_low_level_features(dir_input, dir_image_output: Union[str, pathlib.Path] = None, dir_summary_output: Union[str, pathlib.Path] = None, save_format='json csv', csv_format='long')

   Processes images from the specified input directory or single image file to detect various low-level features,
   which include edge detection, blob detection, blur detection, and HSL color space analysis. It optionally saves
   the processed images and a summary of the features detected.

   :param dir_input: The directory containing images or a single image file path from which
                     to extract features.
   :type dir_input: Union[str, Path]
   :param dir_image_output: The directory where processed images will be saved.
                            If not provided, images will not be saved. Defaults to None.
   :type dir_image_output: Union[str, Path], optional
   :param dir_summary_output: The directory where the summary of detected features
                              will be saved in JSON or CSV format. If not provided, summaries will not be saved. Defaults to None.
   :type dir_summary_output: Union[str, Path], optional
   :param save_format: Specifies the formats in which to save the summary of features. Possible
                       values include "json", "csv", or a combination of both. Defaults to "json csv".
   :type save_format: str, optional
   :param csv_format: Specifies the format of the CSV output. Can be 'long' for long format or
                      'wide' for wide format. Defaults to 'long'.
   :type csv_format: str, optional

   :raises ValueError: If neither `dir_image_output` nor `dir_summary_output` is provided, a ValueError is raised
       indicating that at least one output directory must be specified.

   :returns: The function does not return any value but outputs results to the specified directories.
   :rtype: None


