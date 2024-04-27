:py:mod:`zensvi.cv.segmentation.segmentation`
=============================================

.. py:module:: zensvi.cv.segmentation.segmentation


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   zensvi.cv.segmentation.segmentation.Segmenter




.. py:class:: Segmenter(dataset='cityscapes', task='semantic', device=None)


   A class for performing semantic and panoptic segmentation on images.


   .. py:method:: segment(dir_input: Union[str, pathlib.Path], dir_image_output: Union[str, pathlib.Path, None] = None, dir_summary_output: Union[str, pathlib.Path, None] = None, batch_size=1, save_image_options='segmented_image blend_image', save_format='json csv', csv_format='long', max_workers: Union[int, None] = None)

      Processes a batch of images for segmentation, saves the segmented images and summary statistics.

      This method handles the processing of images for segmentation, managing input/output directories,
      saving options, and parallel processing settings. The method requires specifying an input directory
      or a path to a single image and supports optional saving of output images and segmentation summaries.

      :param dir_input: Path to the directory containing images or a single image file.
      :type dir_input: Union[str, Path]
      :param dir_image_output: Output directory path where segmented images
                               are saved. Defaults to None.
      :type dir_image_output: Union[str, Path, None], optional
      :param dir_summary_output: Output directory path where
                                 segmentation summary files are saved. Defaults to None.
      :type dir_summary_output: Union[str, Path, None], optional
      :param batch_size: Number of images to process in each batch. Defaults to 1.
      :type batch_size: int, optional
      :param save_image_options: Specifies the types of images to save, options include
                                 "segmented_image" and "blend_image". Defaults to "segmented_image blend_image".
      :type save_image_options: str, optional
      :param save_format: Format to save pixel ratios, options include "json" and "csv".
                          Defaults to "json csv".
      :type save_format: str, optional
      :param csv_format: Specifies the format of the CSV output as either "long" or "wide".
                         Defaults to "long".
      :type csv_format: str, optional
      :param max_workers: Maximum number of worker threads for parallel processing.
                          Defaults to None, which lets the system decide.
      :type max_workers: Union[int, None], optional

      :raises ValueError: If neither `dir_image_output` nor `dir_summary_output` is specified.
      :raises ValueError: If `dir_input` is neither a directory nor a file path.

      :returns: The method does not return any value but saves the processed results to specified directories.
      :rtype: None


   .. py:method:: calculate_pixel_ratio_post_process(dir_input, dir_output, save_format='json csv')

      Calculates the pixel ratio of different classes present in the segmented images and saves the results in either JSON or CSV format.

      :param dir_input: A string or Path object representing the input directory containing the segmented images.
      :param dir_output: A string or Path object representing the output directory where the pixel ratio results will be saved.
      :param save_format: A list containing the file formats in which the results will be saved. The allowed file formats are "json" and "csv". The default value is "json csv".

      :returns: None



