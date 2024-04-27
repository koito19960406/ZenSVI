:py:mod:`zensvi.cv`
===================

.. py:module:: zensvi.cv


Subpackages
-----------
.. toctree::
   :titlesonly:
   :maxdepth: 3

   depth_estimation/index.rst


Package Contents
----------------

Classes
~~~~~~~

.. autoapisummary::

   zensvi.cv.Segmenter
   zensvi.cv.ClassifierPlaces365
   zensvi.cv.ClassifierWeather
   zensvi.cv.ClassifierGlare
   zensvi.cv.ClassifierLighting
   zensvi.cv.ClassifierPanorama
   zensvi.cv.ClassifierPlatform
   zensvi.cv.ClassifierQuality
   zensvi.cv.ClassifierReflection
   zensvi.cv.ClassifierViewDirection
   zensvi.cv.DepthEstimator



Functions
~~~~~~~~~

.. autoapisummary::

   zensvi.cv.get_low_level_features



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


.. py:class:: ClassifierPlaces365(device=None)


   Bases: :py:obj:`zensvi.cv.classification.base.BaseClassifier`

   A classifier for identifying places using the Places365 model.

   :param device: The device that the model should be loaded onto. Options are "cpu", "cuda", or "mps".
       If `None`, the model tries to use a GPU if available; otherwise, falls back to CPU.
   :type device: str, optional

   .. py:method:: classify(dir_input: Union[str, pathlib.Path], dir_image_output: Union[str, pathlib.Path, None] = None, dir_summary_output: Union[str, pathlib.Path, None] = None, batch_size: int = 1, save_image_options: str = 'cam_image blend_image', save_format: str = 'json csv', csv_format: str = 'long')

      Classifies images based on scene recognition using the Places365 model. The output file can be saved in JSON and/or CSV format and will contain the scene categories, scene attributes, and environment type (indoor or outdoor) for each image.

      A list of categories can be found at https://github.com/CSAILVision/places365/blob/master/categories_places365.txt and a list of attributes can be found at https://github.com/CSAILVision/places365/blob/master/labels_sunattribute.txt

      Scene categories' values range from 0 to 1, where 1 is the highest probability of the scene category. Scene attributes' values are the responses of the scene attributes, which are the dot product of the scene attributes' weight and the features of the image, and higher values indicate a higher presence of the attribute in the image. The environment type is either "indoor" or "outdoor".

      :param dir_input: directory containing input images.
      :type dir_input: Union[str, Path]
      :param dir_image_output: directory to save output images, defaults to None
      :type dir_image_output: Union[str, Path, None], optional
      :param dir_summary_output: directory to save summary output, defaults to None
      :type dir_summary_output: Union[str, Path, None], optional
      :param batch_size: batch size for inference, defaults to 1
      :type batch_size: int, optional
      :param save_image_options: save options for images, defaults to "cam_image blend_image". Options are "cam_image" and "blend_image". Please add a space between options.
      :type save_image_options: str, optional
      :param save_format: save format for the output, defaults to "json csv". Options are "json" and "csv". Please add a space between options.
      :type save_format: str, optional
      :param csv_format: csv format for the output, defaults to "long". Options are "long" and "wide".
      :type csv_format: str, optional



.. py:class:: ClassifierWeather(device=None)


   Bases: :py:obj:`zensvi.cv.classification.base.BaseClassifier`

   A classifier for identifying weather.

   :param device: The device that the model should be loaded onto. Options are "cpu", "cuda", or "mps".
       If `None`, the model tries to use a GPU if available; otherwise, falls back to CPU.
   :type device: str, optional

   .. py:method:: classify(dir_input: Union[str, pathlib.Path], dir_summary_output: Union[str, pathlib.Path], batch_size=1, save_format='json csv') -> List[str]

      Classifies images based on weather. The output file can be saved in JSON and/or CSV format and will contain weather for each image. The weather categories are "clear", "cloudy", "foggy", "rainy", and "snowy".

      :param dir_input: directory containing input images.
      :type dir_input: Union[str, Path]
      :param dir_summary_output: directory to save summary output.
      :type dir_summary_output: Union[str, Path, None]
      :param batch_size: batch size for inference, defaults to 1
      :type batch_size: int, optional
      :param save_format: save format for the output, defaults to "json csv". Options are "json" and "csv". Please add a space between options.
      :type save_format: str, optional



.. py:class:: ClassifierGlare(device=None)


   Bases: :py:obj:`zensvi.cv.classification.base.BaseClassifier`

   A classifier for identifying glare.

   :param device: The device that the model should be loaded onto. Options are "cpu", "cuda", or "mps".
       If `None`, the model tries to use a GPU if available; otherwise, falls back to CPU.
   :type device: str, optional

   .. py:method:: classify(dir_input: Union[str, pathlib.Path], dir_summary_output: Union[str, pathlib.Path], batch_size=1, save_format='json csv') -> List[str]

      Classifies images based on glare. The output file can be saved in JSON and/or CSV format and will contain glare for each image. The glare categories are "True" and "False".

      :param dir_input: directory containing input images.
      :type dir_input: Union[str, Path]
      :param dir_summary_output: directory to save summary output.
      :type dir_summary_output: Union[str, Path, None]
      :param batch_size: batch size for inference, defaults to 1
      :type batch_size: int, optional
      :param save_format: save format for the output, defaults to "json csv". Options are "json" and "csv". Please add a space between options.
      :type save_format: str, optional



.. py:class:: ClassifierLighting(device=None)


   Bases: :py:obj:`zensvi.cv.classification.base.BaseClassifier`

   A classifier for identifying lighting.

   :param device: The device that the model should be loaded onto. Options are "cpu", "cuda", or "mps".
       If `None`, the model tries to use a GPU if available; otherwise, falls back to CPU.
   :type device: str, optional

   .. py:method:: classify(dir_input: Union[str, pathlib.Path], dir_summary_output: Union[str, pathlib.Path], batch_size=1, save_format='json csv') -> List[str]

      Classifies images based on lighting. The output file can be saved in JSON and/or CSV format and will contain lighting for each image. The lighting categories are "day", "night", and "dawn/dusk".

      :param dir_input: directory containing input images.
      :type dir_input: Union[str, Path]
      :param dir_summary_output: directory to save summary output.
      :type dir_summary_output: Union[str, Path, None]
      :param batch_size: batch size for inference, defaults to 1
      :type batch_size: int, optional
      :param save_format: save format for the output, defaults to "json csv". Options are "json" and "csv". Please add a space between options.
      :type save_format: str, optional



.. py:class:: ClassifierPanorama(device=None)


   Bases: :py:obj:`zensvi.cv.classification.base.BaseClassifier`

   A classifier for identifying panorama.

   :param device: The device that the model should be loaded onto. Options are "cpu", "cuda", or "mps".
       If `None`, the model tries to use a GPU if available; otherwise, falls back to CPU.
   :type device: str, optional

   .. py:method:: classify(dir_input: Union[str, pathlib.Path], dir_summary_output: Union[str, pathlib.Path], batch_size=1, save_format='json csv') -> List[str]

      Classifies images based on panorama. The output file can be saved in JSON and/or CSV format and will contain panorama for each image. The panorama categories are "True" and "False".

      :param dir_input: directory containing input images.
      :type dir_input: Union[str, Path]
      :param dir_summary_output: directory to save summary output.
      :type dir_summary_output: Union[str, Path, None]
      :param batch_size: batch size for inference, defaults to 1
      :type batch_size: int, optional
      :param save_format: save format for the output, defaults to "json csv". Options are "json" and "csv". Please add a space between options.
      :type save_format: str, optional



.. py:class:: ClassifierPlatform(device=None)


   Bases: :py:obj:`zensvi.cv.classification.base.BaseClassifier`

   A classifier for identifying platform.

   :param device: The device that the model should be loaded onto. Options are "cpu", "cuda", or "mps".
       If `None`, the model tries to use a GPU if available; otherwise, falls back to CPU.
   :type device: str, optional

   .. py:method:: classify(dir_input: Union[str, pathlib.Path], dir_summary_output: Union[str, pathlib.Path], batch_size=1, save_format='json csv') -> List[str]

      Classifies images based on platform. The output file can be saved in JSON and/or CSV format and will contain platform for each image. The platform categories are "driving surface", "walking surface", "cycling surface", "tunnel", "field", and "railway".

      :param dir_input: directory containing input images.
      :type dir_input: Union[str, Path]
      :param dir_summary_output: directory to save summary output.
      :type dir_summary_output: Union[str, Path, None]
      :param batch_size: batch size for inference, defaults to 1
      :type batch_size: int, optional
      :param save_format: save format for the output, defaults to "json csv". Options are "json" and "csv". Please add a space between options.
      :type save_format: str, optional



.. py:class:: ClassifierQuality(device=None)


   Bases: :py:obj:`zensvi.cv.classification.base.BaseClassifier`

   A classifier for identifying quality.

   :param device: The device that the model should be loaded onto. Options are "cpu", "cuda", or "mps".
       If `None`, the model tries to use a GPU if available; otherwise, falls back to CPU.
   :type device: str, optional

   .. py:method:: classify(dir_input: Union[str, pathlib.Path], dir_summary_output: Union[str, pathlib.Path], batch_size=1, save_format='json csv') -> List[str]

      Classifies images based on quality. The output file can be saved in JSON and/or CSV format and will contain quality for each image. The quality categories are "good", "slghtly poor", and "very poor".

      :param dir_input: directory containing input images.
      :type dir_input: Union[str, Path]
      :param dir_summary_output: directory to save summary output.
      :type dir_summary_output: Union[str, Path, None]
      :param batch_size: batch size for inference, defaults to 1
      :type batch_size: int, optional
      :param save_format: save format for the output, defaults to "json csv". Options are "json" and "csv". Please add a space between options.
      :type save_format: str, optional



.. py:class:: ClassifierReflection(device=None)


   Bases: :py:obj:`zensvi.cv.classification.base.BaseClassifier`

   A classifier for identifying reflection.

   :param device: The device that the model should be loaded onto. Options are "cpu", "cuda", or "mps".
       If `None`, the model tries to use a GPU if available; otherwise, falls back to CPU.
   :type device: str, optional

   .. py:method:: classify(dir_input: Union[str, pathlib.Path], dir_summary_output: Union[str, pathlib.Path], batch_size=1, save_format='json csv') -> List[str]

      Classifies images based on reflection. The output file can be saved in JSON and/or CSV format and will contain reflection for each image. The reflection categories are "True" and "False".

      :param dir_input: directory containing input images.
      :type dir_input: Union[str, Path]
      :param dir_summary_output: directory to save summary output.
      :type dir_summary_output: Union[str, Path, None]
      :param batch_size: batch size for inference, defaults to 1
      :type batch_size: int, optional
      :param save_format: save format for the output, defaults to "json csv". Options are "json" and "csv". Please add a space between options.
      :type save_format: str, optional



.. py:class:: ClassifierViewDirection(device=None)


   Bases: :py:obj:`zensvi.cv.classification.base.BaseClassifier`

   A classifier for identifying view_direction.

   :param device: The device that the model should be loaded onto. Options are "cpu", "cuda", or "mps".
       If `None`, the model tries to use a GPU if available; otherwise, falls back to CPU.
   :type device: str, optional

   .. py:method:: classify(dir_input: Union[str, pathlib.Path], dir_summary_output: Union[str, pathlib.Path], batch_size=1, save_format='json csv') -> List[str]

      Classifies images based on view_direction. The output file can be saved in JSON and/or CSV format and will contain view_direction for each image. The view_direction categories are "front/back" and "side".

      :param dir_input: directory containing input images.
      :type dir_input: Union[str, Path]
      :param dir_summary_output: directory to save summary output.
      :type dir_summary_output: Union[str, Path, None]
      :param batch_size: batch size for inference, defaults to 1
      :type batch_size: int, optional
      :param save_format: save format for the output, defaults to "json csv". Options are "json" and "csv". Please add a space between options.
      :type save_format: str, optional



.. py:class:: DepthEstimator(device=None, task='relative')


   A class for estimating depth in images. The class uses the DPT model from Hugging Face for relative depth estimation and the ZoeDepth model for absolute depth estimation.

   :param device: device to use for inference, defaults to None
   :type device: str, optional
   :param task: task to perform, either "relative" or "absolute", defaults to "relative"
   :type task: str, optional

   .. py:method:: estimate_depth(dir_input: Union[str, pathlib.Path], dir_image_output: Union[str, pathlib.Path], batch_size: int = 1, max_workers: int = 4)

      Estimates relative depth in the images. Saves the depth maps in the specified directory.

      :param dir_input: directory containing input images.
      :type dir_input: Union[str, Path]
      :param dir_image_output: directory to save the depth maps.
      :type dir_image_output: Union[str, Path]
      :param batch_size: batch size for inference, defaults to 1
      :type batch_size: int, optional
      :param max_workers: maximum number of workers for parallel processing, defaults to 4
      :type max_workers: int, optional



