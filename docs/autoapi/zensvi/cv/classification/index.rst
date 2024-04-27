:orphan:

:py:mod:`zensvi.cv.classification`
==================================

.. py:module:: zensvi.cv.classification


Submodules
----------
.. toctree::
   :titlesonly:
   :maxdepth: 1

   glare/index.rst
   lighting/index.rst
   panorama/index.rst
   places365/index.rst
   platform/index.rst
   quality/index.rst
   reflection/index.rst
   view_direction/index.rst
   weather/index.rst


Package Contents
----------------

Classes
~~~~~~~

.. autoapisummary::

   zensvi.cv.classification.ClassifierPlaces365
   zensvi.cv.classification.ClassifierWeather
   zensvi.cv.classification.ClassifierGlare
   zensvi.cv.classification.ClassifierLighting
   zensvi.cv.classification.ClassifierPanorama
   zensvi.cv.classification.ClassifierPlatform
   zensvi.cv.classification.ClassifierQuality
   zensvi.cv.classification.ClassifierReflection
   zensvi.cv.classification.ClassifierViewDirection




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



