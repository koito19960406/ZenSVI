:py:mod:`zensvi.cv.classification.places365`
============================================

.. py:module:: zensvi.cv.classification.places365


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   zensvi.cv.classification.places365.ClassifierPlaces365




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



