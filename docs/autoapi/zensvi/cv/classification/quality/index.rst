:py:mod:`zensvi.cv.classification.quality`
==========================================

.. py:module:: zensvi.cv.classification.quality


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   zensvi.cv.classification.quality.ClassifierQuality




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



