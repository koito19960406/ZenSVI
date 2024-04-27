:py:mod:`zensvi.cv.depth_estimation`
====================================

.. py:module:: zensvi.cv.depth_estimation


Submodules
----------
.. toctree::
   :titlesonly:
   :maxdepth: 1

   depth_estimation/index.rst


Package Contents
----------------

Classes
~~~~~~~

.. autoapisummary::

   zensvi.cv.depth_estimation.DepthEstimator




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



