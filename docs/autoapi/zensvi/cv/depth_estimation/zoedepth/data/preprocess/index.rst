:py:mod:`zensvi.cv.depth_estimation.zoedepth.data.preprocess`
=============================================================

.. py:module:: zensvi.cv.depth_estimation.zoedepth.data.preprocess


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   zensvi.cv.depth_estimation.zoedepth.data.preprocess.CropParams



Functions
~~~~~~~~~

.. autoapisummary::

   zensvi.cv.depth_estimation.zoedepth.data.preprocess.get_border_params
   zensvi.cv.depth_estimation.zoedepth.data.preprocess.get_white_border
   zensvi.cv.depth_estimation.zoedepth.data.preprocess.get_black_border
   zensvi.cv.depth_estimation.zoedepth.data.preprocess.crop_image
   zensvi.cv.depth_estimation.zoedepth.data.preprocess.crop_images
   zensvi.cv.depth_estimation.zoedepth.data.preprocess.crop_black_or_white_border



.. py:class:: CropParams


   .. py:attribute:: top
      :type: int

      

   .. py:attribute:: bottom
      :type: int

      

   .. py:attribute:: left
      :type: int

      

   .. py:attribute:: right
      :type: int

      


.. py:function:: get_border_params(rgb_image, tolerance=0.1, cut_off=20, value=0, level_diff_threshold=5, channel_axis=-1, min_border=5) -> CropParams


.. py:function:: get_white_border(rgb_image, value=255, **kwargs) -> CropParams

   Crops the white border of the RGB.

   :param rgb: RGB image, shape (H, W, 3).

   :returns: Crop parameters.


.. py:function:: get_black_border(rgb_image, **kwargs) -> CropParams

   Crops the black border of the RGB.

   :param rgb: RGB image, shape (H, W, 3).

   :returns: Crop parameters.


.. py:function:: crop_image(image: numpy.ndarray, crop_params: CropParams) -> numpy.ndarray

   Crops the image according to the crop parameters.

   :param image: RGB or depth image, shape (H, W, 3) or (H, W).
   :param crop_params: Crop parameters.

   :returns: Cropped image.


.. py:function:: crop_images(*images: numpy.ndarray, crop_params: CropParams) -> Tuple[numpy.ndarray]

   Crops the images according to the crop parameters.

   :param images: RGB or depth images, shape (H, W, 3) or (H, W).
   :param crop_params: Crop parameters.

   :returns: Cropped images.


.. py:function:: crop_black_or_white_border(rgb_image, *other_images: numpy.ndarray, tolerance=0.1, cut_off=20, level_diff_threshold=5) -> Tuple[numpy.ndarray]

   Crops the white and black border of the RGB and depth images.

   :param rgb: RGB image, shape (H, W, 3). This image is used to determine the border.
   :param other_images: The other images to crop according to the border of the RGB image.

   :returns: Cropped RGB and other images.


