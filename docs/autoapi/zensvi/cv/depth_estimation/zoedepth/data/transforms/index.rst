:py:mod:`zensvi.cv.depth_estimation.zoedepth.data.transforms`
=============================================================

.. py:module:: zensvi.cv.depth_estimation.zoedepth.data.transforms


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   zensvi.cv.depth_estimation.zoedepth.data.transforms.RandomFliplr
   zensvi.cv.depth_estimation.zoedepth.data.transforms.RandomCrop
   zensvi.cv.depth_estimation.zoedepth.data.transforms.Resize
   zensvi.cv.depth_estimation.zoedepth.data.transforms.ResizeFixed
   zensvi.cv.depth_estimation.zoedepth.data.transforms.Rescale
   zensvi.cv.depth_estimation.zoedepth.data.transforms.NormalizeImage
   zensvi.cv.depth_estimation.zoedepth.data.transforms.DepthToDisparity
   zensvi.cv.depth_estimation.zoedepth.data.transforms.DisparityToDepth
   zensvi.cv.depth_estimation.zoedepth.data.transforms.PrepareForNet



Functions
~~~~~~~~~

.. autoapisummary::

   zensvi.cv.depth_estimation.zoedepth.data.transforms.apply_min_size



.. py:class:: RandomFliplr(probability=0.5)


   Bases: :py:obj:`object`

   Horizontal flip of the sample with given probability.


   .. py:method:: __call__(sample)



.. py:function:: apply_min_size(sample, size, image_interpolation_method=cv2.INTER_AREA)

   Rezise the sample to ensure the given size. Keeps aspect ratio.

   :param sample: sample
   :type sample: dict
   :param size: image size
   :type size: tuple

   :returns: new size
   :rtype: tuple


.. py:class:: RandomCrop(width, height, resize_if_needed=False, image_interpolation_method=cv2.INTER_AREA)


   Bases: :py:obj:`object`

   Get a random crop of the sample with the given size (width, height).


   .. py:method:: __call__(sample)



.. py:class:: Resize(width, height, resize_target=True, keep_aspect_ratio=False, ensure_multiple_of=1, resize_method='lower_bound', image_interpolation_method=cv2.INTER_AREA, letter_box=False)


   Bases: :py:obj:`object`

   Resize sample to given size (width, height).


   .. py:method:: constrain_to_multiple_of(x, min_val=0, max_val=None)


   .. py:method:: get_size(width, height)


   .. py:method:: make_letter_box(sample)


   .. py:method:: __call__(sample)



.. py:class:: ResizeFixed(size)


   Bases: :py:obj:`object`

   .. py:method:: __call__(sample)



.. py:class:: Rescale(max_val=1.0, use_mask=True)


   Bases: :py:obj:`object`

   Rescale target values to the interval [0, max_val].
   If input is constant, values are set to max_val / 2.

   .. py:method:: __call__(sample)



.. py:class:: NormalizeImage(mean, std)


   Bases: :py:obj:`object`

   Normlize image by given mean and std.


   .. py:method:: __call__(sample)



.. py:class:: DepthToDisparity(eps=0.0001)


   Bases: :py:obj:`object`

   Convert depth to disparity. Removes depth from sample.


   .. py:method:: __call__(sample)



.. py:class:: DisparityToDepth(eps=0.0001)


   Bases: :py:obj:`object`

   Convert disparity to depth. Removes disparity from sample.


   .. py:method:: __call__(sample)



.. py:class:: PrepareForNet


   Bases: :py:obj:`object`

   Prepare sample for usage as network input.


   .. py:method:: __call__(sample)



