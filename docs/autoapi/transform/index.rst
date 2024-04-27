:py:mod:`transform`
===================

.. py:module:: transform


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   transform.Resize
   transform.NormalizeImage
   transform.PrepareForNet



Functions
~~~~~~~~~

.. autoapisummary::

   transform.apply_min_size



.. py:function:: apply_min_size(sample, size, image_interpolation_method=cv2.INTER_AREA)

   Rezise the sample to ensure the given size. Keeps aspect ratio.

   :param sample: sample
   :type sample: dict
   :param size: image size
   :type size: tuple

   :returns: new size
   :rtype: tuple


.. py:class:: Resize(width, height, resize_target=True, keep_aspect_ratio=False, ensure_multiple_of=1, resize_method='lower_bound', image_interpolation_method=cv2.INTER_AREA)


   Bases: :py:obj:`object`

   Resize sample to given size (width, height).


   .. py:method:: constrain_to_multiple_of(x, min_val=0, max_val=None)


   .. py:method:: get_size(width, height)


   .. py:method:: __call__(sample)



.. py:class:: NormalizeImage(mean, std)


   Bases: :py:obj:`object`

   Normlize image by given mean and std.


   .. py:method:: __call__(sample)



.. py:class:: PrepareForNet


   Bases: :py:obj:`object`

   Prepare sample for usage as network input.


   .. py:method:: __call__(sample)



