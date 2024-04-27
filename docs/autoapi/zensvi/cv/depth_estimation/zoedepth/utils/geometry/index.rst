:py:mod:`zensvi.cv.depth_estimation.zoedepth.utils.geometry`
============================================================

.. py:module:: zensvi.cv.depth_estimation.zoedepth.utils.geometry


Module Contents
---------------


Functions
~~~~~~~~~

.. autoapisummary::

   zensvi.cv.depth_estimation.zoedepth.utils.geometry.get_intrinsics
   zensvi.cv.depth_estimation.zoedepth.utils.geometry.depth_to_points
   zensvi.cv.depth_estimation.zoedepth.utils.geometry.create_triangles



.. py:function:: get_intrinsics(H, W)

   Intrinsics for a pinhole camera model.
   Assume fov of 55 degrees and central principal point.


.. py:function:: depth_to_points(depth, R=None, t=None)


.. py:function:: create_triangles(h, w, mask=None)

   Reference: https://github.com/google-research/google-research/blob/e96197de06613f1b027d20328e06d69829fa5a89/infinite_nature/render_utils.py#L68
   Creates mesh triangle indices from a given pixel grid size.
       This function is not and need not be differentiable as triangle indices are
       fixed.
   Args:
   h: (int) denoting the height of the image.
   w: (int) denoting the width of the image.
   Returns:
   triangles: 2D numpy array of indices (int) with shape (2(W-1)(H-1) x 3)


