:py:mod:`zensvi.transform`
==========================

.. py:module:: zensvi.transform


Submodules
----------
.. toctree::
   :titlesonly:
   :maxdepth: 1

   transform_image/index.rst


Package Contents
----------------

Classes
~~~~~~~

.. autoapisummary::

   zensvi.transform.ImageTransformer




.. py:class:: ImageTransformer(dir_input, dir_output)


   Transforms images by applying various projections such as fisheye and perspective adjustments.

   :param dir_input: Input directory containing images.
   :type dir_input: Union[str, Path]
   :param dir_output: Output directory where transformed images will be saved.
   :type dir_output: Union[str, Path]

   :raises TypeError: If the input or output directories are not specified as string or Path objects.

   .. py:property:: dir_input

      Property for the input directory.

      :return: dir_input
      :rtype: Path

   .. py:property:: dir_output

      Property for the output directory.

      :return: dir_output
      :rtype: Path

   .. py:method:: perspective(img, FOV, THETA, PHI, height, width)

      Transforms an image to simulate a perspective view from specific angles.

      :param img: Source image to transform.
      :type img: np.ndarray
      :param FOV: Field of view in degrees.
      :type FOV: float
      :param THETA: Rotation around the vertical axis in degrees.
      :type THETA: float
      :param PHI: Tilt angle in degrees.
      :type PHI: float
      :param height: Height of the output image.
      :type height: int
      :param width: Width of the output image.
      :type width: int

      :returns: Transformed image.
      :rtype: np.ndarray


   .. py:method:: equidistant_fisheye(img)

      Transforms an image to an equidistant fisheye projection.

      :param img: Source image to transform.
      :type img: np.ndarray

      :returns: Fisheye projected image.
      :rtype: np.ndarray


   .. py:method:: orthographic_fisheye(img)

      Transforms an image to an orthographic fisheye projection.

      :param img: Source image to transform.
      :type img: np.ndarray

      :returns: Fisheye projected image.
      :rtype: np.ndarray


   .. py:method:: stereographic_fisheye(img)

      Transforms an image to a stereographic fisheye projection.

      :param img: Source image to transform.
      :type img: np.ndarray

      :returns: Fisheye projected image.
      :rtype: np.ndarray


   .. py:method:: equisolid_fisheye(img)

      Transforms an image to an equisolid fisheye projection.

      :param img: Source image to transform.
      :type img: np.ndarray

      :returns: Fisheye projected image.
      :rtype: np.ndarray


   .. py:method:: transform_images(style_list: str = 'perspective equidistant_fisheye orthographic_fisheye stereographic_fisheye equisolid_fisheye', FOV: Union[int, float] = 90, theta: Union[int, float] = 90, phi: Union[int, float] = 0, aspects: tuple = (9, 16), show_size: Union[int, float] = 100)

      Applies specified transformations to all images in the input directory and saves them in the output directory.

      :param style_list: Space-separated list of transformation styles to apply. Valid styles include 'perspective',
                         'equidistant_fisheye', 'orthographic_fisheye', 'stereographic_fisheye', and 'equisolid_fisheye'.
      :type style_list: str
      :param FOV: Field of view for the 'perspective' style in degrees.
      :type FOV: Union[int, float], optional
      :param theta: Rotation step for generating multiple perspective images in degrees.
      :type theta: Union[int, float], optional
      :param phi: Tilt angle for the 'perspective' style in degrees.
      :type phi: Union[int, float], optional
      :param aspects: Aspect ratio of the output images represented as a tuple.
      :type aspects: tuple, optional
      :param show_size: Base size to calculate the dimensions of the output images.
      :type show_size: Union[int, float], optional

      :raises ValueError: If an invalid style is specified in style_list.

      .. rubric:: Notes

      This method processes images concurrently, leveraging multi-threading to speed up the transformation tasks. It
      automatically splits style_list into individual styles and processes each style, creating appropriate subdirectories
      in the output directory for each style.



