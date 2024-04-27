:py:mod:`zensvi.download.utils.imtool`
======================================

.. py:module:: zensvi.download.utils.imtool


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   zensvi.download.utils.imtool.ImageTool




.. py:class:: ImageTool


   .. py:method:: concat_horizontally(im1, im2)
      :staticmethod:

      Description of concat_horizontally
      Horizontally concatenates two images

      :param im1: first PIL image
      :type im1: undefined
      :param im2: second PIL image
      :type im2: undefined


   .. py:method:: concat_vertically(im1, im2)
      :staticmethod:

      Description of concat_vertically
      Vertically concatenates two images

      :param im1: first PIL image
      :type im1: undefined
      :param im2: second PIL image
      :type im2: undefined


   .. py:method:: fetch_image_with_proxy(pano_id, zoom, x, y, ua, proxies)
      :staticmethod:

      Fetches an image using a proxy.

      :param pano_id: GSV panorama id
      :type pano_id: str
      :param zoom: Zoom level for the image
      :type zoom: int
      :param x: The x coordinate of the tile
      :type x: int
      :param y: The y coordinate of the tile
      :type y: int
      :param ua: User agent string
      :type ua: str
      :param proxies: A list of available proxies
      :type proxies: list

      :returns: The fetched image
      :rtype: Image


   .. py:method:: is_bottom_black(image, row_count=3, intensity_threshold=10)
      :staticmethod:

      Check if the bottom 'row_count' rows of the image are near black, with a given intensity threshold.
      This method uses linear computation instead of nested loops for faster execution.

      :param image: The image to check.
      :type image: PIL.Image
      :param row_count: Number of rows to check.
      :type row_count: int
      :param intensity_threshold: The maximum intensity for a pixel to be considered black.
      :type intensity_threshold: int

      :returns: True if the bottom rows are near black, False otherwise.
      :rtype: bool


   .. py:method:: process_image(image, zoom)
      :staticmethod:

      Crop and resize the image based on zoom level if the bottom is black.

      :param image: The image to process.
      :type image: PIL.Image
      :param zoom: The zoom level.
      :type zoom: int

      :returns: The processed image.
      :rtype: PIL.Image


   .. py:method:: get_and_save_image(pano_id, identif, zoom, vertical_tiles, horizontal_tiles, out_path, ua, proxies, cropped=False, full=True)
      :staticmethod:

      Description of get_and_save_image

      Downloads an image tile by tile and composes them together.

      :param pano_id: GSV anorama id
      :type pano_id: undefined
      :param identif: custom identifier
      :type identif: undefined
      :param size: image resolution
      :type size: undefined
      :param vertical_tiles: number of vertical tiles
      :type vertical_tiles: undefined
      :param horizontal_tiles: number of horizontal tiles
      :type horizontal_tiles: undefined
      :param out_path: output path
      :type out_path: undefined
      :param cropped=False: set True if the image split horizontally in half is needed
      :type cropped=False: undefined
      :param full=True: set to True if the full image is needed
      :type full=True: undefined


   .. py:method:: dwl_multiple(panoids, zoom, v_tiles, h_tiles, out_path, uas, proxies, cropped, full, batch_size=1000, logger=None)
      :staticmethod:

      Description of dwl_multiple

      Calls the get_and_save_image function using multiple threads.

      :param panoids: GSV anorama id
      :type panoids: undefined
      :param zoom: image resolution
      :type zoom: undefined
      :param v_tiles: number of vertical tiles
      :type v_tiles: undefined
      :param h_tiles: number of horizontal tiles
      :type h_tiles: undefined
      :param out_path: output path
      :type out_path: undefined
      :param cropped=False: set True if the image split horizontally in half is needed
      :type cropped=False: undefined
      :param full=True: set to True if the full image is needed
      :type full=True: undefined
      :param log_path=None: path to a log file
      :type log_path=None: undefined



