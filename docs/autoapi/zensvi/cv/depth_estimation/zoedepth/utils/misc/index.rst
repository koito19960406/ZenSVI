:py:mod:`zensvi.cv.depth_estimation.zoedepth.utils.misc`
========================================================

.. py:module:: zensvi.cv.depth_estimation.zoedepth.utils.misc

.. autoapi-nested-parse::

   Miscellaneous utility functions.



Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   zensvi.cv.depth_estimation.zoedepth.utils.misc.RunningAverage
   zensvi.cv.depth_estimation.zoedepth.utils.misc.RunningAverageDict
   zensvi.cv.depth_estimation.zoedepth.utils.misc.colors



Functions
~~~~~~~~~

.. autoapisummary::

   zensvi.cv.depth_estimation.zoedepth.utils.misc.denormalize
   zensvi.cv.depth_estimation.zoedepth.utils.misc.colorize
   zensvi.cv.depth_estimation.zoedepth.utils.misc.count_parameters
   zensvi.cv.depth_estimation.zoedepth.utils.misc.compute_errors
   zensvi.cv.depth_estimation.zoedepth.utils.misc.compute_metrics
   zensvi.cv.depth_estimation.zoedepth.utils.misc.parallelize
   zensvi.cv.depth_estimation.zoedepth.utils.misc.printc
   zensvi.cv.depth_estimation.zoedepth.utils.misc.get_image_from_url
   zensvi.cv.depth_estimation.zoedepth.utils.misc.url_to_torch
   zensvi.cv.depth_estimation.zoedepth.utils.misc.pil_to_batched_tensor
   zensvi.cv.depth_estimation.zoedepth.utils.misc.save_raw_16bit



.. py:class:: RunningAverage


   .. py:method:: append(value)


   .. py:method:: get_value()



.. py:function:: denormalize(x)

   Reverses the imagenet normalization applied to the input.

   :param x: input tensor
   :type x: torch.Tensor - shape(N,3,H,W)

   :returns: Denormalized input
   :rtype: torch.Tensor - shape(N,3,H,W)


.. py:class:: RunningAverageDict


   A dictionary of running averages.

   .. py:method:: update(new_dict)


   .. py:method:: get_value()



.. py:function:: colorize(value, vmin=None, vmax=None, cmap='gray_r', invalid_val=-99, invalid_mask=None, background_color=(128, 128, 128, 255), gamma_corrected=False, value_transform=None)

   Converts a depth map to a color image.

   :param value: Input depth map. Shape: (H, W) or (1, H, W) or (1, 1, H, W). All singular dimensions are squeezed
   :type value: torch.Tensor, numpy.ndarry
   :param vmin: vmin-valued entries are mapped to start color of cmap. If None, value.min() is used. Defaults to None.
   :type vmin: float, optional
   :param vmax: vmax-valued entries are mapped to end color of cmap. If None, value.max() is used. Defaults to None.
   :type vmax: float, optional
   :param cmap: matplotlib colormap to use. Defaults to 'magma_r'.
   :type cmap: str, optional
   :param invalid_val: Specifies value of invalid pixels that should be colored as 'background_color'. Defaults to -99.
   :type invalid_val: int, optional
   :param invalid_mask: Boolean mask for invalid regions. Defaults to None.
   :type invalid_mask: numpy.ndarray, optional
   :param background_color: 4-tuple RGB color to give to invalid pixels. Defaults to (128, 128, 128, 255).
   :type background_color: tuple[int], optional
   :param gamma_corrected: Apply gamma correction to colored image. Defaults to False.
   :type gamma_corrected: bool, optional
   :param value_transform: Apply transform function to valid pixels before coloring. Defaults to None.
   :type value_transform: Callable, optional

   :returns: Colored depth map. Shape: (H, W, 4)
   :rtype: numpy.ndarray, dtype - uint8


.. py:function:: count_parameters(model, include_all=False)


.. py:function:: compute_errors(gt, pred)

   Compute metrics for 'pred' compared to 'gt'

   :param gt: Ground truth values
   :type gt: numpy.ndarray
   :param pred: Predicted values
   :type pred: numpy.ndarray
   :param gt.shape should be equal to pred.shape:

   :returns:

             Dictionary containing the following metrics:
                 'a1': Delta1 accuracy: Fraction of pixels that are within a scale factor of 1.25
                 'a2': Delta2 accuracy: Fraction of pixels that are within a scale factor of 1.25^2
                 'a3': Delta3 accuracy: Fraction of pixels that are within a scale factor of 1.25^3
                 'abs_rel': Absolute relative error
                 'rmse': Root mean squared error
                 'log_10': Absolute log10 error
                 'sq_rel': Squared relative error
                 'rmse_log': Root mean squared error on the log scale
                 'silog': Scale invariant log error
   :rtype: dict


.. py:function:: compute_metrics(gt, pred, interpolate=True, garg_crop=False, eigen_crop=True, dataset='nyu', min_depth_eval=0.1, max_depth_eval=10, **kwargs)

   Compute metrics of predicted depth maps. Applies cropping and masking as necessary or specified via arguments. Refer to compute_errors for more details on metrics.



.. py:function:: parallelize(config, model, find_unused_parameters=True)


.. py:class:: colors


   Colors class:
   Reset all colors with colors.reset
   Two subclasses fg for foreground and bg for background.
   Use as colors.subclass.colorname.
   i.e. colors.fg.red or colors.bg.green
   Also, the generic bold, disable, underline, reverse, strikethrough,
   and invisible work with the main class
   i.e. colors.bold

   .. py:class:: fg


      .. py:attribute:: black
         :value: '\x1b[30m'

         

      .. py:attribute:: red
         :value: '\x1b[31m'

         

      .. py:attribute:: green
         :value: '\x1b[32m'

         

      .. py:attribute:: orange
         :value: '\x1b[33m'

         

      .. py:attribute:: blue
         :value: '\x1b[34m'

         

      .. py:attribute:: purple
         :value: '\x1b[35m'

         

      .. py:attribute:: cyan
         :value: '\x1b[36m'

         

      .. py:attribute:: lightgrey
         :value: '\x1b[37m'

         

      .. py:attribute:: darkgrey
         :value: '\x1b[90m'

         

      .. py:attribute:: lightred
         :value: '\x1b[91m'

         

      .. py:attribute:: lightgreen
         :value: '\x1b[92m'

         

      .. py:attribute:: yellow
         :value: '\x1b[93m'

         

      .. py:attribute:: lightblue
         :value: '\x1b[94m'

         

      .. py:attribute:: pink
         :value: '\x1b[95m'

         

      .. py:attribute:: lightcyan
         :value: '\x1b[96m'

         


   .. py:class:: bg


      .. py:attribute:: black
         :value: '\x1b[40m'

         

      .. py:attribute:: red
         :value: '\x1b[41m'

         

      .. py:attribute:: green
         :value: '\x1b[42m'

         

      .. py:attribute:: orange
         :value: '\x1b[43m'

         

      .. py:attribute:: blue
         :value: '\x1b[44m'

         

      .. py:attribute:: purple
         :value: '\x1b[45m'

         

      .. py:attribute:: cyan
         :value: '\x1b[46m'

         

      .. py:attribute:: lightgrey
         :value: '\x1b[47m'

         


   .. py:attribute:: reset
      :value: '\x1b[0m'

      

   .. py:attribute:: bold
      :value: '\x1b[01m'

      

   .. py:attribute:: disable
      :value: '\x1b[02m'

      

   .. py:attribute:: underline
      :value: '\x1b[04m'

      

   .. py:attribute:: reverse
      :value: '\x1b[07m'

      

   .. py:attribute:: strikethrough
      :value: '\x1b[09m'

      

   .. py:attribute:: invisible
      :value: '\x1b[08m'

      


.. py:function:: printc(text, color)


.. py:function:: get_image_from_url(url)


.. py:function:: url_to_torch(url, size=(384, 384))


.. py:function:: pil_to_batched_tensor(img)


.. py:function:: save_raw_16bit(depth, fpath='raw.png')


