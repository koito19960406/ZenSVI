:orphan:

:py:mod:`zensvi.cv.depth_estimation.zoedepth.models.zoedepth_nk`
================================================================

.. py:module:: zensvi.cv.depth_estimation.zoedepth.models.zoedepth_nk


Submodules
----------
.. toctree::
   :titlesonly:
   :maxdepth: 1

   zoedepth_nk_v1/index.rst


Package Contents
----------------

Classes
~~~~~~~

.. autoapisummary::

   zensvi.cv.depth_estimation.zoedepth.models.zoedepth_nk.ZoeDepthNK




Attributes
~~~~~~~~~~

.. autoapisummary::

   zensvi.cv.depth_estimation.zoedepth.models.zoedepth_nk.all_versions
   zensvi.cv.depth_estimation.zoedepth.models.zoedepth_nk.get_version


.. py:class:: ZoeDepthNK(core, bin_conf, bin_centers_type='softplus', bin_embedding_dim=128, n_attractors=[16, 8, 4, 1], attractor_alpha=300, attractor_gamma=2, attractor_kind='sum', attractor_type='exp', min_temp=5, max_temp=50, memory_efficient=False, train_midas=True, is_midas_pretrained=True, midas_lr_factor=1, encoder_lr_factor=10, pos_enc_lr_factor=10, inverse_midas=False, **kwargs)


   Bases: :py:obj:`zoedepth.models.depth_model.DepthModel`

   .. py:method:: forward(x, return_final_centers=False, denorm=False, return_probs=False, **kwargs)

      :param x: Input image tensor of shape (B, C, H, W). Assumes all images are from the same domain.
      :type x: torch.Tensor
      :param return_final_centers: Whether to return the final centers of the attractors. Defaults to False.
      :type return_final_centers: bool, optional
      :param denorm: Whether to denormalize the input image. Defaults to False.
      :type denorm: bool, optional
      :param return_probs: Whether to return the probabilities of the bins. Defaults to False.
      :type return_probs: bool, optional

      :returns:

                Dictionary of outputs with keys:
                    - "rel_depth": Relative depth map of shape (B, 1, H, W)
                    - "metric_depth": Metric depth map of shape (B, 1, H, W)
                    - "domain_logits": Domain logits of shape (B, 2)
                    - "bin_centers": Bin centers of shape (B, N, H, W). Present only if return_final_centers is True
                    - "probs": Bin probabilities of shape (B, N, H, W). Present only if return_probs is True
      :rtype: dict


   .. py:method:: get_lr_params(lr)

      Learning rate configuration for different layers of the model

      :param lr: Base learning rate
      :type lr: float

      :returns: list of parameters to optimize and their learning rates, in the format required by torch optimizers.
      :rtype: list


   .. py:method:: get_conf_parameters(conf_name)

      Returns parameters of all the ModuleDicts children that are exclusively used for the given bin configuration


   .. py:method:: freeze_conf(conf_name)

      Freezes all the parameters of all the ModuleDicts children that are exclusively used for the given bin configuration


   .. py:method:: unfreeze_conf(conf_name)

      Unfreezes all the parameters of all the ModuleDicts children that are exclusively used for the given bin configuration


   .. py:method:: freeze_all_confs()

      Freezes all the parameters of all the ModuleDicts children


   .. py:method:: build(midas_model_type='DPT_BEiT_L_384', pretrained_resource=None, use_pretrained_midas=False, train_midas=False, freeze_midas_bn=True, **kwargs)
      :staticmethod:


   .. py:method:: build_from_config(config)
      :staticmethod:



.. py:data:: all_versions

   

.. py:data:: get_version

   

