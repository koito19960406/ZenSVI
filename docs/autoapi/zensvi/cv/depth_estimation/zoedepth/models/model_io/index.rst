:py:mod:`zensvi.cv.depth_estimation.zoedepth.models.model_io`
=============================================================

.. py:module:: zensvi.cv.depth_estimation.zoedepth.models.model_io


Module Contents
---------------


Functions
~~~~~~~~~

.. autoapisummary::

   zensvi.cv.depth_estimation.zoedepth.models.model_io.load_state_dict
   zensvi.cv.depth_estimation.zoedepth.models.model_io.load_wts
   zensvi.cv.depth_estimation.zoedepth.models.model_io.load_state_dict_from_url
   zensvi.cv.depth_estimation.zoedepth.models.model_io.load_state_from_resource



.. py:function:: load_state_dict(model, state_dict)

   Load state_dict into model, handling DataParallel and DistributedDataParallel. Also checks for "model" key in state_dict.

   DataParallel prefixes state_dict keys with 'module.' when saving.
   If the model is not a DataParallel model but the state_dict is, then prefixes are removed.
   If the model is a DataParallel model but the state_dict is not, then prefixes are added.


.. py:function:: load_wts(model, checkpoint_path)


.. py:function:: load_state_dict_from_url(model, url, **kwargs)


.. py:function:: load_state_from_resource(model, resource: str)

   Loads weights to the model from a given resource. A resource can be of following types:
       1. URL. Prefixed with "url::"
               e.g. url::http(s)://url.resource.com/ckpt.pt

       2. Local path. Prefixed with "local::"
               e.g. local::/path/to/ckpt.pt


   :param model: Model
   :type model: torch.nn.Module
   :param resource: resource string
   :type resource: str

   :returns: Model with loaded weights
   :rtype: torch.nn.Module


