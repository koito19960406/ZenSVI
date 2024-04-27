:py:mod:`zensvi.cv.depth_estimation.zoedepth.utils.config`
==========================================================

.. py:module:: zensvi.cv.depth_estimation.zoedepth.utils.config


Module Contents
---------------


Functions
~~~~~~~~~

.. autoapisummary::

   zensvi.cv.depth_estimation.zoedepth.utils.config.flatten
   zensvi.cv.depth_estimation.zoedepth.utils.config.split_combined_args
   zensvi.cv.depth_estimation.zoedepth.utils.config.parse_list
   zensvi.cv.depth_estimation.zoedepth.utils.config.get_model_config
   zensvi.cv.depth_estimation.zoedepth.utils.config.update_model_config
   zensvi.cv.depth_estimation.zoedepth.utils.config.check_choices
   zensvi.cv.depth_estimation.zoedepth.utils.config.get_config
   zensvi.cv.depth_estimation.zoedepth.utils.config.change_dataset



Attributes
~~~~~~~~~~

.. autoapisummary::

   zensvi.cv.depth_estimation.zoedepth.utils.config.ROOT
   zensvi.cv.depth_estimation.zoedepth.utils.config.HOME_DIR
   zensvi.cv.depth_estimation.zoedepth.utils.config.COMMON_CONFIG
   zensvi.cv.depth_estimation.zoedepth.utils.config.DATASETS_CONFIG
   zensvi.cv.depth_estimation.zoedepth.utils.config.ALL_INDOOR
   zensvi.cv.depth_estimation.zoedepth.utils.config.ALL_OUTDOOR
   zensvi.cv.depth_estimation.zoedepth.utils.config.ALL_EVAL_DATASETS
   zensvi.cv.depth_estimation.zoedepth.utils.config.COMMON_TRAINING_CONFIG
   zensvi.cv.depth_estimation.zoedepth.utils.config.KEYS_TYPE_BOOL


.. py:data:: ROOT

   

.. py:data:: HOME_DIR

   

.. py:data:: COMMON_CONFIG

   

.. py:data:: DATASETS_CONFIG

   

.. py:data:: ALL_INDOOR
   :value: ['nyu', 'ibims', 'sunrgbd', 'diode_indoor', 'hypersim_test']

   

.. py:data:: ALL_OUTDOOR
   :value: ['kitti', 'diml_outdoor', 'diode_outdoor', 'vkitti2', 'ddad']

   

.. py:data:: ALL_EVAL_DATASETS

   

.. py:data:: COMMON_TRAINING_CONFIG

   

.. py:function:: flatten(config, except_keys='bin_conf')


.. py:function:: split_combined_args(kwargs)

   Splits the arguments that are combined with '__' into multiple arguments.
      Combined arguments should have equal number of keys and values.
      Keys are separated by '__' and Values are separated with ';'.
      For example, '__n_bins__lr=256;0.001'

   :param kwargs: key-value pairs of arguments where key-value is optionally combined according to the above format.
   :type kwargs: dict

   :returns: Parsed dict with the combined arguments split into individual key-value pairs.
   :rtype: dict


.. py:function:: parse_list(config, key, dtype=int)

   Parse a list of values for the key if the value is a string. The values are separated by a comma.
   Modifies the config in place.


.. py:function:: get_model_config(model_name, model_version=None)

   Find and parse the .json config file for the model.

   :param model_name: name of the model. The config file should be named config_{model_name}[_{model_version}].json under the models/{model_name} directory.
   :type model_name: str
   :param model_version: Specific config version. If specified config_{model_name}_{model_version}.json is searched for and used. Otherwise config_{model_name}.json is used. Defaults to None.
   :type model_version: str, optional

   :returns: the config dictionary for the model.
   :rtype: easydict


.. py:function:: update_model_config(config, mode, model_name, model_version=None, strict=False)


.. py:function:: check_choices(name, value, choices)


.. py:data:: KEYS_TYPE_BOOL
   :value: ['use_amp', 'distributed', 'use_shared_dict', 'same_lr', 'aug', 'three_phase', 'prefetch',...

   

.. py:function:: get_config(model_name, mode='train', dataset=None, **overwrite_kwargs)

   Main entry point to get the config for the model.

   :param model_name: name of the desired model.
   :type model_name: str
   :param mode: "train" or "infer". Defaults to 'train'.
   :type mode: str, optional
   :param dataset: If specified, the corresponding dataset configuration is loaded as well. Defaults to None.
   :type dataset: str, optional

   Keyword Args: key-value pairs of arguments to overwrite the default config.

   The order of precedence for overwriting the config is (Higher precedence first):
       # 1. overwrite_kwargs
       # 2. "config_version": Config file version if specified in overwrite_kwargs. The corresponding config loaded is config_{model_name}_{config_version}.json
       # 3. "version_name": Default Model version specific config specified in overwrite_kwargs. The corresponding config loaded is config_{model_name}_{version_name}.json
       # 4. common_config: Default config for all models specified in COMMON_CONFIG

   :returns: The config dictionary for the model.
   :rtype: easydict


.. py:function:: change_dataset(config, new_dataset)


