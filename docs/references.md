# References

ZenSVI is a collection of tools developed by many researchers. The following references are provided to give credit to the original authors of the tools and datasets used in ZenSVI. Please cite them when using ZenSVI in your research.

## Main Classes and Functions

### Download Module
- `GSDownloader`: Downloader for Global Streetscape dataset {cite}`hou_global_2024`
- `MLYDownloader`: Downloader for Mapillary dataset {cite}`_mapillary_2024`
- `KVDownloader`: Downloader for Karta View dataset {cite}`_kartaview_`
- `AMSDownloader`: Downloader for Amsterdam Street View Images (SVI) {cite}`_amsterdam_2024`

### Metadata Module
- `MLYMetadata`: Class for processing Mapillary metadata {cite}`_mapillary_2024`

### Computer Vision (CV) Module
- `Segmenter`: Class for semantic/panoptic segmentation {cite}`cheng_maskedattention_2022`
- `ClassifierPlaces365`: Classifier based on the Places365 model {cite}`zhou2017places`
- `ClassifierGlare`: Classifier for detecting glare in images {cite}`hou_global_2024`
- `ClassifierLighting`: Classifier for determining lighting conditions {cite}`hou_global_2024`
- `ClassifierPanorama`: Classifier for identifying panoramic images {cite}`hou_global_2024`
- `ClassifierPlatform`: Classifier for determining the capture platform {cite}`hou_global_2024`
- `ClassifierQuality`: Classifier for assessing image quality {cite}`hou_global_2024`
- `ClassifierReflection`: Classifier for detecting reflections in images {cite}`hou_global_2024`
- `ClassifierViewDirection`: Classifier for determining view direction {cite}`hou_global_2024`
- `ClassifierWeather`: Classifier for identifying weather conditions {cite}`hou_global_2024`
- `ClassifierPerception`: Classifier for perception-based image analysis {cite}`hou_global_2024` {cite}`liang_evaluating_2024`
- `DepthEstimator`: Class for depth estimation in images {cite}`Ranftl2021` {cite}`Ranftl2020` {cite}`depthanything`
- `Embeddings`: Class for generating image embeddings {cite}`safka_christiansafka_2024`

## Bibliography

```{bibliography}