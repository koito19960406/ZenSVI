# Changelog

<!--next-version-placeholder-->

## v1.2.0 (01/03/2025)
- **GSV Metadata Update Capability**: Added new `update_metadata` method for GSVDownloader to update year/month metadata for existing panorama IDs without redoing the entire download process.

## v1.1.0 (28/02/2025)
- **KartaView API Fix**: Fixed KartaView download API and added retry decorator for improved reliability.
- **Object Detection Enhancement**: Added new object detection functionality.
- **Improved Segmentation**: Enhanced pixel ratio calculation in segmentation with improved normalization.
- **Path Handling Improvement**: Using Path() for path_pid existence check for better cross-platform compatibility.
- **Dependency Management**: Updated poetry.lock with new dependencies and version upgrades.
- **Removed Deprecated Modules**: Cleaned up codebase by removing deprecated modules.
- **Added Example Notebooks**: Included Jupyter notebooks demonstrating download, metadata, transform, and visualization capabilities.
- **Verbosity Control**: Added enhanced verbosity control for better user experience.
- **Documentation Update**: Updated README.md with improved documentation.

## v1.0.0 (10/12/2024)
- **License Changed**: Changed license from CC BY-SA 4.0 to MIT License
- **AMSDownloader Introduced**: New downloader for Amsterdam Street View Images (SVI). Usage: `from zensvi.download import AMSDownloader`.
- **PointCloudProcessor Added**: New module to convert Street View Images to point clouds. Usage: `from zensvi.transform import PointCloudProcessor`.
- **Embeddings Module Added**: New functionality for generating image embeddings. Usage: `from zensvi.cv import Embeddings`.
- **Enhanced Documentation**: Improved documentation for embeddings functionality.
- **Enhanced Metadata Return**: Improved metadata return functionality in MLYDownloader.
- **Removed Geopolars Dependency**: Streamlined dependencies by removing geopolars requirement.

## v0.12.0 (15/05/2024)
- **ClassifierPerception Introduced**: New classifier for perception-based image analysis. Usage: `from zensvi.cv import ClassifierPerception`.

## v0.11.0 (27/04/2024)
- **Metadata Module Added**: New module for metadata processing. Usage: `from zensvi import metadata`.
- **ClassifierGlare, ClassifierLighting, ClassifierPanorama, ClassifierPlatform, ClassifierQuality, ClassifierReflection, ClassifierViewDirection, ClassifierWeather Introduced**: New classifiers for various image quality attributes. Usage: `from zensvi.cv import ClassifierGlare, ClassifierLighting, ClassifierPanorama, ClassifierPlatform, ClassifierQuality, ClassifierReflection, ClassifierViewDirection, ClassifierWeather`.
- **DepthEstimator Introduced**: New module for depth estimation. Usage: `from zensvi.cv import DepthEstimator`.

## v0.10.0 (01/04/2024)
- **Visualization Module Added**: Introducing a new module for plotting maps and images. Usage: `from zensvi import visualization`.
- **Low Level Features in CV Module**: New function to quantify low-level features. Usage: `from zensvi.cv import get_low_level_features`.
- **ClassifierPlaces365 Introduced**: New classifier based on Places365 model. Usage: `from zensvi.cv import ClassifierPlaces365`.

## v0.9.0 (28/01/2024)
- **Improved Download Efficiency**: MLYDownloader now uses polygons for improved download efficiency. GSVDownloader improved data cleaning to eliminate overlaps and dark spots.
- **Metadata-Only Downloads**: Added `metadata_only` argument to download only the metadata of SVIs.

## v0.8.0 (25/07/2023)
- **Flexible Image Transformation**: Introduced `theta` argument in `ImageTransformer` to modify view angles without affecting the FOV.

## v0.7.0 (14/07/2023)
- **Sub-Folder Creation in Downloads**: Added `batch_size` argument to organize downloaded SVIs into sub-folders, addressing performance issues with large image counts.

## v0.6.0 (11/06/2023)
- **Panoptic Segmentation Task**: Enabled panoptic segmentation along with semantic segmentation.
- **Flexible Output Formats**: Introduced flexible csv formats and multiple output options including pixel ratios.

## v0.5.0 (05/06/2023)
- **Enhanced Download SVI Arguments**: New arguments for cropping, saving full images or tiles, and OSMnx enhancements.

## v0.4.0 (01/06/2023)
- **Enhanced Image Transformation Options**: Added FOV and aspects arguments, and introduced new fisheye transformation methods.

## v0.3.0 (30/05/2023)
- **New Downloaders for Street Views**: Deprecated older downloader in favor of new GSVDownloader and MLYDownloader.

## v0.2.0 (26/05/2023)
- **New Argument in download_gsv Function**: `input_place_name` for specifying the boundary for downloading GSV.

## v0.1.0 (28/04/2023)
- **First release of `zensvi`!**
- **Improved Spatial Processing**: Enhanced speed and introduced ID columns for better tracking of input and panorama IDs.
