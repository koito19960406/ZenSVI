[![PyPi version](https://img.shields.io/pypi/v/zensvi.svg)](https://pypi.org/project/zensvi/)
[![Python versions](https://img.shields.io/pypi/pyversions/zensvi.svg)](https://pypi.org/project/zensvi/)
[![License](https://img.shields.io/pypi/l/zensvi.svg)](https://pypi.org/project/zensvi/)
[![Downloads](https://pepy.tech/badge/zensvi)](https://pepy.tech/project/zensvi)
[![Downloads](https://pepy.tech/badge/zensvi/month)](https://pepy.tech/project/zensvi)
[![Downloads](https://pepy.tech/badge/zensvi/week)](https://pepy.tech/project/zensvi)
[![Documentation Status](https://readthedocs.org/projects/zensvi/badge/?version=latest)](https://zensvi.readthedocs.io/en/latest/?badge=latest)
[![codecov](https://codecov.io/gh/koito19960406/ZenSVI/graph/badge.svg?token=HAIMJIT9HQ)](https://codecov.io/gh/koito19960406/ZenSVI)

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/koito19960406/ZenSVI/main/docs/_static/logo_zensvi_white2.png">
    <img src="https://raw.githubusercontent.com/koito19960406/ZenSVI/main/docs/_static/logo_zensvi_fixed%202.png" alt="ZenSVI logo" width="400">
  </picture>
</p>

# ZenSVI

**Primary Author:** [Koichi Ito](https://koichiito.com/) (National University of Singapore)

Besides this documentation, we have published a [comprehensive paper](https://arxiv.org/abs/2412.18641) with detailed information and demonstration use cases. The paper provides in-depth insights into the package's architecture, features, and real-world applications.

ZenSVI is a comprehensive Python package for downloading, cleaning, and analyzing street view imagery. For more information about the package or to discuss potential collaborations, please visit my website at [koichiito.com](https://koichiito.com/). The source code is available on [GitHub](https://github.com/koito19960406/ZenSVI).

This package is a one-stop solution for downloading, cleaning, and analyzing street view imagery, with comprehensive API documentation available at [zensvi.readthedocs.io](https://zensvi.readthedocs.io/en/latest/autoapi/index.html).

## Table of Contents

- [ZenSVI](#zensvi)
  - [Table of Contents](#table-of-contents)
  - [Installation of `zensvi`](#installation-of-zensvi)
  - [Installation of `pytorch` and `torchvision`](#installation-of-pytorch-and-torchvision)
  - [Usage](#usage)
    - [Downloading Street View Imagery](#downloading-street-view-imagery)
    - [Analyzing Metadata of Mapillary Images](#analyzing-metadata-of-mapillary-images)
    - [Running Segmentation](#running-segmentation)
    - [Running Places365](#running-places365)
    - [Running PlacePulse 2.0 Prediction](#running-placepulse-20-prediction)
    - [Running Global Streetscapes Prediction](#running-global-streetscapes-prediction)
    - [Running Grounding Object Detection](#running-grounding-object-detection)
    - [Running Depth Estimation](#running-depth-estimation)
    - [Running Embeddings](#running-embeddings)
    - [Running Low-Level Feature Extraction](#running-low-level-feature-extraction)
    - [Transforming Images](#transforming-images)
    - [Creating Point Clouds from Images](#creating-point-clouds-from-images)
    - [Visualizing Results](#visualizing-results)
  - [Contributing](#contributing)
  - [License](#license)
  - [Credits](#credits)

## Installation of `zensvi`

```bash
$ pip install zensvi
```

## Installation of `pytorch` and `torchvision`

Since `zensvi` uses `pytorch` and `torchvision`, you may need to install them separately. Please refer to the [official website](https://pytorch.org/get-started/locally/) for installation instructions.

## Usage
### Downloading Street View Imagery
***Mapillary***

For downloading images from Mapillary, utilize the MLYDownloader. Ensure you have a Mapillary client ID:

```python
from zensvi.download import MLYDownloader

mly_api_key = "YOUR_OWN_MLY_API_KEY"  # Please register your own Mapillary API key
downloader = MLYDownloader(mly_api_key=mly_api_key)
# with lat and lon:
downloader.download_svi("path/to/output_directory", lat=1.290270, lon=103.851959)
# with a csv file with lat and lon:
downloader.download_svi("path/to/output_directory", input_csv_file="path/to/csv_file.csv")
# with a shapefile:
downloader.download_svi("path/to/output_directory", input_shp_file="path/to/shapefile.shp")
# with a place name that works on OpenStreetMap:
downloader.download_svi("path/to/output_directory", input_place_name="Singapore")
```

***KartaView***

For downloading images from KartaView, utilize the KVDownloader:

```python
from zensvi.download import KVDownloader

downloader = KVDownloader()
# with lat and lon:
downloader.download_svi("path/to/output_directory", lat=1.290270, lon=103.851959)
# with a csv file with lat and lon:
downloader.download_svi("path/to/output_directory", input_csv_file="path/to/csv_file.csv")
# with a shapefile:
downloader.download_svi("path/to/output_directory", input_shp_file="path/to/shapefile.shp")
# with a place name that works on OpenStreetMap:
downloader.download_svi("path/to/output_directory", input_place_name="Singapore")
```

***Amsterdam***

For downloading images from Amsterdam, utilize the AMSDownloader:

```python
from zensvi.download import AMSDownloader

downloader = AMSDownloader()
# with lat and lon:
downloader.download_svi("path/to/output_directory", lat=4.899431, lon=52.379189)
# with a csv file with lat and lon:
downloader.download_svi("path/to/output_directory", input_csv_file="path/to/csv_file.csv")
# with a shapefile:
downloader.download_svi("path/to/output_directory", input_shp_file="path/to/shapefile.shp")
# with a place name that works on OpenStreetMap:
downloader.download_svi("path/to/output_directory", input_place_name="Amsterdam")
```

***Global Streetscapes***

For downloading the NUS Global Streetscapes dataset, utilize the GSDownloader:

```python
from zensvi.download import GSDownloader

downloader = GSDownloader()
# Download all data
downloader.download_all_data(local_dir="data/")
# Or download specific subsets
downloader.download_manual_labels(local_dir="manual_labels/")
downloader.download_train(local_dir="manual_labels/train/")
downloader.download_test(local_dir="manual_labels/test/")
downloader.download_img_tar(local_dir="manual_labels/img/")
```

### Analyzing Metadata of Mapillary Images
To analyze the metadata of Mapillary images, use the `MLYMetadata`:

```python
from zensvi.metadata import MLYMetadata

path_input = "path/to/input"
mly_metadata = MLYMetadata(path_input)
mly_metadata.compute_metadata(
    unit="image", # unit of the metadata. Other options are "street" and "grid"
    indicator_list="all", # list of indicators to compute. You can specify a list of indicators in space-separated format, e.g., "year month day" or "all" to compute all indicators
    path_output="path/to/output" # path to the output file
)
```

### Running Segmentation
To perform image segmentation, use the `Segmenter`:

```python
from zensvi.cv import Segmenter

segmenter = Segmenter(dataset="cityscapes", # or "mapillary"
                      task="semantic" # or "panoptic"
                      )
segmenter.segment("path/to/input_directory", 
                  dir_image_output = "path/to/image_output_directory",
                  dir_summary_output = "path/to/segmentation_summary_output"
                  )
```

### Running Places365
To perform scene classification, use the `ClassifierPlaces365`:

```python
from zensvi.cv import ClassifierPlaces365

# initialize the classifier
classifier = ClassifierPlaces365(
    device="cpu",  # device to use (either "cpu", "cuda", or "mps)
)

# set arguments
classifier = ClassifierPlaces365()
classifier.classify(
    "path/to/input_directory",
    dir_image_output="path/to/image_output_directory",
    dir_summary_output="path/to/classification_summary_output"
)
```

### Running PlacePulse 2.0 Prediction
To predict the PlacePulse 2.0 score, use the `ClassifierPerception`:

```python
from zensvi.cv import ClassifierPerception

classifier = ClassifierPerception(
    perception_study="safer", # Other options are "livelier", "wealthier", "more beautiful", "more boring", "more depressing"
)
dir_input = "path/to/input"
dir_summary_output = "path/to/summary_output"
classifier.classify(
    dir_input,
    dir_summary_output=dir_summary_output
)
```

You can also use the ViT version for perception classification:

```python
from zensvi.cv import ClassifierPerceptionViT

classifier = ClassifierPerceptionViT(
    perception_study="safer", # Other options are "livelier", "wealthier", "more beautiful", "more boring", "more depressing"
)
dir_input = "path/to/input"
dir_summary_output = "path/to/summary_output"
classifier.classify(
    dir_input,
    dir_summary_output=dir_summary_output
)
```

### Running Global Streetscapes Prediction
To predict the Global Streetscapes indicators, use:
- `ClassifierGlare`: Whether the image contains glare
- `ClassifierLighting`: The lighting condition of the image
- `ClassifierPanorama`: Whether the image is a panorama
- `ClassifierPlatform`: Platform of the image
- `ClassifierQuality`: Quality of the image
- `ClassifierReflection`: Whether the image contains reflection
- `ClassifierViewDirection`: View direction of the image
- `ClassifierWeather`: Weather condition of the image

```python
from zensvi.cv import ClassifierGlare

classifier = ClassifierGlare()
dir_input = "path/to/input"
dir_summary_output = "path/to/summary_output"
classifier.classify(
    dir_input,
    dir_summary_output=dir_summary_output,
)
```


### Running Grounding Object Detection
To run grounding object detection on the images, use the `ObjectDetector`:

```python
from zensvi.cv import ObjectDetector

detector = ObjectDetector(
    text_prompt="tree",  # specify the object(s) (e.g., single type: "building", multi-type: "car . tree")
    box_threshold=0.35,  # confidence threshold for box detection
    text_threshold=0.25  # confidence threshold for text
)

detector.detect_objects(
    dir_input="path/to/image_input_directory",
    dir_image_output="path/to/image_output_directory",
    dir_summary_output="path/to/detection_summary_output",
    save_format="json" # or "csv"
)
```

### Running Depth Estimation
To estimate the depth of the images, use the `DepthEstimator`:

```python
from zensvi.cv import DepthEstimator

depth_estimator = DepthEstimator(
    device="cpu",  # device to use (either "cpu", "cuda", or "mps")
    task="relative", # task to perform (either "relative" or "absolute")
    encoder="vitl", # encoder variant ("vits", "vitb", "vitl", "vitg")
    max_depth=80.0 # maximum depth for absolute estimation (only used when task="absolute")
)

dir_input = "path/to/input"
dir_image_output = "path/to/image_output" # estimated depth map
depth_estimator.estimate_depth(
    dir_input,
    dir_image_output
)
```

### Running Embeddings
To generate embeddings and search for similar images, use the `Embeddings`:
```python
from zensvi.cv import Embeddings

emb = Embeddings(model_name="resnet-1", cuda=True)
emb.generate_embedding(
    "path/to/image_directory",
    "path/to/output_directory",
    batch_size=1000,
)
results = emb.search_similar_images("path/to/target_image_file", "path/to/embeddings_directory", 20)
```

### Running Low-Level Feature Extraction
To extract low-level features, use the `get_low_level_features`:

```python
from zensvi.cv import get_low_level_features

get_low_level_features(
    "path/to/input_directory",
    dir_image_output="path/to/image_output_directory",
    dir_summary_output="path/to/low_level_feature_summary_output"
)
```

### Transforming Images
Transform images from panoramic to perspective or fisheye views using the `ImageTransformer`:

```python
from zensvi.transform import ImageTransformer

dir_input = "path/to/input"
dir_output = "path/to/output"
image_transformer = ImageTransformer(
    dir_input="path/to/input", 
    dir_output="path/to/output"
)
image_transformer.transform_images(
    style_list="perspective equidistant_fisheye orthographic_fisheye stereographic_fisheye equisolid_fisheye",  # list of projection styles in the form of a string separated by a space
    FOV=90,  # field of view
    theta=120,  # angle of view (horizontal)
    phi=0,  # angle of view (vertical)
    aspects=(9, 16),  # aspect ratio
    show_size=100,  # size of the image to show (i.e. scale factor)
    use_upper_half=True,  # use the upper half of the image for sky view factor calculation
)
```

### Creating Point Clouds from Images
To create a point cloud from images with depth information, use the `PointCloudProcessor`:

```python
from zensvi.transform import PointCloudProcessor
import pandas as pd

processor = PointCloudProcessor(
    image_folder="path/to/image_directory",
    depth_folder="path/to/depth_maps_directory",
    output_coordinate_scale=45,  # scaling factor for output coordinates
    depth_max=255  # maximum depth value for normalization
)

# Create a DataFrame with image information
# The DataFrame should have columns similar to this structure:
data = pd.DataFrame({
    "id": ["Y2y7An1aRCeA5Y4nW7ITrg", "VSsVjWlr4orKerabFRy-dQ"],  # image identifiers
    "heading": [3.627108491916069, 5.209303414492613],           # heading in radians
    "lat": [40.77363963371641, 40.7757528007],                   # latitude
    "lon": [-73.95482278589579, -73.95668603003708],             # longitude
    "x_proj": [4979010.676803163, 4979321.30902424],             # projected x coordinate
    "y_proj": [-8232613.214232705, -8232820.629621736]           # projected y coordinate
})

# Process images and save point clouds
processor.process_multiple_images(
    data=data,
    output_dir="path/to/output_directory",
    save_format="ply"  # output format, can be "pcd", "ply", "npz", or "csv"
)
```

### Creating Point Clouds from Images with VGGT
ZenSVI also supports generating 3D point clouds directly from a collection of images using the Visual Geometry Grounded Transformer (VGGT) model. VGGT is a powerful feed-forward neural network that can infer 3D geometry, including camera parameters and point clouds, from multiple views of a scene. This feature is particularly useful for reconstructing 3D scenes from unordered image collections.

**Installation for VGGT**

To use the VGGT-based point cloud generation, you need to initialize the `vggt` git submodule and install its specific dependencies.

1.  **Initialize the git submodule:**
    If you have cloned the ZenSVI repository, run the following command from the root directory to download the `vggt` submodule:
    ```bash
    git submodule update --init --recursive
    ```

2.  **Install dependencies:**
    Install the required Python packages for `vggt`:
    ```bash
    pip install -r src/zensvi/transform/vggt/requirements.txt
    ```

**Usage**

Once the setup is complete, you can use the `VGGTProcessor` to generate point clouds.

```python
from zensvi.transform import VGGTProcessor

# Initialize the processor. This will download the model weights if not cached.
# Note: VGGT requires a CUDA-enabled GPU.
vggt_processor = VGGTProcessor()

# Define input and output directories
dir_input = "path/to/your/images"
dir_output = "path/to/save/pointclouds"

# Process images to generate point clouds
# The processor will process images in batches and save the resulting point clouds as .ply files.
vggt_processor.process_images_to_pointcloud(
    dir_input=dir_input,
    dir_output=dir_output,
    batch_size=1,  # Adjust batch size based on your GPU memory
    max_workers=4  # Adjust based on your system's capabilities
)
```

### Visualizing Results
To visualize the results, use the `plot_map`, `plot_image`, `plot_hist`, and `plot_kde` functions:

```python
from zensvi.visualization import plot_map, plot_image, plot_hist, plot_kde

# Plotting a map
plot_map(
    path_pid="path/to/pid_file.csv",  # path to the file containing latitudes and longitudes
    variable_name="vegetation", 
    plot_type="point"  # this can be either "point", "line", or "hexagon"
)

# Plotting images in a grid
plot_image(
    dir_image_input="path/to/image_directory", 
    n_row=4,  # number of rows
    n_col=5   # number of columns
)

# Plotting a histogram
plot_hist(
    dir_input="path/to/data.csv",
    columns=["vegetation"],  # list of column names to plot histograms for
    title="Vegetation Distribution by Neighborhood"
)

# Plotting a kernel density estimate
plot_kde(
    dir_input="path/to/data.csv",
    columns=["vegetation"],  # list of column names to plot KDEs for
    title="Vegetation Density by Neighborhood"
)
```

## Contributing

Interested in contributing? Check out the contributing guidelines. Please note that this project is released with a Code of Conduct. By contributing to this project, you agree to abide by its terms.

## License

`zensvi` was created by Koichi Ito. It is licensed under the terms of the MIT License.

Please cite the following paper if you use `zensvi` in a scientific publication:

```bibtex
@article{2025_ceus_zensvi,
  author = {Ito, Koichi and Zhu, Yihan and Abdelrahman, Mahmoud and Liang, Xiucheng and Fan, Zicheng and Hou, Yujun and Zhao, Tianhong and Ma, Rui and Fujiwara, Kunihiko and Ouyang, Jiani and Quintana, Matias and Biljecki, Filip},
  doi = {10.1016/j.compenvurbsys.2025.102283},
  journal = {Computers, Environment and Urban Systems},
  pages = {102283},
  title = {ZenSVI: An open-source software for the integrated acquisition, processing and analysis of street view imagery towards scalable urban science},
  volume = {119},
  year = {2025}
}
```

## Credits
- Logo design by [Kunihiko Fujiwara](https://ual.sg/author/kunihiko-fujiwara/)
- All the packages used in this package: [requirements.txt](requirements.txt)
--------------------------------------------------------------------------------
<br>
<br>
<p align="center">
  <a href="https://ual.sg/">
    <img src="https://raw.githubusercontent.com/koito19960406/ZenSVI/main/docs/_static/ualsg.jpeg" width = 55% alt="Logo">
  </a>
</p>
