[![PyPi version](https://img.shields.io/pypi/v/zensvi.svg)](https://pypi.org/project/zensvi/)
[![Python versions](https://img.shields.io/pypi/pyversions/zensvi.svg)](https://pypi.org/project/zensvi/)
[![License](https://img.shields.io/pypi/l/zensvi.svg)](https://pypi.org/project/zensvi/)
[![Downloads](https://pepy.tech/badge/zensvi)](https://pepy.tech/project/zensvi)
[![Downloads](https://pepy.tech/badge/zensvi/month)](https://pepy.tech/project/zensvi)
[![Downloads](https://pepy.tech/badge/zensvi/week)](https://pepy.tech/project/zensvi)
[![Documentation Status](https://readthedocs.org/projects/zensvi/badge/?version=latest)](https://zensvi.readthedocs.io/en/latest/?badge=latest)

# ZenSVI

This package is a one-stop solution for downloading, cleaning, analyzing street view imagery. Detailed documentation can be found [here](https://zensvi.readthedocs.io/en/latest/).

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
# initialize the classifier
classifier = ClassifierPlaces365(
    device="cpu",  # device to use (either "cpu" or "gpu")
)

# set arguments
classifier = ClassifierPlaces365()
classifier.classify(
    "path/to/input_directory",
    dir_image_output="path/to/image_output_directory",
    dir_summary_output="path/to/classification_summary_output"
)
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
)
```

### Visualizing Results
To visualize the results, use the `plot_map` and `plot_image` functions:

```python
from zensvi.visualization import plot_map, plot_image

# Plotting a map
plot_map(
    "path/to/pid_file.csv",  # path to the file containing latitudes and longitudes
    variable_name="vegetation", 
    plot_type="point"  # this can be either "point", "line", or "hexagon"
)

# Plotting images in a grid
plot_image(
    "path/to/image_directory", 
    4,  # number of rows
    5  # number of columns
)
```

## Contributing

Interested in contributing? Check out the contributing guidelines. Please note that this project is released with a Code of Conduct. By contributing to this project, you agree to abide by its terms.

## License

`zensvi` was created by Koichi Ito. It is licensed under the terms of the CC BY-SA 4.0.

Please cite the following paper if you use `zensvi` in a scientific publication:
***(place holder for the paper citation)***
```bibtex
@article{ito2024zensvi,
  title={ZenSVI: One-Stop Python Package for Integrated Analysis of Street View Imagery},
  author={Ito, Koichi, XXX, XXX, XXX, ...},
  journal={XXX},
  volume={XXX},
  pages={XXX},
  year={2024}
}
```

## Credits

`zensvi` was created with [`cookiecutter`](https://cookiecutter.readthedocs.io/en/latest/) and the `py-pkgs-cookiecutter` [template](https://github.com/py-pkgs/py-pkgs-cookiecutter).
