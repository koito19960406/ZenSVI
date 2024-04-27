# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------

project = "zensvi"
copyright = "2023, koito19960406"
author = "koito19960406"

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
nbsphinx_execute = 'never'

extensions = [
    "myst_nb",
    "autoapi.extension",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
]
autoapi_dirs = ["../src"]

autoapi_options = [
    "members",
    "undoc-members",
    "show-inheritance",
    "imported-members",
    "special-members",
    # "private-members",
    "inherited-members",
    "show-module-summary",
]

autoapi_own_page_level = "class"
# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_rtd_theme"
html_logo = "logo.png"
html_theme_options = {
    "logo_only": True,
    "display_version": False,
}


def skip_util_classes(app, what, name, obj, skip, options):
    if what == "package" and "utils" in name:
        skip = True
    if what == "package" and "mapillary" in name:
        skip = True
    if what == "package" and "classification" in name:
        skip = True
    if what == "package" and "segmentation" in name:
        skip = True
    if what == "package" and "low_level" in name:
        skip = True
    if what == "package" and "zoedepth" in name:
        skip = True
    if what == "package" and "depth_anything" in name:
        skip = True
    if what == "package" and "torchhub" in name:
        skip = True
    if what == "class" and "ImageDataset" in name:
        skip = True
    if what == "class" and "GSVDownloader" in name:
        skip = True
    if what == "module" and "base" in name:
        skip = True
    if what == "module" and "font_property" in name:
        skip = True
    if what == "module" and "gsv" in name:
        skip = True
    if what == "attribute" and "__slots__" in name:
        skip = True
    return skip


def setup(sphinx):
    sphinx.connect("autoapi-skip-member", skip_util_classes)
