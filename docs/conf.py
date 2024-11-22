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
nbsphinx_execute = "never"

extensions = [
    "myst_nb",
    "autoapi.extension",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinxcontrib.bibtex",
]

# Add these lines to configure myst-nb
nb_execution_mode = "off"
# myst_enable_extensions = [
#     "amsmath",
#     "colon_fence",
#     "deflist",
#     "dollarmath",
#     "html_admonition",
#     "html_image",
#     "linkify",
#     "replacements",
#     "smartquotes",
#     "substitution",
#     "tasklist",
# ]

# Bibtex settings
bibtex_bibfiles = ["references.bib"]
bibtex_default_style = "plain"
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
html_theme = "furo"

# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.
html_theme_options = {
    "sidebar_hide_name": False,
    "navigation_with_keys": True,
    # "announcement": "This is a beta version of the documentation.",
    "light_logo": "logo_zensvi_fixed 2.png",
    "dark_logo": "logo_zensvi_white2.png",
}

html_title = "ZenSVI Documentation"
# Remove or comment out the following line:
# html_logo = "logo.png"
html_favicon = "_static/favicon.ico"  # Make sure you have a favicon file

# Add these lines near the other html_* configurations
html_static_path = ["_static"]
html_css_files = ["custom.css"]

def skip_util_classes(app, what, name, obj, skip, options):
    skip_packages = [
        "mapillary",
        "kartaview",
        "classification",
        "segmentation",
        "low_level",
        "depth_estimation",
        "embeddings",
    ]
    skip_modules = [
        "base",
        "font_property",
        "gsv",
        "ams",
        "mly",
        "kv",
        "depth_estimation",
        "embeddings",
        "mly_metadata",
        "image_to_pointcloud",
        "transform_image",
        "hist",
        "image",
        "kde",
        "map",
    ]
    skip_classes = ["ImageDataset", "GSVDownloader", "Logger"]
    skip_keywords = ["utils", "torchhub", "zoedepth", "depth_anything", "dinov2"]

    if what == "package" and any(pkg in name for pkg in skip_packages):
        return True
    if what == "module" and any(mod in name for mod in skip_modules):
        return True
    if what == "class" and any(cls in name for cls in skip_classes):
        return True
    if any(keyword in name for keyword in skip_keywords):
        return True
    if what == "attribute":
        return True

    return skip


def setup(app):
    app.connect("autoapi-skip-member", skip_util_classes)
    app.add_config_value(
        "recommonmark_config",
        {
            "auto_toc_tree_section": "Contents",
            "enable_eval_rst": True,
        },
        True,
    )
    app.add_transform(AutoStructify)


# Add this new import at the top of the file
from recommonmark.transform import AutoStructify
