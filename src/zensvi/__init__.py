# read version from installed package
from importlib.metadata import version
from .zensvi import *

__version__ = version("zensvi")