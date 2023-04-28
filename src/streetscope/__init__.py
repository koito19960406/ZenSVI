# read version from installed package
from importlib.metadata import version
from .streetscope import *

__version__ = version("streetscope")