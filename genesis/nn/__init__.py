"""Neural network modules and functions for Genesis.

This package provides neural network layers, activation functions,
and utilities for building deep learning models with automatic differentiation support.
"""

from . import functional
from .modules import *
from . import utils
from . import parallel  # For torch.nn.parallel API compatibility
