"""
CUDA operations module for Genesis framework.
"""

# Import basic_ops instead of arithmetic for dispatcher registration
from . import basic_ops
from . import reduction_ops
from . import shape_ops
from . import matrix_ops
from . import comparison_ops
from . import tensor_ops
from . import indexing_ops
from . import random_ops
