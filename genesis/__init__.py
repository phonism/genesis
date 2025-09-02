"""Genesis Deep Learning Framework.

A PyTorch-compatible deep learning framework with CUDA acceleration,
automatic differentiation, and optimized tensor operations.
"""

__version__ = "0.1.0"

from .dtypes import (
    float32, float16, float64, bfloat16,
    int32, int64, int16, int8, uint8, bool,
    get_dtype, is_floating_point, is_integer
)

# Global runtime configuration
enable_autocast = False  # Automatic mixed precision training
upgrade = False         # Framework upgrade mode  
use_triton = True       # Enable Triton GPU kernels

from . import utils
from .init import (
        rand,
        randn,
        ones,
        zeros,
        full,
        empty,
        empty_like,
        zeros_like,
        ones_like,
        one_hot,
        eye,
        from_numpy,
)
from .serialization import (
        save, load, 
        save_checkpoint, load_checkpoint
)
from .autograd import Tensor
tensor = Tensor  # PyTorch-style lowercase alias
from . import nn
from . import init
from . import optim
from . import utils
from . import amp  # Automatic mixed precision training
from . import distributed  # Distributed training
from .backend import *
from .functional import *
from . import cuda  # CUDA memory management utilities

# Random number generation - PyTorch-compatible RNG API
from .random import (
    seed, manual_seed, initial_seed,
    get_rng_state, set_rng_state, 
    default_generator, fork_rng, Generator
)

# Device utilities
def cuda_available() -> bool:
    """Check if CUDA is available for use.
    
    Returns:
        bool: True if CUDA device is available and enabled, False otherwise.
    """
    return device('cuda').enabled()
