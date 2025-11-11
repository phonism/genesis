"""Genesis Deep Learning Framework.

A high-performance deep learning framework with CUDA acceleration,
automatic differentiation, and optimized tensor operations.
"""

__version__ = "0.1.0"

from .dtypes import (
    float32, float16, float64, bfloat16,
    int32, int64, int16, int8, uint8, bool,
    get_dtype, is_floating_point, is_integer
)

# Global runtime configuration
enable_autocast: bool = False  # Automatic mixed precision training
upgrade: bool = False           # Framework upgrade mode
use_triton: bool = True         # Enable Triton GPU kernels

from .init import (
    rand,
    randn,
    randint,
    ones,
    zeros,
    full,
    empty,
    empty_like,
    zeros_like,
    ones_like,
    one_hot,
    eye,
    arange,
    outer,
    from_numpy,
)
from .serialization import (
    save, load,
    save_checkpoint, load_checkpoint
)
from .tensor import Tensor, tensor
from . import nn
from . import init
from . import optim
from . import utils
from . import distributed  # Distributed training
from . import ops  # Import ops to register all operations
from .function import Function
from .backends import *
from .functional import *
from . import cuda  # CUDA memory management utilities (includes cuda.amp)

# Bind tensor methods
from .api import bind_tensor_methods, bind_nn_functional_methods
bind_tensor_methods()
bind_nn_functional_methods()

# Random number generation - comprehensive RNG API
from .random import (
    seed, manual_seed, initial_seed,
    get_rng_state, set_rng_state, 
    default_generator, fork_rng, Generator
)

# Device management - import our new Device system
from .device import device

# Gradient context managers
from .grad_mode import no_grad, enable_grad, set_grad_enabled, is_grad_enabled

# Device utilities
def cuda_available() -> bool:
    """Check if CUDA is available for use.
    
    Returns:
        bool: True if CUDA device is available and enabled, False otherwise.
    """
    return cuda.is_available()
