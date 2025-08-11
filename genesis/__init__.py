# Import dtype system - replaces old string constants
from .dtypes import (
    float32, float16, float64, bfloat16,
    int32, int64, int16, int8, uint8, bool,
    get_dtype, is_floating_point, is_integer
)
enable_autocast = False
upgrade = False
use_triton = True

from . import utils
from .init import (
        rand,
        randn,
        ones,
        zeros,
        empty,
        empty_like,
        one_hot,
)
from .serialization import (
        save, load, 
        save_checkpoint, load_checkpoint
)
from .autograd import Tensor
# Add lowercase tensor API for PyTorch compatibility
tensor = Tensor
from . import nn
from . import init
from . import optim
from . import utils
from .backend import *
from .functional import *
