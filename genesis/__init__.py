float32 = "float32"
bfloat16 = "bfloat16"
float16 = "float16"
enable_autocast = False
upgrade = False
use_triton = True

from . import utils
from .init import (
        rand,
        randn,
        ones,
        zeros,
        one_hot,
)
from .serialization import (
        save, load, 
        save_checkpoint, load_checkpoint
)
from .autograd import Tensor
from . import nn
from . import init
from . import optim
from . import utils
from .backend import *
from .functional import *
