"""functional table."""
# Global operator table.
import genesis
import math
from .autograd import Tensor
from .backend import array_api, NDArray
from .nn.functional import *

def triu(a: Tensor, k: int, device=None):
    return Tensor.make_const(array_api.triu(a.data, k))

def empty(*shape, device=None, dtype="float32", requires_grad=False):
    return genesis.init.zeros(*shape, device=device, dtype=dtype, requires_grad=requires_grad)

def arange(*args, dtype=None, device=genesis.device("cuda")):
    if len(args) == 1:
        start, end, step = 0, args[0], 1
    elif len(args) == 2:
        start, end = args
        step = 1
    elif len(args) == 3:
        start, end, step = args
    else:
        raise ValueError("arange requires 1 to 3 positional arguments")
    
    # Delegate to device-specific implementation
    dtype_str = "float32" if dtype is None else dtype.name if hasattr(dtype, 'name') else str(dtype)
    result_data = device.arange(start, end, step, dtype_str)
    return genesis.Tensor(result_data, device=device, dtype=dtype, requires_grad=False)
