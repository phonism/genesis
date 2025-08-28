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


def topk(input: Tensor, k: int, dim: int = -1, largest: bool = True, sorted: bool = True):
    """
    Returns the k largest/smallest elements along a dimension.
    
    Args:
        input: Input tensor
        k: Number of top values to return
        dim: Dimension along which to find top-k values
        largest: If True, return largest values; if False, return smallest
        sorted: If True, return values in sorted order
        
    Returns:
        Tuple of (values, indices) tensors
    """
    values_data, indices_data = array_api.topk(input.data, k, dim, largest, sorted)
    values = genesis.Tensor(values_data, device=input.device, dtype=input.dtype, requires_grad=False)
    indices = genesis.Tensor(indices_data, device=input.device, dtype=genesis.int64, requires_grad=False)
    return values, indices


def isinf(input: Tensor):
    """
    Tests each element of input to see if it is infinite (positive or negative infinity).
    
    Args:
        input: Input tensor
        
    Returns:
        A tensor of the same shape as input with boolean values
    """
    result_data = input.data.isinf()
    return genesis.Tensor(result_data, device=input.device, dtype=genesis.bool, requires_grad=False)


def isnan(input: Tensor):
    """
    Tests each element of input to see if it is NaN (Not a Number).
    
    Args:
        input: Input tensor
        
    Returns:
        A tensor of the same shape as input with boolean values
    """
    result_data = input.data.isnan()
    return genesis.Tensor(result_data, device=input.device, dtype=genesis.bool, requires_grad=False)


def isfinite(input: Tensor):
    """
    Tests each element of input to see if it is finite (not infinite and not NaN).
    
    Args:
        input: Input tensor
        
    Returns:
        A tensor of the same shape as input with boolean values
    """
    result_data = input.data.isfinite()
    return genesis.Tensor(result_data, device=input.device, dtype=genesis.bool, requires_grad=False)


def argsort(input: Tensor, dim: int = -1, descending: bool = False):
    """
    Returns indices that sort a tensor along a dimension.
    
    Args:
        input: Input tensor
        dim: Dimension along which to sort
        descending: If True, sort in descending order
        
    Returns:
        Tensor of indices
    """
    indices_data = array_api.argsort(input.data, dim, descending)
    return genesis.Tensor(indices_data, device=input.device, dtype=genesis.int64, requires_grad=False)


def bincount(input: Tensor, weights=None, minlength: int = 0):
    """
    Count occurrences of each value in integer tensor.
    
    Args:
        input: 1D integer tensor
        weights: Optional weights tensor
        minlength: Minimum length of output
        
    Returns:
        Tensor containing counts
    """
    weights_data = weights.data if weights is not None else None
    result_data = array_api.bincount(input.data, weights_data, minlength)
    dtype = weights.dtype if weights is not None else genesis.int64
    return genesis.Tensor(result_data, device=input.device, dtype=dtype, requires_grad=False)


def allclose(input: Tensor, other: Tensor, rtol: float = 1e-05, atol: float = 1e-08, equal_nan: bool = False):
    """
    Test if all elements of input and other are close.
    
    Args:
        input: First tensor
        other: Second tensor  
        rtol: Relative tolerance
        atol: Absolute tolerance
        equal_nan: Whether to consider NaN values as equal
        
    Returns:
        Boolean scalar tensor indicating if all elements are close
    """
    diff = (input - other).abs()
    threshold = atol + rtol * other.abs()
    close_elements = diff <= threshold
    
    if equal_nan:
        # Handle NaN equality if needed
        input_nan = input != input  # NaN != NaN is True
        other_nan = other != other
        nan_equal = input_nan == other_nan
        close_elements = close_elements | (input_nan & other_nan)
    
    return close_elements.all()
