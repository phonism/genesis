"""functional table."""
# Global operator table.
import genesis
import math
from genesis.tensor import Tensor
from genesis.ops import OperationDispatcher as Dispatcher
from genesis.dtypes import DType
from genesis.device import Device
from typing import Optional, Union, Tuple, Any
from .nn.functional import *

def triu(a: Tensor, k: int = 0, device: Optional[Union[str, Device]] = None) -> Tensor:
    """Return upper triangular part of tensor.

    Args:
        a: Input tensor
        k: Diagonal offset (0 for main diagonal, positive for above, negative for below)
        device: Device placement (unused, kept for compatibility)

    Returns:
        Tensor: Upper triangular tensor
    """
    return Dispatcher.dispatch('triu', a, k)


def tril(a: Tensor, k: int = 0, device: Optional[Union[str, Device]] = None) -> Tensor:
    """Return lower triangular part of tensor.

    Returns a copy of the input tensor with elements above the k-th diagonal zeroed.
    Compatible with PyTorch's torch.tril API.

    Args:
        a: Input tensor (2D or batched)
        k: Diagonal offset (0 for main diagonal, positive for above, negative for below)
        device: Device placement (unused, kept for compatibility)

    Returns:
        Tensor: Lower triangular tensor

    Example:
        >>> x = genesis.ones((3, 3))
        >>> genesis.tril(x)
        tensor([[1., 0., 0.],
                [1., 1., 0.],
                [1., 1., 1.]])
    """
    return Dispatcher.dispatch('tril', a, k)


def empty(*shape: int, device: Optional[Union[str, Device]] = None, dtype: DType = genesis.float32, requires_grad: bool = False) -> Tensor:
    """Create tensor with uninitialized values.

    Args:
        *shape: Shape dimensions of the tensor
        device: Device to place tensor on
        dtype: Data type of the tensor
        requires_grad: Whether to track gradients

    Returns:
        Tensor: Uninitialized tensor with specified shape and dtype
    """
    return genesis.init.zeros(*shape, device=device, dtype=dtype, requires_grad=requires_grad)

def arange(*args: Union[int, float], dtype: Optional[DType] = None, device: Optional[Union[str, Device]] = None) -> Tensor:
    """Create tensor with evenly spaced values within given range.

    Args:
        *args: Either (end,), (start, end), or (start, end, step)
        dtype: Data type of the result tensor
        device: Device to place tensor on

    Returns:
        Tensor: 1D tensor with evenly spaced values

    Raises:
        ValueError: If invalid number of arguments provided
    """
    if len(args) == 1:
        start, end, step = 0, args[0], 1
    elif len(args) == 2:
        start, end = args
        step = 1
    elif len(args) == 3:
        start, end, step = args
    else:
        raise ValueError("arange requires 1 to 3 positional arguments")

    # Handle default device - use CPU by default for better compatibility
    if device is None:
        device = genesis.device("cpu")

    # Use OperationDispatcher for device-specific implementation
    dtype_str = "float32" if dtype is None else dtype.name if hasattr(dtype, 'name') else str(dtype)
    result_data = Dispatcher.dispatch_creation("arange", device, start, end, step, dtype=dtype_str)
    result_data.requires_grad = False
    return result_data


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
    return Dispatcher.dispatch_tuple('topk', input, k, dim, largest, sorted)


def isinf(input: Tensor):
    """
    Tests each element of input to see if it is infinite (positive or negative infinity).
    
    Args:
        input: Input tensor
        
    Returns:
        A tensor of the same shape as input with boolean values
    """
    return Dispatcher.dispatch('isinf', input)


def isnan(input: Tensor):
    """
    Tests each element of input to see if it is NaN (Not a Number).
    
    Args:
        input: Input tensor
        
    Returns:
        A tensor of the same shape as input with boolean values
    """
    return Dispatcher.dispatch('isnan', input)


def isfinite(input: Tensor):
    """
    Tests each element of input to see if it is finite (not infinite and not NaN).
    
    Args:
        input: Input tensor
        
    Returns:
        A tensor of the same shape as input with boolean values
    """
    return Dispatcher.dispatch('isfinite', input)


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
    return Dispatcher.dispatch('argsort', input, dim, descending)


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
    return Dispatcher.dispatch('bincount', input, weights, minlength)


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

def eq(input: Tensor, other):
    """Element-wise equality comparison."""
    return Dispatcher.dispatch("eq", input, other)

def ne(input: Tensor, other):
    """Element-wise not-equal comparison."""  
    return Dispatcher.dispatch("ne", input, other)

def gt(input: Tensor, other):
    """Element-wise greater-than comparison."""
    return Dispatcher.dispatch("gt", input, other)

def ge(input: Tensor, other):
    """Element-wise greater-than-or-equal comparison."""
    return Dispatcher.dispatch("ge", input, other)

def lt(input: Tensor, other):
    """Element-wise less-than comparison."""
    return Dispatcher.dispatch("lt", input, other)

def le(input: Tensor, other):
    """Element-wise less-than-or-equal comparison."""
    return Dispatcher.dispatch("le", input, other)

