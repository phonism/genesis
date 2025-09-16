"""
init
"""

import genesis
import math
from genesis.storage import Storage
from genesis.tensor import Tensor
from genesis.ops.dispatcher import OperationDispatcher
from genesis.dtypes import get_dtype

def rand(*shape, low=0.0, high=1.0, device=None, dtype=genesis.float32, requires_grad=False):
    """ Generate random numbers uniform between low and high """
    # Handle string device
    if device is None:
        device = genesis.device('cpu')
    elif isinstance(device, str):
        device = genesis.device(device)

    # Handle both string and DType for dtype parameter
    if isinstance(dtype, str):
        dtype = get_dtype(dtype)

    # Use dispatch_creation for creation operations like randn
    tensor = OperationDispatcher.dispatch_creation("rand", device, shape, dtype.name, low=low, high=high)
    tensor.requires_grad = requires_grad
    return tensor

def randn(*shape, mean=0.0, std=1.0, device=None, dtype=genesis.float32, requires_grad=False):
    """ Generate random normal with specified mean and std deviation """
    # Handle string device
    if device is None:
        device = genesis.device('cpu')
    elif isinstance(device, str):
        device = genesis.device(device)

    # Handle both string and DType for dtype parameter
    if isinstance(dtype, str):
        dtype = get_dtype(dtype)

    # Use dispatch_creation for creation operations
    tensor = OperationDispatcher.dispatch_creation("randn", device, shape, dtype.name, mean=mean, std=std)
    tensor.requires_grad = requires_grad
    return tensor

def constant(*shape, c=1.0, device=None, dtype=genesis.float32, requires_grad=False):
    """ Generate constant Tensor """
    if device is None:
        device = genesis.device('cpu')
    elif isinstance(device, str):
        device = genesis.device(device)

    # Handle both string and DType for dtype parameter
    if isinstance(dtype, str):
        dtype = get_dtype(dtype)

    # Use Storage.allocate with shape instead of size
    storage = Storage.allocate(shape, dtype, device)
    storage._backend.fill(c)
    tensor = Tensor(storage, shape)
    tensor.requires_grad = requires_grad
    return tensor

def ones(*shape, device=None, dtype=genesis.float32, requires_grad=False):
    """ Generate all-ones Tensor """
    return constant(*shape, c=1.0, device=device, dtype=dtype, requires_grad=requires_grad)

def zeros(*shape, device=None, dtype=genesis.float32, requires_grad=False):
    """ Generate all-zeros Tensor """
    return constant(*shape, c=0.0, device=device, dtype=dtype, requires_grad=requires_grad)

def full(shape, fill_value, device=None, dtype=genesis.float32, requires_grad=False):
    """ Generate Tensor filled with specified value """
    if isinstance(shape, (int,)):
        shape = (shape,)
    elif not isinstance(shape, tuple):
        shape = tuple(shape)
    return constant(*shape, c=fill_value, device=device, dtype=dtype, requires_grad=requires_grad)

def empty(*shape, device=None, dtype=genesis.float32, requires_grad=False):
    """ Generate empty Tensor (uninitialized data) """
    if device is None:
        device = genesis.device('cpu')
    elif isinstance(device, str):
        device = genesis.device(device)
    
    # Use Storage.allocate - empty tensor is just allocated storage without initialization
    storage = Storage.allocate(shape, dtype, device)
    tensor = Tensor(storage, shape)
    tensor.requires_grad = requires_grad
    return tensor

def empty_like(tensor, dtype=None, device=None, requires_grad=None):
    """ Generate empty Tensor with same shape as input tensor """
    if dtype is None:
        dtype = tensor.dtype
    if device is None:
        device = tensor.device
    if requires_grad is None:
        requires_grad = tensor.requires_grad  # Inherit requires_grad from input tensor
    
    # Use genesis.empty function 
    return empty(*tensor.shape, device=device, dtype=dtype, requires_grad=requires_grad)

def zeros_like(tensor, dtype=None, device=None, requires_grad=None):
    """ Generate zeros Tensor with same shape as input tensor """
    if dtype is None:
        dtype = tensor.dtype
    if device is None:
        device = tensor.device
    if requires_grad is None:
        requires_grad = tensor.requires_grad  # Inherit requires_grad from input tensor
    
    return zeros(*tensor.shape, device=device, dtype=dtype, requires_grad=requires_grad)

def randb(*shape, p=0.5, device=None, dtype="bool", requires_grad=False):
    """ Generate binary random Tensor """
    # Handle string device
    if device is None:
        device = genesis.device('cpu')
    elif isinstance(device, str):
        device = genesis.device(device)

    # Handle both string and DType for dtype parameter
    if isinstance(dtype, str):
        dtype = get_dtype(dtype)

    # Generate random uniform tensor and compare with p
    uniform_tensor = OperationDispatcher.dispatch_creation("rand", device, shape, "float32", low=0.0, high=1.0)
    # Convert to boolean: tensor <= p
    bool_tensor = OperationDispatcher.dispatch("le", uniform_tensor, p)
    bool_tensor.requires_grad = requires_grad
    return bool_tensor

def one_hot(n, i, device=None, dtype=genesis.float32, requires_grad=False):
    """ Generate one-hot encoding Tensor """
    device = genesis.device('cpu') if device is None else device
    # i should be an NDArray, pass it directly to device.one_hot
    return genesis.Tensor(device.one_hot(n, i, dtype=genesis.float32), device=device, requires_grad=requires_grad)

def xavier_uniform(fan_in, fan_out, gain=1.0, **kwargs):
    a = gain * math.sqrt(6 / (fan_in + fan_out))
    return rand(fan_in, fan_out, low=-a, high=a, **kwargs)

def xavier_normal(fan_in, fan_out, gain=1.0, **kwargs):
    std = gain * math.sqrt(2 / (fan_in + fan_out))
    return randn(fan_in, fan_out, mean=0, std=std, **kwargs)

def kaiming_uniform(fan_in, fan_out, nonlinearity="relu", **kwargs):
    assert nonlinearity == "relu", "Only relu supported currently"
    gain = math.sqrt(2)
    bound = gain * math.sqrt(3 / fan_in)
    return rand(fan_in, fan_out, low=-bound, high=bound, **kwargs)

def kaiming_normal(fan_in, fan_out, nonlinearity="relu", **kwargs):
    assert nonlinearity == "relu", "Only relu supported currently"
    gain = math.sqrt(2)
    std = gain / math.sqrt(fan_in)
    return randn(fan_in, fan_out, mean=0, std=std, **kwargs)

def eye(n, m=None, device=None, dtype=genesis.float32, requires_grad=False):
    """Generate identity matrix.
    
    Args:
        n: Number of rows
        m: Number of columns (defaults to n for square matrix)
        device: Target device for tensor
        dtype: Data type for tensor elements
        requires_grad: Whether to track gradients
        
    Returns:
        Tensor: Identity matrix of shape (n, m) or (n, n)
    """
    if m is None:
        m = n
    device = genesis.device('cpu') if device is None else device
    
    # Use device-specific eye implementation
    array = device.eye(n, m, dtype=dtype)
    return genesis.Tensor(array, device=device, dtype=dtype, requires_grad=requires_grad)

def ones_like(tensor, dtype=None, device=None, requires_grad=None):
    """Generate ones Tensor with same shape as input tensor.
    
    Args:
        tensor: Input tensor to match shape
        dtype: Data type (defaults to input tensor's dtype)
        device: Target device (defaults to input tensor's device)
        requires_grad: Whether to track gradients (defaults to input tensor's requires_grad)
        
    Returns:
        Tensor: Ones tensor with same shape as input
    """
    if dtype is None:
        dtype = tensor.dtype
    if device is None:
        device = tensor.device
    if requires_grad is None:
        requires_grad = tensor.requires_grad
    
    return ones(*tensor.shape, device=device, dtype=dtype, requires_grad=requires_grad)

def randint(low, high, shape, device=None, dtype=genesis.int64, requires_grad=False):
    """Generate random integers in range [low, high).
    
    Args:
        low: Lowest integer (inclusive)
        high: Highest integer (exclusive)
        shape: Shape of output tensor (tuple or int)
        device: Target device (defaults to CPU)
        dtype: Integer data type (defaults to int64)
        requires_grad: Whether to track gradients
        
    Returns:
        Tensor: Random integer tensor
    """
    if isinstance(shape, int):
        shape = (shape,)
    elif not isinstance(shape, tuple):
        shape = tuple(shape)
        
    # Handle string device
    if device is None:
        device = genesis.device('cpu')
    elif isinstance(device, str):
        device = genesis.device(device)

    # Handle both string and DType for dtype parameter
    if isinstance(dtype, str):
        dtype = get_dtype(dtype)

    # Use dispatch_creation for creation operations
    tensor = OperationDispatcher.dispatch_creation("randint", device, shape, dtype.name, low=low, high=high)
    tensor.requires_grad = requires_grad
    return tensor

def from_numpy(array, device=None, dtype=None, requires_grad=False):
    """Create Tensor from numpy array.
    
    Args:
        array: NumPy array to convert
        device: Target device (defaults to CPU)
        dtype: Target data type (defaults to inferred from numpy array)
        requires_grad: Whether to track gradients
        
    Returns:
        Tensor: Genesis tensor created from numpy array
    """
    import numpy as np
    
    if not isinstance(array, np.ndarray):
        array = np.array(array)
    
    device = genesis.device('cpu') if device is None else device
    
    # Infer dtype from numpy array if not specified
    if dtype is None:
        if array.dtype == np.float32:
            dtype = genesis.float32
        elif array.dtype == np.float64:
            dtype = genesis.float32  # Convert float64 to float32 by default
        elif array.dtype == np.float16:
            dtype = genesis.float16
        elif array.dtype == np.int32:
            dtype = genesis.int32
        elif array.dtype == np.int64:
            dtype = genesis.int64
        elif array.dtype == np.bool_:
            dtype = genesis.bool
        else:
            dtype = genesis.float32  # Default fallback
    
    # Create tensor directly using make_tensor (already imported via genesis.tensor module)
    tensor = genesis.tensor(array, dtype=dtype, device=device, requires_grad=requires_grad)
    return tensor
