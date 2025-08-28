import torch
import genesis
import operator
from functools import reduce


class CPUStorage(torch.Tensor):
    """Storage class for CPU tensors, extending torch.Tensor with to_numpy method."""
    
    def __new__(cls, data):
        """Create a new CPUStorage from a torch.Tensor."""
        if isinstance(data, torch.Tensor):
            # Create a subclass instance from the tensor
            instance = data.as_subclass(cls)
            return instance
        else:
            raise TypeError(f"Expected torch.Tensor, got {type(data)}")
    
    def to_numpy(self):
        """Convert to numpy array."""
        return self.detach().cpu().numpy()


# Since CPUStorage IS a torch.Tensor, all operations work directly!
def add(x, y):
    return x + y

def sub(x, y):
    return x - y

def mul(x, y):
    return x * y

def truediv(x, y):
    return x.__truediv__(y)

def rtruediv(x, y):
    return x.__rtruediv__(y)

def pow(x, scalar):
    return x ** scalar

def log(x):
    return torch.log(x)

def exp(x):
    return torch.exp(x)

def sin(x):
    return torch.sin(x)

def cos(x):
    return torch.cos(x)

def sqrt(x):
    return torch.sqrt(x)

def abs(x):
    return torch.abs(x)

def sign(x):
    return torch.sign(x)

def clamp(x, min_val=None, max_val=None):
    return torch.clamp(x, min=min_val, max=max_val)

def greater_equal(x, y):
    if isinstance(y, torch.Tensor):
        return torch.greater_equal(x, y)
    return torch.greater_equal(x, torch.tensor(y, device=x.device, dtype=x.dtype))

def less_equal(x, y):
    if isinstance(y, torch.Tensor):
        return torch.less_equal(x, y)
    return torch.less_equal(x, torch.tensor(y, device=x.device, dtype=x.dtype))

def ones(shape, device=None, dtype="float32"):
    torch_dtype = torch.float32
    if dtype == "float16":
        torch_dtype = torch.float16
    elif dtype == "bfloat16":
        torch_dtype = torch.bfloat16
    return torch.ones(shape, dtype=torch_dtype)

def maximum(x, y):
    if isinstance(y, torch.Tensor):
        return torch.maximum(x, y)
    return torch.clamp(x, min=y)

def reduce_sum(x, axis=None, keepdims=False):
    return torch.sum(x, axis=axis, keepdims=keepdims)

def reduce_max(x, axis=None, keepdims=False):
    if axis is None:
        return torch.max(x)
    if isinstance(axis, tuple):
        axis = axis[0]
    return torch.max(x, dim=axis, keepdim=keepdims).values

def isinf(x):
    """Tests each element to see if it is infinite."""
    return torch.isinf(x)

def isnan(x):
    """Tests each element to see if it is NaN."""
    return torch.isnan(x)

def isfinite(x):
    """Tests each element to see if it is finite."""
    return torch.isfinite(x)


def reshape(x, new_shape):
    return x.reshape(new_shape)

def view(x, new_shape):
    return x.view(new_shape)

def expand(x, new_shape):
    return x.expand(new_shape)

def permute(x, new_axis):
    return x.permute(new_axis)

def broadcast_to(x, new_shape):
    return x.broadcast_to(new_shape)

def getitem(x, idxs):
    return x.__getitem__(idxs)

def setitem(x, idxs, other):
    # Handle Genesis Tensor indices - extract underlying data
    if hasattr(idxs, 'data') and hasattr(idxs.data, 'data'):
        # Genesis Tensor -> NDArray -> torch.Tensor
        actual_idxs = idxs.data.data
    elif hasattr(idxs, 'data'):
        # NDArray -> torch.Tensor  
        actual_idxs = idxs.data
    else:
        # Direct indexing (int, slice, etc.)
        actual_idxs = idxs
    
    x.__setitem__(actual_idxs, other)
    return x

def fill(tensor, value):
    """Fill tensor with constant value"""
    tensor.fill_(value)
    return tensor

def eq(x, y):
    return x.__eq__(y)

def ge(x, y):
    return x.__ge__(y)

def gt(x, y):
    return x.__gt__(y)

def le(x, y):
    return x.__le__(y)

def lt(x, y):
    return x.__lt__(y)

def matmul(a, b, activation=""):
    return a @ b

def from_numpy(data, device_id=None, dtype=None):
    torch_dtype = None
    if dtype is None or dtype == genesis.float32:
        torch_dtype = torch.float32
    elif dtype == genesis.float16:
        torch_dtype = torch.float16
    elif dtype == genesis.bfloat16:
        torch_dtype = torch.bfloat16
    elif dtype == genesis.bool:
        torch_dtype = torch.bool
    elif dtype == genesis.int64:
        torch_dtype = torch.int64
    tensor = torch.from_numpy(data).to(torch_dtype)
    return CPUStorage(tensor)

def from_tensor(data, device_id=None):
    # If data is already a CPUStorage, return it
    if isinstance(data, CPUStorage):
        return data
    # If it's a torch.Tensor, wrap it
    elif isinstance(data, torch.Tensor):
        return CPUStorage(data)
    # Otherwise return as-is (might be CUDAStorage)
    else:
        return data

def clone(data, device_id=None, dtype=None):
    """Create a deep copy of the tensor data."""
    return data.clone()

def array(shape, device_id=None, dtype=None):
    # Convert dtype to string for comparison (handles both DType objects and strings)
    dtype_str = dtype if isinstance(dtype, str) else (dtype.name if hasattr(dtype, 'name') else str(dtype))
    
    # Handle scalar case (empty shape tuple) - need to pass empty tuple directly
    if len(shape) == 0:
        if dtype is None or dtype_str == "float32":
            arr = torch.empty((), dtype=torch.float32, device=torch.device("cpu"))
        elif dtype_str == "float16":
            arr = torch.empty((), dtype=torch.float16, device=torch.device("cpu"))
        elif dtype_str == "bfloat16":
            arr = torch.empty((), dtype=torch.bfloat16, device=torch.device("cpu"))
        elif dtype_str == "bool":
            arr = torch.empty((), dtype=torch.bool, device=torch.device("cpu"))
        elif dtype_str == "int64":
            arr = torch.empty((), dtype=torch.int64, device=torch.device("cpu"))
        else:
            arr = torch.empty((), dtype=torch.float32, device=torch.device("cpu"))
    else:
        # For non-scalar tensors, unpack the shape tuple
        if dtype is None or dtype_str == "float32":
            arr = torch.empty(*shape, dtype=torch.float32, device=torch.device("cpu"))
        elif dtype_str == "float16":
            arr = torch.empty(*shape, dtype=torch.float16, device=torch.device("cpu"))
        elif dtype_str == "bfloat16":
            arr = torch.empty(*shape, dtype=torch.bfloat16, device=torch.device("cpu"))
        elif dtype_str == "bool":
            arr = torch.empty(*shape, dtype=torch.bool, device=torch.device("cpu"))
        elif dtype_str == "int64":
            arr = torch.empty(*shape, dtype=torch.int64, device=torch.device("cpu"))
        else:
            # Default to float32
            arr = torch.empty(*shape, dtype=torch.float32, device=torch.device("cpu"))
    return CPUStorage(arr)

def cat(arrays, dim=0):
    """
    Concatenate tensors along specified dimension.
    
    Args:
        arrays: List of tensors to concatenate
        dim: Dimension along which to concatenate
        
    Returns:
        torch.Tensor: Concatenated tensor
    """
    return torch.cat(arrays, dim=dim)

def randn(shape, dtype="float32"):
    """
    Create tensor with random normal distribution.
    
    Args:
        shape: Tensor shape
        dtype: Data type
        
    Returns:
        torch.Tensor: Random normal tensor
    """
    torch_dtype = torch.float32
    if dtype == "float16":
        torch_dtype = torch.float16
    elif dtype == "bfloat16":
        torch_dtype = torch.bfloat16
    
    return torch.randn(shape, dtype=torch_dtype, device=torch.device("cpu"))

def rand(shape, dtype="float32"):
    """
    Create tensor with random uniform distribution.
    
    Args:
        shape: Tensor shape
        dtype: Data type
        
    Returns:
        torch.Tensor: Random uniform tensor
    """
    torch_dtype = torch.float32
    if dtype == "float16":
        torch_dtype = torch.float16
    elif dtype == "bfloat16":
        torch_dtype = torch.bfloat16
    
    return torch.rand(shape, dtype=torch_dtype, device=torch.device("cpu"))

def triu(x, k=0):
    """
    Upper triangle of tensor.
    """
    return torch.triu(x, diagonal=k)

def split(x, cnt, dim=None):
    """
    Split tensor along dimension.
    """
    return torch.split(x, cnt, dim=dim)

def squeeze(x, dim=None):
    """
    Remove dimensions of size 1.
    """
    return torch.squeeze(x, dim=dim)

def unsqueeze(x, dim):
    """
    Add a dimension of size 1.
    """
    return torch.unsqueeze(x, dim=dim)

def to_dtype(x, dtype):
    """
    Convert tensor to specified dtype.
    """
    torch_dtype = torch.float32
    if dtype == "float16":
        torch_dtype = torch.float16
    elif dtype == "bfloat16":
        torch_dtype = torch.bfloat16
    elif dtype == "int32":
        torch_dtype = torch.int32
    elif dtype == "int64":
        torch_dtype = torch.int64
    
    return x.to(torch_dtype)

def iadd(x, y):
    """
    In-place addition.
    """
    if isinstance(y, torch.Tensor):
        x.add_(y)
    else:
        x.add_(y)
    return x

def prod(x):
    """
    Product of all elements in x.
    """
    return reduce(operator.mul, x, 1)

def one_hot(n_classes, indices, dtype="float32"):
    """
    Create one-hot encoding using PyTorch.
    
    Args:
        n_classes: Number of classes
        indices: Indices tensor (flat)
        dtype: Data type
        
    Returns:
        torch.Tensor: One-hot encoded tensor
    """
    torch_dtype = torch.float32
    if dtype == "float16":
        torch_dtype = torch.float16
    elif dtype == "bfloat16":
        torch_dtype = torch.bfloat16
    elif dtype == "int32":
        torch_dtype = torch.int32
    elif dtype == "int64":
        torch_dtype = torch.int64
    
    # Convert indices to long for one_hot
    if not isinstance(indices, torch.Tensor):
        indices = torch.tensor(indices)
    indices_long = indices.long()
    
    # Create one-hot encoding
    one_hot_result = torch.nn.functional.one_hot(indices_long, num_classes=n_classes)
    return one_hot_result.to(torch_dtype)

def arange(start, end, step, dtype="float32"):
    """
    Create tensor with values from start to end with step.
    
    Args:
        start: Starting value
        end: Ending value (exclusive)
        step: Step size
        dtype: Data type
        
    Returns:
        torch.Tensor: Range tensor
    """
    torch_dtype = torch.float32
    if dtype == "float16":
        torch_dtype = torch.float16
    elif dtype == "bfloat16":
        torch_dtype = torch.bfloat16
    elif dtype == "int32":
        torch_dtype = torch.int32
    elif dtype == "int64":
        torch_dtype = torch.int64
    
    return torch.arange(start, end, step, dtype=torch_dtype, device=torch.device("cpu"))

def where(condition, x, y):
    """Element-wise selection of values from x or y based on condition."""
    return torch.where(condition, x, y)

def zeros_like(x):
    """Create tensor of zeros with same shape and dtype as x."""
    return torch.zeros_like(x)

def argmax(x, dim=None, keepdim=False):
    """Return indices of maximum values along dimension."""
    if dim is None:
        return torch.argmax(x).unsqueeze(0) if keepdim else torch.argmax(x)
    return torch.argmax(x, dim=dim, keepdim=keepdim)

def argmin(x, dim=None, keepdim=False):
    """Return indices of minimum values along dimension."""
    if dim is None:
        return torch.argmin(x).unsqueeze(0) if keepdim else torch.argmin(x)
    return torch.argmin(x, dim=dim, keepdim=keepdim)

def gather(x, dim, index):
    """Gather values along dimension using indices."""
    return torch.gather(x, dim, index)

def scatter(x, dim, index, src):
    """Scatter values from src along dimension using indices."""
    return x.scatter(dim, index, src)

def broadcast_shapes(shape1, shape2):
    """
    Compute the broadcasted shape of two tensors (NumPy broadcasting rules).
    """
    import numpy as np
    # Use numpy's broadcast_shapes if available (Python 3.10+)
    try:
        return np.broadcast_shapes(shape1, shape2)
    except AttributeError:
        # Fallback implementation
        shape1_rev = list(reversed(shape1))
        shape2_rev = list(reversed(shape2))
        
        max_ndim = max(len(shape1_rev), len(shape2_rev))
        while len(shape1_rev) < max_ndim:
            shape1_rev.append(1)
        while len(shape2_rev) < max_ndim:
            shape2_rev.append(1)
        
        result_shape_rev = []
        for s1, s2 in zip(shape1_rev, shape2_rev):
            if s1 == 1:
                result_shape_rev.append(s2)
            elif s2 == 1:
                result_shape_rev.append(s1)
            elif s1 == s2:
                result_shape_rev.append(s1)
            else:
                raise ValueError(f"Cannot broadcast shapes {tuple(reversed(shape1_rev))} and {tuple(reversed(shape2_rev))}")
        
        return tuple(reversed(result_shape_rev))


def topk(x, k, dim=-1, largest=True, sorted=True):
    """CPU implementation of topk using PyTorch."""
    # Handle case where k is larger than tensor size in the specified dimension
    if dim < 0:
        dim = len(x.shape) + dim
    if k > x.shape[dim]:
        k = x.shape[dim]
    values, indices = torch.topk(x, k, dim=dim, largest=largest, sorted=sorted)
    return values, indices


def argsort(x, dim=-1, descending=False):
    """CPU implementation of argsort using PyTorch."""
    return torch.argsort(x, dim=dim, descending=descending)


def bincount(x, weights=None, minlength=0):
    """CPU implementation of bincount using PyTorch."""
    return torch.bincount(x, weights=weights, minlength=minlength)


def scatter_add(input_tensor, dim, index, src):
    """CPU implementation of scatter_add using PyTorch."""
    return input_tensor.scatter_add(dim, index, src)


def repeat_interleave(x, repeats, dim=None):
    """CPU implementation of repeat_interleave using PyTorch."""
    return torch.repeat_interleave(x, repeats, dim=dim)
