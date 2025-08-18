"""N-dimensional array implementation with device abstraction.

This module provides the core NDArray class and Device abstraction that supports
both CPU (NumPy) and GPU (CUDA) backends for tensor operations.
"""

import operator
import os
import math
from functools import reduce
import numpy as np
import genesis
from typing import Optional, Any
import torch

def prod(x):
    """Return the product of all elements in a list."""
    return reduce(operator.mul, x, 1)

class Device:
    """Device abstraction for CPU and GPU computation backends.
    
    Provides a unified interface for different computational devices,
    supporting both NumPy (CPU) and CUDA (GPU) operations.
    """
    
    def __init__(
        self, 
        name: str, 
        mod: Any, 
        device_id: Optional[int] = None
    ) -> None:
        """Initialize device with backend module.
        
        Args:
            name: Device name ('cpu' or 'cuda')
            mod: Backend module (numpy or cuda module)
            device_id: Optional device ID for multi-GPU systems
        """
        self.name = name
        self.mod = mod
        self.device_id = device_id

    def __eq__(self, other) -> bool:
        if isinstance(other, Device):
            return self.name == other.name
        elif isinstance(other, str):
            return self.name == other
        return False

    def __repr__(self) -> str:
        return self.name + "()"

    def __getattr__(self, name: str):
        return getattr(self.mod, name)

    def enabled(self) -> bool:
        """Return True if the device is enabled."""
        return self.mod is not None

    def randn(self, *shape, dtype: Optional[str] = genesis.float32) -> "NDArray":
        """Create tensor with random normal distribution using backend."""
        # Convert dtype to string format for backend compatibility
        dtype_str = dtype.name if hasattr(dtype, 'name') else str(dtype)
        backend_data = self.mod.randn(shape, dtype=dtype_str)
        return NDArray(backend_data, device=self, dtype=dtype)

    def rand(self, *shape, dtype: Optional[str] = genesis.float32) -> "NDArray":
        """Create tensor with random uniform distribution using backend."""
        # Convert dtype to string format for backend compatibility
        dtype_str = dtype.name if hasattr(dtype, 'name') else str(dtype)
        backend_data = self.mod.rand(shape, dtype=dtype_str)
        return NDArray(backend_data, device=self, dtype=dtype)

    def one_hot(self, n, i, dtype: Optional[str] = genesis.float32) -> "NDArray":
        """
        Device-agnostic one-hot encoding - delegates to device backend
        """
        # Delegate to device-specific implementation
        dtype_str = dtype.name if hasattr(dtype, 'name') else str(dtype)
        result_data = self.mod.one_hot(n, i.data, dtype_str)
        return NDArray(result_data, device=self)

    def empty(self, shape, dtype: Optional[str] = genesis.float32) -> "NDArray":
        dtype = genesis.float32 if dtype is None else dtype
        return NDArray.make(shape, device=self, dtype=dtype)

    def full(self, shape, fill_value, dtype: Optional[str] = genesis.float32) -> "NDArray":
        dtype = genesis.float32 if dtype is None else dtype
        arr = self.empty(shape, dtype)
        arr.fill(fill_value)
        return arr

def device(device_name):
    if isinstance(device_name, int):
        return cuda(device_name)
    if device_name == "cuda":
        return cuda(0)
    elif device_name == "cpu":
        return cpu()
    elif device_name.find("cuda") != -1:
        device_id = int(device_name.split("cuda:")[-1])
        return cuda(device_id)

def cpu():
    """Return cuda device"""
    from . import ndarray_ops_cpu
    return Device("cpu", ndarray_ops_cpu)

def cuda(index=0):
    """Return cuda device"""
    try:
        from . import ndarray_ops_gpu
        return Device("cuda", ndarray_ops_gpu, index)
    except Exception as e:
        print(f"An error occurred: {e}")
        return Device("cuda", None)

def default_device():
    return cpu()

def all_devices():
    """return a list of all available devices"""
    return [cuda()]

class NDArray:
    def __init__(self, data, device=None, dtype=None):
        self._dtype = dtype
        if isinstance(data, np.ndarray):
            device = device if device is not None else default_device()
            self._device = device
            self.data = self._device.from_numpy(data, device_id=self._device.device_id, dtype=dtype)
        elif isinstance(data, NDArray):
            self._device = device
            self.data = self._device.from_tensor(data.data, device_id=self._device.device_id)
        elif isinstance(data, torch.Tensor):
            self._device = device
            self.data = self._device.from_tensor(data, device_id=self._device.device_id)
        elif data.__class__.__name__ == 'CUDAStorage':  # CUDAStorage
            self._device = device
            self.data = data  # Use directly
        else:
            self._device = device
            self.data = self._device.from_numpy(np.array(data, dtype=np.float32), device_id=self._device.device_id, dtype=dtype)

    @staticmethod
    def make(shape, device=None, dtype=genesis.float32):
        array = NDArray.__new__(NDArray)
        array._device = device if device is not None else default_device()
        array._dtype = dtype
        
        # Convert DType to format expected by backend
        if hasattr(dtype, 'name'):  # DType object
            backend_dtype = dtype.name
        else:
            backend_dtype = dtype
            
        array.data = array.device.array(shape, device_id=array._device.device_id, dtype=backend_dtype)
        return array

    def fill(self, value):
        """ Fill (in place) with a constant value. """
        self.data = self.device.fill(self.data, value)

    def __repr__(self):
        return "Gensis::" + self.data.__repr__()

    def __str__(self):
        return self.__repr__()
    
    def cpu(self):
        return self.data.cpu()

    def numpy(self):
        if hasattr(self.data, 'cpu'):
            cpu_data = self.data.cpu()
            if hasattr(cpu_data, 'numpy'):
                return cpu_data.numpy()
            else:
                # cpu() already returned numpy array
                return cpu_data
        else:
            # Fallback for other data types
            return self.data.cpu().numpy()

    @property
    def dtype(self):
        return self._dtype

    @property
    def shape(self):
        return self.data.shape

    @property
    def device(self):
        return self._device

    def numel(self):
        return prod(self.shape)
    
    def is_contiguous(self):
        """Check if tensor has contiguous memory"""
        if hasattr(self.data, 'is_contiguous'):
            return self.data.is_contiguous()
        elif hasattr(self.data, 'is_contiguous'):
            # PyTorch tensor
            return self.data.is_contiguous()
        else:
            # Assume contiguous for unknown types
            return True
    
    def contiguous(self):
        """Return contiguous version of tensor"""
        if hasattr(self.data, 'contiguous'):
            # Use CUDATensor or PyTorch contiguous
            out = NDArray.make(self.shape, device=self.device)
            out.data = self.data.contiguous()
            return out
        else:
            # Already contiguous or unknown type
            return self

    def size(self, dim):
        return self.data.size(dim)

    def triu(self, k=0):
        out = NDArray.make(self.shape, device=self.device)
        # Use device backend for triu operation
        out.data = self.device.triu(self.data, k=k)
        return out

    def split(self, cnt, dim=None):
        result = []
        # Use device backend for split operation
        splits = self.device.split(self.data, cnt, dim=dim if dim is not None else -1)
        for tensor_data in splits:
            out = NDArray.make(tensor_data.shape, device=self.device)
            out.data = tensor_data
            result.append(out)
        return result

    @property
    def flat(self):
        out = NDArray.make(self.shape, device=self.device)
        out.data = self.device.reshape(self.data, (self.numel(),))
        return out

    def __add__(self, y):
        # Optimized: Let device.add() handle the full operation including output creation
        if isinstance(y, NDArray):
            result_data = self.device.add(self.data, y.data)
        else:
            result_data = self.device.add(self.data, y)
        
        # Create minimal NDArray wrapper around the result
        out = NDArray.__new__(NDArray)
        out._device = self.device
        out._dtype = self.dtype
        out.data = result_data
        return out

    def __iadd__(self, y):
        out = NDArray.make(self.shape, device=self.device)
        if isinstance(y, NDArray):
            out.data = self.device.iadd(self.data, y.data)
        else:
            out.data = self.device.iadd(self.data, y)
        return out

    __radd__ = __add__

    def __mul__(self, y):
        # Optimized: Let device.mul() handle the full operation
        if isinstance(y, NDArray):
            result_data = self.device.mul(self.data, y.data)
        else:
            result_data = self.device.mul(self.data, y)
        
        # Create minimal NDArray wrapper around the result
        out = NDArray.__new__(NDArray)
        out._device = self.device
        out._dtype = self.dtype
        out.data = result_data
        return out

    __rmul__ = __mul__

    def __truediv__(self, y):
        # Optimized: Let device.truediv() handle the full operation
        if isinstance(y, NDArray):
            result_data = self.device.truediv(self.data, y.data)
        else:
            result_data = self.device.truediv(self.data, y)
        
        # Create minimal NDArray wrapper around the result
        out = NDArray.__new__(NDArray)
        out._device = self.device
        out._dtype = self.dtype
        out.data = result_data
        return out

    def __rtruediv__(self, y):
        out = NDArray.make(self.shape, device=self.device)
        if isinstance(y, NDArray):
            out.data = self.device.rtruediv(self.data, y.data)
        else:
            out.data = self.device.rtruediv(self.data, y)
        return out

    def maximum(self, y):
        # Optimized: Let device.maximum() handle the full operation
        if isinstance(y, NDArray):
            result_data = self.device.maximum(self.data, y.data)
        else:
            result_data = self.device.maximum(self.data, y)
        
        # Create minimal NDArray wrapper around the result
        out = NDArray.__new__(NDArray)
        out._device = self.device
        out._dtype = self.dtype
        out.data = result_data
        return out
    
    def __neg__(self):
        if self.data.is_contiguous() is False:
            self.data = self.data.contiguous()
        return self * (-1)

    def __sub__(self, other):
        # Optimized: Direct sub operation instead of add + neg
        if isinstance(other, NDArray):
            result_data = self.device.sub(self.data, other.data)
        else:
            result_data = self.device.sub(self.data, other)
        
        # Create minimal NDArray wrapper around the result
        out = NDArray.__new__(NDArray)
        out._device = self.device
        out._dtype = self.dtype
        out.data = result_data
        return out

    def __rsub__(self, other):
        if self.data.is_contiguous() is False:
            self.data = self.data.contiguous()
        if isinstance(other, NDArray):
            if other.data.is_contiguous() is False:
                other.data = other.data.contiguous()
        return other + (-self)

    def __pow__(self, scalar):
        out = NDArray.make(self.shape, device=self.device)
        out.data = self.device.pow(self.data, scalar)
        return out

    def __rpow__(self, scalar):
        out = NDArray.make(self.shape, device=self.device)
        out.data = self.device.pow(scalar, self.data)
        return out

    def log(self):
        # Optimized: avoid NDArray.make() overhead
        result_data = self.device.log(self.data)
        out = NDArray.__new__(NDArray)
        out._device = self.device
        out._dtype = self.dtype
        out.data = result_data
        return out

    def exp(self):
        # Optimized: avoid NDArray.make() overhead
        result_data = self.device.exp(self.data)
        out = NDArray.__new__(NDArray)
        out._device = self.device
        out._dtype = self.dtype
        out.data = result_data
        return out

    def cos(self):
        # Optimized: avoid NDArray.make() overhead
        result_data = self.device.cos(self.data)
        out = NDArray.__new__(NDArray)
        out._device = self.device
        out._dtype = self.dtype
        out.data = result_data
        return out

    def sin(self):
        # Optimized: avoid NDArray.make() overhead
        result_data = self.device.sin(self.data)
        out = NDArray.__new__(NDArray)
        out._device = self.device
        out._dtype = self.dtype
        out.data = result_data
        return out

    def sqrt(self):
        # Optimized: avoid NDArray.make() overhead
        result_data = self.device.sqrt(self.data)
        out = NDArray.__new__(NDArray)
        out._device = self.device
        out._dtype = self.dtype
        out.data = result_data
        return out

    def reshape(self, new_shape):
        out = NDArray.make(new_shape, device=self.device)
        out.data = self.device.reshape(self.data, new_shape)
        return out

    def view(self, new_shape):
        if -1 in new_shape:
            unknown_dim = -1
            known_dim_product = 1
            for dim in new_shape:
                if dim != -1:
                    known_dim_product *= dim
                else:
                    unknown_dim = dim
            inferred_dim = int(np.prod(self.shape) / known_dim_product)
            new_shape = tuple(inferred_dim if dim == -1 else dim for dim in new_shape)
        self.data = self.device.view(self.data, new_shape)
        return self

    def expand(self, new_shape):
        out = NDArray.make(new_shape, device=self.device)
        out.data = self.device.expand(self.data, new_shape)
        return out

    def permute(self, new_axis):
        out = NDArray.make(self.shape, device=self.device)
        out.data = self.device.permute(self.data, new_axis)
        return out

    def contiguous(self):
        out = NDArray.make(self.shape, device=self.device)
        out.data = self.data.contiguous()
        return out

    def float(self):
        out = NDArray.make(self.shape, dtype=genesis.float32, device=self.device)
        out.data = self.data.float()
        return out

    def half(self):
        out = NDArray.make(self.shape, dtype=genesis.float16, device=self.device)
        out.data = self.device.to_dtype(self.data, "float16")
        return out
    
    def long(self):
        """Convert tensor to int64 (long) dtype."""
        out = NDArray.make(self.shape, dtype=genesis.int64, device=self.device)
        out.data = self.device.to_dtype(self.data, "int64")
        return out
    
    def to(self, target):
        """Convert tensor to target device or dtype."""
        if isinstance(target, str):
            if target.startswith('cuda') or target == 'cuda':
                # Move to CUDA device
                import genesis
                target_device = genesis.device(target)
                return NDArray(self.data, device=target_device, dtype=self.dtype)
            else:
                # Assume it's a dtype
                out = NDArray.make(self.shape, dtype=target, device=self.device)
                out.data = self.device.to_dtype(self.data, target)
                return out
        else:
            # Could be device object or dtype object
            if hasattr(target, 'name'):
                # Device object
                return NDArray(self.data, device=target, dtype=self.dtype)
            else:
                # Assume dtype
                out = NDArray.make(self.shape, dtype=target, device=self.device)
                out.data = self.device.to_dtype(self.data, str(target))
                return out

    def broadcast_to(self, new_shape):
        out = NDArray.make(new_shape, dtype=self.dtype, device=self.device)
        out.data = self.device.broadcast_to(self.data, new_shape)
        return out

    def __getitem__(self, idxs):
        # Optimized: avoid NDArray.make() overhead
        if isinstance(idxs, NDArray):
            result_data = self.device.getitem(self.data, idxs.data)
        else:
            result_data = self.device.getitem(self.data, idxs)
        out = NDArray.__new__(NDArray)
        out._device = self.device
        out._dtype = self.dtype
        out.data = result_data
        return out

    def __setitem__(self, idxs, other):
        out = NDArray.make(self.shape, dtype=self.dtype, device=self.device)
        if isinstance(idxs, NDArray):
            idxs = idxs.data
        if isinstance(other, NDArray):
            out.data = self.device.setitem(self.data, idxs, other.data)
        else:
            out.data = self.device.setitem(self.data, idxs, other)
        return out

    def __eq__(self, other):
        out = NDArray.make(self.shape, device=self.device)
        if isinstance(other, NDArray):
            out.data = self.device.eq(self.data, other.data)
        else:
            out.data = self.device.eq(self.data, other)
        return out

    def __ge__(self, other):
        out = NDArray.make(self.shape, device=self.device)
        if isinstance(other, NDArray):
            out.data = self.device.ge(self.data, other.data)
        else:
            out.data = self.device.ge(self.data, other)
        return out

    def __ne__(self, other):
        return 1 - (self == other)

    def __gt__(self, other):
        out = NDArray.make(self.shape, device=self.device)
        if isinstance(other, NDArray):
            out.data = self.device.gt(self.data, other.data)
        else:
            out.data = self.device.gt(self.data, other)
        return out

    def __lt__(self, other):
        out = NDArray.make(self.shape, device=self.device)
        if isinstance(other, NDArray):
            out.data = self.device.lt(self.data, other.data)
        else:
            out.data = self.device.lt(self.data, other)
        return out

    def __le__(self, other):
        out = NDArray.make(self.shape, device=self.device)
        if isinstance(other, NDArray):
            out.data = self.device.le(self.data, other.data)
        else:
            out.data = self.device.le(self.data, other)
        return out

    def __matmul__(self, b, activation=""):
        out = NDArray.make(self.shape, dtype=self.dtype, device=self.device)
        out.data = self.device.matmul(self.data, b.data, activation=activation)
        return out

    def sum(self, axis=None, keepdims=False):
        out = NDArray.make(self.shape, dtype=self.dtype, device=self.device)
        out.data = self.device.reduce_sum(self.data, axis=axis, keepdims=keepdims)
        return out

    def max(self, axis=None, keepdims=False):
        out = NDArray.make(self.shape, device=self.device)
        out.data = self.device.reduce_max(self.data, axis=axis, keepdims=keepdims)
        return out

    def is_floating_point(self):
        return self.data.is_floating_point()

    def data_ptr(self):
        """Return data pointer (Triton compatible)
        
        Returns:
            int: Data pointer address
        """
        if hasattr(self.data, 'ptr'):
            # CUDA data
            return int(self.data.ptr)
        elif hasattr(self.data, 'ctypes'):
            # CPU data (numpy array)
            return self.data.ctypes.data
        else:
            raise RuntimeError("Cannot get data pointer for this NDArray type")
    
    def stride(self, dim=None):
        """Return stride information (in elements, not bytes)
        
        Args:
            dim (int, optional): Specify dimension
            
        Returns:
            int or tuple: stride value or stride tuple
        """
        if hasattr(self.data, 'strides'):
            # CUDATensor has strides attribute
            strides = self.data.strides
        else:
            # Calculate stride
            strides = []
            stride = 1
            for i in reversed(self.shape):
                strides.insert(0, stride)
                stride *= i
            strides = tuple(strides)
        
        if dim is None:
            return strides
        else:
            return strides[dim]
    
    def element_size(self):
        """Return bytes per element
        
        Returns:
            int: Element size (bytes)
        """
        if hasattr(self.data, 'itemsize'):
            return self.data.itemsize
        else:
            # Calculate from dtype
            import numpy as np
            return np.dtype(self._dtype).itemsize
    
    @property
    def is_cuda(self):
        """Check if on CUDA device
        
        Returns:
            bool: True if on CUDA device
        """
        return hasattr(self._device, 'name') and self._device.name == "cuda"

def array(a, dtype=genesis.float32, device=None):
    """ Convenience methods to match numpy a bit more closely."""
    dtype = genesis.float32 if dtype is None else dtype
    return NDArray(a, device=device, dtype=dtype)


def empty(shape, dtype=genesis.float32, device=None):
    device = device if device is not None else default_device()
    return device.empty(shape, dtype)


def full(shape, fill_value, dtype=genesis.float32, device=None):
    device = device if device is not None else default_device()
    return device.full(shape, fill_value, dtype)


def broadcast_to(array, new_shape):
    return array.broadcast_to(new_shape)

def swapaxes(array, x, y):
    new_shape = list(range(len(array.shape)))
    if x < 0:
        x = x + len(new_shape)
    if y < 0:
        y = y + len(new_shape)
    new_shape[x] = y
    new_shape[y] = x
    return array.permute(tuple(new_shape))

def transpose(array, axis):
    x, y = axis
    new_shape = list(range(len(array.shape)))
    if x < 0:
        x = x + len(new_shape)
    if y < 0:
        y = y + len(new_shape)
    new_shape[x] = y
    new_shape[y] = x
    return array.permute(tuple(new_shape))

def norm_axis(a, axis):
    if type(axis) is int:
        axis = (axis,)
    new_axis = []
    for ax in axis:
        if ax < 0:
            ax = ax + len(a.shape)
        new_axis.append(ax)
    return tuple(new_axis)

def sum(a, axis=None, keepdims=False):
    if type(axis) is int:
        axis = (axis, )
    if axis is None:
        return a.sum(axis=axis, keepdims=keepdims)
    axis = norm_axis(a, axis)
    axis = tuple(sorted(list(axis)))
    pre = 0
    for ax in axis:
        if keepdims:
            a = a.sum(axis=ax, keepdims=keepdims)
        else:
            a = a.sum(axis=ax - pre, keepdims=keepdims)
        pre += 1
    return a

def reduce_sum(a, axis=None, keepdims=False):
    return sum(a, axis=axis, keepdims=keepdims)

def max(a, axis=None, keepdims=False):
    if type(axis) is int:
        axis = (axis, )
    if axis is None:
        return a.max(axis=axis, keepdims=keepdims)
    axis = norm_axis(a, axis)
    axis = tuple(sorted(list(axis)))
    pre = 0
    for ax in axis:
        if keepdims:
            a = a.max(axis=ax, keepdims=keepdims)
        else:
            a = a.max(axis=ax - pre, keepdims=keepdims)
        pre += 1
    return a

def reshape(array, new_shape):
    if -1 in new_shape:
        unknown_dim = -1
        known_dim_product = 1
        for dim in new_shape:
            if dim != -1:
                known_dim_product *= dim
            else:
                unknown_dim = dim
        inferred_dim = int(np.prod(array.shape) / known_dim_product)
        new_shape = tuple(inferred_dim if dim == -1 else dim for dim in new_shape)
    return array.reshape(new_shape)

def expand(array, new_shape):
    return array.expand(new_shape)

def negative(array):
    return -array

def divide(a, b, dtype):
    return a.device.truediv(a, b)

def power(a, b):
    return a ** b

def sin(a):
    return a.sin()

def cos(a):
    return a.cos()

def log(a):
    return a.log()

def exp(a):
    return a.exp()

def matmul(a, b):
    return a.matmul(b)

def maximum(a, b):
    return a.maximum(b)

def diag(a):
    return a.diag()

def triu(a, k=0):
    return a.triu(k=k)

def sqrt(a):
    return a.sqrt()

def split(a, cnt, dim=None):
    return a.split(cnt, dim=dim)

def view(a, new_shape):
    return a.view(new_shape)

def cat(arrays, dim=0):
    """
    Concatenate arrays along specified dimension.
    
    Args:
        arrays: List of NDArray objects to concatenate
        dim: Dimension along which to concatenate
        
    Returns:
        NDArray: Concatenated array
    """
    if not arrays:
        raise ValueError("Cannot concatenate empty list of arrays")
    
    # Extract underlying data objects
    data_arrays = []
    for arr in arrays:
        if hasattr(arr.data, 'data'):
            data_arrays.append(arr.data.data)
        else:
            data_arrays.append(arr.data)
    
    # Use device's cat implementation
    result_data = arrays[0].device.cat(data_arrays, dim=dim)
    
    # Wrap in NDArray
    result = NDArray.__new__(NDArray)
    result._device = arrays[0].device
    result._dtype = arrays[0].dtype
    result.data = result_data
    return result
