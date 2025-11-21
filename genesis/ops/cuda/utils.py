"""
Utility functions for GPU operations.
"""
import operator
import numpy as np
from functools import reduce
import genesis
from genesis.backends.cuda import CUDAStorage


def prod(x: list[int]):
    """
    Product of all elements in x.
    """
    return reduce(operator.mul, x, 1)


def broadcast_shapes(shape1, shape2):
    """
    Compute the broadcasted shape of two tensors (NumPy broadcasting rules).
    """
    # Reverse shapes to align from the right (trailing dimensions)
    shape1_rev = list(reversed(shape1))
    shape2_rev = list(reversed(shape2))

    # Pad shorter shape with 1s to make them the same length
    max_ndim = max(len(shape1_rev), len(shape2_rev))
    while len(shape1_rev) < max_ndim:
        shape1_rev.append(1)
    while len(shape2_rev) < max_ndim:
        shape2_rev.append(1)

    # Compute broadcasted shape from right to left
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

    # Reverse back to get final shape
    return tuple(reversed(result_shape_rev))


def from_numpy(data, device_id=0, dtype=None):
    """
    Create CUDAStorage from numpy array.
    """
    np_dtype = None
    if dtype is None or dtype == genesis.float32:
        np_dtype = np.float32
    elif dtype == genesis.float16:
        np_dtype = np.float16
    elif dtype == genesis.bfloat16:
        # bfloat16 is not natively supported in numpy, use float32 for now
        np_dtype = np.float32
    
    if np_dtype and data.dtype != np_dtype:
        data = data.astype(np_dtype)
    
    tensor = CUDAStorage(data.shape, dtype=data.dtype)
    tensor.from_numpy(data)
    return tensor


def from_tensor(data, device_id=0, dtype=None):
    """
    Create CUDAStorage from tensor.
    """
    # If input is already CUDAStorage, return as is
    if isinstance(data, CUDAStorage):
        return data
    
    # Convert PyTorch tensor to CUDAStorage
    if hasattr(data, 'numpy'):  # PyTorch tensor
        numpy_array = data.detach().cpu().numpy()
        return from_numpy(numpy_array)
    
    # Convert other tensor types to CUDAStorage
    # For other types, try to convert to numpy first
    numpy_array = np.asarray(data)
    return from_numpy(numpy_array)


def clone(data, device_id=0, dtype=None):
    """
    Create a deep copy of CUDAStorage data.
    """
    return data.clone()


def array(shape, device_id=0, dtype=None):
    """
    Create empty CUDAStorage array.
    """
    # Convert to string dtype for CUDAStorage compatibility
    if dtype is None or dtype == genesis.float32:
        str_dtype = "float32"
    elif dtype == genesis.float16:
        str_dtype = "float16"
    elif dtype == genesis.bfloat16:
        str_dtype = "bfloat16"
    elif dtype == "int32":
        str_dtype = "int32"
    elif dtype == "bool":
        str_dtype = "bool"
    else:
        str_dtype = str(dtype)  # Convert DType to string
    
    return CUDAStorage(shape, dtype=str_dtype)