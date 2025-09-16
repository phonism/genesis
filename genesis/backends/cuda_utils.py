"""
CUDA utility functions to avoid circular imports.

This module provides factory functions for creating CUDA storage objects
without importing CUDAStorage directly.
"""

import numpy as np

# This will be set by cuda.py to avoid circular import
_cuda_storage_factory = None

def set_cuda_storage_factory(factory_func):
    """Set the CUDAStorage factory function."""
    global _cuda_storage_factory
    _cuda_storage_factory = factory_func

def empty(shape, dtype=np.float32):
    """Create empty CUDAStorage tensor."""
    if _cuda_storage_factory is None:
        raise RuntimeError("CUDAStorage factory not initialized")
    return _cuda_storage_factory(shape, dtype)

def zeros(shape, dtype=np.float32):
    """Create zero-filled CUDAStorage tensor."""
    tensor = empty(shape, dtype)
    tensor.fill(0.0)
    return tensor

def ones(shape, dtype=np.float32):
    """Create ones-filled CUDAStorage tensor."""
    tensor = empty(shape, dtype)
    tensor.fill(1.0)
    return tensor