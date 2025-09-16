"""
CPU storage backend for Genesis framework.

This module provides CPU tensor storage using PyTorch as the underlying engine,
extending PyTorch tensors with additional functionality needed by Genesis.
"""

import torch
import numpy as np
from typing import Union, Tuple, Optional, Any
from .base import Storage


class CPUStorage(torch.Tensor):
    """CPU tensor storage extending PyTorch tensors.
    
    Note: Inherits from torch.Tensor only due to metaclass conflicts.
    Implements Storage interface methods manually.
    """
    
    def __new__(cls, data: Union[torch.Tensor, np.ndarray, list]):
        """Create a new CPUStorage from data.
        
        Args:
            data: Input data as torch.Tensor, numpy array, or list
        """
        if isinstance(data, torch.Tensor):
            # Ensure tensor is on CPU
            if data.is_cuda:
                data = data.cpu()
            return data.as_subclass(cls)
        elif isinstance(data, np.ndarray):
            tensor = torch.from_numpy(data)
            return tensor.as_subclass(cls)
        else:
            # Handle lists and other array-like objects
            tensor = torch.tensor(data)
            return tensor.as_subclass(cls)
    
    def fill(self, value):
        """Fill storage with a constant value."""
        self.fill_(value)
        return self
    
    def to_numpy(self) -> np.ndarray:
        """Convert to numpy array."""
        # Handle different dtypes properly
        if super().dtype == torch.bfloat16:
            # bfloat16 is not supported by numpy, convert to float32
            return torch.Tensor.numpy(super().float().detach().cpu())
        return torch.Tensor.numpy(super().detach().cpu())
    
    def numpy(self) -> np.ndarray:
        """Convert to numpy array (PyTorch-compatible method)."""
        return self.to_numpy()
    
    def clone(self):
        """Create a deep copy of the storage."""
        return super().clone().as_subclass(CPUStorage)
    
    def contiguous(self):
        """Return contiguous version of storage."""
        if self.is_contiguous():
            return self
        return super().contiguous().as_subclass(CPUStorage)
    
    def is_contiguous(self) -> bool:
        """Check if storage is contiguous in memory."""
        return super().is_contiguous()
    
    @property 
    def size_bytes(self) -> int:
        """Return size in bytes."""
        return self.numel() * self.element_size()
    
    @property
    def dtype(self):
        """Return Genesis DType for this storage."""
        from genesis.dtypes import float32, float16, float64, int32, int64, int16, int8, uint8, bool as genesis_bool, bfloat16
        
        torch_dtype = super().dtype
        dtype_map = {
            torch.float32: float32,
            torch.float16: float16,
            torch.float64: float64,
            torch.int32: int32,
            torch.int64: int64,
            torch.int16: int16,
            torch.int8: int8,
            torch.uint8: uint8,
            torch.bool: genesis_bool,
            torch.bfloat16: bfloat16
        }
        return dtype_map.get(torch_dtype, float32)
    
    @classmethod
    def from_numpy(cls, array: np.ndarray, dtype=None):
        """Create CPUStorage from numpy array.
        
        Args:
            array: Input numpy array
            dtype: Optional dtype for type conversion
            
        Returns:
            CPUStorage instance
        """
        if dtype is None:
            return cls(array)
        
        # Handle dtype conversion using torch
        torch_tensor = torch.from_numpy(array)
        if dtype.name == 'bfloat16':
            torch_tensor = torch_tensor.to(torch.bfloat16)
        elif dtype.name == 'float16':
            torch_tensor = torch_tensor.to(torch.float16)
        elif dtype.name == 'float32':
            torch_tensor = torch_tensor.to(torch.float32)
        # Add other dtypes as needed
        
        return cls(torch_tensor)
    
    @classmethod
    def zeros(cls, shape: Tuple[int, ...], dtype=None):
        """Create zero-initialized CPUStorage.
        
        Args:
            shape: Shape of the tensor
            dtype: Data type (optional)
            
        Returns:
            CPUStorage filled with zeros
        """
        tensor = torch.zeros(shape, dtype=dtype)
        return tensor.as_subclass(cls)
    
    @classmethod
    def ones(cls, shape: Tuple[int, ...], dtype=None):
        """Create one-initialized CPUStorage.
        
        Args:
            shape: Shape of the tensor
            dtype: Data type (optional)
            
        Returns:
            CPUStorage filled with ones
        """
        tensor = torch.ones(shape, dtype=dtype)
        return tensor.as_subclass(cls)
    
    @classmethod
    def empty(cls, shape: Tuple[int, ...], dtype=None):
        """Create uninitialized CPUStorage.
        
        Args:
            shape: Shape of the tensor
            dtype: Data type (Genesis dtype name string or torch dtype)
            
        Returns:
            Uninitialized CPUStorage
        """
        # Convert Genesis dtype name to torch dtype if needed
        if isinstance(dtype, str):
            dtype_map = {
                "float32": torch.float32,
                "float16": torch.float16, 
                "float64": torch.float64,
                "int32": torch.int32,
                "int64": torch.int64,
                "int16": torch.int16,
                "int8": torch.int8,
                "uint8": torch.uint8,
                "bool": torch.bool,
                "bfloat16": torch.bfloat16
            }
            dtype = dtype_map.get(dtype, torch.float32)
        
        # Handle nested tuple shape - torch.empty expects shape to be unpacked
        if isinstance(shape, tuple) and len(shape) == 1 and isinstance(shape[0], tuple):
            tensor = torch.empty(shape[0], dtype=dtype)
        else:
            tensor = torch.empty(shape, dtype=dtype)
        return tensor.as_subclass(cls)
    
    def to_cuda(self):
        """Transfer storage to CUDA device.
        
        Returns:
            CUDAStorage instance
        """
        from .cuda import CUDAStorage
        cuda_tensor = self.cuda()
        return CUDAStorage(cuda_tensor)