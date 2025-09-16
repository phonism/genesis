"""
Storage abstraction - clean unified storage
"""
import math
import numpy as np
from typing import Optional, Union
from genesis.dtypes import DType, default_dtype, get_dtype
from genesis.device import Device, device as make_device, cpu, cuda
from genesis.backends.cpu import CPUStorage
from genesis.backends.cuda import CUDAStorage

class Storage:
    """
    Unified storage container - clean PyTorch style
    """
    
    def __init__(self, backend_storage, device: Device):
        """
        Initialize storage with backend storage

        Args:
            backend_storage: CPUStorage or CUDAStorage
            device: Device location
        """
        self._backend = backend_storage
        self.device = device

        # Handle PyTorch tensor case (from CUDA to CPU conversion)
        if hasattr(backend_storage, 'dtype') and str(type(backend_storage)) == "<class 'torch.Tensor'>":
            self.dtype = str(backend_storage.dtype).replace('torch.', '')
            self._size = backend_storage.numel()
            self.size_bytes = backend_storage.numel() * backend_storage.element_size()
        else:
            # Handle regular backend storage
            self.dtype = backend_storage.dtype
            self._size = backend_storage.size
            self.size_bytes = backend_storage.size_bytes  # Get from backend directly
    
    @property
    def numel(self):
        """Total number of elements"""
        return self._size

    def element_size(self) -> int:
        """Return bytes per element"""
        return self._backend.element_size()

    @classmethod 
    def allocate(cls, size: Union[int, tuple], dtype: Optional[DType] = None, device: Optional[Device] = None) -> 'Storage':
        """Allocate new storage space - supports both size and shape"""
        if dtype is None:
            dtype = default_dtype
        if device is None:
            device = cpu
        
        # Handle both size (int) and shape (tuple) parameters
        if isinstance(size, int):
            shape = (size,)
        else:
            shape = size
        
        if device.is_cpu():
            # Use CPUStorage.empty() with correct shape
            backend = CPUStorage.empty(shape, dtype=dtype.name)
        elif device.is_cuda():
            backend = CUDAStorage(shape, dtype)
        else:
            raise NotImplementedError(f"Device not supported: {device}")
        
        return cls(backend, device)
    
    @classmethod
    def make_storage(cls, numpy_data, dtype: Optional[DType] = None, device: Optional[Device] = None) -> 'Storage':
        """Create storage from numpy data - PyTorch compatible API"""
        if dtype is None:
            dtype = default_dtype
        if device is None:
            device = cpu
        
        # Ensure input is numpy array
        if not isinstance(numpy_data, np.ndarray):
            raise ValueError(f"Storage.make_storage expects numpy array, got {type(numpy_data)}")
        
        if device.is_cpu():
            backend = CPUStorage.from_numpy(numpy_data, dtype=dtype)
            return cls(backend, device)
        elif device.is_cuda():
            # Create CUDA storage directly from numpy
            shape = numpy_data.shape if numpy_data.shape else (numpy_data.size,)
            backend = CUDAStorage(shape, dtype.name)
            backend.from_numpy(numpy_data)
            return cls(backend, device)
        else:
            raise NotImplementedError(f"Device not supported: {device}")
    def __getitem__(self, index):
        """Index access - only for CPU storage"""
        if not self.device.is_cpu():
            raise RuntimeError("Cannot index GPU storage directly")
        return self._backend[index]
    
    def __setitem__(self, index, value):
        """Index assignment - only for CPU storage"""
        if not self.device.is_cpu():
            raise RuntimeError("Cannot assign to GPU storage directly")
        self._backend[index] = value
    
    def cpu(self) -> 'Storage':
        """Move storage to CPU"""
        return self.to(cpu)
    
    def cuda(self, device=None) -> 'Storage':
        """Move storage to CUDA"""
        if device is None:
            target_device = cuda
        else:
            target_device = make_device(f'cuda:{device}')
        return self.to(target_device)
    
    def to(self, target_device: Union[str, Device]) -> 'Storage':
        """Move storage to different device"""
        if isinstance(target_device, str):
            target_device = make_device(target_device)
        # If already a Device object, use directly
        
        if self.device == target_device:
            return self
        
        # CPU to CUDA
        if self.device.is_cpu() and target_device.is_cuda():
            cuda_backend = CUDAStorage.from_cpu_storage(self._backend, target_device.index)
            return Storage(cuda_backend, target_device)
        
        # CUDA to CPU
        elif self.device.is_cuda() and target_device.is_cpu():
            # Use CUDAStorage's cpu() method to get PyTorch tensor directly
            cpu_backend = self._backend.cpu()
            return Storage(cpu_backend, target_device)
        
        # CUDA to CUDA
        elif self.device.is_cuda() and target_device.is_cuda():
            cuda_backend = CUDAStorage(self._size, self.dtype, target_device.index)
            self._backend.copy_to_cuda(cuda_backend)
            return Storage(cuda_backend, target_device)
        
        # CPU to CPU
        elif self.device.is_cpu() and target_device.is_cpu():
            new_backend = self._backend.clone()
            return Storage(new_backend, target_device)
        
        else:
            raise NotImplementedError(f"Transfer from {self.device} to {target_device}")
    
    def clone(self) -> 'Storage':
        """Clone storage on same device"""
        cloned_backend = self._backend.clone()
        return Storage(cloned_backend, self.device)
    
    def contiguous(self, shape, stride, offset=0) -> 'Storage':
        """Create contiguous storage from strided data"""
        numel = math.prod(shape)
        
        # Check if already contiguous
        expected_stride = []
        s = 1
        for dim in reversed(shape):
            expected_stride.append(s)
            s *= dim
        expected_stride = tuple(reversed(expected_stride))
        
        if stride == expected_stride and offset == 0:
            return self  # Already contiguous
        
        # Create new contiguous storage
        new_storage = Storage.allocate(numel, self.dtype, self.device)
        
        # Call backend-specific strided copy
        self._backend.copy_strided_to_contiguous(shape, stride, offset, new_storage._backend)
        
        return new_storage
    
    def data_ptr(self):
        """Get pointer to data (PyTorch-style)"""
        return self._backend.data_ptr()
    
    @property
    def data(self):
        """Access to underlying data (for compatibility)"""
        if self.device.is_cpu():
            return self._backend.data
        else:  # CUDA
            return {'ptr': self._backend.ptr}
    
    def to_dtype(self, target_dtype: DType) -> 'Storage':
        """Convert storage to different dtype."""
        if self.dtype == target_dtype:
            return self
            
        # Use backend-specific dtype conversion
        if self.device.is_cuda():
            # CUDA: use Triton-based conversion
            converted_backend = self._backend.to(target_dtype)
        else:
            # CPU: use torch native conversion methods
            if target_dtype == get_dtype("float32"):
                converted_backend = self._backend.float()
            elif target_dtype == get_dtype("float16"):
                converted_backend = self._backend.half()
            elif target_dtype == get_dtype("float64"):
                converted_backend = self._backend.double()
            elif target_dtype == get_dtype("int64"):
                converted_backend = self._backend.long()
            elif target_dtype == get_dtype("int32"):
                converted_backend = self._backend.int()
            else:
                # Fallback for other types
                converted_backend = self._backend.to(target_dtype.torch_dtype)
        
        return Storage(converted_backend, self.device)

    def __repr__(self):
        return f"Storage(size={self._size}, dtype={self.dtype}, device={self.device})"