"""
Pure CUDA storage backend for GPU memory management and operations
Independent of PyTorch, using CUDA Python API directly
"""

try:
    from cuda.bindings import driver as cuda
    from cuda.bindings import nvrtc
except ImportError:
    from cuda import cuda, nvrtc
import numpy as np
from typing import Tuple, List, Optional, Union
import triton
import triton.language as tl
from functools import reduce
import operator
from math import prod
from genesis.dtypes import get_dtype, DType
from dataclasses import dataclass
from enum import Enum

# Optional torch import for cpu() method
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    TORCH_AVAILABLE = False

import os
_USE_TORCH_ALLOC = os.environ.get('GENESIS_USE_TORCH_ALLOCATOR', '0') == '1'

if _USE_TORCH_ALLOC:
    print("[Genesis] Using PyTorch allocator for CUDA memory")
    from genesis.backends.torch_allocator import get_torch_allocator
    _alloc = get_torch_allocator()
    allocate_memory = lambda size, stream=None: _alloc.allocate_memory(size)
    free_memory = _alloc.free_memory
    decrease_ref_count = _alloc.decrease_ref_count
    memory_stats = lambda: {}
    get_memory_manager = lambda: None
    increase_ref_count = lambda ptr: None
    trigger_gc = lambda: None
else:
    # Use lightweight caching allocator optimized for stable training
    from genesis.backends.cuda_memory import (
        allocate_memory, free_memory, memory_stats, get_memory_manager,
        increase_ref_count, decrease_ref_count, trigger_gc
    )
from genesis.backends.base import Storage
from genesis.dtypes import get_dtype
import genesis
from .cuda_error import check_cuda_error

# Import new indexing operations module
from genesis.backends.cuda_kernels import CUDAIndexingOps

# ============= Index Plan Architecture =============

class IndexKind(Enum):
    VIEW = "view"           # Pure view operation
    GATHER = "gather"       # Gather operation
    SCATTER = "scatter"     # Scatter operation  
    COPY = "copy"          # strided copy
    FILL = "fill"          # Fill operation
    MIXED_LIST_SLICE = "mixed_list_slice"  # Mixed list + slice indexing

@dataclass
class IndexPlan:
    """Unified index plan"""
    kind: IndexKind
    # Result metadata for view operations
    result_shape: Optional[Tuple[int, ...]] = None
    result_strides: Optional[Tuple[int, ...]] = None
    ptr_offset_bytes: int = 0
    # Advanced indexing metadata
    index_tensor: Optional['CUDAStorage'] = None
    needs_mask_compaction: bool = False
    # Mixed indexing metadata  
    column_index: Optional['CUDAStorage'] = None
    is_mixed_2d: bool = False
    # Mixed list + slice indexing metadata
    slices: Optional[Tuple] = None

# CUDA initialization handled by memory manager


# ============= Stream Management =============
_default_stream = None

def _ensure_stream():
    """Ensure default stream exists - use lightweight stream initialization"""
    global _default_stream
    if _default_stream is None:
        # Use default CUDA stream (stream 0) to avoid expensive memory manager initialization
        # This avoids the 7+ second initialization cost of get_memory_manager()
        _default_stream = 0  # Default CUDA stream
    return _default_stream


def _allocate_memory(nbytes):
    """Allocate GPU memory using optimized manager"""
    return allocate_memory(nbytes, _ensure_stream())

def _allocate_memory_stream_safe(nbytes, stream):
    """Stream-safe memory allocation"""
    return allocate_memory(nbytes, stream)

def _free_memory(ptr, nbytes):
    """Free GPU memory using optimized manager"""
    free_memory(ptr, nbytes, _ensure_stream())

# ============= User API (PyTorch-like) =============
# ---- helpers ----


# ============= Triton Kernels =============
@triton.jit  
def _fill_kernel(
    output_ptr, fill_value,
    n_elements: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Fill tensor with a constant value - optimized kernel"""
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    tl.store(output_ptr + offsets, fill_value, mask=mask)

# Strided fill kernels for non-contiguous tensors
@triton.jit
def _fill_strided_kernel(
    dst_ptr,
    value,
    total_numel,
    size0: tl.constexpr, stride0: tl.constexpr,
    size1: tl.constexpr, stride1: tl.constexpr,
    size2: tl.constexpr, stride2: tl.constexpr,
    size3: tl.constexpr, stride3: tl.constexpr,
    ndim: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    m = offs < total_numel
    lin = offs.to(tl.int64)
    dst_off = tl.zeros_like(lin)

    if ndim == 4:
        c3 = lin % size3
        lin //= size3
        c2 = lin % size2
        lin //= size2
        c1 = lin % size1
        lin //= size1
        c0 = lin
        dst_off = c0 * stride0 + c1 * stride1 + c2 * stride2 + c3 * stride3
    elif ndim == 3:
        c2 = lin % size2
        lin //= size2
        c1 = lin % size1
        lin //= size1
        c0 = lin
        dst_off = c0 * stride0 + c1 * stride1 + c2 * stride2
    elif ndim == 2:
        c1 = lin % size1
        c0 = lin // size1
        dst_off = c0 * stride0 + c1 * stride1
    else:
        dst_off = lin

    tl.store(dst_ptr + dst_off, value, mask=m)

@triton.jit
def _fill_strided_kernel_general(
    dst_ptr,
    sizes_ptr, strides_ptr,
    value,
    total_numel,
    ndim: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    m = offs < total_numel
    lin = offs.to(tl.int64)
    dst_off = tl.zeros_like(lin)

    for d in range(ndim - 1, -1, -1):
        sz = tl.load(sizes_ptr + d)
        st = tl.load(strides_ptr + d)
        coord = lin % sz
        lin //= sz
        dst_off += coord * st

    tl.store(dst_ptr + dst_off, value, mask=m)

class CUDAStorage(Storage):
    """Pure CUDA implementation of Tensor class"""
    
    def __init__(
        self, 
        shape: Tuple[int, ...], 
        dtype: str = "float32", 
        ptr: Optional[int] = None, 
        strides: Optional[Tuple[int, ...]] = None,
        base: Optional["CUDAStorage"] = None, 
        stream: Optional[int] = None
    ):
        # Normalize shape to tuple of integers
        if isinstance(shape, tuple) and len(shape) == 1 and isinstance(shape[0], tuple):
            self._shape = shape[0]
        elif isinstance(shape, (list, tuple)):
            # Convert list/tuple to tuple, flattening nested structures
            flat_dims = []
            for x in shape:
                if isinstance(x, (list, tuple)):
                    # Nested list/tuple - flatten it
                    flat_dims.extend(x)
                else:
                    flat_dims.append(x)
            self._shape = tuple(int(dim) for dim in flat_dims)
        else:
            # Single value
            self._shape = (int(shape),)
            
        self.base = base
        
        # Data type setup
        # Always expect dtype to be either a DType object or convertible
        self.dtype_obj = dtype if isinstance(dtype, DType) else get_dtype(dtype)
        self._dtype = self.dtype_obj.name
        self._numpy_dtype = self.dtype_obj.numpy_dtype
        self.itemsize = self.dtype_obj.itemsize
        
        # Use property size for nbytes calculation
        self.nbytes = self.size * self.itemsize
        
        # Compute strides (default to C-contiguous)
        if strides is None:
            self.strides = self._compute_strides(self._shape)
            self._is_contiguous = True  # If we computed strides, it's contiguous!
        else:
            self.strides = strides
            self._is_contiguous = None  # Will compute lazily when needed
        
        # Stream management
        self.alloc_stream = stream if stream is not None else _ensure_stream()
        self.last_stream = self.alloc_stream
        self.recorded_streams = {self.alloc_stream}
            
        # GPU memory allocation
        if ptr is None:
            # Memory manager handles CUDA initialization
            self.ptr = _allocate_memory_stream_safe(self.nbytes, self.alloc_stream)
            self.owns_memory = True
        else:
            self.ptr = ptr
            self.owns_memory = False
        
    
    
    def __del__(self):
        """Release GPU memory with reference counting"""
        if hasattr(self, 'owns_memory') and self.owns_memory and hasattr(self, 'ptr') and self.ptr:
            try:
                ptr_value = int(self.ptr)
                if ptr_value != 0:
                    stream = getattr(self, 'last_stream', None)

                    # Use reference counting for better memory pooling
                    if not decrease_ref_count(self.ptr, stream):
                        # If not in ref pool, fall back to direct free
                        nbytes = getattr(self, 'nbytes', 0)
                        free_memory(self.ptr, nbytes, stream)

                    self.ptr = None
            except Exception:
                # Silently ignore errors during shutdown
                # (logging during __del__ is unsafe when Python is shutting down)
                pass

    @classmethod
    def from_cpu_storage(cls, cpu_storage, device_index=0):
        """Create CUDAStorage from CPUStorage by copying data to GPU.

        Args:
            cpu_storage: CPUStorage object (a PyTorch tensor)
            device_index: CUDA device index

        Returns:
            CUDAStorage: New CUDA storage with copied data
        """
        # Get numpy data from CPU storage
        numpy_data = cpu_storage.detach().cpu().numpy()

        # Create CUDA storage with same shape and dtype
        cuda_storage = cls(
            shape=numpy_data.shape,
            dtype=str(cpu_storage.dtype).split('.')[-1]  # Convert torch dtype to string
        )

        # Copy data from numpy to GPU
        cuda_storage.from_numpy(numpy_data)

        return cuda_storage
    
    def record_stream(self, stream: int):
        """Record tensor usage on specified stream to prevent premature deallocation"""
        if hasattr(self, 'recorded_streams'):
            self.recorded_streams.add(stream)
            self.last_stream = stream
    
    def share_memory_(self):
        """Enable memory sharing by increasing reference count"""
        if hasattr(self, 'ptr') and self.ptr:
            increase_ref_count(self.ptr)
            self._is_shared = True
        return self
    
    def is_shared(self) -> bool:
        """Check if memory is shared (has multiple references)"""
        return hasattr(self, '_is_shared') and self._is_shared
    
    def _compute_strides(self, shape: Tuple[int, ...]) -> Tuple[int, ...]:
        """Compute C-contiguous strides"""
        strides = []
        stride = 1
        for dim in reversed(shape):
            strides.append(stride)
            stride *= dim
        return tuple(reversed(strides))
    
    @property
    def shape(self) -> Tuple[int, ...]:
        """Return shape as tuple (BaseStorage interface)."""
        return self._shape
    
    @property
    def dtype(self) -> str:
        """Return dtype as string (BaseStorage interface)."""
        return self._dtype
    
    @property
    def size(self) -> int:
        """Total number of elements"""
        # Ensure we always return an integer, not a list
        result = 1
        for dim in self.shape:
            result *= int(dim)
        return result
    
    @property
    def ndim(self) -> int:
        """Number of dimensions"""
        return len(self.shape)
    
    def numel(self) -> int:
        """Total number of elements (same as size)"""
        return self.size
    
    @property
    def device(self):
        """Device property - returns cuda device"""
        return genesis.device("cuda")
    
    @property
    def size_bytes(self) -> int:
        """Return size in bytes"""
        dtype_obj = get_dtype(self._dtype)
        return self.size * dtype_obj.itemsize

    def element_size(self) -> int:
        """Return bytes per element"""
        dtype_obj = get_dtype(self._dtype)
        return dtype_obj.itemsize
    
    @property
    def data(self):
        """Data property - returns self for compatibility"""
        return self
    
    # ============= Index Parsing Layer (MVP Architecture) =============
    
    
    def is_contiguous(self) -> bool:
        """Check if tensor has contiguous memory - optimized with caching"""
        if self._is_contiguous is None:
            # Compute only once and cache the result
            expected_strides = self._compute_strides(self.shape)
            self._is_contiguous = (self.strides == expected_strides)
        return self._is_contiguous
    
    def contiguous(self) -> 'CUDAStorage':
        """Return contiguous version of tensor"""
        if self.is_contiguous():
            return self
            
        # Create new contiguous tensor
        new_tensor = CUDAStorage(self.shape, self.dtype_obj)
        
        # Use Triton kernel to copy data
        copy_strided_kernel(self, new_tensor)
        return new_tensor
    
    def clone(self) -> 'CUDAStorage':
        """Create a deep copy of the tensor."""
        # Create new tensor with same shape and dtype
        new_tensor = CUDAStorage(self.shape, self.dtype)

        # Copy data using Triton kernel
        copy_strided_kernel(self, new_tensor)
        return new_tensor

    def copy_strided_to_contiguous(self, shape: Tuple[int, ...], stride: Tuple[int, ...], offset: int, dst_storage: 'CUDAStorage'):
        """Copy strided tensor data to contiguous destination storage."""
        # Use existing copy_strided_kernel which handles strided -> contiguous copying
        # Create a temporary view of source tensor with the given shape, stride, offset
        temp_view = CUDAStorage(shape, self.dtype_obj, self.ptr + offset * self.itemsize, stride, base=self)

        # Copy from strided view to contiguous destination
        copy_strided_kernel(temp_view, dst_storage)
        return dst_storage
    
    def reshape(self, *args) -> 'CUDAStorage':
        """Reshape operation - supports both tuple and separate arguments.

        Examples:
            reshape((2, 3))     # tuple argument
            reshape(2, 3)       # separate arguments
        """
        # Handle both tuple and separate arguments
        if len(args) == 1 and isinstance(args[0], (tuple, list)):
            new_shape = list(args[0])
        else:
            new_shape = list(args)
        neg_idx = None
        total_size = 1

        for i, dim in enumerate(new_shape):
            if dim == -1:
                if neg_idx is not None:
                    raise ValueError("Only one dimension can be -1")
                neg_idx = i
            else:
                total_size *= dim

        if neg_idx is not None:
            new_shape[neg_idx] = self.size // total_size

        new_shape = tuple(new_shape)

        # Verify size matches
        if reduce(operator.mul, new_shape, 1) != self.size:
            raise ValueError(f"Cannot reshape array of size {self.size} into shape {new_shape}")

        # If contiguous memory, can create new view directly
        if self.is_contiguous():
            return CUDAStorage(new_shape, self.dtype_obj, self.ptr, None, base=self)
        else:
            # Need to make contiguous first
            contig = self.contiguous()
            # CRITICAL FIX: Keep reference to contig to prevent memory deallocation
            return CUDAStorage(new_shape, self.dtype, contig.ptr, None, base=contig)
    
    def view(self, new_shape: Tuple[int, ...]) -> 'CUDAStorage':
        """View operation (requires contiguous memory)"""
        if not self.is_contiguous():
            raise RuntimeError("view() requires contiguous tensor")
        return self.reshape(new_shape)
    
    def expand(self, new_shape: Tuple[int, ...]) -> 'CUDAStorage':
        """Expand operation (broadcasting)"""
        if len(new_shape) != len(self.shape):
            raise ValueError("Expanded shape must have same number of dimensions")
        
        # Compute new strides
        new_strides = list(self.strides)
        for i, (old_dim, new_dim) in enumerate(zip(self.shape, new_shape)):
            if old_dim == 1 and new_dim > 1:
                # Broadcasting dimension, set stride to 0
                new_strides[i] = 0
            elif old_dim != new_dim:
                raise ValueError(f"Cannot expand dimension {i} from {old_dim} to {new_dim}")
        
        return CUDAStorage(new_shape, self.dtype_obj, self.ptr, tuple(new_strides), base=self)
    
    def permute(self, dims: Tuple[int, ...]) -> 'CUDAStorage':
        """Permute operation (transpose)"""
        if len(dims) != len(self.shape):
            raise ValueError("permute dimensions must match tensor dimensions")
            
        # Compute new shape and strides
        new_shape = tuple(self.shape[i] for i in dims)
        new_strides = tuple(self.strides[i] for i in dims)
        
        # Create new storage - it's likely not contiguous after permute
        result = CUDAStorage(new_shape, self.dtype_obj, self.ptr, new_strides, base=self)
        # Mark as likely non-contiguous (unless it's a no-op permute)
        if dims != tuple(range(len(dims))):
            result._is_contiguous = False
        return result
    
    def transpose(self, dim0: int, dim1: int) -> 'CUDAStorage':
        """Swap two dimensions"""
        dims = list(range(len(self.shape)))
        dims[dim0], dims[dim1] = dims[dim1], dims[dim0]
        return self.permute(tuple(dims))
    
    def unsqueeze(self, dim: int) -> 'CUDAStorage':
        """Add a dimension"""
        if dim < 0:
            dim = len(self.shape) + 1 + dim
            
        new_shape = list(self.shape)
        new_shape.insert(dim, 1)
        
        new_strides = list(self.strides)
        # New dimension's stride is the original stride at that position
        if dim < len(self.strides):
            new_strides.insert(dim, self.strides[dim] if dim < len(self.strides) else 1)
        else:
            new_strides.insert(dim, 1)
            
        return CUDAStorage(tuple(new_shape), self.dtype, self.ptr, tuple(new_strides), base=self)
    
    def squeeze(self, dim: Optional[int] = None) -> 'CUDAStorage':
        """Remove dimensions of size 1"""
        if dim is None:
            # Remove all dimensions of size 1
            new_shape = []
            new_strides = []
            for i, (s, st) in enumerate(zip(self.shape, self.strides)):
                if s != 1:
                    new_shape.append(s)
                    new_strides.append(st)
            return CUDAStorage(tuple(new_shape), self.dtype, self.ptr, tuple(new_strides), base=self)
        else:
            # Remove specified dimension
            if self.shape[dim] != 1:
                raise ValueError(f"Cannot squeeze dimension {dim} of size {self.shape[dim]}")
            new_shape = list(self.shape)
            new_strides = list(self.strides)
            del new_shape[dim]
            del new_strides[dim]
            return CUDAStorage(tuple(new_shape), self.dtype, self.ptr, tuple(new_strides), base=self)
    
    def broadcast_to(self, shape: Tuple[int, ...]) -> 'CUDAStorage':
        """Broadcast to specified shape"""
        # First expand dimensions
        if len(shape) > len(self.shape):
            # Add dimensions at the front
            for _ in range(len(shape) - len(self.shape)):
                self = self.unsqueeze(0)
        
        # Then expand
        return self.expand(shape)
    
    def to_numpy(self) -> np.ndarray:
        """Copy to CPU and convert to numpy array"""
        # Safety check
        if not self.ptr:
            raise RuntimeError("Invalid CUDA pointer - tensor may have been freed")
        
        if self.is_contiguous():
            # Contiguous memory, direct copy
            arr = np.empty(self.shape, dtype=self._numpy_dtype)
            result = cuda.cuMemcpyDtoH(arr, self.ptr, self.nbytes)
            check_cuda_error(result)
            return arr
        else:
            # Non-contiguous memory, need manual handling
            # For simple cases (like expand), we can generate numpy array directly
            if self._is_broadcasted():
                return self._expand_to_numpy()
            else:
                # General case, make contiguous first
                contig = self.contiguous()
                arr = np.empty(self.shape, dtype=self._numpy_dtype)
                result = cuda.cuMemcpyDtoH(arr, contig.ptr, contig.nbytes)
                check_cuda_error(result)
                return arr
    
    def _is_broadcasted(self) -> bool:
        """Check if tensor is broadcasted (some strides are 0)"""
        return any(s == 0 for s in self.strides)
    
    def _expand_to_numpy(self) -> np.ndarray:
        """Handle broadcast/expand cases, generate correct numpy array"""
        # Find dimensions with non-zero stride, read original data
        base_shape = []
        base_strides = []
        for i, (dim, stride) in enumerate(zip(self.shape, self.strides)):
            if stride > 0:
                base_shape.append(dim)
                base_strides.append(stride)
        
        if not base_shape:
            # All dimensions are broadcasted, only one element
            base_shape = [1]
            base_data = np.empty(1, dtype=self._numpy_dtype)
            
            # Add safety checks
            if not self.ptr:
                raise RuntimeError("Invalid CUDA pointer (None)")
            if self.itemsize <= 0:
                raise RuntimeError(f"Invalid itemsize: {self.itemsize}")
            
            # Ensure numpy array is C-contiguous for CUDA
            if not base_data.flags['C_CONTIGUOUS']:
                base_data = np.ascontiguousarray(base_data)
            
            result = cuda.cuMemcpyDtoH(base_data, self.ptr, self.itemsize)
            check_cuda_error(result)
            return np.broadcast_to(base_data.reshape([1] * len(self.shape)), self.shape)
        
        # Calculate size of original data
        base_size = reduce(operator.mul, base_shape, 1)
        base_data = np.empty(base_size, dtype=self._numpy_dtype)
        
        # Copy original data
        result = cuda.cuMemcpyDtoH(base_data, self.ptr, base_size * self.itemsize)
        check_cuda_error(result)
        
        # Reshape to original shape
        original_shape = []
        for i, (dim, stride) in enumerate(zip(self.shape, self.strides)):
            if stride > 0:
                original_shape.append(dim)
            else:
                original_shape.append(1)
        
        reshaped = base_data.reshape(original_shape)
        
        # Broadcast to target shape
        return np.broadcast_to(reshaped, self.shape)
    
    def from_numpy(self, arr: np.ndarray):
        """Copy data from numpy array"""
        if arr.size != self.size:
            raise ValueError("Size mismatch")
        
        # OPTIMIZATION: Always make numpy array contiguous first
        # This avoids slow element-wise copies
        if not arr.flags['C_CONTIGUOUS']:
            arr = np.ascontiguousarray(arr)
        
        if self.is_contiguous():
            # Fast path: destination is contiguous, single memcpy
            result = cuda.cuMemcpyHtoD(self.ptr, arr, self.nbytes)
            check_cuda_error(result)
        else:
            # Destination is non-contiguous
            # Create a temporary contiguous tensor on GPU, copy data, then reshape
            # This is still much faster than element-wise copy
            temp_tensor = CUDAStorage(tuple([self.size]), dtype=self.dtype)
            arr_flat = arr.flatten()
            result = cuda.cuMemcpyHtoD(temp_tensor.ptr, arr_flat, temp_tensor.nbytes)
            check_cuda_error(result)
            
            # Use optimized GPU kernel for contiguous-to-strided copy
            # This is MUCH faster than element-wise CPU-GPU transfers
            copy_strided_reverse_kernel(temp_tensor, self)
            
    @property
    def T(self) -> 'CUDAStorage':
        """Transpose (2D tensor)"""
        if len(self.shape) != 2:
            raise ValueError("T property only works for 2D tensors")
        return self.transpose(0, 1)
        
    
    def data_ptr(self):
        """Return integer address for Triton compatibility"""
        if self.ptr is None:
            raise RuntimeError("CUDAStorage pointer is None - tensor may have been freed")
        return int(self.ptr)
    
    @property
    def __cuda_array_interface__(self):
        """CUDA Array Interface for Triton compatibility"""
        if not self.ptr:
            raise RuntimeError(f"CUDAStorage has null pointer: {self.ptr}")
        
        # For contiguous tensors, strides can be None (like PyTorch)
        strides = None if self.is_contiguous() else tuple(s * self.itemsize for s in self.strides)
        
        interface = {
            'shape': self.shape,
            'typestr': self._get_typestr(),
            'data': (int(self.ptr), False),  # (pointer, read_only)
            'version': 2,  # Use version 2 like PyTorch
            'strides': strides,  # None for contiguous, actual strides for non-contiguous
        }
        
        return interface
    
    def _get_typestr(self):
        """Get numpy-style type string for CUDA array interface"""
        if self.dtype == "float32":
            return '<f4'
        elif self.dtype == "float16":
            return '<f2'
        elif self.dtype == "bfloat16":
            return '<f2'  # Treat as float16 for now
        elif self.dtype == "bool":
            return '|b1'  # bool is 1 byte
        else:
            return '<f4'  # Default to float32
    
    def numpy_dtype(self):
        """Return numpy dtype for creating new tensors"""
        return self._numpy_dtype
    
    def _get_triton_compatible_dtype(self, np_dtype):
        """Convert numpy dtype to Triton-compatible string format"""
        if np_dtype == np.float32:
            return "float32"
        elif np_dtype == np.float16:
            return "float16"
        elif np_dtype == np.int32:
            return "int32"
        elif np_dtype == np.int64:
            return "int64"
        elif np_dtype == np.float64:
            return "float64"
        elif np_dtype == np.bool_ or np_dtype == bool:
            return "bool"
        else:
            return "float32"  # default
    
    def _get_numpy_dtype(self, dtype_str=None):
        """Convert string dtype to numpy dtype"""
        # Use _numpy_dtype set during initialization from DType system
        if dtype_str is None:
            return self._numpy_dtype
        else:
            # For explicit dtype_str, use DType system
            dtype_obj = get_dtype(dtype_str)
            return dtype_obj.numpy_dtype
    
    def fill_(self, value):
        """Fill tensor with a constant value (in-place) using GPU kernel"""
        if self.is_contiguous():
            # Fast path for contiguous tensors
            n_elements = self.size
            grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]), )
            _fill_kernel[grid](
                self, float(value), n_elements, BLOCK_SIZE=1024
            )
        else:
            # Smart strided path - no contiguous() needed!
            n_elements = self.size
            ndim = len(self.shape)

            if ndim <= 4:
                # Use optimized strided kernel for common cases
                grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK"]), )

                # Pad shapes and strides to 4D
                shape_padded = list(self.shape) + [1] * (4 - ndim)
                strides_padded = list(self.strides) + [1] * (4 - ndim)

                _fill_strided_kernel[grid](
                    self, float(value), n_elements,
                    shape_padded[0], strides_padded[0],
                    shape_padded[1], strides_padded[1],
                    shape_padded[2], strides_padded[2],
                    shape_padded[3], strides_padded[3],
                    ndim, BLOCK=1024
                )
            else:
                # Use general strided kernel for high-dimensional tensors
                grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK"]), )

                # Create GPU arrays for sizes and strides
                sizes_gpu = from_numpy(np.array(self.shape, dtype=np.int64))
                strides_gpu = from_numpy(np.array(self.strides, dtype=np.int64))

                _fill_strided_kernel_general[grid](
                    self, sizes_gpu, strides_gpu,
                    float(value), n_elements, ndim, BLOCK=1024
                )
        return self

    def fill(self, value):
        """Fill tensor with a constant value (in-place) using GPU kernel"""
        # Special optimization for zeros on contiguous tensors
        if value == 0.0 and self.is_contiguous():
            result = cuda.cuMemsetD8(self.ptr, 0, self.nbytes)
            check_cuda_error(result)
            return self

        # For all other cases, use the optimized fill_ method
        return self.fill_(value)
    
    
    def _preprocess_index_key(self, key):
        """Convert lists in index keys to tensors before passing to indexing operations"""
        if isinstance(key, CUDAStorage):
            # If it's already a CUDAStorage, return as-is
            return key
        elif isinstance(key, list):
            # Check if list contains CUDAStorage objects (advanced indexing)
            if key and isinstance(key[0], CUDAStorage):
                # It's a list of CUDAStorage tensors - convert to tuple for multi-dimensional indexing
                return tuple(key)
            # Otherwise convert list to tensor
            key_array = np.array(key)
            if key_array.dtype == np.bool_ or key_array.dtype == bool:
                return from_numpy(key_array.astype(np.bool_))
            else:
                return from_numpy(key_array.astype(np.int64))

        elif isinstance(key, tuple):
            # Process each element in tuple
            processed_key = []
            for idx in key:
                if isinstance(idx, CUDAStorage):
                    # If it's already a CUDAStorage tensor, keep it as-is
                    processed_key.append(idx)
                elif isinstance(idx, list):
                    key_array = np.array(idx)
                    if key_array.dtype == np.bool_ or key_array.dtype == bool:
                        processed_key.append(from_numpy(key_array.astype(np.bool_)))
                    else:
                        processed_key.append(from_numpy(key_array.astype(np.int64)))
                else:
                    processed_key.append(idx)
            return tuple(processed_key)

        else:
            # Return key as-is (int, slice, tensor, etc.)
            return key
    
    def __getitem__(self, key):
        """GPU native getitem implementation - using extracted indexing operations"""
        # Pre-process lists to tensors
        key = self._preprocess_index_key(key)
        plan = CUDAIndexingOps.parse_index(self, key)
        return CUDAIndexingOps.execute_getitem(self, plan)
    
    
    
    def _copy_data_to_view(self, target_view, value):
        """Copy data to target view"""
        if isinstance(value, CUDAStorage):
            # Tensor to Tensor copy
            if target_view.shape != value.shape:
                raise ValueError(f"Shape mismatch: {target_view.shape} vs {value.shape}")
            
            if target_view.is_contiguous() and value.is_contiguous():
                # Both contiguous: direct memcpy
                result = cuda.cuMemcpyDtoD(target_view.ptr, value.ptr, target_view.nbytes)
                check_cuda_error(result)
            else:
                # GPU strided copy using cuMemcpy2D
                self._gpu_strided_copy(target_view, value)
        elif isinstance(value, (int, float)):
            # Scalar assignment: use fill operation
            self._fill_view(target_view, value)
        else:
            raise TypeError(f"Cannot assign {type(value)} to CUDAStorage")
    
    def _gpu_strided_copy(self, target_view, value):
        """GPU-only strided copy without CPU fallback (kernel-based)."""
        if target_view.shape != value.shape:
            raise ValueError(f"Shape mismatch: {target_view.shape} vs {value.shape}")
        # Keep fast 2D path via cuMemcpy2D
        if target_view.ndim == 2 and value.ndim == 2:
            self._gpu_2d_strided_copy(target_view, value)
            return
        # General path: single Triton kernel
        copy_strided_to_strided_kernel(value, target_view)
    
    def _gpu_1d_strided_copy(self, target_view, value):
        """GPU 1D strided copy (kernel)."""
        copy_strided_to_strided_kernel(value, target_view)
    
    def _gpu_flattened_copy(self, target_view, value):
        """Unified path via strided kernel."""
        copy_strided_to_strided_kernel(value, target_view)
    
    def _gpu_strided_copy_fast(self, target_view, value):
        """Fast GPU strided copy for non-contiguous views"""
        # For now, prioritize correctness over performance
        # Use existing working 1D/2D strided copy when possible
        
        if target_view.ndim == 1 and value.ndim == 1:
            # 1D case: use existing fast implementation
            self._gpu_1d_strided_copy(target_view, value)
        elif target_view.ndim == 2 and value.ndim == 2:
            # 2D case: use existing fast implementation
            self._gpu_2d_strided_copy(target_view, value)
        else:
            # Higher dimensions: use safe element-wise copy for small tensors
            if target_view.size <= 1000:
                self._gpu_elementwise_copy(target_view, value)
            else:
                # For large tensors, try to decompose into 2D operations
                self._gpu_decompose_to_2d_copy(target_view, value)
    
    def _gpu_decompose_to_2d_copy(self, target_view, value):
        """Unified path via strided kernel."""
        copy_strided_to_strided_kernel(value, target_view)

    def _gpu_2d_strided_copy(self, target_view, value):
        """GPU 2D strided copy using cuMemcpy2D - restored from old architecture"""
        height, width = target_view.shape

        # Special case: if height=1, this is essentially a 1D copy operation
        if height == 1:
            # Flatten to 1D and use 1D copy
            target_1d = target_view.reshape((-1,))
            value_1d = value.reshape((-1,))
            self._gpu_1d_strided_copy(target_1d, value_1d)
            return

        # Calculate strides in bytes
        target_pitch = target_view.stride()[0] * target_view.itemsize
        value_pitch = value.stride()[0] * value.itemsize
        width_bytes = width * target_view.itemsize

        # Handle broadcasted tensors (stride = 0) - fallback to elementwise copy
        if value_pitch == 0 or any(s == 0 for s in value.stride()):
            # This is a broadcasted tensor, use elementwise copy instead
            return self._gpu_elementwise_copy(target_view, value)

        # Use cuMemcpy2D for efficient 2D copy
        copy_params = cuda.CUDA_MEMCPY2D()
        copy_params.srcMemoryType = cuda.CUmemorytype.CU_MEMORYTYPE_DEVICE
        copy_params.srcDevice = value.ptr
        copy_params.srcPitch = value_pitch

        copy_params.dstMemoryType = cuda.CUmemorytype.CU_MEMORYTYPE_DEVICE
        copy_params.dstDevice = target_view.ptr
        copy_params.dstPitch = target_pitch

        copy_params.WidthInBytes = width_bytes
        copy_params.Height = height

        result = cuda.cuMemcpy2D(copy_params)
        check_cuda_error(result)

    def _gpu_decompose_to_2d_copy(self, target_view, value):
        """Decompose high-dimensional copy into lower-dimensional operations - pixel-level copy from old arch"""
        # For non-contiguous tensors, we can't safely reshape to 2D
        # Instead, decompose recursively along dimensions

        if target_view.ndim == 3 and value.ndim == 3:
            # Handle 3D tensors by iterating over the first dimension
            for i in range(target_view.shape[0]):
                target_2d = target_view[i]
                value_2d = value[i]
                if target_2d.is_contiguous() and value_2d.is_contiguous():
                    # Both contiguous: direct copy
                    result = cuda.cuMemcpyDtoD(target_2d.ptr, value_2d.ptr, target_2d.nbytes)
                    check_cuda_error(result)
                else:
                    self._gpu_2d_strided_copy(target_2d, value_2d)

        elif target_view.ndim == 4 and value.ndim == 4:
            # Handle 4D tensors by iterating over the first dimension
            for i in range(target_view.shape[0]):
                target_3d = target_view[i]
                value_3d = value[i]
                # Recursively handle 3D case
                self._gpu_decompose_to_2d_copy(target_3d, value_3d)

        elif target_view.ndim > 4:
            # For higher dimensions, iterate over the first dimension
            for i in range(target_view.shape[0]):
                target_sub = target_view[i]
                value_sub = value[i]
                self._gpu_decompose_to_2d_copy(target_sub, value_sub)

        else:
            # For other cases, fallback to element-wise copy
            self._gpu_elementwise_copy(target_view, value)

    def _gpu_elementwise_copy(self, target_view, value):
        """Element-wise copy for small tensors - restored from old architecture"""
        value_flat = value.reshape((-1,))
        for i in range(value_flat.size):
            # Convert linear index to multi-dimensional indices
            indices = []
            remaining = i
            for dim_size in reversed(target_view.shape):
                indices.append(remaining % dim_size)
                remaining //= dim_size
            indices.reverse()

            # Set individual element
            target_view[tuple(indices)] = float(value_flat[i].to_numpy().item())

    def _gpu_lastdim_slice_copy(self, target_view, value):
        """Optimized copy for last-dimension slicing (e.g., x[..., :n])"""
        # Calculate number of "rows" (all dimensions except the last)
        num_rows = 1
        for i in range(len(target_view.shape) - 1):
            num_rows *= target_view.shape[i]
        
        # Size of each row in elements and bytes
        row_elements = target_view.shape[-1]
        row_bytes = row_elements * target_view.itemsize
        
        # Source and target strides for row beginnings  
        target_row_stride = target_view.stride()[-2] * target_view.itemsize if len(target_view.shape) > 1 else row_bytes
        value_row_stride = value.stride()[-2] * value.itemsize if len(value.shape) > 1 else row_bytes
        
        # Copy each row using direct CUDA memcpy
        for row in range(num_rows):
            # Convert CUDA pointers to integers for arithmetic
            target_row_ptr = int(target_view.ptr) + row * target_row_stride
            value_row_ptr = int(value.ptr) + row * value_row_stride
            
            result = cuda.cuMemcpyDtoD(target_row_ptr, value_row_ptr, row_bytes)
            check_cuda_error(result)

    def _fill_view(self, target_view, value):
        """Fill view with scalar value"""
        if target_view.is_contiguous():
            # Contiguous memory: memset or simple fill kernel
            if target_view.dtype == "float32" and value == 0.0:
                result = cuda.cuMemsetD8(target_view.ptr, 0, target_view.nbytes)
                check_cuda_error(result)
                return
            n_elements = target_view.size
            grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
            _fill_kernel[grid](target_view, float(value), n_elements, BLOCK_SIZE=1024)
            return
        # Non-contiguous: launch strided fill kernel
        ndim = len(target_view.shape)
        numel = target_view.size
        if numel == 0:
            return
        if numel < 1024:
            BLOCK = 256; num_warps = 2
        elif numel < 1024 * 1024:
            BLOCK = 512; num_warps = 4
        else:
            BLOCK = 1024; num_warps = 8
        grid = (triton.cdiv(numel, BLOCK),)
        if ndim <= 4:
            sizes = list(target_view.shape) + [1] * (4 - ndim)
            strides = list(target_view.stride()) + [0] * (4 - ndim)
            _fill_strided_kernel[grid](
                target_view, float(value), numel,
                size0=sizes[0], stride0=strides[0],
                size1=sizes[1], stride1=strides[1],
                size2=sizes[2], stride2=strides[2],
                size3=sizes[3], stride3=strides[3],
                ndim=ndim,
                BLOCK=BLOCK,
                num_warps=num_warps,
            )
        else:
            cache_key = (tuple(target_view.shape), tuple(target_view.stride()))
            if cache_key not in _metadata_cache:
                sizes_gpu = empty((ndim,), np.int64)
                strides_gpu = empty((ndim,), np.int64)
                _metadata_cache[cache_key] = (sizes_gpu, strides_gpu)
                s = np.array(target_view.shape, dtype=np.int64)
                st = np.array(target_view.stride(), dtype=np.int64)
                r = cuda.cuMemcpyHtoD(sizes_gpu.ptr, s, s.nbytes); check_cuda_error(r)
                r = cuda.cuMemcpyHtoD(strides_gpu.ptr, st, st.nbytes); check_cuda_error(r)
            sizes_gpu, strides_gpu = _metadata_cache[cache_key]
            _fill_strided_kernel_general[grid](
                target_view, sizes_gpu, strides_gpu,
                float(value), numel,
                ndim=ndim,
                BLOCK=BLOCK,
                num_warps=num_warps,
            )
    
    
    def _copy_to_view(self, target_view, value):
        """Copy value to target view efficiently on GPU"""
        if isinstance(value, CUDAStorage):
            # Direct GPU-to-GPU copy
            if target_view.shape != value.shape:
                raise ValueError(f"Shape mismatch: {target_view.shape} vs {value.shape}")
            
            # Use CUDA memcpy for contiguous tensors, or strided copy kernel
            if target_view.is_contiguous() and value.is_contiguous():
                # Direct memory copy
                result = cuda.cuMemcpyDtoD(target_view.ptr, value.ptr, target_view.nbytes)
                check_cuda_error(result)
            else:
                # GPU strided copy (single kernel when possible)
                self._gpu_strided_copy(target_view, value)
        
        elif isinstance(value, np.ndarray):
            # Copy from numpy array
            if target_view.shape != value.shape:
                value = np.broadcast_to(value, target_view.shape)
            target_view.from_numpy(value)
        
        else:
            # Scalar or other - convert and copy
            value_np = np.array(value, dtype=target_view._numpy_dtype)
            if value_np.shape != target_view.shape:
                value_np = np.broadcast_to(value_np, target_view.shape)
            target_view.from_numpy(value_np)
    
    def __setitem__(self, key, value):
        """GPU native setitem implementation - using extracted indexing operations"""
        # Pre-process lists to tensors  
        key = self._preprocess_index_key(key)
        plan = CUDAIndexingOps.parse_index(self, key)
        return CUDAIndexingOps.execute_setitem(self, plan, value)
    
    def item(self):
        """Return the value of a single-element tensor as a Python scalar."""
        if self.size != 1:
            raise ValueError("only one element tensors can be converted to Python scalars")
        # Copy to CPU and get scalar value
        return self.to_numpy().item()
    
    def float(self):
        """Convert tensor to float32 type using GPU-native conversion"""
        return _convert_dtype_gpu(self, "float32")
    
    def half(self):
        """Convert tensor to float16 type using GPU-native conversion"""
        return _convert_dtype_gpu(self, "float16")
    
    def long(self):
        """Convert tensor to int64 type using GPU-native conversion"""
        return _convert_dtype_gpu(self, "int64")
    
    def detach(self):
        """Detach tensor from computation graph (for PyTorch compatibility)"""
        return self  # CUDAStorage doesn't have gradients, so just return self
    
    def cpu(self):
        """Move tensor to CPU and convert to PyTorch tensor"""
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is not available. Cannot convert to CPU tensor.")
        np_data = self.to_numpy()
        
        # Handle read-only numpy arrays (e.g., from broadcast operations)
        # PyTorch requires writable tensors, so create a copy if needed
        if not np_data.flags.writeable:
            np_data = np_data.copy()
            
        return torch.from_numpy(np_data)
    
    def numpy(self):
        """Convert to numpy array (alias for to_numpy for PyTorch compatibility)"""
        return self.to_numpy()
    
    def stride(self, dim=None):
        """Get stride for specific dimension (PyTorch compatibility)"""
        if dim is None:
            return self.strides
        else:
            return self.strides[dim]
    
    

    def to(self, target_dtype):
        """Convert tensor to specified dtype"""
        # Handle different input types and convert to our string format
        target_dtype_str = None
        
        # Handle torch.dtype objects
        if str(target_dtype).startswith('torch.'):
            if str(target_dtype) == 'torch.float32':
                target_dtype_str = "float32"
            elif str(target_dtype) == 'torch.float16':
                target_dtype_str = "float16"
            elif str(target_dtype) == 'torch.int32':
                target_dtype_str = "int32"
            elif str(target_dtype) == 'torch.int64':
                target_dtype_str = "int64"
            elif str(target_dtype) == 'torch.float64':
                target_dtype_str = "float64"
            else:
                target_dtype_str = "float32"  # default
        elif isinstance(target_dtype, np.dtype):
            # Convert numpy dtype to string
            target_dtype_str = self._get_triton_compatible_dtype(target_dtype)
        elif isinstance(target_dtype, DType):
            # Genesis DType object
            target_dtype_str = target_dtype.name
        else:
            # Assume it's already string or convertible
            target_dtype_str = str(target_dtype)
        
        if self.dtype == target_dtype_str:
            return self  # Already correct type
        
        # Convert dtype using GPU-native method
        return _convert_dtype_gpu(self, target_dtype_str)
    
    
    def split(self, split_size_or_sections, dim=None):
        """Split tensor along specified dimension
        
        Args:
            split_size_or_sections: Either a single int (split into chunks of this size)
                                  or a list of ints (split into chunks of these sizes)
            dim: Dimension to split along
        """
        if dim is None:
            dim = 0
        if dim < 0:
            dim = len(self.shape) + dim
            
        if isinstance(split_size_or_sections, int):
            # Split into chunks of equal size
            cnt = split_size_or_sections
            if self.shape[dim] % cnt != 0:
                raise ValueError(f"Tensor size {self.shape[dim]} in dimension {dim} is not divisible by {cnt}")
            
            chunk_size = self.shape[dim] // cnt
            result = []
            
            for i in range(cnt):
                # Create a slice for this chunk
                start_idx = i * chunk_size
                end_idx = start_idx + chunk_size
                
                # Calculate new shape
                new_shape = list(self.shape)
                new_shape[dim] = chunk_size
                
                # Calculate offset for the slice
                offset = start_idx * self.strides[dim] * self.itemsize
                new_ptr = int(self.ptr) + offset
                
                # Create new tensor for this chunk
                chunk = CUDAStorage(tuple(new_shape), dtype=self.dtype, ptr=new_ptr, strides=self.strides, base=self)
                result.append(chunk)
                
        elif isinstance(split_size_or_sections, (list, tuple)):
            # Split into chunks of specified sizes
            sizes = split_size_or_sections
            if sum(sizes) != self.shape[dim]:
                raise ValueError(f"Sum of split sizes {sum(sizes)} doesn't match tensor size {self.shape[dim]} in dimension {dim}")
            
            result = []
            start_idx = 0
            
            for size in sizes:
                # Calculate new shape
                new_shape = list(self.shape)
                new_shape[dim] = size
                
                # Calculate offset for the slice
                offset = start_idx * self.strides[dim] * self.itemsize
                new_ptr = int(self.ptr) + offset
                
                # Create new tensor for this chunk
                chunk = CUDAStorage(tuple(new_shape), dtype=self.dtype, ptr=new_ptr, strides=self.strides, base=self)
                result.append(chunk)
                
                start_idx += size
        else:
            raise TypeError(f"split_size_or_sections must be int or list/tuple, got {type(split_size_or_sections)}")
            
        return result
    
    @staticmethod
    def cat(tensors, dim=0):
        """
        Concatenate tensors along specified dimension - optimized GPU implementation
        
        Args:
            tensors: List of CUDAStorage tensors to concatenate
            dim: Dimension along which to concatenate (default: 0)
            
        Returns:
            CUDAStorage: Concatenated tensor
        """
        if not tensors:
            raise ValueError("Cannot concatenate empty list of tensors")
        
        if len(tensors) == 1:
            return tensors[0]  # No concatenation needed
        
        # All tensors must be CUDAStorage
        if not all(isinstance(t, CUDAStorage) for t in tensors):
            raise TypeError("All tensors must be CUDAStorage instances")
        
        # Get reference tensor properties
        first = tensors[0]
        dtype = first.dtype
        ndim = len(first.shape)
        
        # Normalize dimension
        if dim < 0:
            dim = ndim + dim
        if dim < 0 or dim >= ndim:
            raise ValueError(f"Dimension {dim} out of range for {ndim}D tensor")
        
        # Verify shape compatibility
        for i, tensor in enumerate(tensors[1:], 1):
            if len(tensor.shape) != ndim:
                raise ValueError(f"Tensor {i} has {len(tensor.shape)} dimensions, expected {ndim}")
            if tensor.dtype != dtype:
                raise ValueError(f"All tensors must have same dtype, got {tensor.dtype}, expected {dtype}")
            
            # Check all dimensions except concat dimension
            for j in range(ndim):
                if j != dim and tensor.shape[j] != first.shape[j]:
                    raise ValueError(f"All tensors must have same size in dimension {j}, "
                                   f"tensor {i} has {tensor.shape[j]}, expected {first.shape[j]}")
        
        # Calculate output shape
        output_shape = list(first.shape)
        output_shape[dim] = sum(t.shape[dim] for t in tensors)
        output_shape = tuple(output_shape)
        
        # Create output tensor
        result = CUDAStorage(output_shape, dtype=dtype)
        
        # Handle empty tensors in input
        non_empty_tensors = [t for t in tensors if t.shape[dim] > 0]
        if not non_empty_tensors:
            # All input tensors are empty along concat dimension
            return result
        
        # Optimized concatenation using direct memory copy
        current_offset = 0
        
        for tensor in non_empty_tensors:
            if not tensor.is_contiguous():
                tensor = tensor.contiguous()
            
            # Calculate copy parameters
            concat_size = tensor.shape[dim]
            if concat_size == 0:
                continue
            
            # Simple but effective approach: copy tensor data directly
            # Calculate number of elements to copy and where to place them
            
            if dim == 0:
                # Concatenating along first dimension - straightforward
                elements_to_copy = tensor.size
                src_ptr = int(tensor.ptr)
                dst_offset_bytes = current_offset * prod(output_shape[1:]) * result.itemsize
                dst_ptr = int(result.ptr) + dst_offset_bytes
                
                # Direct memory copy
                check_cuda_error(cuda.cuMemcpy(
                    dst_ptr,
                    src_ptr, 
                    elements_to_copy * tensor.itemsize
                ))
                
            elif dim == ndim - 1:
                # Concatenating along last dimension - use GPU-optimized path for 2 tensors
                if len(non_empty_tensors) == 2 and current_offset == 0:
                    # Special case: exactly 2 tensors, use optimized kernel
                    return _cat_last_dim_gpu(non_empty_tensors[0], non_empty_tensors[1])
                else:
                    # Fallback to row-by-row copy for complex cases
                    elements_per_row = tensor.shape[dim]  # Elements in concat dimension
                    num_rows = tensor.size // elements_per_row  # Number of rows
                    output_elements_per_row = output_shape[dim]  # Output row size
                    
                    src_ptr = int(tensor.ptr)
                    element_size = tensor.itemsize
                    
                    for row in range(num_rows):
                        src_row_ptr = src_ptr + row * elements_per_row * element_size
                        dst_row_ptr = int(result.ptr) + (row * output_elements_per_row + current_offset) * element_size
                        
                        check_cuda_error(cuda.cuMemcpy(
                            dst_row_ptr,
                            src_row_ptr,
                            elements_per_row * element_size
                        ))
            else:
                # For middle dimensions, use a simpler iterative approach
                # Copy slice by slice
                outer_size = prod(tensor.shape[:dim])
                inner_size = prod(tensor.shape[dim+1:])
                concat_size = tensor.shape[dim]
                
                src_ptr = int(tensor.ptr)
                element_size = tensor.itemsize
                
                for outer_idx in range(outer_size):
                    for concat_idx in range(concat_size):
                        for inner_idx in range(inner_size):
                            # Calculate source position
                            src_offset = (outer_idx * concat_size * inner_size + 
                                        concat_idx * inner_size + inner_idx) * element_size
                            
                            # Calculate destination position  
                            dst_offset = (outer_idx * output_shape[dim] * inner_size +
                                        (current_offset + concat_idx) * inner_size + 
                                        inner_idx) * element_size
                            
                            check_cuda_error(cuda.cuMemcpy(
                                int(result.ptr) + dst_offset,
                                src_ptr + src_offset,
                                element_size
                            ))
            
            current_offset += concat_size
        
        return result
    
    
    def __repr__(self):
        try:
            return f"CUDAStorage(shape={self.shape}, dtype={self.dtype}, ptr=0x{int(self.ptr):x})"
        except:
            return f"CUDAStorage(shape={self.shape}, dtype={self.dtype})"


# Utility functions: create tensors
def empty(shape: Tuple[int, ...], dtype: np.dtype = np.float32) -> CUDAStorage:
    """Create uninitialized tensor"""
    return CUDAStorage(shape, dtype)

def zeros(shape: Tuple[int, ...], dtype: np.dtype = np.float32) -> CUDAStorage:
    """Create tensor filled with zeros"""
    tensor = CUDAStorage(shape, dtype)
    result = cuda.cuMemsetD8(tensor.ptr, 0, tensor.nbytes)
    check_cuda_error(result)
    return tensor

def ones(shape: Tuple[int, ...], dtype: np.dtype = np.float32) -> CUDAStorage:
    """Create tensor filled with ones"""
    # Simplified implementation, should use kernel
    arr = np.ones(shape, dtype=dtype)
    tensor = CUDAStorage(shape, dtype)
    tensor.from_numpy(arr)
    return tensor

def from_numpy(arr: np.ndarray) -> CUDAStorage:
    """Create tensor from numpy array"""
    tensor = CUDAStorage(arr.shape, arr.dtype)
    tensor.from_numpy(arr)
    return tensor


# ===================== Triton kernels for strided copy operations =====================

@triton.jit
def _copy_strided_to_contig_optimized(
    src_ptr, dst_ptr,
    total_numel,
    # Pass sizes and strides as compile-time constants for common cases
    size0: tl.constexpr, stride0: tl.constexpr,
    size1: tl.constexpr, stride1: tl.constexpr, 
    size2: tl.constexpr, stride2: tl.constexpr,
    size3: tl.constexpr, stride3: tl.constexpr,
    ndim: tl.constexpr,
    BLOCK: tl.constexpr
):
    """
    Optimized Triton kernel for strided copy.
    Uses compile-time constants and vectorized operations for better performance.
    """
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < total_numel
    
    # Optimized address calculation using compile-time constants
    linear = offs.to(tl.int64)
    src_off = tl.zeros_like(linear)
    
    # Unroll common dimension cases for performance
    if ndim == 4:
        coord3 = linear % size3
        linear //= size3
        coord2 = linear % size2  
        linear //= size2
        coord1 = linear % size1
        linear //= size1
        coord0 = linear
        src_off = coord0 * stride0 + coord1 * stride1 + coord2 * stride2 + coord3 * stride3
    elif ndim == 3:
        coord2 = linear % size2
        linear //= size2
        coord1 = linear % size1
        linear //= size1
        coord0 = linear
        src_off = coord0 * stride0 + coord1 * stride1 + coord2 * stride2
    elif ndim == 2:
        coord1 = linear % size1
        coord0 = linear // size1
        src_off = coord0 * stride0 + coord1 * stride1
    else:
        # General case for other dimensions
        if ndim >= 1:
            coord0 = linear % size0
            linear //= size0
            src_off += coord0 * stride0
        if ndim >= 2:
            coord1 = linear % size1
            linear //= size1
            src_off += coord1 * stride1
        if ndim >= 3:
            coord2 = linear % size2
            linear //= size2
            src_off += coord2 * stride2
        if ndim >= 4:
            coord3 = linear % size3
            src_off += coord3 * stride3
    
    # Vectorized read/write with prefetch hint and bounds check
    # Ensure src_off is within valid range
    valid_src_mask = mask & (src_off >= 0)
    data = tl.load(src_ptr + src_off, mask=valid_src_mask)
    tl.store(dst_ptr + offs, data, mask=valid_src_mask)

@triton.jit
def _copy_strided_to_contig_general(
    src_ptr, dst_ptr,
    sizes_ptr, strides_ptr,
    total_numel,
    ndim: tl.constexpr,
    BLOCK: tl.constexpr
):
    """
    General fallback kernel for arbitrary dimensions.
    """
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < total_numel
    
    linear = offs.to(tl.int64)
    src_off = tl.zeros_like(linear)
    
    # Compute address using original approach (load on demand)
    for d in range(ndim - 1, -1, -1):
        size_d = tl.load(sizes_ptr + d)
        stride_d = tl.load(strides_ptr + d) 
        coord = linear % size_d
        linear //= size_d
        src_off += coord * stride_d
    
    data = tl.load(src_ptr + src_off, mask=mask)
    tl.store(dst_ptr + offs, data, mask=mask)

@triton.jit
def _copy_contig_to_strided_optimized(
    src_ptr, dst_ptr,
    total_numel,
    # Pass sizes and strides as compile-time constants for common cases
    size0: tl.constexpr, stride0: tl.constexpr,
    size1: tl.constexpr, stride1: tl.constexpr, 
    size2: tl.constexpr, stride2: tl.constexpr,
    size3: tl.constexpr, stride3: tl.constexpr,
    ndim: tl.constexpr,
    BLOCK: tl.constexpr
):
    """
    Optimized Triton kernel for contiguous-to-strided copy (reverse operation).
    """
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < total_numel
    
    # Calculate destination strided offset for each element
    linear = offs.to(tl.int64)
    dst_off = tl.zeros_like(linear)
    
    # Unroll common dimension cases for performance
    if ndim == 4:
        coord3 = linear % size3
        linear //= size3
        coord2 = linear % size2  
        linear //= size2
        coord1 = linear % size1
        linear //= size1
        coord0 = linear
        dst_off = coord0 * stride0 + coord1 * stride1 + coord2 * stride2 + coord3 * stride3
    elif ndim == 3:
        coord2 = linear % size2
        linear //= size2
        coord1 = linear % size1
        linear //= size1
        coord0 = linear
        dst_off = coord0 * stride0 + coord1 * stride1 + coord2 * stride2
    elif ndim == 2:
        coord1 = linear % size1
        coord0 = linear // size1
        dst_off = coord0 * stride0 + coord1 * stride1
    else:
        # General case for other dimensions
        if ndim >= 1:
            coord0 = linear % size0
            linear //= size0
            dst_off += coord0 * stride0
        if ndim >= 2:
            coord1 = linear % size1
            linear //= size1
            dst_off += coord1 * stride1
        if ndim >= 3:
            coord2 = linear % size2
            linear //= size2
            dst_off += coord2 * stride2
        if ndim >= 4:
            coord3 = linear % size3
            dst_off += coord3 * stride3
    
    # Vectorized read from contiguous source, write to strided destination
    data = tl.load(src_ptr + offs, mask=mask)
    tl.store(dst_ptr + dst_off, data, mask=mask)

@triton.jit
def _copy_contig_to_strided_general(
    src_ptr, dst_ptr,
    sizes_ptr, strides_ptr,
    total_numel,
    ndim: tl.constexpr,
    BLOCK: tl.constexpr
):
    """General kernel for contiguous-to-strided copy with arbitrary dimensions"""
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < total_numel
    
    linear = offs.to(tl.int64)
    dst_off = tl.zeros_like(linear)
    
    # General multi-dimensional index calculation
    for d in range(ndim - 1, -1, -1):
        size_d = tl.load(sizes_ptr + d)
        stride_d = tl.load(strides_ptr + d) 
        coord = linear % size_d
        linear //= size_d
        dst_off += coord * stride_d
    
    data = tl.load(src_ptr + offs, mask=mask)
    tl.store(dst_ptr + dst_off, data, mask=mask)

# ===================== Data type conversion kernels =====================

@triton.jit
def _dtype_convert_kernel(input_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    """
    GPU-native data type conversion kernel.
    Triton handles type conversion automatically based on pointer types.
    """
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load and convert (Triton handles type conversion automatically)
    input_vals = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    tl.store(output_ptr + offsets, input_vals, mask=mask)

def _convert_dtype_gpu(self, target_dtype: str) -> "CUDAStorage":
    """
    GPU-native dtype conversion using Triton kernel.
    Avoids expensive CPU round-trip.
    """
    if self.dtype == target_dtype:
        return self
    
    output = CUDAStorage(self.shape, dtype=target_dtype)
    
    if not self.is_contiguous():
        src = self.contiguous()
    else:
        src = self
    
    n_elements = src.size
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    
    _dtype_convert_kernel[grid](src, output, n_elements, BLOCK_SIZE=1024)
    
    return output

@triton.jit
def _cat_last_dim_kernel(
    tensor1_ptr, tensor2_ptr, output_ptr,
    num_rows, size1, size2, output_size,
    BLOCK_SIZE: tl.constexpr
):
    """
    Optimized concatenation along last dimension using Triton.
    Replaces the CPU loop for last dimension concatenation.
    """
    pid = tl.program_id(axis=0)
    row_idx = pid
    
    if row_idx >= num_rows:
        return
    
    # Copy first tensor data
    for i in range(0, size1, BLOCK_SIZE):
        offset_range = i + tl.arange(0, BLOCK_SIZE)
        mask = offset_range < size1
        
        src_ptr = tensor1_ptr + row_idx * size1 + offset_range
        dst_ptr = output_ptr + row_idx * output_size + offset_range
        
        data = tl.load(src_ptr, mask=mask)
        tl.store(dst_ptr, data, mask=mask)
    
    # Copy second tensor data
    for i in range(0, size2, BLOCK_SIZE):
        offset_range = i + tl.arange(0, BLOCK_SIZE)
        mask = offset_range < size2
        
        src_ptr = tensor2_ptr + row_idx * size2 + offset_range
        dst_ptr = output_ptr + row_idx * output_size + size1 + offset_range
        
        data = tl.load(src_ptr, mask=mask)
        tl.store(dst_ptr, data, mask=mask)

def _cat_last_dim_gpu(tensor1: "CUDAStorage", tensor2: "CUDAStorage") -> "CUDAStorage":
    """
    GPU-native concatenation along last dimension for two tensors.
    Optimized for the common case in RoPE (rotate_half).
    """
    # Verify inputs
    if tensor1.shape[:-1] != tensor2.shape[:-1]:
        raise ValueError("All dimensions except last must match")
    if tensor1.dtype != tensor2.dtype:
        raise ValueError("Tensors must have same dtype")
    
    # Calculate output shape
    output_shape = list(tensor1.shape)
    output_shape[-1] = tensor1.shape[-1] + tensor2.shape[-1]
    output_shape = tuple(output_shape)
    
    # Create output tensor
    result = CUDAStorage(output_shape, dtype=tensor1.dtype)
    
    # Ensure contiguous
    t1 = tensor1.contiguous() if not tensor1.is_contiguous() else tensor1
    t2 = tensor2.contiguous() if not tensor2.is_contiguous() else tensor2
    
    # Launch kernel
    num_rows = prod(t1.shape[:-1])
    size1 = t1.shape[-1]
    size2 = t2.shape[-1]
    output_size = size1 + size2
    
    BLOCK_SIZE = 64  # Good for typical hidden sizes
    grid = (num_rows,)
    
    _cat_last_dim_kernel[grid](
        t1, t2, result,
        num_rows, size1, size2, output_size,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return result

# Cache for metadata to avoid repeated GPU allocations
_metadata_cache = {}

def copy_strided_kernel(src: CUDAStorage, dst: CUDAStorage):
    """Optimized high-performance strided copy kernel"""
    assert src.size == dst.size
    assert dst.is_contiguous(), "Destination must be contiguous"
    
    if src.is_contiguous():
        # Fast path: both contiguous, direct cudaMemcpy
        result = cuda.cuMemcpyDtoD(dst.ptr, src.ptr, src.nbytes)
        check_cuda_error(result)
        return
    
    ndim = len(src.shape)
    numel = src.size
    
    # Choose optimal block size based on problem size
    if numel < 1024:
        BLOCK = 256
        num_warps = 2
    elif numel < 1024 * 1024:
        BLOCK = 512 
        num_warps = 4
    else:
        BLOCK = 1024
        num_warps = 8
    
    grid = (triton.cdiv(numel, BLOCK),)
    
    # Use optimized kernel for common cases (ndim <= 4)
    if ndim <= 4:
        # Pad with zeros for unused dimensions
        sizes = list(src.shape) + [1] * (4 - ndim)
        strides = list(src.strides) + [0] * (4 - ndim)
        
        _copy_strided_to_contig_optimized[grid](
            src, dst,
            numel,
            size0=sizes[0], stride0=strides[0],
            size1=sizes[1], stride1=strides[1], 
            size2=sizes[2], stride2=strides[2],
            size3=sizes[3], stride3=strides[3],
            ndim=ndim,
            BLOCK=BLOCK,
            num_warps=num_warps
        )
    else:
        # General case for ndim > 4
        cache_key = (tuple(src.shape), tuple(src.strides))
        if cache_key not in _metadata_cache:
            sizes_gpu = empty((ndim,), np.int64)
            strides_gpu = empty((ndim,), np.int64)
            _metadata_cache[cache_key] = (sizes_gpu, strides_gpu)
            
            # Copy metadata to GPU once
            sizes = np.array(src.shape, dtype=np.int64)
            strides = np.array(src.strides, dtype=np.int64)
            result = cuda.cuMemcpyHtoD(sizes_gpu.ptr, sizes, sizes.nbytes)
            check_cuda_error(result)
            result = cuda.cuMemcpyHtoD(strides_gpu.ptr, strides, strides.nbytes)
            check_cuda_error(result)
        
        sizes_gpu, strides_gpu = _metadata_cache[cache_key]
        
        _copy_strided_to_contig_general[grid](
            src, dst,
            sizes_gpu, strides_gpu,
            numel,
            ndim=ndim,
            BLOCK=BLOCK,
            num_warps=num_warps
        )

def copy_strided_reverse_kernel(src: CUDAStorage, dst: CUDAStorage):
    """Optimized copy from contiguous source to strided destination"""
    assert src.size == dst.size
    assert src.is_contiguous(), "Source must be contiguous"
    
    if dst.is_contiguous():
        # Fast path: both contiguous, direct cudaMemcpy
        result = cuda.cuMemcpyDtoD(dst.ptr, src.ptr, src.nbytes)
        check_cuda_error(result)
        return
    
    ndim = len(dst.shape)
    numel = dst.size
    
    # Choose optimal block size based on problem size
    if numel < 1024:
        BLOCK = 256
        num_warps = 2
    elif numel < 1024 * 1024:
        BLOCK = 512 
        num_warps = 4
    else:
        BLOCK = 1024
        num_warps = 8
    
    grid = (triton.cdiv(numel, BLOCK),)
    
    # Use optimized kernel for common cases (ndim <= 4)
    if ndim <= 4:
        # Pad with zeros for unused dimensions
        sizes = list(dst.shape) + [1] * (4 - ndim)
        strides = list(dst.strides) + [0] * (4 - ndim)
        
        _copy_contig_to_strided_optimized[grid](
            src, dst,
            numel,
            size0=sizes[0], stride0=strides[0],
            size1=sizes[1], stride1=strides[1], 
            size2=sizes[2], stride2=strides[2],
            size3=sizes[3], stride3=strides[3],
            ndim=ndim,
            BLOCK=BLOCK,
            num_warps=num_warps
        )
    else:
        # General case for ndim > 4
        cache_key = (tuple(dst.shape), tuple(dst.strides))
        if cache_key not in _metadata_cache:
            sizes_gpu = empty((ndim,), np.int64)
            strides_gpu = empty((ndim,), np.int64)
            _metadata_cache[cache_key] = (sizes_gpu, strides_gpu)
            
            # Copy metadata to GPU once
            sizes = np.array(dst.shape, dtype=np.int64)
            strides = np.array(dst.strides, dtype=np.int64)
            result = cuda.cuMemcpyHtoD(sizes_gpu.ptr, sizes, sizes.nbytes)
            check_cuda_error(result)
            result = cuda.cuMemcpyHtoD(strides_gpu.ptr, strides, strides.nbytes)
            check_cuda_error(result)
        
        sizes_gpu, strides_gpu = _metadata_cache[cache_key]
        
        _copy_contig_to_strided_general[grid](
            src, dst,
            sizes_gpu, strides_gpu,
            numel,
            ndim=ndim,
            BLOCK=BLOCK,
            num_warps=num_warps
        )

# ===================== Strided -> Strided copy kernels =====================
@triton.jit
def _copy_strided_to_strided_optimized(
    src_ptr, dst_ptr, total_numel,
    s_sz0: tl.constexpr, s_st0: tl.constexpr,
    s_sz1: tl.constexpr, s_st1: tl.constexpr,
    s_sz2: tl.constexpr, s_st2: tl.constexpr,
    s_sz3: tl.constexpr, s_st3: tl.constexpr,
    d_sz0: tl.constexpr, d_st0: tl.constexpr,
    d_sz1: tl.constexpr, d_st1: tl.constexpr,
    d_sz2: tl.constexpr, d_st2: tl.constexpr,
    d_sz3: tl.constexpr, d_st3: tl.constexpr,
    ndim: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    m = offs < total_numel
    lin = offs.to(tl.int64)
    src_off = tl.zeros_like(lin)
    dst_off = tl.zeros_like(lin)
    if ndim == 4:
        c3 = lin % s_sz3; lin //= s_sz3
        c2 = lin % s_sz2; lin //= s_sz2
        c1 = lin % s_sz1; lin //= s_sz1
        c0 = lin
        src_off = c0 * s_st0 + c1 * s_st1 + c2 * s_st2 + c3 * s_st3
        dst_off = c0 * d_st0 + c1 * d_st1 + c2 * d_st2 + c3 * d_st3
    elif ndim == 3:
        c2 = lin % s_sz2; lin //= s_sz2
        c1 = lin % s_sz1; lin //= s_sz1
        c0 = lin
        src_off = c0 * s_st0 + c1 * s_st1 + c2 * s_st2
        dst_off = c0 * d_st0 + c1 * d_st1 + c2 * d_st2
    elif ndim == 2:
        c1 = lin % s_sz1
        c0 = lin // s_sz1
        src_off = c0 * s_st0 + c1 * s_st1
        dst_off = c0 * d_st0 + c1 * d_st1
    else:
        src_off = lin * s_st0
        dst_off = lin * d_st0
    val = tl.load(src_ptr + src_off, mask=m)
    tl.store(dst_ptr + dst_off, val, mask=m)

@triton.jit
def _copy_strided_to_strided_general(
    src_ptr, dst_ptr,
    s_sizes_ptr, s_strides_ptr,
    d_sizes_ptr, d_strides_ptr,
    total_numel,
    ndim: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    m = offs < total_numel
    lin = offs.to(tl.int64)
    src_off = tl.zeros_like(lin)
    dst_off = tl.zeros_like(lin)
    for d in range(ndim - 1, -1, -1):
        sz = tl.load(s_sizes_ptr + d)
        sst = tl.load(s_strides_ptr + d)
        dsts = tl.load(d_strides_ptr + d)
        coord = lin % sz
        lin //= sz
        src_off += coord * sst
        dst_off += coord * dsts
    val = tl.load(src_ptr + src_off, mask=m)
    tl.store(dst_ptr + dst_off, val, mask=m)

def copy_strided_to_strided_kernel(src: CUDAStorage, dst: CUDAStorage):
    """Copy from possibly non-contiguous src to possibly non-contiguous dst using a single kernel."""
    assert src.size == dst.size
    if src.size == 0:
        return
    if src.is_contiguous() and dst.is_contiguous():
        r = cuda.cuMemcpyDtoD(dst.ptr, src.ptr, src.nbytes)
        check_cuda_error(r)
        return
    ndim = len(src.shape)
    numel = src.size
    if numel < 1024:
        BLOCK = 256; num_warps = 2
    elif numel < 1024 * 1024:
        BLOCK = 512; num_warps = 4
    else:
        BLOCK = 1024; num_warps = 8
    grid = (triton.cdiv(numel, BLOCK),)
    if ndim <= 4:
        s_sizes = list(src.shape) + [1] * (4 - ndim)
        s_strides = list(src.strides) + [0] * (4 - ndim)
        d_sizes = list(dst.shape) + [1] * (4 - ndim)
        d_strides = list(dst.strides) + [0] * (4 - ndim)
        _copy_strided_to_strided_optimized[grid](
            src, dst, numel,
            s_sz0=s_sizes[0], s_st0=s_strides[0],
            s_sz1=s_sizes[1], s_st1=s_strides[1],
            s_sz2=s_sizes[2], s_st2=s_strides[2],
            s_sz3=s_sizes[3], s_st3=s_strides[3],
            d_sz0=d_sizes[0], d_st0=d_strides[0],
            d_sz1=d_sizes[1], d_st1=d_strides[1],
            d_sz2=d_sizes[2], d_st2=d_strides[2],
            d_sz3=d_sizes[3], d_st3=d_strides[3],
            ndim=ndim,
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return
    # General path for ndim > 4
    cache_key_src = (tuple(src.shape), tuple(src.strides))
    cache_key_dst = (tuple(dst.shape), tuple(dst.strides))
    if cache_key_src not in _metadata_cache:
        s_sz_gpu = empty((ndim,), np.int64)
        s_st_gpu = empty((ndim,), np.int64)
        _metadata_cache[cache_key_src] = (s_sz_gpu, s_st_gpu)
        s = np.array(src.shape, dtype=np.int64)
        st = np.array(src.strides, dtype=np.int64)
        r = cuda.cuMemcpyHtoD(s_sz_gpu.ptr, s, s.nbytes); check_cuda_error(r)
        r = cuda.cuMemcpyHtoD(s_st_gpu.ptr, st, st.nbytes); check_cuda_error(r)
    if cache_key_dst not in _metadata_cache:
        d_sz_gpu = empty((ndim,), np.int64)
        d_st_gpu = empty((ndim,), np.int64)
        _metadata_cache[cache_key_dst] = (d_sz_gpu, d_st_gpu)
        ds = np.array(dst.shape, dtype=np.int64)
        dsts = np.array(dst.strides, dtype=np.int64)
        r = cuda.cuMemcpyHtoD(d_sz_gpu.ptr, ds, ds.nbytes); check_cuda_error(r)
        r = cuda.cuMemcpyHtoD(d_st_gpu.ptr, dsts, dsts.nbytes); check_cuda_error(r)
    s_sz_gpu, s_st_gpu = _metadata_cache[cache_key_src]
    d_sz_gpu, d_st_gpu = _metadata_cache[cache_key_dst]
    _copy_strided_to_strided_general[grid](
        src, dst,
        s_sz_gpu, s_st_gpu,
        d_sz_gpu, d_st_gpu,
        numel,
        ndim=ndim,
        BLOCK=BLOCK,
        num_warps=num_warps,
    )


# Set up factory function for cuda_utils to avoid circular import
from . import cuda_utils
cuda_utils.set_cuda_storage_factory(CUDAStorage)