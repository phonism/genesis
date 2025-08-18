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
from ..dtypes import get_dtype
from dataclasses import dataclass
from enum import Enum

from .cuda_memory_manager import allocate_memory, free_memory, memory_stats, get_memory_manager

# ============= Index Plan Architecture =============

class IndexKind(Enum):
    VIEW = "view"           # Pure view operation
    GATHER = "gather"       # Gather operation
    SCATTER = "scatter"     # Scatter operation  
    COPY = "copy"          # strided copy
    FILL = "fill"          # Fill operation

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

# CUDA error checking
def check_cuda_error(result):
    if isinstance(result, tuple):
        err = result[0]
        if err != cuda.CUresult.CUDA_SUCCESS:
            error_name = cuda.cuGetErrorName(err)[1].decode() if len(cuda.cuGetErrorName(err)) > 1 else "Unknown"
            error_string = cuda.cuGetErrorString(err)[1].decode() if len(cuda.cuGetErrorString(err)) > 1 else "Unknown error"
            raise RuntimeError(f"CUDA error: {error_name} - {error_string}")
        return result[1:] if len(result) > 1 else None
    else:
        if result != cuda.CUresult.CUDA_SUCCESS:
            raise RuntimeError(f"CUDA error: {result}")

# CUDA initialization handled by memory manager


# ============= Stream Management =============
_default_stream = None

def _ensure_stream():
    """Ensure default stream exists - use memory manager's stream"""
    global _default_stream
    if _default_stream is None:
        # Use memory manager's default stream to avoid multi-stream issues
        manager = get_memory_manager()
        _default_stream = manager.default_stream
    return _default_stream


def _allocate_memory(nbytes):
    """Allocate GPU memory using optimized manager"""
    return allocate_memory(nbytes, _ensure_stream())

def _allocate_memory_stream_safe(nbytes, stream):
    """Stream-safe memory allocation"""
    return allocate_memory(nbytes, stream)

def _free_memory(ptr, nbytes):
    """Free GPU memory using optimized manager"""
    free_memory(ptr, _ensure_stream())

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

class CUDAStorage:
    """Pure CUDA implementation of Tensor class"""
    
    def __init__(self, shape: Tuple[int, ...], dtype = "float32", 
                 ptr: Optional[int] = None, strides: Optional[Tuple[int, ...]] = None,
                 base: Optional['CUDAStorage'] = None, stream: Optional[int] = None):
        # Flatten nested tuple if necessary
        if isinstance(shape, tuple) and len(shape) == 1 and isinstance(shape[0], tuple):
            self.shape = shape[0]
        else:
            self.shape = tuple(shape)
        self.base = base
        
        # Data type setup
        self.dtype_obj = get_dtype(dtype)
        self.dtype = self.dtype_obj.name
        self._numpy_dtype = self.dtype_obj.numpy_dtype
        self.itemsize = self.dtype_obj.itemsize
        
        # Use property size for nbytes calculation
        self.nbytes = self.size * self.itemsize
        
        # Compute strides (default to C-contiguous)
        if strides is None:
            self.strides = self._compute_strides(self.shape)
        else:
            self.strides = strides
        
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
        """Release GPU memory"""
        if hasattr(self, 'owns_memory') and self.owns_memory and hasattr(self, 'ptr') and self.ptr:
            try:
                ptr_value = int(self.ptr)
                if ptr_value != 0:
                    stream = getattr(self, 'last_stream', None)
                    # Pass size information for caching allocator
                    nbytes = getattr(self, 'nbytes', 0)
                    free_memory(self.ptr, nbytes, stream)
                    self.ptr = None
            except:
                pass
    
    def record_stream(self, stream: int):
        """Record tensor usage on specified stream to prevent premature deallocation"""
        if hasattr(self, 'recorded_streams'):
            self.recorded_streams.add(stream)
            self.last_stream = stream
    
    def _compute_strides(self, shape: Tuple[int, ...]) -> Tuple[int, ...]:
        """Compute C-contiguous strides"""
        strides = []
        stride = 1
        for dim in reversed(shape):
            strides.append(stride)
            stride *= dim
        return tuple(reversed(strides))
    
    @property
    def size(self) -> int:
        """Total number of elements"""
        return reduce(operator.mul, self.shape, 1)
    
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
        import genesis
        return genesis.cuda()
    
    @property
    def data(self):
        """Data property - returns self for compatibility"""
        return self
    
    # ============= Index Parsing Layer (MVP Architecture) =============
    
    def _parse_index(self, key) -> IndexPlan:
        """Parse index key into unified IndexPlan"""
        # Handle CUDAStorage as index
        if isinstance(key, CUDAStorage):
            if key.dtype == "bool":
                # Boolean indexing
                if key.shape != self.shape:
                    raise ValueError("Boolean mask must have same shape as tensor")
                return IndexPlan(kind=IndexKind.GATHER, needs_mask_compaction=True, index_tensor=key)
            else:
                # Integer tensor indexing
                return IndexPlan(kind=IndexKind.GATHER, index_tensor=key)
        
        # Handle list indexing
        if isinstance(key, list):
            # Convert list to numpy array then to CUDAStorage
            key_array = np.array(key)
            if key_array.dtype == np.bool_ or key_array.dtype == bool:
                mask_tensor = from_numpy(key_array.astype(np.bool_))
                return IndexPlan(kind=IndexKind.GATHER, needs_mask_compaction=True, index_tensor=mask_tensor)
            else:
                # Integer list indexing
                idx_tensor = from_numpy(key_array.astype(np.int64))
                return IndexPlan(kind=IndexKind.GATHER, index_tensor=idx_tensor)
        
        # Handle tuple containing advanced indexing
        if isinstance(key, tuple):
            has_advanced = any(isinstance(idx, (list, CUDAStorage)) or 
                              (hasattr(idx, 'shape') and hasattr(idx, 'dtype')) 
                              for idx in key if idx is not None)
            
            if has_advanced:
                # Mixed indexing - not supported
                return self._parse_mixed_index(key)
        
        # Handle numpy array indexing
        if hasattr(key, 'shape') and hasattr(key, 'dtype'):
            if key.dtype == np.bool_ or key.dtype == bool:
                # Convert to CUDAStorage
                mask_tensor = from_numpy(key.astype(np.bool_))
                return IndexPlan(kind=IndexKind.GATHER, needs_mask_compaction=True, index_tensor=mask_tensor)
            elif np.issubdtype(key.dtype, np.integer):
                # Integer array indexing
                idx_tensor = from_numpy(key.astype(np.int64))
                return IndexPlan(kind=IndexKind.GATHER, index_tensor=idx_tensor)
        
        # Handle basic indexing (int, slice, tuple, etc.)
        return self._parse_basic_index(key)
    
    def _parse_mixed_index(self, key) -> IndexPlan:
        """Handle mixed indexing (tuple with both basic and advanced indexing)"""
        raise NotImplementedError("Mixed indexing not supported - use separate basic or advanced indexing")
    
    def _parse_basic_index(self, key) -> IndexPlan:
        """Parse basic indexing into view operations"""
        if isinstance(key, int):
            # Single integer index
            if key < 0:
                key += self.shape[0]
            if key >= self.shape[0] or key < 0:
                raise IndexError(f"Index {key} out of bounds")
            
            result_shape = self.shape[1:]
            result_strides = self.strides[1:] 
            ptr_offset_bytes = key * self.strides[0] * self.itemsize
            
            return IndexPlan(
                kind=IndexKind.VIEW,
                result_shape=result_shape,
                result_strides=result_strides,
                ptr_offset_bytes=ptr_offset_bytes
            )
        
        elif isinstance(key, slice):
            # Slice indexing
            start, stop, step = key.indices(self.shape[0])
            
            if step > 0:
                length = max(0, (stop - start + step - 1) // step)
            else:
                length = max(0, (start - stop - step - 1) // (-step))
            
            result_shape = (length,) + self.shape[1:]
            result_strides = (self.strides[0] * step,) + self.strides[1:]
            ptr_offset_bytes = start * self.strides[0] * self.itemsize
            
            return IndexPlan(
                kind=IndexKind.VIEW,
                result_shape=result_shape,
                result_strides=result_strides,
                ptr_offset_bytes=ptr_offset_bytes
            )
        
        elif isinstance(key, tuple):
            # Check if this is mixed indexing (contains advanced indexing elements)
            has_advanced = any(isinstance(idx, (list, CUDAStorage)) or 
                              (hasattr(idx, 'shape') and hasattr(idx, 'dtype')) 
                              for idx in key)
            
            if has_advanced:
                # Mixed indexing - this should be handled as advanced indexing, not basic
                raise NotImplementedError("Mixed indexing should be handled by advanced indexing path")
            
            # Multi-dimensional indexing - simplified version, only handle all int/slice cases
            # Count non-None indices to validate against tensor dimensions
            non_none_count = sum(1 for idx in key if idx is not None)
            if non_none_count > len(self.shape):
                raise IndexError("Too many indices for tensor")
            
            # Build combined view
            result_shape = []
            result_strides = []
            ptr_offset_bytes = 0
            
            # Handle Ellipsis expansion first
            if Ellipsis in key:
                # Expand Ellipsis
                ell_idx = key.index(Ellipsis)
                left = list(key[:ell_idx])
                right = list(key[ell_idx+1:])
                missing = len(self.shape) - (len(left) + len(right))
                if missing < 0:
                    raise IndexError("too many indices for tensor")
                full_key = left + [slice(None)] * missing + right
            else:
                # Pad dimensions
                full_key = list(key) + [slice(None)] * (len(self.shape) - len(key))
            
            tensor_dim = 0  # Track which tensor dimension we're processing
            for key_idx, idx in enumerate(full_key):
                if idx is None:
                    # None (newaxis) - add new dimension of size 1
                    result_shape.append(1)
                    result_strides.append(0)
                    # Don't advance tensor_dim counter since this doesn't consume a tensor dimension
                    continue
                
                # For non-None indices, we process actual tensor dimensions
                if tensor_dim >= len(self.shape):
                    raise IndexError("Too many indices for tensor")
                    
                if isinstance(idx, int):
                    # Integer index - eliminate dimension
                    if idx < 0:
                        idx += self.shape[tensor_dim]
                    if idx >= self.shape[tensor_dim] or idx < 0:
                        raise IndexError(f"Index {idx} out of bounds for dimension {tensor_dim}")
                    
                    ptr_offset_bytes += idx * self.strides[tensor_dim] * self.itemsize
                    # Don't add to result_shape (dimension eliminated)
                    tensor_dim += 1
                    
                elif isinstance(idx, slice):
                    # Slice index - modify dimension
                    start, stop, step = idx.indices(self.shape[tensor_dim])
                    
                    if step > 0:
                        length = max(0, (stop - start + step - 1) // step)
                    else:
                        length = max(0, (start - stop - step - 1) // (-step))
                    
                    ptr_offset_bytes += start * self.strides[tensor_dim] * self.itemsize
                    result_shape.append(length)
                    result_strides.append(self.strides[tensor_dim] * step)
                    tensor_dim += 1
                    
                else:
                    raise NotImplementedError(f"Indexing with {type(idx)} not implemented yet")
            
            # Add any remaining tensor dimensions that weren't indexed
            while tensor_dim < len(self.shape):
                result_shape.append(self.shape[tensor_dim])
                result_strides.append(self.strides[tensor_dim])
                tensor_dim += 1
            
            return IndexPlan(
                kind=IndexKind.VIEW,
                result_shape=tuple(result_shape),
                result_strides=tuple(result_strides),
                ptr_offset_bytes=ptr_offset_bytes
            )
        
        elif key is None:
            # None (newaxis) - add dimension of size 1 at the beginning
            result_shape = (1,) + self.shape
            result_strides = (0,) + self.strides
            return IndexPlan(
                kind=IndexKind.VIEW,
                result_shape=result_shape,
                result_strides=result_strides,
                ptr_offset_bytes=0
            )
        
        # Other cases return copy for now
        return IndexPlan(kind=IndexKind.COPY)
    
    def is_contiguous(self) -> bool:
        """Check if tensor has contiguous memory"""
        expected_strides = self._compute_strides(self.shape)
        return self.strides == expected_strides
    
    def contiguous(self) -> 'CUDAStorage':
        """Return contiguous version of tensor"""
        if self.is_contiguous():
            return self
            
        # Create new contiguous tensor
        new_tensor = CUDAStorage(self.shape, self.dtype)
        
        # Use Triton kernel to copy data
        copy_strided_kernel(self, new_tensor)
        return new_tensor
    
    def reshape(self, new_shape: Tuple[int, ...]) -> 'CUDAStorage':
        """Reshape operation"""
        # Handle -1 case
        new_shape = list(new_shape)
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
            return CUDAStorage(new_shape, self.dtype, self.ptr, None, base=self)
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
        
        return CUDAStorage(new_shape, self.dtype, self.ptr, tuple(new_strides), base=self)
    
    def permute(self, dims: Tuple[int, ...]) -> 'CUDAStorage':
        """Permute operation (transpose)"""
        if len(dims) != len(self.shape):
            raise ValueError("permute dimensions must match tensor dimensions")
            
        # Compute new shape and strides
        new_shape = tuple(self.shape[i] for i in dims)
        new_strides = tuple(self.strides[i] for i in dims)
        
        return CUDAStorage(new_shape, self.dtype, self.ptr, new_strides, base=self)
    
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
        if not self.is_contiguous():
            # For non-contiguous tensors, make contiguous first
            contig = self.contiguous()
            self.data = contig.data
            self.strides = contig.strides
        
        n_elements = self.size
        grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]), )
        
        _fill_kernel[grid](
            self, float(value), n_elements, BLOCK_SIZE=1024
        )
        return self
    
    def fill(self, value):
        """Fill tensor with a constant value (in-place) using GPU kernel"""
        if not self.is_contiguous():
            # For non-contiguous tensors, make contiguous first
            contig = self.contiguous()
            self.data = contig.data
            self.strides = contig.strides
        
        n_elements = self.size
        grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]), )
        
        _fill_kernel[grid](
            self, float(value), n_elements, BLOCK_SIZE=1024
        )
        return self
    
    def __getitem__(self, key):
        """GPU native getitem implementation - based on IndexPlan architecture"""
        plan = self._parse_index(key)
        
        if plan.kind == IndexKind.VIEW:
            # Zero-copy view
            result_ptr = int(self.ptr) + plan.ptr_offset_bytes if self.ptr else None
            return CUDAStorage(
                shape=plan.result_shape,
                dtype=self.dtype,
                ptr=result_ptr,
                strides=plan.result_strides,
                base=self
            )
        
        elif plan.kind == IndexKind.GATHER:
            if plan.needs_mask_compaction:
                # Boolean indexing
                if tuple(plan.index_tensor.shape) != tuple(self.shape):
                    raise ValueError("boolean mask must have the same shape as tensor")
                lin_idx = _boolean_mask_to_linear_indices(plan.index_tensor)
                return _gather_linear(self, lin_idx)
            else:
                # Integer tensor indexing
                idx = plan.index_tensor
                if idx.dtype != "int64":
                    idx = idx.to("int64")
                
                # Convert row indices to linear indices accounting for row size
                row_size = int(np.prod(self.shape[1:]))  # elements per row
                
                # Use GPU-native expansion
                idx_flat = idx.reshape((-1,))  # flatten index tensor
                linear_idx_tensor = _expand_row_indices_gpu(idx_flat, row_size)
                
                flat_src = self.contiguous()
                flat_src = CUDAStorage((flat_src.size,), dtype=self.dtype, ptr=flat_src.ptr, strides=(1,), base=flat_src)
                out = _gather_linear(flat_src, linear_idx_tensor)
                
                # Correct shape: index_shape + remaining_original_dims  
                result_shape = tuple(plan.index_tensor.shape) + tuple(self.shape[1:])
                return out.reshape(result_shape)
        
        else:
            raise NotImplementedError(f"Unsupported indexing operation: {type(key)}")
    
    
    def __setitem__(self, key, value):
        """GPU native setitem implementation - based on IndexPlan architecture"""
        # Handle CUDAStorage indexing first
        if isinstance(key, CUDAStorage):
            if key.dtype == "bool":
                return self._setitem_boolean_mask(key, value)
            elif key.dtype in ["int32", "int64"]:
                return self._setitem_integer_indices(key, value)
        
        # Try to parse with IndexPlan
        plan = self._parse_index(key)
        
        if plan.kind == IndexKind.VIEW:
            # View assignment: get target view, then copy data
            target_view = self[key]  # Reuse getitem to get view
            self._copy_data_to_view(target_view, value)
            return
        
        elif plan.kind == IndexKind.GATHER:
            # Advanced indexing assignment - not implemented yet
            raise NotImplementedError("Advanced indexing assignment not implemented")
        
        # Handle basic indexing cases
        if isinstance(key, int):
            if key < 0:
                key = self.shape[0] + key
            target_view = self[key]
            self._copy_data_to_view(target_view, value)
            return
            
        elif isinstance(key, slice):
            target_view = self[key]
            self._copy_data_to_view(target_view, value)
            return
            
        elif isinstance(key, tuple):
            target_view = self[key]
            self._copy_data_to_view(target_view, value)
            return
            
        elif isinstance(key, list):
            # List indexing: treat as integer indices
            if isinstance(value, (int, float)):
                for idx in key:
                    self[idx] = value
            else:
                if hasattr(value, '__len__') and len(value) == len(key):
                    for i, idx in enumerate(key):
                        self[idx] = value[i] if hasattr(value, '__getitem__') else value
                else:
                    for idx in key:
                        self[idx] = value
            return
            
        else:
            raise NotImplementedError(f"Unsupported indexing type: {type(key)}")
    
    def _setitem_boolean_mask(self, mask, value):
        """Set values using boolean mask - Triton-optimized GPU implementation"""
        # Use Triton kernel for efficient boolean mask setitem
        
        # Convert to flat tensors
        self_flat = self.reshape((-1,))
        mask_flat = mask.reshape((-1,))
        
        if isinstance(value, (int, float)):
            # Scalar value: use Triton kernel
            self._triton_boolean_setitem_scalar(self_flat, mask_flat, float(value))
        else:
            # Array value: need to handle properly
            if isinstance(value, CUDAStorage):
                value_flat = value.reshape((-1,))
            else:
                # Convert to CUDAStorage
                import numpy as np
                value_np = np.array(value, dtype=self._numpy_dtype)
                value_flat = from_numpy(value_np).reshape((-1,))
            
            self._triton_boolean_setitem_array(self_flat, mask_flat, value_flat)
        
        return self
    
    def _triton_boolean_setitem_scalar(self, target_flat, mask_flat, value):
        """Triton kernel for boolean mask setitem with scalar value"""
        n_elements = target_flat.size
        
        @triton.jit
        def boolean_setitem_scalar_kernel(
            target_ptr, mask_ptr, value,
            n_elements,
            BLOCK_SIZE: tl.constexpr,
        ):
            pid = tl.program_id(axis=0)
            block_start = pid * BLOCK_SIZE
            offsets = block_start + tl.arange(0, BLOCK_SIZE)
            mask = offsets < n_elements
            
            # Load mask values
            mask_vals = tl.load(mask_ptr + offsets, mask=mask, other=0)
            
            # Load current target values
            target_vals = tl.load(target_ptr + offsets, mask=mask, other=0.0)
            
            # Set value where mask is true
            new_vals = tl.where(mask_vals, value, target_vals)
            
            # Store back
            tl.store(target_ptr + offsets, new_vals, mask=mask)
        
        BLOCK_SIZE = 1024
        grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
        
        boolean_setitem_scalar_kernel[grid](
            target_flat, mask_flat, value,
            n_elements,
            BLOCK_SIZE=BLOCK_SIZE,
        )
    
    def _triton_boolean_setitem_array(self, target_flat, mask_flat, value_flat):
        """Triton kernel for boolean mask setitem with array values"""
        # For array values, we need to track which value element to use
        # Current implementation: element-wise assignment (can be optimized with Triton kernel)
        value_idx = 0
        for i in range(mask_flat.size):
            mask_val = mask_flat[i]
            if hasattr(mask_val, 'to_numpy'):
                is_true = bool(mask_val.to_numpy().item())
            else:
                is_true = bool(mask_val)
                
            if is_true and value_idx < value_flat.size:
                target_flat[i] = value_flat[value_idx]
                value_idx += 1
    
    def _setitem_integer_indices(self, indices, value):
        """Set values using integer indices - efficient GPU implementation"""
        # Use existing __getitem__ to get indexed elements, then bulk copy
        
        # Get the integer-indexed view  
        selected = self[indices]  # This uses existing efficient integer indexing
        
        # Assign value using existing copy operations
        if isinstance(value, (int, float)):
            # Scalar: use efficient fill
            selected.fill(value)
        elif isinstance(value, CUDAStorage):
            # Another tensor: use efficient copy
            if selected.size != value.size:
                if value.size == 1:
                    selected.fill(float(value.flat()[0]))
                else:
                    raise ValueError(f"Cannot broadcast value of size {value.size} to {selected.size} positions")
            else:
                result = cuda.cuMemcpyDtoD(selected.ptr, value.ptr, selected.nbytes)
                check_cuda_error(result)
        else:
            self._copy_data_to_view(selected, value)
        
        return self
    
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
        
        elif hasattr(value, '__array__'):
            # numpy arrays etc: convert to CUDAStorage first then copy
            import numpy as np
            value_array = np.array(value, dtype=target_view._numpy_dtype)
            if value_array.shape != target_view.shape:
                value_array = np.broadcast_to(value_array, target_view.shape)
            value_tensor = from_numpy(value_array)
            self._copy_data_to_view(target_view, value_tensor)
        
        else:
            raise TypeError(f"Cannot assign {type(value)} to CUDAStorage")
    
    def _gpu_strided_copy(self, target_view, value):
        """GPU-only strided copy without CPU fallback"""
        if target_view.shape != value.shape:
            raise ValueError(f"Shape mismatch: {target_view.shape} vs {value.shape}")
        
        # For 1D tensors, use element-wise copy with stride
        if target_view.ndim == 1 and value.ndim == 1:
            self._gpu_1d_strided_copy(target_view, value)
        elif target_view.ndim == 2 and value.ndim == 2:
            self._gpu_2d_strided_copy(target_view, value) 
        else:
            # For higher dimensions, flatten and copy
            self._gpu_flattened_copy(target_view, value)
    
    def _gpu_1d_strided_copy(self, target_view, value):
        """GPU 1D strided copy"""
        size = target_view.size
        target_stride = target_view.stride()[0] * target_view.itemsize
        value_stride = value.stride()[0] * value.itemsize
        itemsize = target_view.itemsize
        
        if target_stride == itemsize and value_stride == itemsize:
            # Both contiguous, direct copy
            result = cuda.cuMemcpyDtoD(target_view.ptr, value.ptr, size * itemsize)
            check_cuda_error(result)
        else:
            # Strided copy - copy element by element
            for i in range(size):
                src_ptr = value.ptr + i * value_stride
                dst_ptr = target_view.ptr + i * target_stride
                result = cuda.cuMemcpyDtoD(dst_ptr, src_ptr, itemsize)
                check_cuda_error(result)
    
    def _gpu_2d_strided_copy(self, target_view, value):
        """GPU 2D strided copy using cuMemcpy2D"""
        height, width = target_view.shape
        
        # Calculate strides in bytes
        target_pitch = target_view.stride()[0] * target_view.itemsize
        value_pitch = value.stride()[0] * value.itemsize
        width_bytes = width * target_view.itemsize
        
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
    
    def _gpu_flattened_copy(self, target_view, value):
        """GPU copy by flattening tensors"""
        if target_view.is_contiguous():
            # Contiguous case: use fast reshape+copy
            target_flat = target_view.reshape((-1,))
            value_flat = value.reshape((-1,))
            self._gpu_1d_strided_copy(target_flat, value_flat)
        else:
            # Non-contiguous case: use optimized strided copy
            self._gpu_strided_copy_fast(target_view, value)
    
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
        """Decompose high-dimensional copy into 2D operations"""
        # Try to reshape to 2D while preserving memory layout
        if target_view.ndim >= 3:
            # Merge all but the first dimension
            outer_size = target_view.shape[0]
            inner_size = target_view.size // outer_size
            
            # Check if we can safely create 2D views
            if (target_view.stride()[0] * outer_size <= target_view.base.size if target_view.base else target_view.size):
                try:
                    # Create 2D view of target
                    target_2d = CUDAStorage(
                        shape=(outer_size, inner_size),
                        dtype=target_view.dtype,
                        ptr=target_view.ptr,
                        strides=(target_view.stride()[0], 1),  # Assume inner stride = 1
                        base=target_view.base or target_view
                    )
                    # Reshape value to 2D
                    value_2d = value.reshape((outer_size, inner_size))
                    
                    self._gpu_2d_strided_copy(target_2d, value_2d)
                    return
                except:
                    pass
        
        # Fallback to element-wise copy
        self._gpu_elementwise_copy(target_view, value)
    
    def _gpu_elementwise_copy(self, target_view, value):
        """Element-wise copy for small tensors"""
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

    def _fill_view(self, target_view, value):
        """Fill view with scalar value"""
        if target_view.is_contiguous():
            # Contiguous memory: simple implementation
            if target_view.dtype == "float32" and value == 0.0:
                # Zero fill: can use cuMemsetD8
                result = cuda.cuMemsetD8(target_view.ptr, 0, target_view.nbytes)
                check_cuda_error(result)
            else:
                # Other values: create data on CPU first, then copy
                import numpy as np
                fill_data = np.full(target_view.shape, value, dtype=target_view._numpy_dtype)
                result = cuda.cuMemcpyHtoD(target_view.ptr, fill_data, fill_data.nbytes)
                check_cuda_error(result)
        else:
            # Non-contiguous memory: implement GPU fill for non-contiguous tensors
            # For now, disable non-contiguous fill operations
            raise NotImplementedError("Non-contiguous fill not implemented in GPU-only mode")
    
    
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
                # GPU strided copy using cuMemcpy2D
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
        import torch
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


# ===================== Triton kernels for indexing =====================

@triton.jit
def _gather_linear_kernel(src_ptr, idx_ptr, out_ptr, N, BLOCK: tl.constexpr):
    pid  = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    m    = offs < N
    idx  = tl.load(idx_ptr + offs, mask=m).to(tl.int64)
    val  = tl.load(src_ptr + idx, mask=m)
    tl.store(out_ptr + offs, val, mask=m)

@triton.jit
def _expand_row_indices_kernel(
    row_indices_ptr, linear_indices_ptr,
    num_rows, row_size,
    BLOCK_SIZE: tl.constexpr
):
    """
    GPU-native expansion of row indices to linear indices.
    Replaces CPU loop for advanced indexing.
    """
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Calculate which output element we're computing
    output_idx = offsets
    mask = output_idx < (num_rows * row_size)
    
    # Calculate which row and which column within row
    row_idx = output_idx // row_size
    col_idx = output_idx % row_size
    
    # Load the actual row index from input tensor
    row_indices = tl.load(row_indices_ptr + row_idx, mask=mask, other=0)
    
    # Calculate linear index: actual_row * row_size + col_offset
    linear_idx = row_indices * row_size + col_idx
    
    tl.store(linear_indices_ptr + output_idx, linear_idx, mask=mask)

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

@triton.jit
def _compact_mask_atomic_i32(
    mask_ptr,              # *u8 / *i1
    out_idx_ptr_i32,       # *i32
    counter_ptr_i32,       # *i32 (global counter with length=1)
    N,                     # Runtime parameter, avoid repeated JIT compilation
    BLOCK: tl.constexpr,
):
    """Efficient implementation using single atomic reservation + prefix sum, avoiding MLIR type validation issues"""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    inb = offs < N

    # Read mask -> bool
    v = tl.load(mask_ptr + offs, mask=inb, other=0)
    active = inb & (v != 0)

    # Prefix sum needs numeric type; convert bool -> i32
    act_i32 = active.to(tl.int32)

    # "Active indices" within this block (exclusive prefix sum)
    # tl.cumsum is inclusive, so subtract self to get exclusive
    local = tl.cumsum(act_i32, axis=0) - act_i32      # [BLOCK] i32

    # Active count of this block (get scalar i32)
    cnt = tl.sum(act_i32, axis=0)                     # () i32

    # Global single atomic add, reserve contiguous space
    base = tl.atomic_add(counter_ptr_i32, cnt)        # () i32

    # Only compute/store for active lanes
    idx = base + local                                # [BLOCK] i32 (only effective for active)
    tl.store(out_idx_ptr_i32 + idx, offs, mask=active)

@triton.jit
def _widen_i32_to_i64(src_i32, dst_i64, n, BLOCK: tl.constexpr):
    """Widen int32 array to int64"""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    val_i32 = tl.load(src_i32 + offs, mask=mask, other=0)
    val_i64 = val_i32.to(tl.int64)  # This is safe since we're widening from i32 to i64
    tl.store(dst_i64 + offs, val_i64, mask=mask)

def _flatten_view(t: "CUDAStorage") -> "CUDAStorage":
    # Keep zero-copy flatten (only change shape/strides)
    return CUDAStorage((t.size,), dtype=t.dtype, ptr=t.ptr, strides=(1,), base=t)

def _boolean_mask_to_linear_indices(mask: "CUDAStorage") -> "CUDAStorage":
    """Compress bool mask with same shape as data to linear indices (return 1D int64).
       Use full int32 pipeline to avoid MLIR crashes, then convert to int64 at the end."""
    assert mask.dtype == "bool", "mask must be boolean"
    m = mask
    if not m.is_contiguous():
        m = m.contiguous()
    flat = _flatten_view(m)

    N = flat.size
    # Pre-allocate index buffer of max length N (using int32)
    idx_buf_i32 = empty((N,), np.int32)
    # Counter (int32)
    counter = empty((1,), np.int32)
    # Zero the counter
    check_cuda_error(cuda.cuMemsetD8(counter.ptr, 0, 4))

    BLOCK = 1024
    grid  = (triton.cdiv(N, BLOCK),)
    _compact_mask_atomic_i32[grid](flat, idx_buf_i32, counter, N, BLOCK=BLOCK, num_warps=4)

    # Read back counter (4 bytes), this is the only tiny DtoH (negligible)
    import numpy as _np
    k_host = _np.empty(1, dtype=_np.int32)
    check_cuda_error(cuda.cuMemcpyDtoH(k_host, counter.ptr, 4))
    k = int(k_host[0])

    # Convert int32 indices to int64 (avoid doing conversion in kernel)
    # Create int64 buffer
    idx_buf_i64 = empty((k,), np.int64)
    
    if k > 0:
        # Use externally defined widen kernel
        WIDEN_BLOCK = 1024
        widen_grid = (triton.cdiv(k, WIDEN_BLOCK),)
        _widen_i32_to_i64[widen_grid](idx_buf_i32, idx_buf_i64, k, BLOCK=WIDEN_BLOCK)

    # Return int64 version of the indices
    return CUDAStorage((k,), dtype="int64", ptr=idx_buf_i64.ptr, strides=(1,), base=idx_buf_i64)

def _gather_linear(src: "CUDAStorage", linear_idx: "CUDAStorage") -> "CUDAStorage":
    """Gather from contiguous src using linear indices, return 1D contiguous vector."""
    idx = linear_idx
    if idx.dtype != "int64":
        idx = idx.to("int64")
    if not src.is_contiguous():
        src = src.contiguous()

    N = int(idx.size)
    out = empty((N,), src._numpy_dtype)

    BLOCK = 1024
    grid  = (triton.cdiv(N, BLOCK),)
    _gather_linear_kernel[grid](src, idx, out, N, BLOCK=BLOCK, num_warps=4)
    return out

def _expand_row_indices_gpu(row_indices: "CUDAStorage", row_size: int) -> "CUDAStorage":
    """
    GPU-native expansion of row indices to linear indices.
    Replaces the CPU loop in advanced indexing.
    """
    if row_indices.dtype != "int64":
        row_indices = row_indices.to("int64")
    
    num_rows = row_indices.size
    total_elements = num_rows * row_size
    
    # Create output tensor
    linear_indices = CUDAStorage((total_elements,), dtype="int64")
    
    # Launch kernel
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(total_elements, BLOCK_SIZE),)
    
    _expand_row_indices_kernel[grid](
        row_indices, linear_indices,
        num_rows, row_size,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return linear_indices

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
    
    # Vectorized read/write with prefetch hint
    data = tl.load(src_ptr + src_off, mask=mask)
    tl.store(dst_ptr + offs, data, mask=mask)

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
