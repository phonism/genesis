"""
Pure CUDA implementation of Tensor memory management and operations
Independent of PyTorch, using CUDA Python API directly
"""

try:
    # Try new API first
    from cuda.bindings import driver as cuda
    from cuda.bindings import nvrtc
except ImportError:
    # Fall back to old API if new one not available
    from cuda import cuda, nvrtc
import numpy as np
from typing import Tuple, List, Optional, Union
import itertools
import triton
import triton.language as tl
from functools import reduce
import operator
from ..dtypes import get_dtype
from dataclasses import dataclass
from enum import Enum

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
    index_tensor: Optional['CUDATensor'] = None
    needs_mask_compaction: bool = False
    # Temporary buffer requirements
    temp_memory_bytes: int = 0

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

# CUDA initialization - will be done lazily
_cuda_initialized = False
_cuda_device = None
_cuda_context = None

# ============= Advanced Memory Pool (PyTorch-inspired) =============
import threading
from collections import defaultdict
from typing import Dict, List

class CUDAMemoryPool:
    """PyTorch-style CUDA memory pool"""
    
    def __init__(self):
        self.lock = threading.Lock()  # Thread safety
        
        # Memory alignment size (512 bytes, consistent with PyTorch)
        self.alignment = 512
        
        # Hierarchical memory pool: size -> [ptr_list]  
        self.pools: Dict[int, List[int]] = defaultdict(list)
        
        # Memory statistics
        self.allocated_bytes = 0
        self.cached_bytes = 0
        self.max_pool_size_per_bucket = 10  # Max 10 blocks cached per size
        self.max_total_cached = 100 * 1024 * 1024  # Max 100MB cached
        
        # Predefined size buckets (similar to PyTorch)
        self.size_buckets = self._create_size_buckets()
    
    def _create_size_buckets(self):
        """Create size buckets, similar to PyTorch strategy"""
        buckets = []
        
        # Small blocks: 512B - 64KB, doubling each time
        size = 512
        while size <= 64 * 1024:
            buckets.append(size)
            size *= 2
            
        # Medium blocks: 128KB - 1MB, step size 64KB
        size = 128 * 1024
        while size <= 1024 * 1024:
            buckets.append(size)
            size += 64 * 1024
            
        # Large blocks: 2MB+, step size 1MB
        size = 2 * 1024 * 1024
        while size <= 64 * 1024 * 1024:  # Max 64MB
            buckets.append(size)
            size += 1024 * 1024
            
        return sorted(buckets)
    
    def _round_up_to_bucket(self, nbytes):
        """Round up request size to appropriate bucket"""
        # First align to alignment
        aligned_bytes = ((nbytes + self.alignment - 1) // self.alignment) * self.alignment
        
        # Find first bucket >= aligned_bytes
        for bucket_size in self.size_buckets:
            if bucket_size >= aligned_bytes:
                return bucket_size
        
        # Exceeds max bucket, use aligned_bytes directly
        return aligned_bytes
    
    def allocate(self, nbytes):
        """Allocate memory"""
        if nbytes == 0:
            return None
            
        bucket_size = self._round_up_to_bucket(nbytes)
        
        with self.lock:
            # Try to get from pool
            if self.pools[bucket_size]:
                ptr = self.pools[bucket_size].pop()
                self.cached_bytes -= bucket_size
                self.allocated_bytes += bucket_size
                return ptr
        
        # Not in pool, allocate new
        result = cuda.cuMemAlloc(bucket_size)
        mem_result = check_cuda_error(result)
        ptr = mem_result[0] if mem_result else None
        
        if ptr:
            with self.lock:
                self.allocated_bytes += bucket_size
        
        return ptr
    
    def deallocate(self, ptr, nbytes):
        """Deallocate memory"""
        if not ptr or nbytes == 0:
            return
            
        bucket_size = self._round_up_to_bucket(nbytes)
        
        with self.lock:
            self.allocated_bytes -= bucket_size
            
            # Check if should cache
            should_cache = (
                len(self.pools[bucket_size]) < self.max_pool_size_per_bucket and
                self.cached_bytes + bucket_size <= self.max_total_cached
            )
            
            if should_cache:
                self.pools[bucket_size].append(ptr)
                self.cached_bytes += bucket_size
                return
        
        # Don't cache, free directly
        try:
            cuda.cuMemFree(ptr)
        except:
            pass
    
    def empty_cache(self):
        """Empty cache (similar to torch.cuda.empty_cache)"""
        with self.lock:
            for bucket_size, ptr_list in self.pools.items():
                for ptr in ptr_list:
                    try:
                        cuda.cuMemFree(ptr)
                    except:
                        pass
                ptr_list.clear()
            self.cached_bytes = 0
    
    def memory_stats(self):
        """Get memory statistics"""
        with self.lock:
            return {
                'allocated_bytes': self.allocated_bytes,
                'cached_bytes': self.cached_bytes,
                'pool_sizes': {size: len(ptrs) for size, ptrs in self.pools.items() if ptrs}
            }

# ============= CUDA Async Memory Pool Support =============
_use_async_pool = False
_async_pool = None
_default_stream = None

def _detect_async_pool_support():
    """Detect if CUDA async memory pool is supported"""
    try:
        if not _cuda_initialized:
            return False
        
        # Check if default memory pool exists
        pool_result = cuda.cuDeviceGetDefaultMemPool(_cuda_device)
        if pool_result[0] != cuda.CUresult.CUDA_SUCCESS:
            return False
        
        return True
    except Exception:
        return False

def _enable_async_pool(device=0, release_threshold_bytes=8<<30):
    """Enable CUDA async memory pool and set parameters"""
    global _use_async_pool, _async_pool
    try:
        if not _detect_async_pool_support():
            print("CUDA async memory pool not supported, using traditional memory pool")
            return False
        
        # Get default memory pool
        pool_result = cuda.cuDeviceGetDefaultMemPool(_cuda_device)
        if pool_result[0] != cuda.CUresult.CUDA_SUCCESS:
            return False
        
        _async_pool = pool_result[1]
        
        # Enable async memory pool directly, skip threshold setting for now
        _use_async_pool = True
        print(f"✅ CUDA async memory pool enabled (default config)")
        
        return True
        
    except Exception as e:
        print(f"Async memory pool enable failed: {e}")
        return False

def _ensure_stream():
    """Ensure default stream exists"""
    global _default_stream
    if _default_stream is None:
        result = cuda.cuStreamCreate(0)
        if result[0] == cuda.CUresult.CUDA_SUCCESS:
            _default_stream = result[1]
        else:
            raise RuntimeError(f"Cannot create CUDA stream: {result[0]}")
    return _default_stream

# Global memory pool instance
_memory_pool = CUDAMemoryPool()

def _allocate_memory(nbytes):
    """Use hybrid memory allocation strategy"""
    if _use_async_pool:
        # Async memory pool path
        try:
            stream = _ensure_stream()
            ptr_result = cuda.cuMemAllocAsync(nbytes, stream)
            if ptr_result[0] == cuda.CUresult.CUDA_SUCCESS:
                ptr = ptr_result[1]
                # Sync to ensure allocation complete
                sync_result = cuda.cuStreamSynchronize(stream)
                check_cuda_error(sync_result)
                return ptr
        except Exception:
            pass
    
    # Traditional memory pool path (fallback)
    return _memory_pool.allocate(nbytes)

def _free_memory(ptr, nbytes):
    """Use hybrid memory deallocation strategy"""
    if _use_async_pool:
        # Async memory pool path
        try:
            stream = _ensure_stream()
            free_result = cuda.cuMemFreeAsync(ptr, stream)
            if free_result == cuda.CUresult.CUDA_SUCCESS:
                # Sync to ensure deallocation complete
                sync_result = cuda.cuStreamSynchronize(stream)
                check_cuda_error(sync_result)
                return
        except Exception:
            pass
    
    # Traditional memory pool path (fallback)
    _memory_pool.deallocate(ptr, nbytes)

# ============= User API (PyTorch-like) =============
def empty_cache():
    """Empty memory cache, similar to torch.cuda.empty_cache()"""
    _memory_pool.empty_cache()

def memory_stats():
    """Get memory statistics"""
    return _memory_pool.memory_stats()

def memory_allocated():
    """Get currently allocated memory amount (bytes)"""
    return _memory_pool.allocated_bytes

def memory_cached():
    """Get currently cached memory amount (bytes)"""
    return _memory_pool.cached_bytes

def _ensure_cuda_initialized():
    """Lazy CUDA initialization - only initialize when first needed"""
    global _cuda_initialized, _cuda_device, _cuda_context
    if _cuda_initialized:
        return _cuda_device, _cuda_context
    
    # Initialize CUDA
    result = cuda.cuInit(0)
    check_cuda_error(result)
    
    result = cuda.cuDeviceGet(0)
    _cuda_device = check_cuda_error(result)[0]
    # Bind/reuse Primary Context (consistent with Triton)
    
    result = cuda.cuDevicePrimaryCtxRetain(_cuda_device)
    _cuda_context = check_cuda_error(result)[0]
    check_cuda_error(cuda.cuCtxSetCurrent(_cuda_context))
    
    _cuda_initialized = True
    
    # Enable CUDA async memory pool optimization
    _enable_async_pool(device=0, release_threshold_bytes=8<<30)
    
    return _cuda_device, _cuda_context

# ---- helpers ----
import ctypes

def ensure_ctx(context):
    check_cuda_error(cuda.cuCtxSetCurrent(context))

def check_ptr_accessible(ptr, context):
    """Simplified pointer accessibility check"""
    ensure_ctx(context)
    if not ptr or int(ptr) == 0:
        raise RuntimeError(f"Invalid pointer: {ptr}")
    return True

class CUDATensor:
    """Pure CUDA implementation of Tensor class"""
    
    def __init__(self, shape: Tuple[int, ...], dtype = "float32", 
                 ptr: Optional[int] = None, strides: Optional[Tuple[int, ...]] = None,
                 base: Optional['CUDATensor'] = None):
        # Flatten nested tuple if necessary
        if isinstance(shape, tuple) and len(shape) == 1 and isinstance(shape[0], tuple):
            self.shape = shape[0]
        else:
            self.shape = tuple(shape)
        # Keep reference to base tensor to prevent memory from being freed
        self.base = base
        
        # Use original DType system (restored from rollback)
        self.dtype_obj = get_dtype(dtype)
        self.dtype = self.dtype_obj.name
        self._numpy_dtype = self.dtype_obj.numpy_dtype
        self.itemsize = self.dtype_obj.itemsize
        self.nbytes = int(self.size * self.itemsize)
        
        # Compute strides (default to C-contiguous)
        if strides is None:
            self.strides = self._compute_strides(self.shape)
        else:
            self.strides = strides
            
        # GPU memory
        if ptr is None:
            # Allocate new memory - ensure CUDA is initialized first
            # Only initialize if not already done (avoid duplicate initialization)
            if not _cuda_initialized:
                _ensure_cuda_initialized()
            self.ptr = _allocate_memory(self.nbytes)
            self.owns_memory = True
        else:
            # Use existing memory
            self.ptr = ptr
            self.owns_memory = False
    
    
    def __del__(self):
        """Release GPU memory"""
        if hasattr(self, 'owns_memory') and self.owns_memory and hasattr(self, 'ptr') and self.ptr:
            try:
                # Check if the pointer is still valid before freeing
                ptr_value = int(self.ptr)
                if ptr_value != 0:
                    # Use memory pool for deallocation
                    _free_memory(self.ptr, getattr(self, 'nbytes', 0))
                    self.ptr = None  # Mark as freed
            except:
                pass  # Ignore errors during cleanup
    
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
        # Handle CUDATensor as index
        if isinstance(key, CUDATensor):
            if key.dtype == "bool":
                # Boolean indexing
                if key.shape != self.shape:
                    raise ValueError("Boolean mask must have same shape as tensor")
                return IndexPlan(kind=IndexKind.GATHER, needs_mask_compaction=True, index_tensor=key)
            else:
                # Integer tensor indexing
                return IndexPlan(kind=IndexKind.GATHER, index_tensor=key)
        
        # Handle numpy array indexing
        if hasattr(key, 'shape') and hasattr(key, 'dtype'):
            if key.dtype == np.bool_ or key.dtype == bool:
                # Convert to CUDATensor
                mask_tensor = from_numpy(key.astype(np.bool_))
                return IndexPlan(kind=IndexKind.GATHER, needs_mask_compaction=True, index_tensor=mask_tensor)
            elif np.issubdtype(key.dtype, np.integer):
                # Integer array indexing
                idx_tensor = from_numpy(key.astype(np.int64))
                return IndexPlan(kind=IndexKind.GATHER, index_tensor=idx_tensor)
        
        # Handle basic indexing (int, slice, tuple, etc.)
        return self._parse_basic_index(key)
    
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
        
        # Other cases return copy for now
        return IndexPlan(kind=IndexKind.COPY)
    
    def is_contiguous(self) -> bool:
        """Check if tensor has contiguous memory"""
        expected_strides = self._compute_strides(self.shape)
        return self.strides == expected_strides
    
    def contiguous(self) -> 'CUDATensor':
        """Return contiguous version of tensor"""
        if self.is_contiguous():
            return self
            
        # Create new contiguous tensor
        new_tensor = CUDATensor(self.shape, self.dtype)
        
        # Use Triton kernel to copy data
        copy_strided_kernel(self, new_tensor)
        return new_tensor
    
    def reshape(self, new_shape: Tuple[int, ...]) -> 'CUDATensor':
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
            return CUDATensor(new_shape, self.dtype, self.ptr, None, base=self)
        else:
            # Need to make contiguous first
            contig = self.contiguous()
            # CRITICAL FIX: Keep reference to contig to prevent memory deallocation
            return CUDATensor(new_shape, self.dtype, contig.ptr, None, base=contig)
    
    def view(self, new_shape: Tuple[int, ...]) -> 'CUDATensor':
        """View operation (requires contiguous memory)"""
        if not self.is_contiguous():
            raise RuntimeError("view() requires contiguous tensor")
        return self.reshape(new_shape)
    
    def expand(self, new_shape: Tuple[int, ...]) -> 'CUDATensor':
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
        
        return CUDATensor(new_shape, self.dtype, self.ptr, tuple(new_strides), base=self)
    
    def permute(self, dims: Tuple[int, ...]) -> 'CUDATensor':
        """Permute operation (transpose)"""
        if len(dims) != len(self.shape):
            raise ValueError("permute dimensions must match tensor dimensions")
            
        # Compute new shape and strides
        new_shape = tuple(self.shape[i] for i in dims)
        new_strides = tuple(self.strides[i] for i in dims)
        
        return CUDATensor(new_shape, self.dtype, self.ptr, new_strides, base=self)
    
    def transpose(self, dim0: int, dim1: int) -> 'CUDATensor':
        """Swap two dimensions"""
        dims = list(range(len(self.shape)))
        dims[dim0], dims[dim1] = dims[dim1], dims[dim0]
        return self.permute(tuple(dims))
    
    def unsqueeze(self, dim: int) -> 'CUDATensor':
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
            
        return CUDATensor(tuple(new_shape), self.dtype, self.ptr, tuple(new_strides), base=self)
    
    def squeeze(self, dim: Optional[int] = None) -> 'CUDATensor':
        """Remove dimensions of size 1"""
        if dim is None:
            # Remove all dimensions of size 1
            new_shape = []
            new_strides = []
            for i, (s, st) in enumerate(zip(self.shape, self.strides)):
                if s != 1:
                    new_shape.append(s)
                    new_strides.append(st)
            return CUDATensor(tuple(new_shape), self.dtype, self.ptr, tuple(new_strides), base=self)
        else:
            # Remove specified dimension
            if self.shape[dim] != 1:
                raise ValueError(f"Cannot squeeze dimension {dim} of size {self.shape[dim]}")
            new_shape = list(self.shape)
            new_strides = list(self.strides)
            del new_shape[dim]
            del new_strides[dim]
            return CUDATensor(tuple(new_shape), self.dtype, self.ptr, tuple(new_strides), base=self)
    
    def broadcast_to(self, shape: Tuple[int, ...]) -> 'CUDATensor':
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
            temp_tensor = CUDATensor(tuple([self.size]), dtype=self.dtype)
            arr_flat = arr.flatten()
            result = cuda.cuMemcpyHtoD(temp_tensor.ptr, arr_flat, temp_tensor.nbytes)
            check_cuda_error(result)
            
            # Now we need to copy from temp (contiguous) to self (strided)
            # For now, fall back to element-wise but with GPU source
            # TODO: Use a CUDA kernel for this strided copy
            import itertools
            flat_idx = 0
            for dst_idx in itertools.product(*[range(dim) for dim in self.shape]):
                # Get value from temp tensor
                src_offset = flat_idx * self.itemsize
                value_bytes = bytearray(self.itemsize)
                
                # Copy from GPU to CPU (single element)
                result = cuda.cuMemcpyDtoH(value_bytes, int(temp_tensor.ptr) + src_offset, self.itemsize)
                check_cuda_error(result)
                
                # Calculate destination offset
                dst_offset = sum(idx * stride for idx, stride in zip(dst_idx, self.strides)) * self.itemsize
                
                # Copy to destination on GPU
                dst_ptr_addr = int(self.ptr) + dst_offset
                result = cuda.cuMemcpyHtoD(dst_ptr_addr, value_bytes, self.itemsize)
                check_cuda_error(result)
                
                flat_idx += 1
            
    @property
    def T(self) -> 'CUDATensor':
        """Transpose (2D tensor)"""
        if len(self.shape) != 2:
            raise ValueError("T property only works for 2D tensors")
        return self.transpose(0, 1)
        
    
    def data_ptr(self):
        """Return integer address for Triton compatibility"""
        if self.ptr is None:
            raise RuntimeError("CUDATensor pointer is None - tensor may have been freed")
        return int(self.ptr)
    
    @property
    def __cuda_array_interface__(self):
        """CUDA Array Interface for Triton compatibility"""
        if not self.ptr:
            raise RuntimeError(f"CUDATensor has null pointer: {self.ptr}")
        
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
        """Fill tensor with a constant value (in-place)"""
        # Create a numpy array filled with the value
        numpy_dtype = self._numpy_dtype
        fill_data = np.full(self.shape, value, dtype=numpy_dtype)
        # Copy to GPU
        self.from_numpy(fill_data)
        return self
    
    def __getitem__(self, key):
        """GPU native getitem implementation - based on IndexPlan architecture + preserves original advanced indexing functionality"""
        # First try to parse as IndexPlan
        try:
            plan = self._parse_index(key)
            
            if plan.kind == IndexKind.VIEW:
                # Zero-copy view
                result_ptr = int(self.ptr) + plan.ptr_offset_bytes if self.ptr else None
                return CUDATensor(
                    shape=plan.result_shape,
                    dtype=self.dtype,
                    ptr=result_ptr,
                    strides=plan.result_strides,
                    base=self
                )
            
            elif plan.kind == IndexKind.GATHER:
                if plan.needs_mask_compaction:
                    # Boolean indexing: use original implementation
                    if tuple(plan.index_tensor.shape) != tuple(self.shape):
                        raise NotImplementedError("boolean mask must have the same shape as tensor")
                    lin_idx = _boolean_mask_to_linear_indices(plan.index_tensor)
                    return _gather_linear(self, lin_idx)
                else:
                    # Integer tensor indexing: need to handle multi-dimensional indexing properly
                    # For tensor[2D_indices], each index selects a row, so we need to:
                    # 1. Convert row indices to linear indices accounting for row size
                    # 2. Gather rows, not individual elements
                    
                    idx = plan.index_tensor
                    if idx.dtype != "int64":
                        idx = idx.to("int64")
                    
                    # Convert row indices to byte offsets for full rows  
                    row_size = int(np.prod(self.shape[1:]))  # elements per row
                    flat_indices = []
                    
                    # Create indices for all elements in selected rows
                    idx_flat = idx.reshape((-1,))  # flatten index tensor
                    idx_np = idx_flat.to_numpy()  # convert to numpy for iteration
                    for row_idx in idx_np:
                        row_idx = int(row_idx)  # ensure it's Python int
                        for col_idx in range(row_size):
                            flat_indices.append(row_idx * row_size + col_idx)
                    
                    # Convert to tensor and gather
                    linear_idx_tensor = from_numpy(np.array(flat_indices, dtype=np.int64))
                    
                    flat_src = self.contiguous()
                    flat_src = CUDATensor((flat_src.size,), dtype=self.dtype, ptr=flat_src.ptr, strides=(1,), base=flat_src)
                    out = _gather_linear(flat_src, linear_idx_tensor)
                    
                    # Correct shape: index_shape + remaining_original_dims  
                    result_shape = tuple(plan.index_tensor.shape) + tuple(self.shape[1:])
                    return out.reshape(result_shape)
            
            else:
                raise NotImplementedError(f"Unsupported indexing operation: {type(key)}")
                
        except (NotImplementedError, TypeError):
            raise
    
    def _getitem_fallback(self, key):
        """Original complex getitem implementation as fallback"""
        # ---- helpers ----
        def _norm_tuple(key_obj):
            # Handle Ellipsis / pad dimensions
            if key_obj is Ellipsis:
                key_list = [slice(None)] * self.ndim
            elif not isinstance(key_obj, tuple):
                key_list = [key_obj]
            else:
                key_list = list(key_obj)

            # Expand Ellipsis
            if Ellipsis in key_list:
                ell_idx = key_list.index(Ellipsis)
                left  = key_list[:ell_idx]
                right = key_list[ell_idx+1:]
                missing = self.ndim - (len(left) + len(right))
                if missing < 0:
                    raise IndexError("too many indices for tensor")
                key_list = left + [slice(None)] * missing + right

            # Pad to ndim
            if len(key_list) < self.ndim:
                key_list += [slice(None)] * (self.ndim - len(key_list))
            return key_list

        def _slice_len(start, stop, step):
            if step > 0:
                if start >= stop: return 0
                return (stop - start + step - 1) // step
            else:
                if start <= stop: return 0
                stepn = -step
                return (start - stop + stepn - 1) // stepn

        # ---- Handle CUDATensor/numpy/list "array index keys" convert to CUDATensor (on device) ----
        def _as_device_index(obj):
            if isinstance(obj, CUDATensor):
                return obj
            import numpy as _np
            if isinstance(obj, _np.ndarray):
                # bool or integer
                if obj.dtype == bool or obj.dtype == _np.bool_:
                    return from_numpy(obj.astype(_np.bool_))
                elif _np.issubdtype(obj.dtype, _np.integer):
                    return from_numpy(obj.astype(_np.int64))
                else:
                    raise TypeError(f"unsupported index array dtype: {obj.dtype}")
            if isinstance(obj, (list, tuple)):
                # Check if it's a mixed tuple with slices/ints (not array indexing)
                if isinstance(obj, tuple) and any(isinstance(x, (slice, int, type(None), type(Ellipsis))) for x in obj):
                    return None  # This is tuple indexing, not array indexing
                try:
                    a = _np.array(obj)
                    # Only proceed if it's a proper numeric array
                    if a.dtype == object:
                        return None  # Not a numeric array
                    return _as_device_index(a)
                except:
                    return None
            return None  # Non-array type

        # ---- Boolean/integer array indexing (advanced indexing) priority handling ----
        dev_idx = _as_device_index(key)
        if dev_idx is not None:
            # Array indexing as a whole is treated as "advanced indexing"
            if dev_idx.dtype == "bool":
                # Require same-shape boolean mask; result returns 1D
                if tuple(dev_idx.shape) != tuple(self.shape):
                    raise NotImplementedError("boolean mask must have the same shape as tensor")
                lin_idx = _boolean_mask_to_linear_indices(dev_idx)
                return _gather_linear(self, lin_idx)  # 1D contiguous
            else:
                # Integer index tensor: treat as linear indexing on "flattened tensor"
                flat_src = self.contiguous()
                flat_src = CUDATensor((flat_src.size,), dtype=self.dtype, ptr=flat_src.ptr, strides=(1,), base=flat_src)
                idx = dev_idx
                if idx.dtype != "int64":
                    idx = idx.to("int64")
                out = _gather_linear(flat_src, idx.reshape((idx.size,)))
                # Reshape by original index shape (zero-copy)
                return CUDATensor(tuple(dev_idx.shape), dtype=self.dtype, ptr=out.ptr, strides=(1,), base=out)

        # ---- Scalar / slice / tuple mixed (zero-copy view) ----
        key_list = _norm_tuple(key)

        # Calculate new ptr/shape/strides
        new_shape = []
        new_strides = []
        offset_bytes = 0

        tensor_dim = 0  # Track which tensor dimension we're processing
        for key_idx, idx in enumerate(key_list):
            if idx is None:
                # Support None(newaxis) - adds a new dimension of size 1 with stride 0
                new_shape.append(1)
                new_strides.append(0)
                continue

            # For non-None indices, we process actual tensor dimensions
            if tensor_dim >= len(self.shape):
                raise IndexError(f"Too many indices for tensor")
                
            size_d = self.shape[tensor_dim]
            stride_d = self.strides[tensor_dim]

            if isinstance(idx, int):
                i = idx + size_d if idx < 0 else idx
                if i < 0 or i >= size_d:
                    raise IndexError(f"index {idx} out of bounds for dim {tensor_dim} with size {size_d}")
                offset_bytes += i * stride_d * self.itemsize
                # This dimension is eliminated, not added to shape/stride
                tensor_dim += 1
                continue

            if isinstance(idx, slice):
                start, stop, step = idx.indices(size_d)
                length = _slice_len(start, stop, step)
                if length == 0:
                    # Still return valid 0-length view
                    new_shape.append(0)
                    new_strides.append(stride_d * step)
                    offset_bytes += start * stride_d * self.itemsize
                else:
                    new_shape.append(length)
                    new_strides.append(stride_d * step)
                    offset_bytes += start * stride_d * self.itemsize
                tensor_dim += 1
                continue

            raise TypeError(f"unsupported index type at key_idx {key_idx}: {type(idx)}")

        # Add any remaining tensor dimensions that weren't indexed
        while tensor_dim < len(self.shape):
            new_shape.append(self.shape[tensor_dim])
            new_strides.append(self.strides[tensor_dim])
            tensor_dim += 1

        new_ptr = int(self.ptr) + offset_bytes
        return CUDATensor(tuple(new_shape), dtype=self.dtype, ptr=new_ptr, strides=tuple(new_strides), base=self)
    
    def __setitem__(self, key, value):
        """GPU native setitem implementation - based on IndexPlan architecture + maintaining original functionality"""
        # First try to parse with IndexPlan
        try:
            plan = self._parse_index(key)
            
            if plan.kind == IndexKind.VIEW:
                # View assignment: get target view, then copy data
                target_view = self[key]  # Reuse getitem to get view
                self._copy_data_to_view(target_view, value)
                return
            
            elif plan.kind == IndexKind.GATHER:
                # Advanced indexing assignment: use scatter operation (not implemented yet, fall back to original method)
                pass
                
        except (NotImplementedError, TypeError):
            # Fall back to original implementation
            pass
        
        # Basic setitem implementation for backward compatibility
        # Handle simple cases that are commonly needed
        
        if isinstance(key, CUDATensor):
            # Boolean or integer tensor indexing
            if key.dtype == "bool":
                # Boolean indexing setitem: x[mask] = value
                # Convert boolean mask to linear indices and use scatter
                return self._setitem_boolean_mask(key, value)
            elif key.dtype in ["int32", "int64"]:
                # Integer indexing setitem: x[indices] = value  
                return self._setitem_integer_indices(key, value)
        
        # For other cases, fall back to CPU temporarily
        # This is not efficient but ensures correctness
        cpu_self = self.cpu()
        if isinstance(value, CUDATensor):
            cpu_value = value.cpu()
        else:
            cpu_value = value
            
        # Perform setitem on CPU
        cpu_self.data.numpy()[key] = cpu_value.data.numpy() if hasattr(cpu_value, 'data') else cpu_value
        
        # Copy result back to GPU (in-place update)
        updated_cuda = from_numpy(cpu_self.data.numpy())
        # Copy data from updated tensor to self
        result = cuda.cuMemcpy(self.ptr, updated_cuda.ptr, self.size * self.itemsize)
        check_cuda_error(result)
        
        return self
    
    def _setitem_boolean_mask(self, mask, value):
        """Set values using boolean mask"""
        # For now, use CPU fallback for boolean setitem
        # TODO: Implement efficient CUDA scatter operation
        cpu_self = self.cpu()
        cpu_mask = mask.cpu() if isinstance(mask, CUDATensor) else mask
        
        if isinstance(value, CUDATensor):
            cpu_value = value.cpu()
        else:
            cpu_value = value
            
        # Perform boolean setitem on CPU
        cpu_self.data.numpy()[cpu_mask.data.numpy()] = cpu_value.data.numpy() if hasattr(cpu_value, 'data') else cpu_value
        
        # Copy result back to GPU (in-place update)
        # Create new CUDA tensor and copy data
        updated_cuda = from_numpy(cpu_self.data.numpy())
        # Copy data from updated tensor to self
        result = cuda.cuMemcpy(self.ptr, updated_cuda.ptr, self.size * self.itemsize)
        check_cuda_error(result)
        return self
    
    def _setitem_integer_indices(self, indices, value):
        """Set values using integer indices"""  
        # For now, use CPU fallback for integer setitem
        # TODO: Implement efficient CUDA scatter operation
        cpu_self = self.cpu()
        cpu_indices = indices.cpu() if isinstance(indices, CUDATensor) else indices
        
        if isinstance(value, CUDATensor):
            cpu_value = value.cpu()
        else:
            cpu_value = value
            
        # Perform integer setitem on CPU
        cpu_self.data.numpy()[cpu_indices.data.numpy()] = cpu_value.data.numpy() if hasattr(cpu_value, 'data') else cpu_value
        
        # Copy result back to GPU (in-place update)
        updated_cuda = from_numpy(cpu_self.data.numpy())
        # Copy data from updated tensor to self
        result = cuda.cuMemcpy(self.ptr, updated_cuda.ptr, self.size * self.itemsize)
        check_cuda_error(result)
        return self
    
    def _copy_data_to_view(self, target_view, value):
        """Copy data to target view"""
        if isinstance(value, CUDATensor):
            # Tensor to Tensor copy
            if target_view.shape != value.shape:
                raise ValueError(f"Shape mismatch: {target_view.shape} vs {value.shape}")
            
            if target_view.is_contiguous() and value.is_contiguous():
                # Both contiguous: direct memcpy
                result = cuda.cuMemcpyDtoD(target_view.ptr, value.ptr, target_view.nbytes)
                check_cuda_error(result)
            else:
                # Need strided copy (temporarily use original method)
                target_view.from_numpy(value.to_numpy())
        
        elif isinstance(value, (int, float)):
            # Scalar assignment: use fill operation
            self._fill_view(target_view, value)
        
        elif hasattr(value, '__array__'):
            # numpy arrays etc: convert to CUDATensor first then copy
            import numpy as np
            value_array = np.array(value, dtype=target_view._numpy_dtype)
            if value_array.shape != target_view.shape:
                value_array = np.broadcast_to(value_array, target_view.shape)
            value_tensor = from_numpy(value_array)
            self._copy_data_to_view(target_view, value_tensor)
        
        else:
            raise TypeError(f"Cannot assign {type(value)} to CUDATensor")
    
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
            # Non-contiguous memory: temporarily use original method
            import numpy as np
            fill_data = np.full(target_view.shape, value, dtype=target_view._numpy_dtype)
            target_view.from_numpy(fill_data)
    
    
    def _setitem_fallback(self, key, value):
        """Support tensor slice assignment with GPU acceleration"""
        # Handle CUDATensor as key (convert to numpy for processing)
        if isinstance(key, CUDATensor):
            key_np = key.to_numpy()
            if key_np.ndim == 0:  # scalar
                key = int(key_np.item())
            else:
                # For array indexing, use numpy array but process on GPU
                key = key_np
        
        if isinstance(key, int):
            # Single integer index - zero-copy slice assignment
            if key < 0:
                key = self.shape[0] + key
            if key >= self.shape[0]:
                raise IndexError(f"Index {key} out of bounds for dimension 0 with size {self.shape[0]}")
            
            # Get the slice view (zero-copy)
            slice_view = self[key]
            
            # GPU-native copy
            self._copy_to_view(slice_view, value)
        
        elif isinstance(key, slice):
            # Slice assignment - zero-copy view with GPU copy
            slice_view = self[key]
            self._copy_to_view(slice_view, value)
            
        elif isinstance(key, tuple):
            # Multi-dimensional indexing for assignment - zero-copy view
            slice_view = self[key]
            self._copy_to_view(slice_view, value)
            
        elif isinstance(key, (list, tuple)) or (hasattr(key, 'shape') and hasattr(key, 'dtype')):
            # Array-like indexing (boolean or integer arrays)
            if hasattr(key, 'shape') and hasattr(key, 'dtype'):
                key_np = key
            else:
                import numpy as np
                key_np = np.array(key)
            
            if key_np.dtype == bool or key_np.dtype == 'bool':
                # Boolean indexing assignment - GPU implementation
                self._setitem_boolean_gpu(key_np, value)
            else:
                # Integer array indexing assignment - GPU implementation  
                self._setitem_integer_array_gpu(key_np, value)
        
        else:
            raise NotImplementedError(f"Unsupported indexing type for assignment: {type(key)}")
    
    def _copy_to_view(self, target_view, value):
        """Copy value to target view efficiently on GPU"""
        if isinstance(value, CUDATensor):
            # Direct GPU-to-GPU copy
            if target_view.shape != value.shape:
                raise ValueError(f"Shape mismatch: {target_view.shape} vs {value.shape}")
            
            # Use CUDA memcpy for contiguous tensors, or strided copy kernel
            if target_view.is_contiguous() and value.is_contiguous():
                # Direct memory copy
                result = cuda.cuMemcpyDtoD(target_view.ptr, value.ptr, target_view.nbytes)
                check_cuda_error(result)
            else:
                # Use strided copy (would need to implement strided copy kernel)
                # For now, fallback to CPU method
                target_view.from_numpy(value.to_numpy())
        
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
    
    def _setitem_boolean_gpu(self, mask_np, value):
        """Boolean indexing assignment using GPU kernels"""
        from . import ndarray_ops_gpu
        
        # Convert mask to CUDATensor
        mask_tensor = from_numpy(mask_np.astype(np.bool_))
        
        # Get indices of True elements
        indices = ndarray_ops_gpu.compact_boolean_mask(mask_tensor)
        
        # Prepare value tensor
        if isinstance(value, CUDATensor):
            value_tensor = value
        elif isinstance(value, np.ndarray):
            value_tensor = from_numpy(value)
        else:
            # Scalar - broadcast to match number of True elements
            value_np = np.full(indices.size, value, dtype=self._numpy_dtype)
            value_tensor = from_numpy(value_np)
        
        # Scatter values to target positions
        ndarray_ops_gpu.scatter_by_indices(value_tensor, indices, self)
    
    def _setitem_integer_array_gpu(self, indices_np, value):
        """Integer array indexing assignment using GPU kernels"""
        from . import ndarray_ops_gpu
        
        # Convert indices to CUDATensor
        indices_tensor = from_numpy(indices_np.astype(np.int64))
        
        # Prepare value tensor
        if isinstance(value, CUDATensor):
            value_tensor = value
        elif isinstance(value, np.ndarray):
            value_tensor = from_numpy(value)
        else:
            # Scalar - broadcast to match number of indices
            value_np = np.full(indices_np.size, value, dtype=self._numpy_dtype)
            value_tensor = from_numpy(value_np)
        
        # Scatter values to target positions
        ndarray_ops_gpu.scatter_by_indices(value_tensor, indices_tensor, self)
    
    def float(self):
        """Convert tensor to float32 type"""
        if self.dtype == "float32":
            return self  # Already float32
        
        # Convert to float32
        np_data = self.to_numpy().astype(np.float32)
        result = CUDATensor(self.shape, "float32")
        result.from_numpy(np_data)
        return result
    
    def half(self):
        """Convert tensor to float16 type"""
        if self.dtype == "float16":
            return self  # Already float16
        
        # Convert to float16
        np_data = self.to_numpy().astype(np.float16)
        result = CUDATensor(self.shape, "float16")
        result.from_numpy(np_data)
        return result
    
    def long(self):
        """Convert tensor to int64 type"""
        if self.dtype == "int64":
            return self  # Already int64
        
        # Convert to int64
        np_data = self.to_numpy().astype(np.int64)
        result = CUDATensor(self.shape, "int64")
        result.from_numpy(np_data)
        return result
    
    def detach(self):
        """Detach tensor from computation graph (for PyTorch compatibility)"""
        return self  # CUDATensor doesn't have gradients, so just return self
    
    def cpu(self):
        """Move tensor to CPU and convert to PyTorch tensor"""
        import torch
        np_data = self.to_numpy()
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
        
        # Convert dtype
        target_dtype_obj = get_dtype(target_dtype_str)
        target_np_dtype = target_dtype_obj.numpy_dtype
        np_data = self.to_numpy().astype(target_np_dtype)
        result = CUDATensor(self.shape, target_dtype_str)
        result.from_numpy(np_data)
        return result
    
    
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
                chunk = CUDATensor(tuple(new_shape), dtype=self.dtype, ptr=new_ptr, strides=self.strides, base=self)
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
                chunk = CUDATensor(tuple(new_shape), dtype=self.dtype, ptr=new_ptr, strides=self.strides, base=self)
                result.append(chunk)
                
                start_idx += size
        else:
            raise TypeError(f"split_size_or_sections must be int or list/tuple, got {type(split_size_or_sections)}")
            
        return result
    
    def __repr__(self):
        try:
            return f"CUDATensor(shape={self.shape}, dtype={self.dtype}, ptr=0x{int(self.ptr):x})"
        except:
            return f"CUDATensor(shape={self.shape}, dtype={self.dtype})"


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

def _flatten_view(t: "CUDATensor") -> "CUDATensor":
    # Keep zero-copy flatten (only change shape/strides)
    return CUDATensor((t.size,), dtype=t.dtype, ptr=t.ptr, strides=(1,), base=t)

def _boolean_mask_to_linear_indices(mask: "CUDATensor") -> "CUDATensor":
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
    return CUDATensor((k,), dtype="int64", ptr=idx_buf_i64.ptr, strides=(1,), base=idx_buf_i64)

def _gather_linear(src: "CUDATensor", linear_idx: "CUDATensor") -> "CUDATensor":
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

@triton.jit
def _copy_strided_to_contig(
    src_ptr, dst_ptr,            # Pointers to source / destination data
    sizes_ptr, strides_ptr,      # size / stride for each dimension (int64)
    total_numel,                 # Total number of tensor elements
    ndim: tl.constexpr,          # Number of dimensions
    BLOCK: tl.constexpr          # How many elements each CTA processes
):
    """
    Triton kernel based on user's successful solution
    Uses vectorized operations, avoiding loops and break statements
    """
    # Linear index range handled by current CTA
    pid  = tl.program_id(axis=0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < total_numel     # Mask out-of-bounds threads

    # Linear idx → multi-dimensional coordinates → source offset
    linear  = offs.to(tl.int64)
    src_off = tl.zeros_like(linear)

    for d in range(ndim - 1, -1, -1):
        size_d   = tl.load(sizes_ptr + d)
        stride_d = tl.load(strides_ptr + d)
        coord    = linear % size_d
        linear  //= size_d
        src_off += coord * stride_d

    # Read & write back
    tl.store(dst_ptr + offs,
             tl.load(src_ptr + src_off, mask=mask),
             mask=mask)

def copy_strided_kernel(src: CUDATensor, dst: CUDATensor):
    """High-performance strided copy using proven Triton kernel (based on user's working solution)"""
    assert src.size == dst.size
    assert dst.is_contiguous(), "Destination must be contiguous"
    
    if src.is_contiguous():
        # Fast path: both contiguous, direct cudaMemcpy
        result = cuda.cuMemcpyDtoD(dst.ptr, src.ptr, src.nbytes)
        check_cuda_error(result)
    else:
        # Use proven Triton kernel approach
        ndim = len(src.shape)
        BLOCK = 1024
        numel = src.size
        grid = (triton.cdiv(numel, BLOCK),)

        # Create GPU tensors for shapes and strides (int64) - pass CUDATensor objects directly
        sizes_gpu = empty((ndim,), np.int64)
        strides_gpu = empty((ndim,), np.int64)
        
        # Copy metadata to GPU
        sizes = np.array(src.shape, dtype=np.int64)
        strides = np.array(src.strides, dtype=np.int64)
        result = cuda.cuMemcpyHtoD(sizes_gpu.ptr, sizes, sizes.nbytes)
        check_cuda_error(result)
        result = cuda.cuMemcpyHtoD(strides_gpu.ptr, strides, strides.nbytes)
        check_cuda_error(result)

        # Launch kernel - pass CUDATensor objects directly like torch tensors
        _copy_strided_to_contig[grid](
            src, dst,
            sizes_gpu, strides_gpu,
            numel,
            ndim=ndim,
            BLOCK=BLOCK,
            num_warps=4,
        )

# Utility functions: create tensors
def empty(shape: Tuple[int, ...], dtype: np.dtype = np.float32) -> CUDATensor:
    """Create uninitialized tensor"""
    return CUDATensor(shape, dtype)

def zeros(shape: Tuple[int, ...], dtype: np.dtype = np.float32) -> CUDATensor:
    """Create tensor filled with zeros"""
    tensor = CUDATensor(shape, dtype)
    result = cuda.cuMemsetD8(tensor.ptr, 0, tensor.nbytes)
    check_cuda_error(result)
    return tensor

def ones(shape: Tuple[int, ...], dtype: np.dtype = np.float32) -> CUDATensor:
    """Create tensor filled with ones"""
    # Simplified implementation, should use kernel
    arr = np.ones(shape, dtype=dtype)
    tensor = CUDATensor(shape, dtype)
    tensor.from_numpy(arr)
    return tensor

def from_numpy(arr: np.ndarray) -> CUDATensor:
    """Create tensor from numpy array"""
    tensor = CUDATensor(arr.shape, arr.dtype)
    tensor.from_numpy(arr)
    return tensor
