"""CUDA Indexing Operations - Extracted from CUDAStorage.

This module contains all indexing-related operations that were originally
part of the CUDAStorage class. This separation improves code organization
and maintainability while preserving all existing functionality.
"""

import numpy as np
from enum import Enum
from dataclasses import dataclass
from typing import Optional, Tuple, Any, Union, List
import triton
import triton.language as tl

# Import CUDA bindings
try:
    from cuda.bindings import driver as cuda
    from cuda.bindings import nvrtc
except ImportError:
    from cuda import cuda, nvrtc

from ..dtypes import get_dtype


class IndexKind(Enum):
    """Types of indexing operations supported by CUDA tensors."""
    VIEW = "view"           # Pure view operation
    GATHER = "gather"       # Gather operation
    SCATTER = "scatter"     # Scatter operation  
    COPY = "copy"          # strided copy
    FILL = "fill"          # Fill operation
    MIXED_LIST_SLICE = "mixed_list_slice"  # Mixed list + slice indexing


@dataclass
class IndexPlan:
    """Unified index plan for all indexing operations."""
    kind: IndexKind
    # Result metadata for view operations
    result_shape: Optional[Tuple[int, ...]] = None
    result_strides: Optional[Tuple[int, ...]] = None
    ptr_offset_bytes: int = 0
    # Advanced indexing metadata
    index_tensor: Optional[Any] = None  # Any storage-like object
    needs_mask_compaction: bool = False
    # Mixed indexing metadata  
    column_index: Optional[Any] = None  # Any storage-like object
    is_mixed_2d: bool = False
    # Mixed list + slice indexing metadata
    slices: Optional[Tuple] = None


class CUDAIndexingOps:
    """CUDA tensor indexing operations extracted from CUDAStorage.
    
    This class contains all the indexing logic that was originally part
    of the CUDAStorage class, providing a cleaner separation of concerns.
    """
    
    @staticmethod
    def parse_index(storage, key) -> IndexPlan:
        """Parse index key into unified IndexPlan"""
        # Handle CUDAStorage as index
        if isinstance(key, type(storage)):  # Explicit type check for CUDAStorage
            if key.dtype == "bool":
                # Boolean indexing
                if key.shape != storage.shape:
                    raise ValueError("Boolean mask must have same shape as tensor")
                return IndexPlan(kind=IndexKind.GATHER, needs_mask_compaction=True, index_tensor=key)
            else:
                # Integer tensor indexing
                return IndexPlan(kind=IndexKind.GATHER, index_tensor=key)
        
        # List should be pre-converted by CUDAStorage before calling this method
        if isinstance(key, list):
            raise ValueError("Lists should be converted to tensors before calling parse_index")
        
        # Handle tuple containing advanced indexing
        if isinstance(key, tuple):
            has_advanced = any(isinstance(idx, list) or (hasattr(idx, 'shape') and hasattr(idx, 'dtype')) 
                              for idx in key if idx is not None)
            
            if has_advanced:
                return CUDAIndexingOps._parse_mixed_index(storage, key)
            else:
                return CUDAIndexingOps._parse_basic_index(storage, key)
        
        # Handle single basic indices
        return CUDAIndexingOps._parse_basic_index(storage, key)
    
    @staticmethod
    def _parse_mixed_index(storage, key) -> IndexPlan:
        """Handle mixed indexing (tuple with both basic and advanced indexing)"""
        # Special case: 2D mixed indexing like tensor[row_indices, col_indices]
        if (len(key) == 2 and len(storage.shape) == 2 and 
            all(isinstance(idx, (list, type(storage))) for idx in key)):
            
            row_idx, col_idx = key
            
            # Lists should be pre-converted by CUDAStorage
            if isinstance(row_idx, list) or isinstance(col_idx, list):
                raise ValueError("Lists should be converted to tensors before calling parse_index")
            
            return IndexPlan(
                kind=IndexKind.GATHER, 
                index_tensor=row_idx,
                column_index=col_idx,
                is_mixed_2d=True
            )
        
        # Mixed list/tensor + slice indexing like A[indices, :2]
        if (len(key) == 2 and 
            hasattr(key[0], 'shape') and hasattr(key[0], 'dtype') and
            isinstance(key[1], slice)):
            
            index_tensor, slice_idx = key
            
            return IndexPlan(
                kind=IndexKind.MIXED_LIST_SLICE,
                index_tensor=index_tensor,
                slices=(index_tensor, slice_idx)
            )
        
        # For now, other mixed indexing cases are not fully implemented
        raise NotImplementedError("Complex mixed indexing not yet supported")
    
    @staticmethod
    def _parse_basic_index(storage, key) -> IndexPlan:
        """Parse basic indexing (integers, slices, ellipsis, None) into view plan"""
        if isinstance(key, int):
            # Single integer index
            if key < 0:
                key += storage.shape[0]
            if key >= storage.shape[0] or key < 0:
                raise IndexError(f"Index {key} out of bounds")
            
            result_shape = storage.shape[1:]
            result_strides = storage.strides[1:] 
            ptr_offset_bytes = key * storage.strides[0] * storage.itemsize
            
            return IndexPlan(
                kind=IndexKind.VIEW,
                result_shape=result_shape,
                result_strides=result_strides,
                ptr_offset_bytes=ptr_offset_bytes
            )
        
        elif isinstance(key, slice):
            # Slice indexing
            start, stop, step = key.indices(storage.shape[0])
            
            if step > 0:
                length = max(0, (stop - start + step - 1) // step)
            else:
                length = max(0, (start - stop - step - 1) // (-step))
            
            result_shape = (length,) + storage.shape[1:]
            result_strides = (storage.strides[0] * step,) + storage.strides[1:]
            ptr_offset_bytes = start * storage.strides[0] * storage.itemsize
            
            return IndexPlan(
                kind=IndexKind.VIEW,
                result_shape=result_shape,
                result_strides=result_strides,
                ptr_offset_bytes=ptr_offset_bytes
            )
        
        elif isinstance(key, tuple):
            # Multi-dimensional indexing - detailed implementation follows original logic
            return CUDAIndexingOps._parse_multidim_basic(storage, key)
        
        elif key is None:
            # None (newaxis) - add dimension of size 1 at the beginning
            result_shape = (1,) + storage.shape
            result_strides = (0,) + storage.strides
            return IndexPlan(
                kind=IndexKind.VIEW,
                result_shape=result_shape,
                result_strides=result_strides,
                ptr_offset_bytes=0
            )
        
        # Other cases return copy for now
        return IndexPlan(kind=IndexKind.COPY)
    
    @staticmethod
    def _parse_multidim_basic(storage, key) -> IndexPlan:
        """Parse multi-dimensional basic indexing"""
        # Count non-None indices to validate against tensor dimensions
        non_none_count = sum(1 for idx in key if idx is not None)
        if non_none_count > len(storage.shape):
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
            missing = len(storage.shape) - (len(left) + len(right))
            if missing < 0:
                raise IndexError("too many indices for tensor")
            full_key = left + [slice(None)] * missing + right
        else:
            # Pad dimensions
            full_key = list(key) + [slice(None)] * (len(storage.shape) - len(key))
        
        tensor_dim = 0  # Track which tensor dimension we're processing
        for key_idx, idx in enumerate(full_key):
            if idx is None:
                # None (newaxis) - add new dimension of size 1
                result_shape.append(1)
                result_strides.append(0)
                continue
            
            # For non-None indices, we process actual tensor dimensions
            if tensor_dim >= len(storage.shape):
                raise IndexError("Too many indices for tensor")
                
            if isinstance(idx, int):
                # Integer index - eliminate dimension
                if idx < 0:
                    idx += storage.shape[tensor_dim]
                if idx >= storage.shape[tensor_dim] or idx < 0:
                    raise IndexError(f"Index {idx} out of bounds for dimension {tensor_dim}")
                
                ptr_offset_bytes += idx * storage.strides[tensor_dim] * storage.itemsize
                # Don't add to result_shape (dimension eliminated)
                tensor_dim += 1
                
            elif isinstance(idx, slice):
                # Slice index - modify dimension
                start, stop, step = idx.indices(storage.shape[tensor_dim])
                
                if step > 0:
                    length = max(0, (stop - start + step - 1) // step)
                else:
                    length = max(0, (start - stop - step - 1) // (-step))
                
                ptr_offset_bytes += start * storage.strides[tensor_dim] * storage.itemsize
                result_shape.append(length)
                result_strides.append(storage.strides[tensor_dim] * step)
                tensor_dim += 1
                
            else:
                raise NotImplementedError(f"Indexing with {type(idx)} not implemented yet")
        
        # Add any remaining tensor dimensions that weren't indexed
        while tensor_dim < len(storage.shape):
            result_shape.append(storage.shape[tensor_dim])
            result_strides.append(storage.strides[tensor_dim])
            tensor_dim += 1
        
        return IndexPlan(
            kind=IndexKind.VIEW,
            result_shape=tuple(result_shape),
            result_strides=tuple(result_strides),
            ptr_offset_bytes=ptr_offset_bytes
        )
    
    @staticmethod
    def execute_getitem(storage, plan: IndexPlan):
        """Execute getitem operation using parsed index plan"""
        if plan.kind == IndexKind.VIEW:
            # Zero-copy view
            result_ptr = int(storage.ptr) + plan.ptr_offset_bytes if storage.ptr else None
            return storage.__class__(
                shape=plan.result_shape,
                dtype=storage.dtype,
                ptr=result_ptr,
                strides=plan.result_strides,
                base=storage
            )
        
        elif plan.kind == IndexKind.GATHER:
            if plan.needs_mask_compaction:
                # Boolean indexing
                if tuple(plan.index_tensor.shape) != tuple(storage.shape):
                    raise ValueError("boolean mask must have the same shape as tensor")
                return CUDAIndexingOps._getitem_boolean_gather(storage, plan.index_tensor)
            elif plan.is_mixed_2d:
                # Mixed 2D indexing
                return CUDAIndexingOps._getitem_mixed_2d(storage, plan.index_tensor, plan.column_index)
            else:
                # Regular tensor indexing
                return CUDAIndexingOps._getitem_tensor_gather(storage, plan.index_tensor)
        
        elif plan.kind == IndexKind.MIXED_LIST_SLICE:
            # Handle mixed list + slice indexing
            return CUDAIndexingOps._getitem_mixed_list_slice(storage, plan)
        
        else:
            raise NotImplementedError(f"Getitem not implemented for plan kind: {plan.kind}")
    
    @staticmethod
    def execute_setitem(storage, plan: IndexPlan, value):
        """Execute setitem operation using parsed index plan"""
        if plan.kind == IndexKind.VIEW:
            # View assignment: get target view, then copy data
            target_view = CUDAIndexingOps.execute_getitem(storage, plan)
            CUDAIndexingOps._copy_data_to_view(storage, target_view, value)
            return
        
        elif plan.kind == IndexKind.GATHER:
            # Advanced indexing assignment - implement scatter operation
            return CUDAIndexingOps._setitem_gather(storage, plan, value)
        
        else:
            raise NotImplementedError(f"Setitem not implemented for plan kind: {plan.kind}")
    
    @staticmethod
    def _getitem_mixed_2d(storage, row_idx, col_idx):
        """Handle mixed 2D indexing: tensor[row_indices, col_indices]"""
        # Compute linear indices: row_idx * n_cols + col_idx
        n_cols = storage.shape[1]
        linear_indices = CUDAIndexingOps._compute_linear_indices_2d(storage, row_idx, col_idx, n_cols)
        
        # Use linear indexing to get values
        flat_storage = storage.reshape((-1,))
        result = flat_storage[linear_indices]
        
        # Reshape to match row_idx shape
        if hasattr(row_idx, 'shape'):
            result = result.reshape(row_idx.shape)
        
        return result
    
    @staticmethod
    def _compute_linear_indices_2d(storage, row_idx, col_idx, n_cols):
        """Compute linear indices for 2D mixed indexing using Triton"""
        # Ensure indices are compatible
        if hasattr(row_idx, 'shape') and hasattr(col_idx, 'shape'):
            if row_idx.shape != col_idx.shape:
                raise ValueError("row and column indices must have the same shape")
        
        n_elements = row_idx.size if hasattr(row_idx, 'size') else len(row_idx)
        
        # Create output tensor
        linear_indices = storage.__class__(shape=(n_elements,), dtype="int64")
        
        @triton.jit
        def compute_linear_indices_kernel(
            row_idx, col_idx, linear_indices, n_cols, n_elements, BLOCK_SIZE: tl.constexpr
        ):
            pid = tl.program_id(axis=0)
            block_start = pid * BLOCK_SIZE
            offsets = block_start + tl.arange(0, BLOCK_SIZE)
            mask = offsets < n_elements
            
            row_vals = tl.load(row_idx + offsets, mask=mask)
            col_vals = tl.load(col_idx + offsets, mask=mask) 
            linear_vals = row_vals * n_cols + col_vals
            
            tl.store(linear_indices + offsets, linear_vals, mask=mask)
        
        # Launch kernel
        grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
        compute_linear_indices_kernel[grid](
            row_idx, col_idx, linear_indices, n_cols, n_elements, BLOCK_SIZE=1024
        )
        
        return linear_indices
    
    @staticmethod
    def _getitem_boolean_gather(storage, mask):
        """Boolean mask indexing - extract values where mask is True"""
        lin_idx = CUDAIndexingOps._boolean_mask_to_linear_indices(mask)
        return CUDAIndexingOps._gather_linear(storage, lin_idx)
    
    @staticmethod
    def _getitem_tensor_gather(storage, indices):
        """Integer tensor indexing"""
        idx = indices
        if idx.dtype != "int64":
            idx = idx.to("int64")
        
        # Convert row indices to linear indices accounting for row size
        row_size = int(np.prod(storage.shape[1:]))  # elements per row
        
        # Use GPU-native expansion
        idx_flat = idx.reshape((-1,))  # flatten index tensor
        linear_idx_tensor = CUDAIndexingOps._expand_row_indices_gpu(idx_flat, row_size)
        
        flat_src = storage.contiguous()
        flat_src = storage.__class__((flat_src.size,), dtype=storage.dtype, ptr=flat_src.ptr, strides=(1,), base=flat_src)
        out = CUDAIndexingOps._gather_linear(flat_src, linear_idx_tensor)
        
        # Correct shape: index_shape + remaining_original_dims  
        result_shape = tuple(indices.shape) + tuple(storage.shape[1:])
        return out.reshape(result_shape)
    
    @staticmethod
    def _setitem_gather(storage, plan, value):
        """Implement advanced indexing assignment (scatter operation)"""
        if plan.needs_mask_compaction:
            # Boolean indexing assignment
            return CUDAIndexingOps._setitem_gather_boolean(storage, plan.index_tensor, value)
        elif plan.is_mixed_2d:
            # Mixed 2D indexing assignment: tensor[row_indices, col_indices] = values
            return CUDAIndexingOps._setitem_mixed_2d(storage, plan, value)
        else:
            # Integer tensor indexing assignment
            return CUDAIndexingOps._setitem_gather_integer(storage, plan.index_tensor, value)
    
    @staticmethod
    def _setitem_gather_boolean(storage, mask, value):
        """Set values using boolean mask indexing (scatter with boolean mask)"""
        if tuple(mask.shape) != tuple(storage.shape):
            raise ValueError("boolean mask must have the same shape as tensor")
        
        # Convert boolean mask to linear indices
        lin_idx = CUDAIndexingOps._boolean_mask_to_linear_indices(mask)
        
        # Use integer scatter operation
        CUDAIndexingOps._setitem_scatter_linear(storage, lin_idx, value)
        return storage
    
    @staticmethod
    def _setitem_mixed_2d(storage, plan, value):
        """Set values using mixed 2D indexing: tensor[row_indices, col_indices] = values"""
        row_idx = plan.index_tensor
        col_idx = plan.column_index
        
        # Convert to same linear indices as in forward pass
        n_cols = storage.shape[1]
        linear_indices = CUDAIndexingOps._compute_linear_indices_2d(storage, row_idx, col_idx, n_cols)
        
        # Use linear scatter
        CUDAIndexingOps._setitem_scatter_linear(storage, linear_indices, value)
        return storage
    
    @staticmethod
    def _setitem_gather_integer(storage, indices, value):
        """Set values using integer tensor indexing (scatter operation)"""
        idx = indices
        if idx.dtype != "int64":
            idx = idx.to("int64")
        
        # Convert row indices to linear indices
        row_size = int(np.prod(storage.shape[1:]))  # elements per row
        
        if row_size == 1:
            # 1D case: indices are already linear
            linear_idx = idx
        else:
            # Multi-dimensional case: expand row indices to linear indices
            idx_flat = idx.reshape((-1,))
            linear_idx = CUDAIndexingOps._expand_row_indices_gpu(idx_flat, row_size)
        
        CUDAIndexingOps._setitem_scatter_linear(storage, linear_idx, value)
        return storage
    
    @staticmethod
    def _copy_data_to_view(storage, target_view, value):
        """Copy data to view - placeholder implementation"""
        # This would be moved from the original implementation
        # For now, use a simple fallback
        if hasattr(storage, '_copy_data_to_view'):
            return storage._copy_data_to_view(target_view, value)
        else:
            raise NotImplementedError("Copy data to view not yet moved")
    
    @staticmethod
    def _setitem_scatter_linear(storage, linear_indices, value):
        """Scatter values to linear indices"""
        if isinstance(value, (int, float)):
            # Scalar value: use Triton kernel for scatter
            CUDAIndexingOps._triton_scatter_scalar(storage, linear_indices, float(value))
        else:
            # Array value: scatter array values
            if isinstance(value, type(storage)):  # CUDAStorage object
                value_flat = value.reshape((-1,))
            else:
                # This should not happen in normal usage since values should be preprocessed
                raise TypeError(f"Unsupported value type for scatter operation: {type(value)}")
            
            CUDAIndexingOps._triton_scatter_array(storage, linear_indices, value_flat)
    
    @staticmethod
    def _triton_scatter_scalar(storage, indices, value):
        """Triton kernel for scatter operation with scalar value"""
        n_indices = indices.size
        
        # Ensure storage is contiguous for linear indexing
        if not storage.is_contiguous():
            raise ValueError("Target tensor must be contiguous for scatter operation")
        
        BLOCK_SIZE = 1024
        grid = lambda meta: (triton.cdiv(n_indices, meta['BLOCK_SIZE']),)
        
        CUDAIndexingOps._scatter_scalar_kernel[grid](
            storage, indices, value,
            n_indices,
            BLOCK_SIZE=BLOCK_SIZE,
        )
    
    @staticmethod
    def _triton_scatter_array(storage, indices, values):
        """Triton kernel for scatter operation with array values"""
        n_indices = indices.size
        n_values = values.size
        
        if n_indices != n_values:
            raise ValueError(f"Number of indices ({n_indices}) must match number of values ({n_values})")
        
        # Ensure storage is contiguous for linear indexing
        if not storage.is_contiguous():
            raise ValueError("Target tensor must be contiguous for scatter operation")
        
        BLOCK_SIZE = 1024
        grid = lambda meta: (triton.cdiv(n_indices, meta['BLOCK_SIZE']),)
        
        CUDAIndexingOps._scatter_array_kernel[grid](
            storage, indices, values,
            n_indices,
            BLOCK_SIZE=BLOCK_SIZE,
        )
    
    @staticmethod
    @triton.jit
    def _scatter_scalar_kernel(
        target_ptr, indices_ptr, value,
        n_indices,
        BLOCK_SIZE: tl.constexpr,
    ):
        """Scatter scalar value to target tensor at specified indices"""
        pid = tl.program_id(axis=0)
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_indices
        
        # Load indices
        indices = tl.load(indices_ptr + offsets, mask=mask).to(tl.int64)
        
        # Store value at indexed positions
        tl.store(target_ptr + indices, value, mask=mask)
    
    @staticmethod
    @triton.jit
    def _scatter_array_kernel(
        target_ptr, indices_ptr, values_ptr,
        n_indices,
        BLOCK_SIZE: tl.constexpr,
    ):
        """Scatter array values to target tensor at specified indices"""
        pid = tl.program_id(axis=0)
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_indices
        
        # Load indices and values
        indices = tl.load(indices_ptr + offsets, mask=mask).to(tl.int64)
        values = tl.load(values_ptr + offsets, mask=mask)
        
        # Store values at indexed positions
        tl.store(target_ptr + indices, values, mask=mask)
    
    @staticmethod  
    def _boolean_mask_to_linear_indices(mask):
        """Compress bool mask with same shape as data to linear indices (return 1D int64)"""
        assert mask.dtype == "bool", "mask must be boolean"
        m = mask
        if not m.is_contiguous():
            m = m.contiguous()
        flat = CUDAIndexingOps._flatten_view(m)

        N = flat.size
        # Pre-allocate index buffer of max length N (using int32)  
        from .cuda_storage import empty
        idx_buf_i32 = empty((N,), np.int32)
        # Counter (int32)
        counter = empty((1,), np.int32)
        # Zero the counter
        try:
            from cuda.bindings import driver as cuda
        except ImportError:
            from cuda import cuda
        cuda.cuMemsetD8(counter.ptr, 0, 4)

        BLOCK = 1024
        grid = (triton.cdiv(N, BLOCK),)
        CUDAIndexingOps._compact_mask_atomic_i32[grid](flat, idx_buf_i32, counter, N, BLOCK=BLOCK, num_warps=4)

        # Read back counter (4 bytes)
        import numpy as _np
        k_host = _np.empty(1, dtype=_np.int32)
        cuda.cuMemcpyDtoH(k_host, counter.ptr, 4)
        k = int(k_host[0])

        # Convert int32 indices to int64
        idx_buf_i64 = empty((k,), np.int64)
        
        @triton.jit
        def i32_to_i64_kernel(src, dst, n, BLOCK_SIZE: tl.constexpr):
            pid = tl.program_id(axis=0)
            offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
            mask = offset < n
            vals_i32 = tl.load(src + offset, mask=mask)
            vals_i64 = vals_i32.to(tl.int64)
            tl.store(dst + offset, vals_i64, mask=mask)
        
        grid_conv = lambda meta: (triton.cdiv(k, meta["BLOCK_SIZE"]),)
        i32_to_i64_kernel[grid_conv](idx_buf_i32, idx_buf_i64, k, BLOCK_SIZE=1024)
        
        return idx_buf_i64
    
    @staticmethod
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
    
    @staticmethod
    def _flatten_view(storage):
        """Create a flattened view of storage"""
        return storage.reshape((-1,))
        
    @staticmethod
    @triton.jit
    def _gather_linear_kernel(src_ptr, idx_ptr, out_ptr, N, BLOCK: tl.constexpr):
        """Triton kernel for linear gather operation"""
        pid  = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        m    = offs < N
        idx  = tl.load(idx_ptr + offs, mask=m).to(tl.int64)
        val  = tl.load(src_ptr + idx, mask=m)
        tl.store(out_ptr + offs, val, mask=m)
    
    @staticmethod
    def _gather_linear(src, linear_idx):
        """Gather from contiguous src using linear indices, return 1D contiguous vector."""
        idx = linear_idx
        if idx.dtype != "int64":
            idx = idx.to("int64")
        if not src.is_contiguous():
            src = src.contiguous()

        N = int(idx.size)
        from .cuda_storage import empty
        out = empty((N,), src._numpy_dtype)

        BLOCK = 1024
        grid  = (triton.cdiv(N, BLOCK),)
        CUDAIndexingOps._gather_linear_kernel[grid](src, idx, out, N, BLOCK=BLOCK, num_warps=4)
        return out
    
    @staticmethod
    @triton.jit
    def _expand_row_indices_kernel(
        row_indices_ptr, linear_indices_ptr,
        num_rows, row_size,
        BLOCK_SIZE: tl.constexpr
    ):
        """GPU-native expansion of row indices to linear indices."""
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
    
    @staticmethod
    def _expand_row_indices_gpu(row_indices, row_size: int):
        """GPU-native expansion of row indices to linear indices."""
        if row_indices.dtype != "int64":
            row_indices = row_indices.to("int64")
        
        num_rows = row_indices.numel()
        total_elements = num_rows * row_size
        
        # Create output tensor
        linear_indices = row_indices.__class__((total_elements,), dtype="int64")
        
        # Launch kernel
        BLOCK_SIZE = 1024
        grid = (triton.cdiv(total_elements, BLOCK_SIZE),)
        
        CUDAIndexingOps._expand_row_indices_kernel[grid](
            row_indices, linear_indices,
            num_rows, row_size,
            BLOCK_SIZE=BLOCK_SIZE
        )
        
        return linear_indices
    
    @staticmethod
    def _getitem_mixed_list_slice(storage, plan):
        """Handle mixed list + slice indexing: A[indices, :2]"""
        # First apply the list indexing on the first dimension
        idx = plan.index_tensor
        if idx.dtype != "int64":
            idx = idx.to("int64")
        
        # Get the selected rows
        row_size = int(np.prod(storage.shape[1:]))  # elements per row
        idx_flat = idx.reshape((-1,))  # flatten index tensor
        linear_idx_tensor = CUDAIndexingOps._expand_row_indices_gpu(idx_flat, row_size)
        
        flat_src = storage.contiguous()
        flat_src = storage.__class__((flat_src.size,), dtype=storage.dtype, ptr=flat_src.ptr, strides=(1,), base=flat_src)
        gathered = CUDAIndexingOps._gather_linear(flat_src, linear_idx_tensor)
        
        # Reshape to get selected rows
        intermediate_shape = tuple(idx.shape) + tuple(storage.shape[1:])
        gathered = gathered.reshape(intermediate_shape)
        
        # Now apply the slice on the second dimension
        _, slice_idx = plan.slices
        if slice_idx != slice(None, None, None):
            # Apply the slice to the second dimension
            gathered = gathered[(slice(None),) + (slice_idx,) + (slice(None),) * (len(gathered.shape) - 2)]
        
        return gathered
