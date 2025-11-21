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

from genesis.dtypes import get_dtype
from . import cuda_utils
from .cuda_error import check_cuda_error


class IndexKind(Enum):
    """Types of indexing operations supported by CUDA tensors."""
    VIEW = "view"           # Pure view operation
    GATHER = "gather"       # Gather operation
    SCATTER = "scatter"     # Scatter operation  
    COPY = "copy"           # strided copy
    FILL = "fill"           # Fill operation
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
                # Allow 1D mask to index first dimension of multi-dimensional tensor
                if key.shape != storage.shape:
                    if len(key.shape) == 1 and len(storage.shape) > 1 and key.shape[0] == storage.shape[0]:
                        # 1D mask indexing first dimension - this is valid
                        pass
                    else:
                        raise ValueError(f"Boolean mask shape {key.shape} incompatible with tensor shape {storage.shape}")
                return IndexPlan(kind=IndexKind.GATHER, needs_mask_compaction=True, index_tensor=key)
            else:
                # Integer tensor indexing
                return IndexPlan(kind=IndexKind.GATHER, index_tensor=key)
        
        # List should be pre-converted by CUDAStorage before calling this method
        if isinstance(key, list):
            raise ValueError("Lists should be converted to tensors before calling parse_index")
        
        # Handle tuple containing advanced indexing
        if isinstance(key, tuple):
            has_advanced = any(isinstance(idx, list) or (hasattr(idx, "shape") and hasattr(idx, "dtype")) 
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
            hasattr(key[0], "shape") and hasattr(key[0], "dtype") and
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
                # Allow 1D mask to index first dimension of multi-dimensional tensor
                if tuple(plan.index_tensor.shape) != tuple(storage.shape):
                    mask_shape = plan.index_tensor.shape
                    tensor_shape = storage.shape
                    if not (len(mask_shape) == 1 and len(tensor_shape) > 1 and mask_shape[0] == tensor_shape[0]):
                        raise ValueError(f"Boolean mask shape {mask_shape} incompatible with tensor shape {tensor_shape}")
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
        """Handle mixed 2D indexing: tensor[row_indices, col_indices] with a single kernel."""
        # Ensure indices compatible
        if hasattr(row_idx, "shape") and hasattr(col_idx, "shape"):
            if tuple(row_idx.shape) != tuple(col_idx.shape):
                raise ValueError("row and column indices must have the same shape")
        
        # Flatten indices
        row_flat = row_idx.reshape((-1,))
        col_flat = col_idx.reshape((-1,))
        K = int(row_flat.size)
        n_cols = int(storage.shape[1])

        # Flat contiguous source
        flat_src = storage.contiguous()
        flat_src = storage.__class__((flat_src.size,), dtype=storage.dtype, ptr=flat_src.ptr, strides=(1,), base=flat_src)

        # Allocate output 1D then reshape to indices shape
        out_1d = cuda_utils.empty((K,), flat_src._numpy_dtype)

        BLOCK = 1024
        grid = (triton.cdiv(K, BLOCK),)
        CUDAIndexingOps._gather_mixed_2d_kernel[grid](
            flat_src, row_flat, col_flat, out_1d, K, n_cols, BLOCK_SIZE=BLOCK
        )

        # Reshape to match row_idx shape
        result = out_1d
        if hasattr(row_idx, "shape"):
            result = result.reshape(row_idx.shape)
        return result
    
    @staticmethod
    def _compute_linear_indices_2d(storage, row_idx, col_idx, n_cols):
        """Compute linear indices for 2D mixed indexing using Triton"""
        # Ensure indices are compatible
        if hasattr(row_idx, "shape") and hasattr(col_idx, "shape"):
            if row_idx.shape != col_idx.shape:
                raise ValueError("row and column indices must have the same shape")
        
        n_elements = row_idx.size if hasattr(row_idx, "size") else len(row_idx)
        
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
        """Boolean mask indexing - fused compact+gather implementation"""
        return CUDAIndexingOps._gather_mask_fused(storage, mask)
    
    @staticmethod
    def _getitem_tensor_gather(storage, indices):
        """Integer tensor indexing"""
        idx = indices
        if idx.dtype != "int64":
            idx = idx.to("int64")

        # Flatten indices and make source a flat contiguous view
        idx_flat = idx.reshape((-1,))
        flat_src = storage.contiguous()
        flat_src = storage.__class__((flat_src.size,), dtype=storage.dtype, ptr=flat_src.ptr, strides=(1,), base=flat_src)

        num_rows = int(idx_flat.size)
        row_size = int(np.prod(storage.shape[1:])) if len(storage.shape) > 1 else 1
        n_src_rows = int(storage.shape[0]) if len(storage.shape) > 0 else 1

        # Allocate output 1D and launch row-wise gather
        out_1d = cuda_utils.empty((num_rows * row_size,), flat_src._numpy_dtype)
        COL_BLOCK = 256
        grid = (num_rows, triton.cdiv(row_size, COL_BLOCK))
        CUDAIndexingOps._gather_rows_kernel[grid](
            flat_src, idx_flat, out_1d,
            num_rows, row_size, n_src_rows,
            COL_BLOCK=COL_BLOCK,
        )

        result_shape = tuple(indices.shape) + tuple(storage.shape[1:])
        return out_1d.reshape(result_shape)
    
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
        """Set values using boolean mask indexing (fused scatter for mask)."""
        CUDAIndexingOps._scatter_mask(storage, mask, value)
        return storage
    
    @staticmethod
    def _setitem_mixed_2d(storage, plan, value):
        """Set values using mixed 2D indexing: tensor[row_indices, col_indices] = values"""
        row_idx = plan.index_tensor
        col_idx = plan.column_index
        
        # Only supports 2D tensors by design of parse_mixed_index
        n_cols = int(storage.shape[1])
        
        # Flatten indices to 1D
        if hasattr(row_idx, "reshape"):
            row_flat = row_idx.reshape((-1,))
        else:
            raise TypeError("row indices must be CUDAStorage tensor")
        if hasattr(col_idx, "reshape"):
            col_flat = col_idx.reshape((-1,))
        else:
            raise TypeError("col indices must be CUDAStorage tensor")
        
        K = int(row_flat.size)
        if K != int(col_flat.size):
            raise ValueError("row and column indices must have the same number of elements")
        
        # Destination must be contiguous for linear addressing
        if not storage.is_contiguous():
            raise ValueError("Target tensor must be contiguous for mixed 2D scatter")
        dst_flat = storage.reshape((-1,)) if len(storage.shape) != 1 else storage
        
        if isinstance(value, (int, float)):
            BLOCK = 1024
            grid = (triton.cdiv(K, BLOCK),)
            CUDAIndexingOps._scatter_mixed_2d_scalar_kernel[grid](
                dst_flat, row_flat, col_flat, K, n_cols, float(value), BLOCK_SIZE=BLOCK
            )
            return storage
        
        if not hasattr(value, "shape") or not hasattr(value, "ptr"):
            raise TypeError("Unsupported value type for mixed 2D scatter: expected CUDAStorage or scalar")
        val_flat = value.reshape((-1,))
        if int(val_flat.size) != K:
            raise ValueError(f"values size ({int(val_flat.size)}) must match number of indices ({K})")
        
        BLOCK = 1024
        grid = (triton.cdiv(K, BLOCK),)
        CUDAIndexingOps._scatter_mixed_2d_array_kernel[grid](
            dst_flat, row_flat, col_flat, val_flat, K, n_cols, BLOCK_SIZE=BLOCK
        )
        return storage
    
    @staticmethod
    def _setitem_gather_integer(storage, indices, value):
        """Set values using integer tensor indexing (scatter operation)"""
        idx = indices
        if idx.dtype != "int64":
            idx = idx.to("int64")
        # Require target contiguous for linear addressing
        if not storage.is_contiguous():
            raise ValueError("Target tensor must be contiguous for integer scatter operation")

        row_size = int(np.prod(storage.shape[1:])) if len(storage.shape) > 1 else 1

        if row_size == 1:
            # 1D case: indices are already linear
            linear_idx = idx
            CUDAIndexingOps._setitem_scatter_linear(storage, linear_idx, value)
            return storage

        # 2D+ case: use row-wise scatter to avoid building huge linear index buffers
        idx_flat = idx.reshape((-1,))
        num_rows = int(idx_flat.size)
        n_src_rows = int(storage.shape[0])

        # Ensure flat destination pointer
        dst_flat = storage
        if len(storage.shape) != 1:
            dst_flat = storage.reshape((-1,))

        if isinstance(value, (int, float)):
            # Scalar assignment: write scalar across entire selected rows
            COL_BLOCK = 256
            grid = (num_rows, triton.cdiv(row_size, COL_BLOCK))
            CUDAIndexingOps._scatter_rows_scalar_kernel[grid](
                dst_flat, idx_flat, float(value),
                num_rows, row_size, n_src_rows,
                COL_BLOCK=COL_BLOCK,
            )
            return storage

        if not hasattr(value, "shape") or not hasattr(value, "ptr"):
            raise TypeError("Unsupported value type for integer scatter: expected CUDAStorage or scalar")

        # Array assignment: values shape must be idx.shape + storage.shape[1:]
        val_flat = value.reshape((-1,))
        expected = num_rows * row_size
        if val_flat.size != expected:
            raise ValueError(f"scatter values size ({val_flat.size}) must match indices*row_size ({expected})")

        COL_BLOCK = 256
        grid = (num_rows, triton.cdiv(row_size, COL_BLOCK))
        CUDAIndexingOps._scatter_rows_array_kernel[grid](
            dst_flat, idx_flat, val_flat,
            num_rows, row_size, n_src_rows,
            COL_BLOCK=COL_BLOCK,
        )
        return storage
    
    @staticmethod
    def _copy_data_to_view(storage, target_view, value):
        """Copy data to target view - pixel-level copy from old architecture"""

        # Pixel-level copy of old architecture logic (line 688-706 from cuda_storage.py)
        if hasattr(value, "shape") and hasattr(value, "ptr"):  # CUDAStorage check
            # Tensor to Tensor copy
            if target_view.shape != value.shape:
                raise ValueError(f"Shape mismatch: {target_view.shape} vs {value.shape}")

            if target_view.is_contiguous() and value.is_contiguous():
                # Both contiguous: direct memcpy
                result = cuda.cuMemcpyDtoD(target_view.ptr, value.ptr, target_view.nbytes)
                check_cuda_error(result)
            else:
                # GPU strided copy using cuMemcpy2D
                storage._gpu_strided_copy(target_view, value)
        elif isinstance(value, (int, float)):
            # Scalar assignment: use fill operation
            storage._fill_view(target_view, value)
        else:
            raise TypeError(f"Cannot assign {type(value)} to CUDAStorage")
    
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
        grid = lambda meta: (triton.cdiv(n_indices, meta["BLOCK_SIZE"]),)
        
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
        grid = lambda meta: (triton.cdiv(n_indices, meta["BLOCK_SIZE"]),)
        
        CUDAIndexingOps._scatter_array_kernel[grid](
            storage, indices, values,
            n_indices,
            BLOCK_SIZE=BLOCK_SIZE,
        )

    @staticmethod
    @triton.jit
    def _scatter_rows_scalar_kernel(
        dst_ptr, idx_ptr, value,
        num_rows, row_size, n_src_rows,
        COL_BLOCK: tl.constexpr,
    ):
        rid = tl.program_id(0)
        cb  = tl.program_id(1)
        cols = cb * COL_BLOCK + tl.arange(0, COL_BLOCK)
        m_row = rid < num_rows
        m_col = cols < row_size
        m = m_row & m_col
        ridx = tl.load(idx_ptr + rid, mask=m_row, other=0).to(tl.int64)
        valid_r = (ridx >= 0) & (ridx < n_src_rows)
        m = m & valid_r
        out_base = ridx * row_size
        tl.store(dst_ptr + out_base + cols, value, mask=m)

    @staticmethod
    @triton.jit
    def _scatter_rows_array_kernel(
        dst_ptr, idx_ptr, values_ptr,
        num_rows, row_size, n_src_rows,
        COL_BLOCK: tl.constexpr,
    ):
        rid = tl.program_id(0)
        cb  = tl.program_id(1)
        cols = cb * COL_BLOCK + tl.arange(0, COL_BLOCK)
        m_row = rid < num_rows
        m_col = cols < row_size
        m = m_row & m_col
        ridx = tl.load(idx_ptr + rid, mask=m_row, other=0).to(tl.int64)
        valid_r = (ridx >= 0) & (ridx < n_src_rows)
        m = m & valid_r
        out_base = ridx * row_size
        val_base = rid * row_size
        vals = tl.load(values_ptr + val_base + cols, mask=m, other=0)
        tl.store(dst_ptr + out_base + cols, vals, mask=m)

    @staticmethod
    def _scatter_mask(storage, mask, value):
        """Fused boolean-mask scatter that avoids building index buffers.

        - If value is scalar: single pass kernel writes at mask positions.
        - If value is array: count K first, validate size, then single pass kernel maps values sequentially.
        """
        if tuple(mask.shape) != tuple(storage.shape):
            raise ValueError("boolean mask must have the same shape as tensor")
        if mask.dtype != "bool":
            raise ValueError("mask must be boolean")

        # Flatten contiguous views
        dst = storage
        if not dst.is_contiguous():
            raise ValueError("Target tensor must be contiguous for boolean scatter")
        dst_flat = dst
        if len(dst.shape) != 1:
            dst_flat = dst.reshape((-1,))

        m = mask
        if not m.is_contiguous():
            m = m.contiguous()
        mflat = m.reshape((-1,))

        N = mflat.size

        if isinstance(value, (int, float)):
            # Scalar assignment: direct masked store
            BLOCK = 1024
            grid = (triton.cdiv(N, BLOCK),)
            CUDAIndexingOps._scatter_mask_scalar_kernel[grid](
                dst_flat, mflat, float(value), N, BLOCK_SIZE=BLOCK
            )
            return

        # Array assignment: need to map values sequentially to True positions
        if not hasattr(value, "shape") or not hasattr(value, "ptr"):
            raise TypeError("Unsupported value type for boolean scatter: expected CUDAStorage or scalar")
        val_flat = value.reshape((-1,))

        # Count K true elements without allocating index buffers
        counter = cuda_utils.empty((1,), np.int32)
        cuda.cuMemsetD8(counter.ptr, 0, 4)
        BLOCK = 1024
        grid = (triton.cdiv(N, BLOCK),)
        CUDAIndexingOps._count_mask_kernel[grid](mflat, counter, N, BLOCK_SIZE=BLOCK)

        # Read back K and validate
        k_host = np.empty(1, dtype=np.int32)
        cuda.cuMemcpyDtoH(k_host, counter.ptr, 4)
        k = int(k_host[0])
        if val_flat.size != k:
            raise ValueError(f"boolean scatter values size ({val_flat.size}) must match mask true count ({k})")

        # Reset counter and scatter
        cuda.cuMemsetD8(counter.ptr, 0, 4)
        CUDAIndexingOps._scatter_mask_array_kernel[grid](
            dst_flat, mflat, val_flat, counter, N, BLOCK_SIZE=BLOCK
        )

    @staticmethod
    @triton.jit
    def _count_mask_kernel(mask_ptr, counter_ptr_i32, N, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        inb = offs < N
        v = tl.load(mask_ptr + offs, mask=inb, other=0)
        active = (v != 0) & inb
        cnt = tl.sum(active.to(tl.int32), axis=0)
        tl.atomic_add(counter_ptr_i32, cnt)

    @staticmethod
    @triton.jit
    def _scatter_mask_scalar_kernel(target_ptr, mask_ptr, value, N, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        inb = offs < N
        v = tl.load(mask_ptr + offs, mask=inb, other=0)
        active = (v != 0) & inb
        tl.store(target_ptr + offs, value, mask=active)

    @staticmethod
    @triton.jit
    def _scatter_mask_array_kernel(target_ptr, mask_ptr, values_ptr, counter_ptr_i32, N, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        inb = offs < N
        v = tl.load(mask_ptr + offs, mask=inb, other=0)
        active = (v != 0) & inb
        act_i32 = active.to(tl.int32)
        local = tl.cumsum(act_i32, axis=0) - act_i32
        cnt = tl.sum(act_i32, axis=0)
        base = tl.atomic_add(counter_ptr_i32, cnt)
        idx = base + local
        vals = tl.load(values_ptr + idx, mask=active, other=0)
        tl.store(target_ptr + offs, vals, mask=active)
    
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
    @triton.jit
    def _scatter_mixed_2d_scalar_kernel(
        dst_ptr, row_idx_ptr, col_idx_ptr,
        K, n_cols, value,
        BLOCK_SIZE: tl.constexpr,
    ):
        """Scatter scalar to dst[row[i], col[i]] without building linear index buffer."""
        pid = tl.program_id(0)
        offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        m = offs < K
        r = tl.load(row_idx_ptr + offs, mask=m, other=0).to(tl.int64)
        c = tl.load(col_idx_ptr + offs, mask=m, other=0).to(tl.int64)
        inb = (r >= 0) & (c >= 0) & (c < n_cols)
        lin = r * n_cols + c
        tl.store(dst_ptr + lin, value, mask=m & inb)

    @staticmethod
    @triton.jit
    def _scatter_mixed_2d_array_kernel(
        dst_ptr, row_idx_ptr, col_idx_ptr, vals_ptr,
        K, n_cols,
        BLOCK_SIZE: tl.constexpr,
    ):
        """Scatter array values to dst[row[i], col[i]]; vals[i] pairs with (row[i], col[i])."""
        pid = tl.program_id(0)
        offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        m = offs < K
        r = tl.load(row_idx_ptr + offs, mask=m, other=0).to(tl.int64)
        c = tl.load(col_idx_ptr + offs, mask=m, other=0).to(tl.int64)
        inb = (r >= 0) & (c >= 0) & (c < n_cols)
        lin = r * n_cols + c
        vals = tl.load(vals_ptr + offs, mask=m & inb, other=0)
        tl.store(dst_ptr + lin, vals, mask=m & inb)
    
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
        idx_buf_i32 = cuda_utils.empty((N,), np.int32)
        # Counter (int32)
        counter = cuda_utils.empty((1,), np.int32)
        # Zero the counter
        cuda.cuMemsetD8(counter.ptr, 0, 4)

        BLOCK = 1024
        grid = (triton.cdiv(N, BLOCK),)
        CUDAIndexingOps._compact_mask_atomic_i32[grid](flat, idx_buf_i32, counter, N, BLOCK=BLOCK, num_warps=4)

        # Read back counter (4 bytes)
        k_host = np.empty(1, dtype=np.int32)
        cuda.cuMemcpyDtoH(k_host, counter.ptr, 4)
        k = int(k_host[0])

        # Convert int32 indices to int64
        idx_buf_i64 = cuda_utils.empty((k,), np.int64)
        
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
        out = cuda_utils.empty((N,), src._numpy_dtype)

        BLOCK = 1024
        grid  = (triton.cdiv(N, BLOCK),)
        CUDAIndexingOps._gather_linear_kernel[grid](src, idx, out, N, BLOCK=BLOCK, num_warps=4)
        return out

    @staticmethod
    @triton.jit
    def _gather_mask_fused_kernel(
        src_ptr, mask_ptr, out_ptr,
        counter_ptr_i32,
        N,
        BLOCK: tl.constexpr,
    ):
        """Fused boolean-mask gather: compact + gather in one pass.

        - Uses single atomic per block to reserve output space.
        - Writes selected src elements directly into out at computed positions.
        """
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        inb = offs < N

        v = tl.load(mask_ptr + offs, mask=inb, other=0)
        active = inb & (v != 0)

        act_i32 = active.to(tl.int32)
        local = tl.cumsum(act_i32, axis=0) - act_i32
        cnt = tl.sum(act_i32, axis=0)
        base = tl.atomic_add(counter_ptr_i32, cnt)

        idx = base + local
        vals = tl.load(src_ptr + offs, mask=active, other=0)
        tl.store(out_ptr + idx, vals, mask=active)

    @staticmethod
    def _gather_rows_by_bool_mask(storage, mask):
        """Gather rows from storage where mask is True.

        Args:
            storage: Multi-dimensional CUDAStorage
            mask: 1D boolean mask with length == storage.shape[0]

        Returns:
            CUDAStorage with selected rows
        """
        # First, compact the boolean mask to get indices of True values
        # Use a simple kernel to build index array
        n_rows = mask.shape[0]
        m = mask.contiguous().reshape((-1,))

        # Allocate output arrays
        indices_full = cuda_utils.empty((n_rows,), np.int64)
        counter = cuda_utils.empty((1,), np.int32)
        cuda.cuMemsetD8(counter.ptr, 0, 4)

        # Compact boolean mask to indices
        BLOCK = 1024
        grid = (triton.cdiv(n_rows, BLOCK),)
        CUDAIndexingOps._compact_bool_to_indices_kernel[grid](
            m, indices_full, counter, n_rows, BLOCK=BLOCK, num_warps=4
        )

        # Read back count
        k_host = np.empty(1, dtype=np.int32)
        cuda.cuMemcpyDtoH(k_host, counter.ptr, 4)
        k = int(k_host[0])

        if k == 0:
            # No True values, return empty tensor
            row_size = int(np.prod(storage.shape[1:]))
            return storage.__class__((0, row_size) if len(storage.shape) > 1 else (0,),
                                    dtype=storage.dtype, ptr=None)

        # Create view of indices with actual count
        indices = storage.__class__((k,), dtype="int64", ptr=indices_full.ptr,
                                   strides=(1,), base=indices_full)

        # Now use regular row gathering
        return CUDAIndexingOps._getitem_tensor_gather(storage, indices)

    @staticmethod
    @triton.jit
    def _compact_bool_to_indices_kernel(mask_ptr, out_ptr, counter_ptr, N, BLOCK: tl.constexpr):
        """Compact boolean mask to integer indices.

        Each thread processes one element. If the mask is True,
        atomically increment counter to get output position.
        """
        pid = tl.program_id(0)
        offsets = pid * BLOCK + tl.arange(0, BLOCK)
        mask_valid = offsets < N

        # Load boolean values
        bool_vals = tl.load(mask_ptr + offsets, mask=mask_valid, other=False)

        # For each True value, atomically get an index and write position
        # Use where to only do atomic ops for True values
        true_mask = mask_valid & bool_vals

        # Atomic add returns the OLD value before increment
        # We need one atomic operation per True element
        output_indices = tl.zeros([BLOCK], dtype=tl.int32)
        output_indices = tl.where(true_mask, tl.atomic_add(counter_ptr, 1, sem="relaxed"), -1)

        # Store positions for True values
        valid_output = output_indices >= 0
        tl.store(out_ptr + output_indices, offsets.to(tl.int64), mask=valid_output)

    @staticmethod
    def _gather_mask_fused(storage, mask):
        """Fused boolean mask gather that avoids building linear index tensor.

        Supports two modes:
        1. Mask same shape as storage: element-wise selection
        2. 1D mask on multi-dim storage: row-wise selection
        """
        if mask.dtype != "bool":
            raise ValueError("mask must be boolean")

        # Check if this is 1D mask on multi-dimensional tensor (row selection)
        if len(mask.shape) == 1 and len(storage.shape) > 1 and mask.shape[0] == storage.shape[0]:
            # Row-wise boolean indexing: convert mask to indices and gather rows
            return CUDAIndexingOps._gather_rows_by_bool_mask(storage, mask)

        # Element-wise boolean indexing: mask must match storage shape exactly
        if tuple(mask.shape) != tuple(storage.shape):
            raise ValueError("boolean mask must have the same shape as tensor")

        # Ensure contiguous flattened views
        src = storage.contiguous()
        src = storage.__class__((src.size,), dtype=storage.dtype, ptr=src.ptr, strides=(1,), base=src)
        m = mask
        if not m.is_contiguous():
            m = m.contiguous()
        mflat = m.reshape((-1,))

        N = mflat.size
        # Allocate output upper bound and counter
        out_full = cuda_utils.empty((N,), src._numpy_dtype)
        counter = cuda_utils.empty((1,), np.int32)
        cuda.cuMemsetD8(counter.ptr, 0, 4)

        BLOCK = 1024
        grid = (triton.cdiv(N, BLOCK),)
        CUDAIndexingOps._gather_mask_fused_kernel[grid](
            src, mflat, out_full, counter, N, BLOCK=BLOCK, num_warps=4
        )

        # Read back the count
        k_host = np.empty(1, dtype=np.int32)
        cuda.cuMemcpyDtoH(k_host, counter.ptr, 4)
        k = int(k_host[0])

        # Create a size-k view over the same device buffer to avoid extra D2D copy
        if k == N:
            return out_full
        out_view = src.__class__((k,), dtype=src.dtype, ptr=out_full.ptr, strides=(1,), base=out_full)
        return out_view

    @staticmethod
    @triton.jit
    def _gather_rows_kernel(
        src_ptr, idx_ptr, out_ptr,
        num_rows, row_size, n_src_rows,
        COL_BLOCK: tl.constexpr,
    ):
        """2D row-wise gather without building a huge linear-index tensor.

        Grid:
          - program_id(0): row id in [0, num_rows)
          - program_id(1): column block id covering [0, row_size)
        """
        rid = tl.program_id(0)
        cb  = tl.program_id(1)
        cols = cb * COL_BLOCK + tl.arange(0, COL_BLOCK)

        m_row = rid < num_rows
        m_col = cols < row_size
        m = m_row & m_col

        ridx = tl.load(idx_ptr + rid, mask=m_row, other=0).to(tl.int64)
        valid_r = (ridx >= 0) & (ridx < n_src_rows)
        m = m & valid_r

        src_base = ridx * row_size
        out_base = rid * row_size

        vals = tl.load(src_ptr + src_base + cols, mask=m, other=0)
        tl.store(out_ptr + out_base + cols, vals, mask=m)

    @staticmethod
    @triton.jit
    def _gather_mixed_2d_kernel(
        src_ptr, row_idx_ptr, col_idx_ptr, out_ptr,
        K, n_cols,
        BLOCK_SIZE: tl.constexpr,
    ):
        """Gather values for pairs (row[i], col[i]) without building linear index buffer."""
        pid = tl.program_id(0)
        offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        m = offs < K
        r = tl.load(row_idx_ptr + offs, mask=m, other=0).to(tl.int64)
        c = tl.load(col_idx_ptr + offs, mask=m, other=0).to(tl.int64)
        inb = (r >= 0) & (c >= 0) & (c < n_cols)
        lin = r * n_cols + c
        vals = tl.load(src_ptr + lin, mask=m & inb, other=0)
        tl.store(out_ptr + offs, vals, mask=m & inb)
    
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
