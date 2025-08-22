"""
Indexing and manipulation operations for GPU backend.
"""
import triton
import triton.language as tl
from ..cuda_storage import CUDAStorage


# =============================================================================
# TRITON KERNELS
# =============================================================================

@triton.jit  
def fill_kernel(
    output_ptr, fill_value,
    n_elements: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fill tensor with a constant value.
    """
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    tl.store(output_ptr + offsets, fill_value, mask=mask)


@triton.jit
def expand_row_indices_kernel(
    row_indices_ptr, linear_indices_ptr,
    num_rows, row_size,
    BLOCK_SIZE: tl.constexpr
):
    """
    Expand row indices to linear indices for all elements in selected rows.
    Each row index becomes row_size consecutive linear indices.
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
def compact_kernel(
    input_ptr, mask_ptr, output_ptr, output_indices_ptr,
    n_elements: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Compact kernel: copy elements where mask is True to contiguous output.
    """
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    input_mask = offsets < n_elements
    
    # Load mask and input values
    mask_vals = tl.load(mask_ptr + offsets, mask=input_mask, other=False)
    input_vals = tl.load(input_ptr + offsets, mask=input_mask, other=0.0)
    
    # Load output indices (computed by prefix sum)
    output_idxs = tl.load(output_indices_ptr + offsets, mask=input_mask, other=-1)
    
    # Store to output where mask is True and we have valid output index
    valid_output_mask = input_mask & mask_vals & (output_idxs >= 0)
    tl.store(output_ptr + output_idxs, input_vals, mask=valid_output_mask)


@triton.jit
def boolean_count_kernel(
    mask_ptr, count_ptr,
    n_elements: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Count number of True elements in boolean mask.
    """
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load mask values and count True elements
    mask_vals = tl.load(mask_ptr + offsets, mask=mask, other=0.0)
    # Convert boolean to int and sum
    count = tl.sum(mask_vals.to(tl.int32))
    
    # Store count (only one thread should do this)
    if pid == 0:
        tl.store(count_ptr, count)


@triton.jit
def count_true_kernel(
    mask_ptr, count_ptr, n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Count number of True elements in boolean mask.
    """
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load mask values and count True elements in this block
    m = tl.load(mask_ptr + offsets, mask=mask, other=False)
    block_count = tl.sum(m.to(tl.int32))
    
    # Atomically add to global count
    if block_count > 0:
        tl.atomic_add(count_ptr, block_count)


@triton.jit  
def extract_indices_kernel(
    mask_ptr, out_idx_ptr, n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Extract indices where mask is True (simplified version).
    """
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load mask values
    m = tl.load(mask_ptr + offsets, mask=mask, other=False)
    
    # Simple approach: each thread writes its own index if True
    # This will have gaps but we'll compact later
    true_mask = mask & m
    tl.store(out_idx_ptr + offsets, offsets.to(tl.int64), mask=true_mask)


@triton.jit
def gather_linear_kernel(
    src_ptr, idx_ptr, out_ptr, n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Gather elements by linear indices.
    """
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load indices and gather data
    indices = tl.load(idx_ptr + offsets, mask=mask)
    values = tl.load(src_ptr + indices, mask=mask)
    tl.store(out_ptr + offsets, values, mask=mask)


@triton.jit
def scatter_linear_kernel(
    src_ptr, idx_ptr, out_ptr, n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Scatter elements by linear indices.
    """
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load source values and indices
    val = tl.load(src_ptr + offsets, mask=mask)
    idx = tl.load(idx_ptr + offsets, mask=mask)
    
    # Convert indices to int64 for pointer arithmetic
    idx_int64 = idx.to(tl.int64)
    
    # Scatter to output (last write wins for duplicate indices)
    tl.store(out_ptr + idx_int64, val, mask=mask)


@triton.jit
def gather_kernel(
    input_ptr, index_ptr, output_ptr,
    input_stride_0, input_stride_1, input_stride_2,
    index_stride_0, index_stride_1, index_stride_2,
    output_stride_0, output_stride_1, output_stride_2,
    input_size_0, input_size_1, input_size_2,
    index_size_0, index_size_1, index_size_2,
    gather_dim: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Gather operation along specified dimension.
    """
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Calculate total elements in output
    n_elements = index_size_0 * index_size_1 * index_size_2
    mask = offsets < n_elements
    
    # Convert linear offset to 3D coordinates
    idx_2 = offsets % index_size_2
    tmp = offsets // index_size_2
    idx_1 = tmp % index_size_1
    idx_0 = tmp // index_size_1
    
    # Load indices
    index_offset = (idx_0 * index_stride_0 + 
                   idx_1 * index_stride_1 + 
                   idx_2 * index_stride_2)
    indices = tl.load(index_ptr + index_offset, mask=mask)
    
    # Convert indices to int64 for address calculation
    indices_int64 = indices.to(tl.int64)
    
    # Calculate input offset based on gather dimension
    if gather_dim == 0:
        input_offset = (indices_int64 * input_stride_0 + 
                       idx_1 * input_stride_1 + 
                       idx_2 * input_stride_2)
    elif gather_dim == 1:
        input_offset = (idx_0 * input_stride_0 + 
                       indices_int64 * input_stride_1 + 
                       idx_2 * input_stride_2)
    else:  # gather_dim == 2
        input_offset = (idx_0 * input_stride_0 + 
                       idx_1 * input_stride_1 + 
                       indices_int64 * input_stride_2)
    
    # Load input values and store to output
    values = tl.load(input_ptr + input_offset, mask=mask)
    output_offset = (idx_0 * output_stride_0 + 
                    idx_1 * output_stride_1 + 
                    idx_2 * output_stride_2)
    tl.store(output_ptr + output_offset, values, mask=mask)


@triton.jit  
def scatter_kernel(
    input_ptr, index_ptr, src_ptr, output_ptr,
    input_stride_0, input_stride_1, input_stride_2,
    index_stride_0, index_stride_1, index_stride_2,
    src_stride_0, src_stride_1, src_stride_2,
    output_stride_0, output_stride_1, output_stride_2,
    index_size_0, index_size_1, index_size_2,
    scatter_dim: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Scatter operation along specified dimension.
    """
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Calculate total elements to scatter
    n_elements = index_size_0 * index_size_1 * index_size_2
    mask = offsets < n_elements
    
    # Convert linear offset to 3D coordinates
    idx_2 = offsets % index_size_2
    tmp = offsets // index_size_2
    idx_1 = tmp % index_size_1
    idx_0 = tmp // index_size_1
    
    # Load indices and source values
    index_offset = (idx_0 * index_stride_0 + 
                   idx_1 * index_stride_1 + 
                   idx_2 * index_stride_2)
    indices = tl.load(index_ptr + index_offset, mask=mask)
    
    src_offset = (idx_0 * src_stride_0 + 
                 idx_1 * src_stride_1 + 
                 idx_2 * src_stride_2)
    src_values = tl.load(src_ptr + src_offset, mask=mask)
    
    # Convert indices to int64 for address calculation
    indices_int64 = indices.to(tl.int64)
    
    # Calculate output offset based on scatter dimension
    if scatter_dim == 0:
        output_offset = (indices_int64 * output_stride_0 + 
                        idx_1 * output_stride_1 + 
                        idx_2 * output_stride_2)
    elif scatter_dim == 1:
        output_offset = (idx_0 * output_stride_0 + 
                        indices_int64 * output_stride_1 + 
                        idx_2 * output_stride_2)
    else:  # scatter_dim == 2
        output_offset = (idx_0 * output_stride_0 + 
                        idx_1 * output_stride_1 + 
                        indices_int64 * output_stride_2)
    
    # Store values to output
    tl.store(output_ptr + output_offset, src_values, mask=mask)


# =============================================================================
# GPU OPERATIONS
# =============================================================================


def getitem(x, idxs):
    """
    Get tensor elements by indices with optimizations for common cases.
    """
    if not isinstance(x, CUDAStorage):
        return x.__getitem__(idxs)
    
    # For simple int/slice indexing, CUDAStorage already handles efficiently (no CPU roundtrip)
    if isinstance(idxs, (int, slice)):
        return x.__getitem__(idxs)
    
    # For tuple of int/slice, also efficient
    if isinstance(idxs, tuple) and all(isinstance(idx, (int, slice)) for idx in idxs):
        return x.__getitem__(idxs)
    
    # For boolean indexing with CUDAStorage mask, try to optimize
    if isinstance(idxs, CUDAStorage) and idxs.dtype == "bool":
        # This could be optimized with GPU kernels in the future
        # For now, use the existing implementation but it's isolated here
        return x.__getitem__(idxs)
    
    # For other complex cases, fallback to CUDAStorage implementation
    return x.__getitem__(idxs)


def setitem(x, idxs, other):
    """
    Set tensor elements by indices with optimizations for common cases.
    Supports broadcasting to match CPU/PyTorch behavior.
    """
    if not isinstance(x, CUDAStorage):
        return x.__setitem__(idxs, other)
    
    # Handle broadcasting for CUDAStorage (to match CPU/PyTorch behavior)
    if isinstance(other, CUDAStorage):
        # Get target shape by creating a temporary view
        target_view = x[idxs]
        target_shape = target_view.shape
        
        # Check if broadcasting is needed
        if other.shape != target_shape:
            # Broadcast other to target shape
            other = other.broadcast_to(target_shape)
    
    # For simple cases, CUDAStorage handles efficiently
    if isinstance(idxs, (int, slice)):
        return x.__setitem__(idxs, other)
    
    if isinstance(idxs, tuple) and all(isinstance(idx, (int, slice)) for idx in idxs):
        return x.__setitem__(idxs, other)
    
    # For complex cases, fallback to CUDAStorage implementation  
    return x.__setitem__(idxs, other)


def fill(tensor, value):
    """
    Fill tensor with constant value.
    """
    if not isinstance(tensor, CUDAStorage):
        tensor.fill_(value)
        return tensor
    
    if not tensor.is_contiguous():
        tensor = tensor.contiguous()
    
    n_elements = tensor.size
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]), )
    fill_kernel[grid](tensor, value, n_elements, BLOCK_SIZE=1024)
    
    return tensor


def fill_tensor(tensor, value):
    """
    Fill tensor with constant value (alias for fill).
    """
    return fill(tensor, value)


def gather(input_tensor, dim, index):
    """
    Gather values along dimension using indices.
    
    Args:
        input_tensor: Input CUDAStorage tensor
        dim: Dimension to gather along
        index: CUDAStorage tensor with indices
        
    Returns:
        CUDAStorage: Gathered values
    """
    if not isinstance(input_tensor, CUDAStorage) or not isinstance(index, CUDAStorage):
        # Fallback for non-CUDA tensors
        import torch
        return torch.gather(input_tensor, dim, index)
    
    # Create output tensor with same shape as index
    output = CUDAStorage(index.shape, dtype=input_tensor.dtype)
    
    # Handle up to 3D tensors (extend strides with 1s for lower dimensions)
    input_shape = list(input_tensor.shape) + [1] * (3 - len(input_tensor.shape))
    index_shape = list(index.shape) + [1] * (3 - len(index.shape))
    
    input_strides = list(input_tensor.strides) + [1] * (3 - len(input_tensor.strides))
    index_strides = list(index.strides) + [1] * (3 - len(index.strides))
    output_strides = list(output.strides) + [1] * (3 - len(output.strides))
    
    n_elements = index.size
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    
    gather_kernel[grid](
        input_tensor, index, output,
        input_strides[0], input_strides[1], input_strides[2],
        index_strides[0], index_strides[1], index_strides[2],
        output_strides[0], output_strides[1], output_strides[2],
        input_shape[0], input_shape[1], input_shape[2],
        index_shape[0], index_shape[1], index_shape[2],
        gather_dim=dim,
        BLOCK_SIZE=1024
    )
    
    return output


def scatter(input_tensor, dim, index, src):
    """
    Scatter values from src along dimension using indices.
    
    Args:
        input_tensor: Input CUDAStorage tensor to scatter into
        dim: Dimension to scatter along
        index: CUDAStorage tensor with indices
        src: CUDAStorage tensor with source values
        
    Returns:
        CUDAStorage: Scattered values
    """
    if not all(isinstance(t, CUDAStorage) for t in [input_tensor, index, src]):
        # Fallback for non-CUDA tensors
        import torch
        return input_tensor.scatter(dim, index, src)
    
    # Create output tensor as copy of input
    output = input_tensor.clone()
    
    # Handle up to 3D tensors (extend strides with 1s for lower dimensions)
    input_shape = list(input_tensor.shape) + [1] * (3 - len(input_tensor.shape))
    index_shape = list(index.shape) + [1] * (3 - len(index.shape))
    src_shape = list(src.shape) + [1] * (3 - len(src.shape))
    
    input_strides = list(input_tensor.strides) + [1] * (3 - len(input_tensor.strides))
    index_strides = list(index.strides) + [1] * (3 - len(index.strides))
    src_strides = list(src.strides) + [1] * (3 - len(src.strides))
    output_strides = list(output.strides) + [1] * (3 - len(output.strides))
    
    n_elements = index.size
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    
    scatter_kernel[grid](
        input_tensor, index, src, output,
        input_strides[0], input_strides[1], input_strides[2],
        index_strides[0], index_strides[1], index_strides[2],
        src_strides[0], src_strides[1], src_strides[2],
        output_strides[0], output_strides[1], output_strides[2],
        index_shape[0], index_shape[1], index_shape[2],
        scatter_dim=dim,
        BLOCK_SIZE=1024
    )
    
    return output


def cat(arrays, dim=0):
    """
    Concatenate CUDAStorage arrays along specified dimension.
    
    Args:
        arrays: List of CUDAStorage objects to concatenate
        dim: Dimension along which to concatenate
        
    Returns:
        CUDAStorage: Concatenated array
    """
    return CUDAStorage.cat(arrays, dim=dim)