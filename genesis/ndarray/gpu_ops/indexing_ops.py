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
    """
    if not isinstance(x, CUDAStorage):
        return x.__setitem__(idxs, other)
    
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