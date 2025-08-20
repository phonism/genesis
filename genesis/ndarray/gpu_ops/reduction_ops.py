"""
Reduction operations for GPU backend.
"""
import triton
import triton.language as tl
from functools import reduce as functools_reduce
import operator
from ..cuda_storage import CUDAStorage


# =============================================================================
# TRITON KERNELS
# =============================================================================

@triton.jit
def sum_kernel(x_ptr, output_ptr, M, N, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    """
    Optimized sum kernel with better memory access patterns.
    """
    pid_m = tl.program_id(axis=0)
    m_offset = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    m_mask = m_offset < M
    out = tl.zeros((BLOCK_M,), dtype=tl.float32)
    
    for start in range(0, N, BLOCK_N):
        n_offset = start + tl.arange(0, BLOCK_N)
        offset = m_offset[:, None] * N + n_offset[None, :]
        n_mask = n_offset < N
        mask = m_mask[:, None] & n_mask[None, :]
        inp = tl.load(x_ptr + offset, mask=mask, other=0.0)
        out += tl.sum(inp, axis=1)

    tl.store(output_ptr + m_offset, out, mask=m_mask)


@triton.jit 
def sum_full_kernel(x_ptr, output_ptr, N, BLOCK_SIZE: tl.constexpr):
    """
    Optimized kernel for full tensor reduction - single scalar output.
    """
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    
    # Load and sum this block
    values = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    block_sum = tl.sum(values)
    
    # Atomic add to output (single scalar)
    tl.atomic_add(output_ptr, block_sum)


@triton.jit
def max_kernel(x_ptr, output_ptr, M, N, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    """
    Optimized max kernel with better memory access patterns.
    """
    pid_m = tl.program_id(axis=0)
    m_offset = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    m_mask = m_offset < M
    out = tl.full((BLOCK_M,), -float("inf"), dtype=tl.float32)
    
    for start in range(0, N, BLOCK_N):
        n_offset = start + tl.arange(0, BLOCK_N)
        offset = m_offset[:, None] * N + n_offset[None, :]
        n_mask = n_offset < N
        mask = m_mask[:, None] & n_mask[None, :]
        inp = tl.load(x_ptr + offset, mask=mask, other=-float("inf"))
        out = tl.maximum(out, tl.max(inp, axis=1))

    tl.store(output_ptr + m_offset, out, mask=m_mask)


@triton.jit 
def max_full_kernel(x_ptr, output_ptr, N, BLOCK_SIZE: tl.constexpr):
    """
    Optimized kernel for full tensor max reduction - single scalar output.
    Uses atomic max for combining results across blocks.
    """
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    
    # Load and find max in this block
    values = tl.load(x_ptr + offsets, mask=mask, other=-float("inf"))
    block_max = tl.max(values)
    
    # Use atomic max to combine with global max
    # Note: We need to implement this carefully since atomic_max may not be available
    # Use a workaround with atomic compare-and-swap
    tl.atomic_max(output_ptr, block_max)


# All specialized 2D/3D kernels removed - only using general kernels for stability


# =============================================================================
# GPU OPERATIONS
# =============================================================================


def reduce_sum(x, axis=None, keepdims=False):
    """
    Optimized reduce sum operation.
    """    
    shape = x.shape
    ndim = len(shape)
    
    # Fast path: full tensor reduction to scalar
    if axis is None:
        # Ensure contiguous
        if not x.is_contiguous():
            x = x.contiguous()
            
        n_elements = x.size
        output_shape = (1,) if keepdims else ()
        output = CUDAStorage(output_shape, dtype=x.dtype)
        
        # Initialize output to zero
        if output.size > 0:  # Avoid indexing empty tensors
            output.fill(0.0)  # Use fill instead of slice assignment
        
        # Standard Triton pattern: use power-of-2 block size with masking
        # Block size is always power of 2 for performance
        block_size = 1024  # Fixed power-of-2 block size
        grid = (triton.cdiv(n_elements, block_size),)
        
        sum_full_kernel[grid](x, output, n_elements, block_size)
        return output
    
    # Normalize axis
    if isinstance(axis, int):
        axis = (axis,)
    else:
        axis = tuple(axis)
    
    # TEMPORARY: Force all operations to use general fallback
    # Skip all 2D/3D optimizations to test if they cause the issue
    # Fast path: single axis reduction
    if len(axis) == 1:
        ax = axis[0]
        if ax < 0:
            ax += ndim
            
        # Calculate output shape
        output_shape = list(shape)
        if keepdims:
            output_shape[ax] = 1
        else:
            output_shape.pop(ax)
        output_shape = tuple(output_shape)
        
        # Ensure contiguous
        if not x.is_contiguous():
            x = x.contiguous()
        
        # All specialized 2D/3D kernels removed for stability
        # Use only the general fallback implementation for all cases
        
        # All other cases use general fallback
        
        # General fallback: use reshape approach for other dimensions
        if ax == ndim - 1:  # Last axis - coalesced memory access
            m = functools_reduce(operator.mul, shape[:-1], 1)
            n = shape[-1]
        else:  # Other axes - need permute
            axes_to_keep = tuple(i for i in range(ndim) if i != ax)
            new_order = axes_to_keep + (ax,)
            x = x.permute(new_order)
            m = functools_reduce(operator.mul, [shape[i] for i in axes_to_keep], 1)
            n = shape[ax]
        
        x = x.reshape((m, n))
        temp_output = CUDAStorage((m,), dtype=x.dtype)
        
        # Optimized block configuration - both must be power of 2
        block_m = 32  # Fixed power-of-2 for M dimension
        block_n = min(2048, triton.next_power_of_2(n))  # Power-of-2 for N dimension
        grid = (triton.cdiv(m, block_m),)
        
        sum_kernel[grid](x, temp_output, m, n, block_m, block_n)
        
        # Reshape to final output shape
        output = temp_output.reshape(output_shape)
        return output
    
    # Multi-axis reduction: fallback to original implementation but optimized
    axes_to_keep = tuple(i for i in range(ndim) if i not in axis)
    new_order = axes_to_keep + axis

    x = x.permute(new_order)
    new_shape = tuple(shape[i] for i in axes_to_keep) + tuple(shape[i] for i in axis)

    m = functools_reduce(operator.mul, new_shape[:len(axes_to_keep)], 1)
    n = functools_reduce(operator.mul, new_shape[len(axes_to_keep):], 1)
    x = x.reshape((m, n))
    
    output_shape = tuple(new_shape[i] for i in range(len(axes_to_keep)))
    if keepdims:
        output_shape = list(shape)
        for i in axis:
            output_shape[i] = 1
        output_shape = tuple(output_shape)
    
    output = CUDAStorage(output_shape, dtype=x.dtype)

    # Ensure contiguous once
    if not x.is_contiguous():
        x = x.contiguous()
    
    # Better block configuration - both must be power of 2
    block_m = 32  # Fixed power-of-2 for M dimension  
    block_n = min(2048, triton.next_power_of_2(n))  # Power-of-2 for N dimension
    grid = (triton.cdiv(m, block_m),)
    
    sum_kernel[grid](x, output, m, n, block_m, block_n)
    return output


def reduce_max(x, axis=None, keepdims=False):
    """
    Optimized reduce max operation.
    """    
    shape = x.shape
    ndim = len(shape)
    
    # Fast path: full tensor reduction to scalar
    if axis is None:
        # Ensure contiguous
        if not x.is_contiguous():
            x = x.contiguous()
            
        n_elements = x.size
        output_shape = (1,) if keepdims else ()
        output = CUDAStorage(output_shape, dtype=x.dtype)
        
        # Initialize output to -inf
        if output.size > 0:  # Avoid indexing empty tensors
            output.fill(float('-inf'))  # Use fill for initialization
        
        # Standard Triton pattern: use power-of-2 block size with masking
        # Block size is always power of 2 for performance
        block_size = 1024  # Fixed power-of-2 block size
        grid = (triton.cdiv(n_elements, block_size),)
        
        max_full_kernel[grid](x, output, n_elements, block_size)
        return output
    
    # Normalize axis
    if isinstance(axis, int):
        axis = (axis,)
    else:
        axis = tuple(axis)
    
    # Fast path: single axis reduction
    if len(axis) == 1:
        ax = axis[0]
        if ax < 0:
            ax += ndim
            
        # Calculate output shape
        output_shape = list(shape)
        if keepdims:
            output_shape[ax] = 1
        else:
            output_shape.pop(ax)
        output_shape = tuple(output_shape)
        
        # Ensure contiguous
        if not x.is_contiguous():
            x = x.contiguous()
        
        # All specialized 2D kernels removed for stability
        
        # All specialized 3D kernels removed for stability
        
        # General fallback: use reshape approach for other dimensions
        if ax == ndim - 1:  # Last axis - coalesced memory access
            m = functools_reduce(operator.mul, shape[:-1], 1)
            n = shape[-1]
        else:  # Other axes - need permute
            axes_to_keep = tuple(i for i in range(ndim) if i != ax)
            new_order = axes_to_keep + (ax,)
            x = x.permute(new_order)
            m = functools_reduce(operator.mul, [shape[i] for i in axes_to_keep], 1)
            n = shape[ax]
        
        x = x.reshape((m, n))
        temp_output = CUDAStorage((m,), dtype=x.dtype)
        
        # Optimized block configuration - both must be power of 2
        block_m = 32  # Fixed power-of-2 for M dimension
        block_n = min(2048, triton.next_power_of_2(n))  # Power-of-2 for N dimension
        grid = (triton.cdiv(m, block_m),)
        
        max_kernel[grid](x, temp_output, m, n, block_m, block_n)
        
        # Reshape to final output shape
        output = temp_output.reshape(output_shape)
        return output
    
    # Multi-axis reduction: fallback to original implementation but optimized
    axes_to_keep = tuple(i for i in range(ndim) if i not in axis)
    new_order = axes_to_keep + axis

    x = x.permute(new_order)
    new_shape = tuple(shape[i] for i in axes_to_keep) + tuple(shape[i] for i in axis)

    m = functools_reduce(operator.mul, new_shape[:len(axes_to_keep)], 1)
    n = functools_reduce(operator.mul, new_shape[len(axes_to_keep):], 1)
    x = x.reshape((m, n))
    
    output_shape = tuple(new_shape[i] for i in range(len(axes_to_keep)))
    if keepdims:
        output_shape = list(shape)
        for i in axis:
            output_shape[i] = 1
        output_shape = tuple(output_shape)
    
    output = CUDAStorage(output_shape, dtype=x.dtype)

    # Ensure contiguous once
    if not x.is_contiguous():
        x = x.contiguous()
    
    # Better block configuration - both must be power of 2
    block_m = 32  # Fixed power-of-2 for M dimension  
    block_n = min(2048, triton.next_power_of_2(n))  # Power-of-2 for N dimension
    grid = (triton.cdiv(m, block_m),)
    
    max_kernel[grid](x, output, m, n, block_m, block_n)
    return output