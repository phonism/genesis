"""
Reduction operations for GPU backend.
"""
import triton
import triton.language as tl
from functools import reduce as functools_reduce
import operator
import os
import math
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
# TWO-STAGE REDUCTION KERNELS (Flag-Gems inspired)
# =============================================================================

@triton.jit
def sum_kernel_two_stage_1(
    inp_ptr, 
    mid_ptr, 
    M, 
    BLOCK_SIZE: tl.constexpr,
):
    """
    Two-stage reduction: Stage 1 - Compute partial sums for each block.
    Each block processes BLOCK_SIZE elements and stores partial sum in mid_ptr.
    """
    # Determine data type for computation
    if tl.constexpr(inp_ptr.dtype.element_ty == tl.float16) or tl.constexpr(
        inp_ptr.dtype.element_ty == tl.bfloat16
    ):
        cdtype = tl.float32
    else:
        cdtype = inp_ptr.dtype.element_ty

    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < M

    # Load values and compute partial sum
    inp_val = tl.load(inp_ptr + offset, mask=mask, other=0.0).to(cdtype)
    sum_val = tl.sum(inp_val)
    
    # Store partial sum
    tl.store(mid_ptr + pid, sum_val)


@triton.jit
def sum_kernel_two_stage_2(
    mid_ptr, 
    out_ptr, 
    mid_size, 
    BLOCK_MID: tl.constexpr
):
    """
    Two-stage reduction: Stage 2 - Sum all partial results.
    Combines all partial sums from Stage 1 into final result.
    """
    # Determine data type for computation
    if tl.constexpr(mid_ptr.dtype.element_ty == tl.float16) or tl.constexpr(
        mid_ptr.dtype.element_ty == tl.bfloat16
    ):
        cdtype = tl.float32
    else:
        cdtype = mid_ptr.dtype.element_ty

    offset = tl.arange(0, BLOCK_MID)
    mask = offset < mid_size
    
    mid_val = tl.load(mid_ptr + offset, mask=mask, other=0.0).to(cdtype)
    sum_val = tl.sum(mid_val)
    
    tl.store(out_ptr, sum_val)


@triton.jit
def sum_kernel_inner_dim(
    output_ptr,
    input_ptr,
    M,
    N,
    TILE_N: tl.constexpr,
    ONE_TILE_PER_CTA: tl.constexpr,
):
    """
    Specialized kernel for inner dimension reduction.
    Optimized for reducing the last dimension of tensors.
    """
    if tl.constexpr(input_ptr.dtype.element_ty == tl.float16) or tl.constexpr(
        input_ptr.dtype.element_ty == tl.bfloat16
    ):
        cdtype = tl.float32
    else:
        cdtype = input_ptr.dtype.element_ty

    pid_m = tl.program_id(0)
    
    if ONE_TILE_PER_CTA:
        # Single tile handles entire N dimension
        n_offsets = tl.arange(0, TILE_N)
        inp_offset = pid_m * N + n_offsets
        mask = n_offsets < N
        inp = tl.load(input_ptr + inp_offset, mask=mask, other=0.0).to(cdtype)
        out = tl.sum(inp)
        tl.store(output_ptr + pid_m, out)
    else:
        # Multiple tiles needed for N dimension - use scalar accumulation for better precision
        sum_acc = tl.zeros((), dtype=cdtype)  # Scalar accumulator
        for start_n in range(0, N, TILE_N):
            n_offsets = start_n + tl.arange(0, TILE_N)
            inp_offsets = pid_m * N + n_offsets
            mask = n_offsets < N
            inp = tl.load(input_ptr + inp_offsets, mask=mask, other=0.0).to(cdtype)
            # Sum this tile and add to scalar accumulator for better numerical stability
            tile_sum = tl.sum(inp)
            sum_acc += tile_sum
        tl.store(output_ptr + pid_m, sum_acc)


@triton.jit
def sum_kernel_non_inner_dim(
    output_ptr,
    input_ptr,
    M,
    N,
    K,
    TILE_N: tl.constexpr,
    TILE_K: tl.constexpr,
    ONE_TILE_PER_CTA: tl.constexpr,
):
    """
    Specialized kernel for non-inner dimension reduction.
    Handles reduction along middle dimensions with shape [M, N, K].
    """
    if tl.constexpr(input_ptr.dtype.element_ty == tl.float16) or tl.constexpr(
        input_ptr.dtype.element_ty == tl.bfloat16
    ):
        cdtype = tl.float32
    else:
        cdtype = input_ptr.dtype.element_ty

    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)

    k_offsets = pid_k * TILE_K + tl.arange(0, TILE_K)[None, :]

    if ONE_TILE_PER_CTA:
        # Single tile handles entire N dimension
        n_offsets = tl.arange(0, TILE_N)[:, None]
        inp_offset = pid_m * N * K + n_offsets * K + k_offsets
        mask = (n_offsets < N) & (k_offsets < K)
        inp = tl.load(input_ptr + inp_offset, mask=mask, other=0.0).to(cdtype)
        out = tl.sum(inp, axis=0, keep_dims=True)
        out_offset = pid_m * K + k_offsets
        tl.store(output_ptr + out_offset, out, mask=k_offsets < K)
    else:
        # Multiple tiles needed for N dimension
        sum_acc = tl.zeros([TILE_N, TILE_K], dtype=cdtype)
        for start_n in range(0, N, TILE_N):
            n_offsets = start_n + tl.arange(0, TILE_N)[:, None]
            inp_offsets = pid_m * N * K + n_offsets * K + k_offsets
            mask = (n_offsets < N) & (k_offsets < K)
            inp = tl.load(input_ptr + inp_offsets, mask=mask, other=0.0).to(cdtype)
            sum_acc += inp
        out = tl.sum(sum_acc, axis=0, keep_dims=True)
        out_offset = pid_m * K + k_offsets
        tl.store(output_ptr + out_offset, out, mask=k_offsets < K)


# =============================================================================
# TWO-STAGE MAX REDUCTION KERNELS (Flag-Gems inspired)
# =============================================================================

@triton.jit
def max_kernel_two_stage_1(
    inp_ptr, 
    partial_ptr,
    N,
    BLOCK_SIZE: tl.constexpr,
):
    """Stage 1: Compute partial max values for blocks."""
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < N
    
    # Load data with -inf for out-of-bounds
    vals = tl.load(inp_ptr + offset, mask=mask, other=-float("inf"))
    
    # Compute block max
    block_max = tl.max(vals)
    
    # Store partial result
    tl.store(partial_ptr + pid, block_max)


@triton.jit  
def max_kernel_two_stage_2(
    partial_ptr, 
    output_ptr, 
    num_blocks,
    BLOCK_SIZE: tl.constexpr,
):
    """Stage 2: Reduce partial max values to final result."""
    pid = tl.program_id(0)
    
    if pid == 0:  # Only one thread block for stage 2
        offset = tl.arange(0, BLOCK_SIZE)
        mask = offset < num_blocks
        
        # Load partial results with -inf for padding
        vals = tl.load(partial_ptr + offset, mask=mask, other=-float("inf"))
        
        # Final reduction
        result = tl.max(vals)
        
        # Store final result
        tl.store(output_ptr, result)


@triton.jit
def max_kernel_inner_dim(
    output_ptr,
    input_ptr,
    M,
    N,
    TILE_M: tl.constexpr,
    TILE_N: tl.constexpr,
    ONE_TILE_PER_CTA: tl.constexpr,
):
    """Optimized kernel for max reduction along inner dimension."""
    pid_m = tl.program_id(0)
    
    m_offsets = pid_m * TILE_M + tl.arange(0, TILE_M)[:, None]
    
    if ONE_TILE_PER_CTA:
        # Single tile handles entire N dimension
        n_offsets = tl.arange(0, TILE_N)[None, :]
        inp_offset = m_offsets * N + n_offsets
        mask = (m_offsets < M) & (n_offsets < N)
        inp = tl.load(input_ptr + inp_offset, mask=mask, other=-float("inf"))
        out = tl.max(inp, axis=1, keep_dims=True)
        tl.store(output_ptr + m_offsets, out, mask=m_offsets < M)
    else:
        # Multiple tiles needed for N dimension
        max_acc = tl.full([TILE_M, TILE_N], -float("inf"), dtype=tl.float32)
        for start_n in range(0, N, TILE_N):
            n_offsets = start_n + tl.arange(0, TILE_N)[None, :]
            inp_offsets = m_offsets * N + n_offsets
            mask = (m_offsets < M) & (n_offsets < N)
            inp = tl.load(input_ptr + inp_offsets, mask=mask, other=-float("inf"))
            max_acc = tl.maximum(max_acc, inp)
        out = tl.max(max_acc, axis=1, keep_dims=True)
        tl.store(output_ptr + m_offsets, out, mask=m_offsets < M)


@triton.jit
def max_kernel_non_inner_dim(
    output_ptr,
    input_ptr,
    M,
    N,
    K,
    TILE_N: tl.constexpr,
    TILE_K: tl.constexpr,
    ONE_TILE_PER_CTA: tl.constexpr,
):
    """Optimized kernel for max reduction along non-inner dimensions."""
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)

    k_offsets = pid_k * TILE_K + tl.arange(0, TILE_K)[None, :]

    if ONE_TILE_PER_CTA:
        # Single tile handles entire N dimension
        n_offsets = tl.arange(0, TILE_N)[:, None]
        inp_offset = pid_m * N * K + n_offsets * K + k_offsets
        mask = (n_offsets < N) & (k_offsets < K)
        inp = tl.load(input_ptr + inp_offset, mask=mask, other=-float("inf"))
        out = tl.max(inp, axis=0, keep_dims=True)
        out_offset = pid_m * K + k_offsets
        tl.store(output_ptr + out_offset, out, mask=k_offsets < K)
    else:
        # Multiple tiles needed for N dimension
        max_acc = tl.full([TILE_N, TILE_K], -float("inf"), dtype=tl.float32)
        for start_n in range(0, N, TILE_N):
            n_offsets = start_n + tl.arange(0, TILE_N)[:, None]
            inp_offsets = pid_m * N * K + n_offsets * K + k_offsets
            mask = (n_offsets < N) & (k_offsets < K)
            inp = tl.load(input_ptr + inp_offsets, mask=mask, other=-float("inf"))
            max_acc = tl.maximum(max_acc, inp)
        out = tl.max(max_acc, axis=0, keep_dims=True)
        out_offset = pid_m * K + k_offsets
        tl.store(output_ptr + out_offset, out, mask=k_offsets < K)


# =============================================================================
# GPU OPERATIONS
# =============================================================================


def reduce_sum_v1(x, axis=None, keepdims=False):
    """
    Original reduce sum implementation (v1).
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


def reduce_sum_v2(x, axis=None, keepdims=False):
    """
    Two-stage reduction implementation (v2) - Flag-Gems inspired.
    """    
    shape = x.shape
    ndim = len(shape)
    
    # Fast path: full tensor reduction to scalar using two-stage approach
    if axis is None:
        # Ensure contiguous
        if not x.is_contiguous():
            x = x.contiguous()
            
        n_elements = x.size
        output_shape = (1,) if keepdims else ()
        output = CUDAStorage(output_shape, dtype=x.dtype)
        
        # Two-stage reduction strategy
        # Stage 1: Calculate optimal block size (Flag-Gems approach)
        block_size = triton.next_power_of_2(math.ceil(math.sqrt(n_elements)))
        block_size = min(block_size, 1024)  # Cap at 1024 for stability
        mid_size = triton.cdiv(n_elements, block_size)
        
        # Create intermediate storage for partial sums
        mid = CUDAStorage((mid_size,), dtype=x.dtype)
        
        # Stage 1: Compute partial sums
        grid1 = (mid_size,)
        sum_kernel_two_stage_1[grid1](x, mid, n_elements, block_size)
        
        # Stage 2: Sum partial results
        block_mid = triton.next_power_of_2(mid_size)
        grid2 = (1,)
        sum_kernel_two_stage_2[grid2](mid, output, mid_size, block_mid)
        
        return output
    
    # Normalize axis
    if isinstance(axis, int):
        axis = (axis,)
    else:
        axis = tuple(axis)
    
    # Fast path: single axis reduction with optimized kernels
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
        
        # Use specialized kernels for different cases
        if ax == ndim - 1:  # Last axis - use inner dimension kernel
            m = functools_reduce(operator.mul, shape[:-1], 1)
            n = shape[-1]
            
            output = CUDAStorage(output_shape, dtype=x.dtype)
            x = x.reshape((m, n))
            
            # Choose tile size and mode based on N
            tile_n = min(triton.next_power_of_2(n), 2048)
            one_tile_per_cta = (tile_n >= n)
            
            grid = (m,)
            sum_kernel_inner_dim[grid](output, x, m, n, tile_n, one_tile_per_cta)
            
            return output.reshape(output_shape)
            
        else:  # Other axes - need permute, use non-inner dimension kernel
            axes_to_keep = tuple(i for i in range(ndim) if i != ax)
            new_order = axes_to_keep + (ax,)
            x = x.permute(new_order)
            
            # Reshape to [M, N, K] format for non-inner kernel
            m = functools_reduce(operator.mul, [shape[i] for i in axes_to_keep], 1) 
            n = shape[ax]
            k = 1  # For single axis, K is always 1
            
            x = x.reshape((m, n, k))
            temp_output = CUDAStorage((m, k), dtype=x.dtype)
            
            # Use non-inner dimension kernel
            tile_n = min(triton.next_power_of_2(n), 1024)
            tile_k = 1  # K=1 for single axis reduction
            one_tile_per_cta = (tile_n >= n)
            
            grid = (m, 1)  # K dimension grid is 1
            sum_kernel_non_inner_dim[grid](
                temp_output, x, m, n, k, tile_n, tile_k, one_tile_per_cta
            )
            
            # Reshape to final output shape
            return temp_output.reshape(output_shape)
    
    # Multi-axis reduction: fallback to v1 approach for now
    return reduce_sum_v1(x, axis, keepdims)


def reduce_sum_v3(x, axis=None, keepdims=False):
    """
    Advanced reduction implementation (v3) - Full Flag-Gems optimizations.
    """    
    shape = x.shape
    ndim = len(shape)
    
    # Fast path: full tensor reduction using improved two-stage approach
    if axis is None:
        # Ensure contiguous
        if not x.is_contiguous():
            x = x.contiguous()
            
        n_elements = x.size
        output_shape = (1,) if keepdims else ()
        output = CUDAStorage(output_shape, dtype=x.dtype)
        
        # Enhanced two-stage reduction strategy with better block size selection
        if n_elements <= 1024:
            # Small tensors: use single stage with optimal block size
            block_size = triton.next_power_of_2(n_elements)
            grid = (1,)
            sum_kernel_two_stage_2[grid](x, output, n_elements, block_size)
        else:
            # Large tensors: use two-stage with adaptive block size
            # Use a more sophisticated block size calculation
            optimal_blocks = min(triton.cdiv(n_elements, 256), 512)  # Limit max blocks
            block_size = triton.cdiv(n_elements, optimal_blocks)
            block_size = triton.next_power_of_2(block_size)
            block_size = max(block_size, 64)  # Minimum block size
            
            mid_size = triton.cdiv(n_elements, block_size)
            mid = CUDAStorage((mid_size,), dtype=x.dtype)
            
            # Stage 1: Compute partial sums
            grid1 = (mid_size,)
            sum_kernel_two_stage_1[grid1](x, mid, n_elements, block_size)
            
            # Stage 2: Sum partial results with optimal block size
            block_mid = triton.next_power_of_2(mid_size)
            block_mid = min(block_mid, 1024)  # Cap block size
            grid2 = (1,)
            sum_kernel_two_stage_2[grid2](mid, output, mid_size, block_mid)
        
        return output
    
    # Normalize axis
    if isinstance(axis, int):
        axis = (axis,)
    else:
        axis = tuple(axis)
    
    # Enhanced single axis reduction with better kernel selection
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
        
        # Enhanced kernel selection based on reduction characteristics
        if ax == ndim - 1:  # Last axis - use optimized inner dimension kernel
            m = functools_reduce(operator.mul, shape[:-1], 1)
            n = shape[-1]
            
            output = CUDAStorage(output_shape, dtype=x.dtype)
            x = x.reshape((m, n))
            
            # Adaptive tile size based on N and hardware characteristics
            if n <= 128:
                tile_n = triton.next_power_of_2(n)
                one_tile_per_cta = True
            elif n <= 2048:
                tile_n = min(triton.next_power_of_2(n), 1024)
                one_tile_per_cta = (tile_n >= n)
            else:
                # Large N: use multiple tiles with optimal size
                tile_n = 1024
                one_tile_per_cta = False
            
            grid = (m,)
            sum_kernel_inner_dim[grid](output, x, m, n, tile_n, one_tile_per_cta)
            
            return output.reshape(output_shape)
            
        else:  # Non-inner axes - use enhanced non-inner dimension kernel
            axes_to_keep = tuple(i for i in range(ndim) if i != ax)
            new_order = axes_to_keep + (ax,)
            x = x.permute(new_order)
            
            # Enhanced reshape for non-inner kernel
            m = functools_reduce(operator.mul, [shape[i] for i in axes_to_keep], 1)
            n = shape[ax]
            k = 1  # Single axis reduction
            
            x = x.reshape((m, n, k))
            temp_output = CUDAStorage((m, k), dtype=x.dtype)
            
            # Adaptive tiling for non-inner reduction
            if n <= 256:
                tile_n = triton.next_power_of_2(n)
                one_tile_per_cta = True
            else:
                # Larger N: use optimized tiling
                tile_n = min(512, triton.next_power_of_2(min(n, 512)))
                one_tile_per_cta = (tile_n >= n)
            
            tile_k = 1  # K=1 for single axis
            
            grid = (m, 1)
            sum_kernel_non_inner_dim[grid](
                temp_output, x, m, n, k, tile_n, tile_k, one_tile_per_cta
            )
            
            return temp_output.reshape(output_shape)
    
    # Multi-axis reduction with enhanced strategy
    axes_to_keep = tuple(i for i in range(ndim) if i not in axis)
    new_order = axes_to_keep + axis

    x = x.permute(new_order)
    new_shape = tuple(shape[i] for i in axes_to_keep) + tuple(shape[i] for i in axis)

    m = functools_reduce(operator.mul, new_shape[:len(axes_to_keep)], 1)
    n = functools_reduce(operator.mul, new_shape[len(axes_to_keep):], 1)
    
    # For multi-axis, check if it's worth using specialized approach
    if n <= 4096:
        # Small reduction dimension: use two-stage approach
        x = x.reshape((m, n))
        temp_storage = CUDAStorage((m,), dtype=x.dtype)
        
        # Use enhanced two-stage approach
        if m == 1:
            # Single row: direct reduction
            output_shape_temp = (1,) if keepdims else ()
            final_output = CUDAStorage(output_shape_temp, dtype=x.dtype)
            
            block_size = min(triton.next_power_of_2(n), 1024)
            mid_size = triton.cdiv(n, block_size)
            
            if mid_size == 1:
                grid = (1,)
                sum_kernel_two_stage_2[grid](x, final_output, n, block_size)
                return final_output
            else:
                mid = CUDAStorage((mid_size,), dtype=x.dtype)
                grid1 = (mid_size,)
                sum_kernel_two_stage_1[grid1](x, mid, n, block_size)
                
                block_mid = triton.next_power_of_2(mid_size)
                grid2 = (1,)
                sum_kernel_two_stage_2[grid2](mid, final_output, mid_size, block_mid)
                return final_output
        else:
            # Multiple rows: use optimized kernel
            tile_n = min(triton.next_power_of_2(n), 1024)
            one_tile_per_cta = (tile_n >= n)
            
            grid = (m,)
            sum_kernel_inner_dim[grid](temp_storage, x, m, n, tile_n, one_tile_per_cta)
            
            output_shape = tuple(new_shape[i] for i in range(len(axes_to_keep)))
            if keepdims:
                output_shape = list(shape)
                for i in axis:
                    output_shape[i] = 1
                output_shape = tuple(output_shape)
            
            return temp_storage.reshape(output_shape)
    else:
        # Large reduction: fallback to v2
        return reduce_sum_v2(x, axis, keepdims)


def reduce_sum(x, axis=None, keepdims=False):
    """
    Main reduce sum function with version control.
    """
    version = os.environ.get('GENESIS_REDUCTION_VERSION', 'v3')
    
    if version == 'v3':
        return reduce_sum_v3(x, axis, keepdims)
    elif version == 'v2':
        return reduce_sum_v2(x, axis, keepdims)
    else:
        # Default to v1 (original implementation)
        return reduce_sum_v1(x, axis, keepdims)


def reduce_max_v1(x, axis=None, keepdims=False):
    """
    Original reduce max operation (v1).
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


def reduce_max_v2(x, axis=None, keepdims=False):
    """
    Two-stage reduce max operation (v2) - Flag-Gems inspired.
    Uses two-stage reduction for better performance on large tensors.
    """
    import math
    
    shape = x.shape
    ndim = len(shape)
    
    # Fast path: full tensor reduction to scalar with two-stage approach
    if axis is None:
        if not x.is_contiguous():
            x = x.contiguous()
            
        n_elements = x.size
        output_shape = (1,) if keepdims else ()
        
        # Flag-Gems style: adaptive block size
        block_size = triton.next_power_of_2(min(1024, max(32, int(math.ceil(math.sqrt(n_elements))))))
        num_blocks = triton.cdiv(n_elements, block_size)
        
        if num_blocks == 1:
            # Single block - use direct reduction
            output = CUDAStorage(output_shape, dtype=x.dtype)
            output.fill(-float("inf"))
            max_full_kernel[block_size,](x, output, n_elements, block_size)
            return output
        else:
            # Two-stage reduction
            partial_results = CUDAStorage((num_blocks,), dtype=x.dtype)
            
            # Stage 1: Compute partial max values
            grid1 = (num_blocks,)
            max_kernel_two_stage_1[grid1](x, partial_results, n_elements, block_size)
            
            # Stage 2: Reduce partial results
            output = CUDAStorage(output_shape, dtype=x.dtype)
            stage2_block_size = triton.next_power_of_2(min(1024, num_blocks))
            grid2 = (1,)
            max_kernel_two_stage_2[grid2](partial_results, output, num_blocks, stage2_block_size)
            
            return output
    
    # For axis-specific reductions, fall back to v1 for now
    return reduce_max_v1(x, axis, keepdims)


def reduce_max_v3(x, axis=None, keepdims=False):
    """
    Advanced reduce max operation (v3) - Flag-Gems inspired with specialized kernels.
    Uses specialized kernels for inner vs non-inner dimension reductions.
    """
    import math
    
    shape = x.shape
    ndim = len(shape)
    
    # Fast path: full tensor reduction to scalar (same as v2)
    if axis is None:
        return reduce_max_v2(x, axis, keepdims)
    
    # Normalize axis
    if isinstance(axis, int):
        axis = (axis,)
    else:
        axis = tuple(axis)
    
    # Single axis reduction with specialized kernels
    if len(axis) == 1:
        ax = axis[0]
        if ax < 0:
            ax += ndim
            
        if not x.is_contiguous():
            x = x.contiguous()
        
        # Calculate output shape
        output_shape = list(shape)
        if keepdims:
            output_shape[ax] = 1
        else:
            output_shape.pop(ax)
        output_shape = tuple(output_shape)
        
        # Specialized kernel selection based on reduced dimension
        if ax == ndim - 1:  # Inner dimension reduction
            M = functools_reduce(operator.mul, shape[:-1], 1)
            N = shape[-1]
            
            output = CUDAStorage(output_shape, dtype=x.dtype)
            x_2d = x.reshape((M, N))
            temp_output = CUDAStorage((M,), dtype=x.dtype)
            
            # Adaptive tiling for inner dimension
            TILE_M = 32
            TILE_N = min(2048, triton.next_power_of_2(N))
            ONE_TILE_PER_CTA = TILE_N >= N
            
            grid = (triton.cdiv(M, TILE_M),)
            max_kernel_inner_dim[grid](temp_output, x_2d, M, N, TILE_M, TILE_N, ONE_TILE_PER_CTA)
            
            return temp_output.reshape(output_shape)
        
        else:  # Non-inner dimension reduction
            axes_to_keep = tuple(i for i in range(ndim) if i != ax)
            new_order = axes_to_keep + (ax,)
            x = x.permute(new_order)
            
            # Enhanced reshape for non-inner kernel (same as sum_v3)
            m = functools_reduce(operator.mul, [shape[i] for i in axes_to_keep], 1)
            n = shape[ax]
            k = 1  # Single axis reduction
            
            x = x.reshape((m, n, k))
            temp_output = CUDAStorage((m, k), dtype=x.dtype)
            
            # Adaptive tiling for non-inner reduction
            if n <= 256:
                tile_n = triton.next_power_of_2(n)
                one_tile_per_cta = True
            else:
                # Larger N: use optimized tiling
                tile_n = min(512, triton.next_power_of_2(min(n, 512)))
                one_tile_per_cta = (tile_n >= n)
            
            tile_k = 1  # K=1 for single axis
            
            grid = (m, 1)
            max_kernel_non_inner_dim[grid](
                temp_output, x, m, n, k, tile_n, tile_k, one_tile_per_cta
            )
            
            return temp_output.reshape(output_shape)
    
    # Multi-axis reduction: fall back to v1
    return reduce_max_v1(x, axis, keepdims)


def reduce_max(x, axis=None, keepdims=False):
    """
    Optimized reduce max operation with version control.
    
    Version can be controlled via GENESIS_REDUCTION_VERSION environment variable:
    - v1: Original implementation
    - v2: Two-stage reduction (Flag-Gems inspired)  
    - v3: Advanced optimizations with specialized kernels (default)
    """
    
    version = os.environ.get('GENESIS_REDUCTION_VERSION', 'v3')
    
    if version == 'v1':
        return reduce_max_v1(x, axis, keepdims)
    elif version == 'v2':
        return reduce_max_v2(x, axis, keepdims)
    elif version == 'v3':
        return reduce_max_v3(x, axis, keepdims)
    else:
        # Default to v3 (advanced optimizations)
        return reduce_max_v3(x, axis, keepdims)

