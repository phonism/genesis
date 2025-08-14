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
    Sum kernel.
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
        inp = tl.load(x_ptr + offset, mask=mask, other=0)
        out += tl.sum(inp, axis=1)

    tl.store(output_ptr + m_offset, out, mask=m_mask)


@triton.jit
def max_kernel(x_ptr, output_ptr, M, N, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    """
    Max kernel.
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


# =============================================================================
# GPU OPERATIONS
# =============================================================================


def reduce_sum(x, axis=None, keepdims=False):
    """
    Reduce sum operation.
    """    
    shape = x.shape
    ndim = len(shape)
    if axis is None:
        axis = tuple(range(ndim))
    elif isinstance(axis, int):
        axis = (axis,)
    else:
        axis = tuple(axis)
    
    # Create a permutation that moves all axes to be reduced to the end
    axes_to_keep = tuple(i for i in range(ndim) if i not in axis)
    new_order = axes_to_keep + axis

    x = x.permute(new_order)

    # Calculate the new shape after permutation
    new_shape = tuple(shape[i] for i in axes_to_keep) + tuple(shape[i] for i in axis)

    # Determine the dimensions for reduction
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

    if not x.is_contiguous():
        x = x.contiguous()

    block_m = 4
    block_n = min(triton.next_power_of_2(n), 1024)
    grid = (triton.cdiv(m, block_m), 1, 1)
    
    # As per expert advice, ensure tensor is contiguous
    if not x.is_contiguous():
        x = x.contiguous()
    if not output.is_contiguous():
        output = output.contiguous()
    
    # Ensure we pass CUDAStorage objects directly, not .ptr or others
    sum_kernel[grid](x, output, m, n, block_m, block_n)
    return output


def reduce_max(x, axis=None, keepdims=False):
    """
    Reduce max operation.
    """    
    shape = x.shape
    ndim = len(shape)
    if axis is None:
        axis = tuple(range(ndim))
    elif isinstance(axis, int):
        axis = (axis,)
    else:
        axis = tuple(axis)
    
    # Create a permutation that moves all axes to be reduced to the end
    axes_to_keep = tuple(i for i in range(ndim) if i not in axis)
    new_order = axes_to_keep + axis

    x = x.permute(new_order)

    # Calculate the new shape after permutation
    new_shape = tuple(shape[i] for i in axes_to_keep) + tuple(shape[i] for i in axis)

    # Determine the dimensions for reduction
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

    if not x.is_contiguous():
        x = x.contiguous()

    block_m = 4
    block_n = min(triton.next_power_of_2(n), 1024)
    grid = (triton.cdiv(m, block_m), 1, 1)
    
    # Ensure tensor is contiguous
    if not x.is_contiguous():
        x = x.contiguous()
    if not output.is_contiguous():
        output = output.contiguous()
    
    # Use max_kernel for reduction
    max_kernel[grid](x, output, m, n, block_m, block_n)
    return output