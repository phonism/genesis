"""
Comparison operations for GPU backend.
"""
import triton
import triton.language as tl
from genesis.backends.cuda import CUDAStorage
from genesis.ops.dispatcher import register_cuda


# =============================================================================
# TRITON KERNELS
# =============================================================================

@triton.jit
def compare_kernel(
    x_ptr, y_ptr, output_ptr, n_elements,
    op_type: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Element-wise comparison kernel.
    """
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    
    if op_type == 0:  # eq
        result = x == y
    elif op_type == 1:  # lt
        result = x < y
    elif op_type == 2:  # le  
        result = x <= y
    elif op_type == 3:  # gt
        result = x > y
    elif op_type == 4:  # ge
        result = x >= y
    else:  # ne
        result = x != y
    
    tl.store(output_ptr + offsets, result, mask=mask)


@triton.jit
def compare_scalar_kernel(
    x_ptr, output_ptr, n_elements, scalar_val,
    op_type: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Element-wise comparison with Python scalar kernel.
    """
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)

    if op_type == 0:  # eq
        result = x == scalar_val
    elif op_type == 1:  # lt
        result = x < scalar_val
    elif op_type == 2:  # le
        result = x <= scalar_val
    elif op_type == 3:  # gt
        result = x > scalar_val
    elif op_type == 4:  # ge
        result = x >= scalar_val
    else:  # ne
        result = x != scalar_val

    tl.store(output_ptr + offsets, result, mask=mask)


@triton.jit
def compare_broadcast_kernel(
    x_ptr, y_ptr, output_ptr, n_elements,
    op_type: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Element-wise comparison with 0-D tensor (broadcast scalar from GPU memory).
    """
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr)  # Load single scalar element from GPU

    if op_type == 0:  # eq
        result = x == y
    elif op_type == 1:  # lt
        result = x < y
    elif op_type == 2:  # le
        result = x <= y
    elif op_type == 3:  # gt
        result = x > y
    elif op_type == 4:  # ge
        result = x >= y
    else:  # ne
        result = x != y

    tl.store(output_ptr + offsets, result, mask=mask)


# =============================================================================
# GPU OPERATIONS
# =============================================================================


def _compare_op(x, y, op_type: int):
    """
    Generic comparison operation supporting tensor, 0-D tensor, and scalar.

    Args:
        x: Input CUDAStorage tensor.
        y: CUDAStorage tensor, 0-D tensor, or Python scalar.
        op_type: Comparison type (0=eq, 1=lt, 2=le, 3=gt, 4=ge, 5=ne).

    Returns:
        CUDAStorage with boolean result.
    """
    if not x.is_contiguous():
        x = x.contiguous()

    output = CUDAStorage(x.shape, dtype="bool")
    n_elements = output.size
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]), )

    if isinstance(y, CUDAStorage):
        if not y.is_contiguous():
            y = y.contiguous()

        # Check if y is a 0-D tensor or size-1 tensor (broadcast as scalar)
        if y.shape == () or y.size == 1:
            # Use broadcast kernel - reads scalar from GPU memory
            compare_broadcast_kernel[grid](x, y, output, n_elements, op_type, BLOCK_SIZE=1024)
        elif x.shape != y.shape:
            raise ValueError(f"Shape mismatch: {x.shape} vs {y.shape}")
        else:
            # Same-shape tensor comparison
            compare_kernel[grid](x, y, output, n_elements, op_type, BLOCK_SIZE=1024)
    else:
        # Python scalar comparison
        compare_scalar_kernel[grid](x, output, n_elements, float(y), op_type, BLOCK_SIZE=1024)

    return output


@register_cuda("eq")
def eq(x, y):
    """Element-wise equality comparison."""
    return _compare_op(x, y, 0)


@register_cuda("ge")
def ge(x, y):
    """Element-wise greater-than-or-equal comparison."""
    return _compare_op(x, y, 4)


@register_cuda("gt")
def gt(x, y):
    """Element-wise greater-than comparison."""
    return _compare_op(x, y, 3)


@register_cuda("le")
def le(x, y):
    """Element-wise less-than-or-equal comparison."""
    return _compare_op(x, y, 2)


@register_cuda("lt")
def lt(x, y):
    """Element-wise less-than comparison."""
    return _compare_op(x, y, 1)


@register_cuda("ne")
def ne(x, y):
    """Element-wise not-equal comparison."""
    return _compare_op(x, y, 5)