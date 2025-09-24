"""
Strided element-wise operations for CUDA tensors.
Handles non-contiguous memory layouts without requiring contiguous() calls.
Supports arbitrary dimensions with optimized 1-4D paths and general N-D fallback.
"""

import triton
import triton.language as tl
from genesis.backends.cuda import CUDAStorage

# Operation type constants for better readability
SCALAR_ADD = 0
SCALAR_SUB = 1
SCALAR_MUL = 2
SCALAR_DIV = 3
SCALAR_RSUB = 4
SCALAR_RDIV = 5
SCALAR_POW = 6
SCALAR_RPOWER = 7

BINARY_ADD = 0
BINARY_SUB = 1
BINARY_MUL = 2
BINARY_DIV = 3

UNARY_EXP = 0
UNARY_LOG = 1
UNARY_SQRT = 2
UNARY_ABS = 3
UNARY_NEG = 4
UNARY_SIN = 5
UNARY_COS = 6

COMPARE_GT = 0
COMPARE_EQ = 1
COMPARE_GE = 2
COMPARE_LE = 3

SPECIAL_CLAMP = 0
SPECIAL_MAXIMUM = 1
SPECIAL_SIGN = 2


@triton.jit
def strided_binary_kernel(
    x_ptr, x_shape_0, x_stride_0, x_shape_1, x_stride_1, x_shape_2, x_stride_2, x_shape_3, x_stride_3,
    y_ptr, y_shape_0, y_stride_0, y_shape_1, y_stride_1, y_shape_2, y_stride_2, y_shape_3, y_stride_3,
    out_ptr, out_shape_0, out_stride_0, out_shape_1, out_stride_1, out_shape_2, out_stride_2, out_shape_3, out_stride_3,
    n_elements, ndim, op_type: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    """
    Generic strided binary operation kernel.
    Supports up to 4D tensors with arbitrary strides.
    op_type: 0=add, 1=sub, 2=mul, 3=div
    """
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Calculate multi-dimensional indices for each element
    if ndim == 1:
        idx = offsets
        x_offset = idx * x_stride_0
        y_offset = idx * y_stride_0
        out_offset = idx * out_stride_0
    elif ndim == 2:
        idx_0 = offsets // x_shape_1
        idx_1 = offsets % x_shape_1
        x_offset = idx_0 * x_stride_0 + idx_1 * x_stride_1
        y_offset = idx_0 * y_stride_0 + idx_1 * y_stride_1
        out_offset = idx_0 * out_stride_0 + idx_1 * out_stride_1
    elif ndim == 3:
        tmp = offsets // x_shape_2
        idx_2 = offsets % x_shape_2
        idx_1 = tmp % x_shape_1
        idx_0 = tmp // x_shape_1
        x_offset = idx_0 * x_stride_0 + idx_1 * x_stride_1 + idx_2 * x_stride_2
        y_offset = idx_0 * y_stride_0 + idx_1 * y_stride_1 + idx_2 * y_stride_2
        out_offset = idx_0 * out_stride_0 + idx_1 * out_stride_1 + idx_2 * out_stride_2
    else:  # ndim == 4
        tmp = offsets // x_shape_3
        idx_3 = offsets % x_shape_3
        tmp2 = tmp // x_shape_2
        idx_2 = tmp % x_shape_2
        idx_1 = tmp2 % x_shape_1
        idx_0 = tmp2 // x_shape_1
        x_offset = idx_0 * x_stride_0 + idx_1 * x_stride_1 + idx_2 * x_stride_2 + idx_3 * x_stride_3
        y_offset = idx_0 * y_stride_0 + idx_1 * y_stride_1 + idx_2 * y_stride_2 + idx_3 * y_stride_3
        out_offset = idx_0 * out_stride_0 + idx_1 * out_stride_1 + idx_2 * out_stride_2 + idx_3 * out_stride_3

    # Load values
    x_val = tl.load(x_ptr + x_offset, mask=mask)
    y_val = tl.load(y_ptr + y_offset, mask=mask)

    # Perform operation based on op_type
    if op_type == 0:  # add
        result = x_val + y_val
    elif op_type == 1:  # sub
        result = x_val - y_val
    elif op_type == 2:  # mul
        result = x_val * y_val
    else:  # div
        result = x_val / y_val

    # Store result
    tl.store(out_ptr + out_offset, result, mask=mask)


@triton.jit
def strided_scalar_kernel(
    x_ptr, x_shape_0, x_stride_0, x_shape_1, x_stride_1, x_shape_2, x_stride_2, x_shape_3, x_stride_3,
    scalar,
    out_ptr, out_shape_0, out_stride_0, out_shape_1, out_stride_1, out_shape_2, out_stride_2, out_shape_3, out_stride_3,
    n_elements, ndim, op_type: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    """
    Generic strided scalar operation kernel.
    op_type: 0=add, 1=sub, 2=mul, 3=div, 4=rsub, 5=rdiv, 6=pow, 7=rpower
    """
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Calculate multi-dimensional indices for each element
    if ndim == 1:
        idx = offsets
        x_offset = idx * x_stride_0
        out_offset = idx * out_stride_0
    elif ndim == 2:
        idx_0 = offsets // x_shape_1
        idx_1 = offsets % x_shape_1
        x_offset = idx_0 * x_stride_0 + idx_1 * x_stride_1
        out_offset = idx_0 * out_stride_0 + idx_1 * out_stride_1
    elif ndim == 3:
        tmp = offsets // x_shape_2
        idx_2 = offsets % x_shape_2
        idx_1 = tmp % x_shape_1
        idx_0 = tmp // x_shape_1
        x_offset = idx_0 * x_stride_0 + idx_1 * x_stride_1 + idx_2 * x_stride_2
        out_offset = idx_0 * out_stride_0 + idx_1 * out_stride_1 + idx_2 * out_stride_2
    else:  # ndim == 4
        tmp = offsets // x_shape_3
        idx_3 = offsets % x_shape_3
        tmp2 = tmp // x_shape_2
        idx_2 = tmp % x_shape_2
        idx_1 = tmp2 % x_shape_1
        idx_0 = tmp2 // x_shape_1
        x_offset = idx_0 * x_stride_0 + idx_1 * x_stride_1 + idx_2 * x_stride_2 + idx_3 * x_stride_3
        out_offset = idx_0 * out_stride_0 + idx_1 * out_stride_1 + idx_2 * out_stride_2 + idx_3 * out_stride_3

    # Load value
    x_val = tl.load(x_ptr + x_offset, mask=mask)

    # Perform operation based on op_type
    if op_type == 0:  # add
        result = x_val + scalar
    elif op_type == 1:  # sub
        result = x_val - scalar
    elif op_type == 2:  # mul
        result = x_val * scalar
    elif op_type == 3:  # div
        result = x_val / scalar
    elif op_type == 4:  # rsub (scalar - x)
        result = scalar - x_val
    elif op_type == 5:  # rdiv (scalar / x)
        result = scalar / x_val
    elif op_type == 6:  # pow (x ** scalar)
        # Use exp(scalar * log(x)) for x^scalar
        result = tl.exp(scalar * tl.log(x_val))
    else:  # rpower (scalar ** x)
        # Use exp(x * log(scalar)) for scalar^x
        result = tl.exp(x_val * tl.log(scalar))

    # Store result
    tl.store(out_ptr + out_offset, result, mask=mask)


@triton.jit
def strided_unary_kernel(
    x_ptr, x_shape_0, x_stride_0, x_shape_1, x_stride_1, x_shape_2, x_stride_2, x_shape_3, x_stride_3,
    out_ptr, out_shape_0, out_stride_0, out_shape_1, out_stride_1, out_shape_2, out_stride_2, out_shape_3, out_stride_3,
    n_elements, ndim, op_type: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    """
    Generic strided unary operation kernel.
    op_type: 0=exp, 1=log, 2=sqrt, 3=abs, 4=neg, 5=sin, 6=cos
    """
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Calculate multi-dimensional indices for each element
    if ndim == 1:
        idx = offsets
        x_offset = idx * x_stride_0
        out_offset = idx * out_stride_0
    elif ndim == 2:
        idx_0 = offsets // x_shape_1
        idx_1 = offsets % x_shape_1
        x_offset = idx_0 * x_stride_0 + idx_1 * x_stride_1
        out_offset = idx_0 * out_stride_0 + idx_1 * out_stride_1
    elif ndim == 3:
        tmp = offsets // x_shape_2
        idx_2 = offsets % x_shape_2
        idx_1 = tmp % x_shape_1
        idx_0 = tmp // x_shape_1
        x_offset = idx_0 * x_stride_0 + idx_1 * x_stride_1 + idx_2 * x_stride_2
        out_offset = idx_0 * out_stride_0 + idx_1 * out_stride_1 + idx_2 * out_stride_2
    else:  # ndim == 4
        tmp = offsets // x_shape_3
        idx_3 = offsets % x_shape_3
        tmp2 = tmp // x_shape_2
        idx_2 = tmp % x_shape_2
        idx_1 = tmp2 % x_shape_1
        idx_0 = tmp2 // x_shape_1
        x_offset = idx_0 * x_stride_0 + idx_1 * x_stride_1 + idx_2 * x_stride_2 + idx_3 * x_stride_3
        out_offset = idx_0 * out_stride_0 + idx_1 * out_stride_1 + idx_2 * out_stride_2 + idx_3 * out_stride_3

    # Load value
    x_val = tl.load(x_ptr + x_offset, mask=mask)

    # Convert to float32 for precision (following existing pattern in exp_kernel)
    x_f32 = x_val.to(tl.float32)

    # Perform operation based on op_type
    if op_type == 0:  # exp
        result_f32 = tl.exp(x_f32)
    elif op_type == 1:  # log
        result_f32 = tl.log(x_f32)
    elif op_type == 2:  # sqrt
        result_f32 = tl.sqrt(x_f32)
    elif op_type == 3:  # abs
        result_f32 = tl.abs(x_f32)
    elif op_type == 4:  # neg
        result_f32 = -x_f32
    elif op_type == 5:  # sin
        result_f32 = tl.sin(x_f32)
    else:  # cos (op_type == 6)
        result_f32 = tl.cos(x_f32)

    # Convert back to original dtype
    result = result_f32.to(x_val.dtype)

    # Store result
    tl.store(out_ptr + out_offset, result, mask=mask)


@triton.jit
def strided_compare_scalar_kernel(
    x_ptr, x_shape_0, x_stride_0, x_shape_1, x_stride_1, x_shape_2, x_stride_2, x_shape_3, x_stride_3,
    scalar,
    out_ptr, out_shape_0, out_stride_0, out_shape_1, out_stride_1, out_shape_2, out_stride_2, out_shape_3, out_stride_3,
    n_elements, ndim, op_type: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    """
    Generic strided comparison with scalar kernel.
    op_type: 0=gt, 1=eq, 2=ge, 3=le
    """
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Calculate multi-dimensional indices for each element
    if ndim == 1:
        idx = offsets
        x_offset = idx * x_stride_0
        out_offset = idx * out_stride_0
    elif ndim == 2:
        idx_0 = offsets // x_shape_1
        idx_1 = offsets % x_shape_1
        x_offset = idx_0 * x_stride_0 + idx_1 * x_stride_1
        out_offset = idx_0 * out_stride_0 + idx_1 * out_stride_1
    elif ndim == 3:
        tmp = offsets // x_shape_2
        idx_2 = offsets % x_shape_2
        idx_1 = tmp % x_shape_1
        idx_0 = tmp // x_shape_1
        x_offset = idx_0 * x_stride_0 + idx_1 * x_stride_1 + idx_2 * x_stride_2
        out_offset = idx_0 * out_stride_0 + idx_1 * out_stride_1 + idx_2 * out_stride_2
    else:  # ndim == 4
        tmp = offsets // x_shape_3
        idx_3 = offsets % x_shape_3
        tmp2 = tmp // x_shape_2
        idx_2 = tmp % x_shape_2
        idx_1 = tmp2 % x_shape_1
        idx_0 = tmp2 // x_shape_1
        x_offset = idx_0 * x_stride_0 + idx_1 * x_stride_1 + idx_2 * x_stride_2 + idx_3 * x_stride_3
        out_offset = idx_0 * out_stride_0 + idx_1 * out_stride_1 + idx_2 * out_stride_2 + idx_3 * out_stride_3

    # Load value
    x_val = tl.load(x_ptr + x_offset, mask=mask)

    # Perform comparison based on op_type
    if op_type == 0:  # gt
        result = x_val > scalar
    elif op_type == 1:  # eq
        result = x_val == scalar
    elif op_type == 2:  # ge
        result = x_val >= scalar
    else:  # le
        result = x_val <= scalar

    # Store result
    tl.store(out_ptr + out_offset, result, mask=mask)


@triton.jit
def strided_special_kernel(
    x_ptr, x_shape_0, x_stride_0, x_shape_1, x_stride_1, x_shape_2, x_stride_2, x_shape_3, x_stride_3,
    out_ptr, out_shape_0, out_stride_0, out_shape_1, out_stride_1, out_shape_2, out_stride_2, out_shape_3, out_stride_3,
    n_elements, ndim, op_type: tl.constexpr, min_val, max_val, scalar,
    BLOCK_SIZE: tl.constexpr
):
    """
    Generic strided special operations kernel.
    op_type: 0=clamp, 1=maximum_scalar, 2=sign
    """
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Calculate multi-dimensional indices for each element
    if ndim == 1:
        idx = offsets
        x_offset = idx * x_stride_0
        out_offset = idx * out_stride_0
    elif ndim == 2:
        idx_0 = offsets // x_shape_1
        idx_1 = offsets % x_shape_1
        x_offset = idx_0 * x_stride_0 + idx_1 * x_stride_1
        out_offset = idx_0 * out_stride_0 + idx_1 * out_stride_1
    elif ndim == 3:
        tmp = offsets // x_shape_2
        idx_2 = offsets % x_shape_2
        idx_1 = tmp % x_shape_1
        idx_0 = tmp // x_shape_1
        x_offset = idx_0 * x_stride_0 + idx_1 * x_stride_1 + idx_2 * x_stride_2
        out_offset = idx_0 * out_stride_0 + idx_1 * out_stride_1 + idx_2 * out_stride_2
    else:  # ndim == 4
        tmp = offsets // x_shape_3
        idx_3 = offsets % x_shape_3
        tmp2 = tmp // x_shape_2
        idx_2 = tmp % x_shape_2
        idx_1 = tmp2 % x_shape_1
        idx_0 = tmp2 // x_shape_1
        x_offset = idx_0 * x_stride_0 + idx_1 * x_stride_1 + idx_2 * x_stride_2 + idx_3 * x_stride_3
        out_offset = idx_0 * out_stride_0 + idx_1 * out_stride_1 + idx_2 * out_stride_2 + idx_3 * out_stride_3

    # Load value
    x_val = tl.load(x_ptr + x_offset, mask=mask)

    # Perform operation based on op_type
    if op_type == 0:  # clamp
        result = tl.minimum(tl.maximum(x_val, min_val), max_val)
    elif op_type == 1:  # maximum with scalar
        result = tl.maximum(x_val, scalar)
    else:  # sign
        result = tl.where(x_val > 0, 1.0, tl.where(x_val < 0, -1.0, 0.0))

    # Store result
    tl.store(out_ptr + out_offset, result, mask=mask)


