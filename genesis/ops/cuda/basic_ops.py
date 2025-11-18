"""
Basic arithmetic operations for GPU backend.
"""
import numpy as np
import triton
import triton.language as tl
from genesis.backends.cuda import CUDAStorage
from .utils import broadcast_shapes
from genesis.ops.dispatcher import register_cuda
from .strided_ops import (
    strided_binary_kernel, strided_scalar_kernel, strided_unary_kernel, strided_compare_scalar_kernel, strided_special_kernel,
    SCALAR_ADD, SCALAR_SUB, SCALAR_MUL, SCALAR_DIV, SCALAR_RSUB, SCALAR_RDIV, SCALAR_POW, SCALAR_RPOWER,
    BINARY_ADD, BINARY_SUB, BINARY_MUL, BINARY_DIV,
    UNARY_EXP, UNARY_LOG, UNARY_SQRT, UNARY_ABS, UNARY_NEG, UNARY_SIN, UNARY_COS,
    COMPARE_GT, COMPARE_EQ, COMPARE_GE, COMPARE_LE,
    SPECIAL_CLAMP, SPECIAL_MAXIMUM, SPECIAL_SIGN
)
from .shape_ops import broadcast_to


# =============================================================================
# TRITON KERNELS
# =============================================================================

@triton.jit
def add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    """
    Add kernel with fixed block size for stable performance.
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    x = tl.load(x_ptr + offs, mask=mask)
    y = tl.load(y_ptr + offs, mask=mask)
    output = x + y
    tl.store(output_ptr + offs, output, mask=mask)


@triton.jit
def copy_kernel(src_ptr, dst_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    """
    Copy data from src to dst.
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    data = tl.load(src_ptr + offs, mask=mask)
    tl.store(dst_ptr + offs, data, mask=mask)


@triton.jit
def add_inplace_kernel(x_ptr, y_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    """
    In-place add kernel: x += y (modifies x directly)
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    x = tl.load(x_ptr + offs, mask=mask)
    y = tl.load(y_ptr + offs, mask=mask)
    result = x + y
    tl.store(x_ptr + offs, result, mask=mask)  # Store back to x_ptr


@triton.jit
def sub_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    """
    Sub kernel for same-shape tensors.
    """
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x - y
    tl.store(output_ptr + offsets, output, mask=mask)


@triton.jit
def add_scalar_kernel(x_ptr, scalar, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    """
    Add scalar kernel.
    """
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    # Simply add scalar directly - Triton should handle type compatibility
    output = x + scalar
    tl.store(output_ptr + offsets, output, mask=mask)


@triton.jit
def add_scalar_inplace_kernel(x_ptr, scalar, n_elements, BLOCK_SIZE: tl.constexpr):
    """
    In-place add scalar kernel: x += scalar
    """
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    result = x + scalar
    tl.store(x_ptr + offsets, result, mask=mask)  # Store back to x_ptr


@triton.jit
def sub_inplace_kernel(x_ptr, y_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    """
    In-place sub kernel: x -= y (modifies x directly)
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    x = tl.load(x_ptr + offs, mask=mask)
    y = tl.load(y_ptr + offs, mask=mask)
    result = x - y
    tl.store(x_ptr + offs, result, mask=mask)  # Store back to x_ptr


@triton.jit
def sub_scalar_inplace_kernel(x_ptr, scalar, n_elements, BLOCK_SIZE: tl.constexpr):
    """
    In-place sub scalar kernel: x -= scalar
    """
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    result = x - scalar
    tl.store(x_ptr + offsets, result, mask=mask)  # Store back to x_ptr


@triton.jit
def mul_inplace_kernel(x_ptr, y_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    """
    In-place mul kernel: x *= y (modifies x directly)
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    x = tl.load(x_ptr + offs, mask=mask)
    y = tl.load(y_ptr + offs, mask=mask)
    result = x * y
    tl.store(x_ptr + offs, result, mask=mask)  # Store back to x_ptr


@triton.jit
def mul_scalar_inplace_kernel(x_ptr, scalar, n_elements, BLOCK_SIZE: tl.constexpr):
    """
    In-place mul scalar kernel: x *= scalar
    """
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    result = x * scalar
    tl.store(x_ptr + offsets, result, mask=mask)  # Store back to x_ptr


@triton.jit
def mul_scalar_kernel(x_ptr, scalar, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    """
    Mul scalar kernel.
    """
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    output = x * scalar
    tl.store(output_ptr + offsets, output, mask=mask)


@triton.jit
def mul_kernel(x_ptr, y_ptr, output_ptr, N, BLOCK_SIZE: tl.constexpr):
    """
    Mul kernel.
    """
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x * y
    tl.store(output_ptr + offsets, output, mask=mask)


@triton.jit
def div_scalar_kernel(x_ptr, scalar, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    """
    Div scalar kernel.
    """
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    output = x / scalar
    tl.store(output_ptr + offsets, output, mask=mask)


@triton.jit
def rdiv_scalar_kernel(x_ptr, scalar, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    """
    Rdiv scalar kernel.
    """
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    output = scalar / x
    tl.store(output_ptr + offsets, output, mask=mask)


@triton.jit
def rsub_scalar_kernel(x_ptr, scalar, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    """
    Rsub scalar kernel (scalar - tensor).
    """
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    output = scalar - x
    tl.store(output_ptr + offsets, output, mask=mask)


@triton.jit
def rpower_scalar_kernel(x_ptr, scalar, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    """
    Rpower scalar kernel (scalar ** tensor).
    """
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    # Use exp(log) to compute power: scalar^x = exp(x * log(scalar))
    log_scalar = tl.log(scalar)
    output = tl.exp(x * log_scalar)
    tl.store(output_ptr + offsets, output, mask=mask)


@triton.jit
def gt_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    """Greater than comparison kernel."""
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    result = x > y
    tl.store(output_ptr + offsets, result, mask=mask)


@triton.jit
def gt_scalar_kernel(x_ptr, scalar, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    """Greater than scalar comparison kernel."""
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    x = tl.load(x_ptr + offsets, mask=mask)
    result = x > scalar
    tl.store(output_ptr + offsets, result, mask=mask)


@triton.jit
def eq_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    """Equality comparison kernel."""
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    result = x == y
    tl.store(output_ptr + offsets, result, mask=mask)


@triton.jit
def eq_scalar_kernel(x_ptr, scalar, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    """Equality scalar comparison kernel."""
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    x = tl.load(x_ptr + offsets, mask=mask)
    result = x == scalar
    tl.store(output_ptr + offsets, result, mask=mask)


@triton.jit
def div_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    """
    Div kernel.
    """
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x / y
    tl.store(output_ptr + offsets, output, mask=mask)


@triton.jit
def log_kernel(x_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    """
    Log kernel.
    """
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    # Compute log in float32 for precision, then convert back to original type
    x_f32 = x.to(tl.float32)
    log_result = tl.log(x_f32)
    output = log_result.to(x.dtype)
    tl.store(output_ptr + offsets, output, mask=mask)


@triton.jit
def exp_kernel(x_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    """
    Exp kernel.
    """
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    # Compute exp in float32 for precision, then convert back to original type
    x_f32 = x.to(tl.float32)
    exp_result = tl.exp(x_f32)
    output = exp_result.to(x.dtype)
    tl.store(output_ptr + offsets, output, mask=mask)


@triton.jit
def cos_kernel(x_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    """
    Cos kernel.
    """
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    output = tl.cos(x)
    tl.store(output_ptr + offsets, output, mask=mask)


@triton.jit
def sin_kernel(x_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    """
    Sin kernel.
    """
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    output = tl.sin(x)
    tl.store(output_ptr + offsets, output, mask=mask)


@triton.jit
def sqrt_kernel(x_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    """
    Sqrt kernel.
    """
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    output = tl.sqrt(x)
    tl.store(output_ptr + offsets, output, mask=mask)


@triton.jit
def pow_kernel(x_ptr, scalar, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    """
    Pow kernel.
    """
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    output = tl.pow(x, scalar)
    tl.store(output_ptr + offsets, output, mask=mask)


@triton.jit
def maximum_scalar_kernel(x_ptr, scalar, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    """
    Maximum scalar kernel.
    """
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    output = tl.maximum(x, scalar)
    tl.store(output_ptr + offsets, output, mask=mask)


@triton.jit
def arange_kernel(output_ptr, start, step, n_elements, BLOCK_SIZE: tl.constexpr):
    """
    Arange kernel - GPU implementation of range generation.
    """
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Calculate values: start + step * index
    values = start + step * offsets.to(tl.float32)
    tl.store(output_ptr + offsets, values, mask=mask)


@triton.jit
def one_hot_kernel(indices_ptr, output_ptr, n_classes, n_indices, 
                   BLOCK_SIZE: tl.constexpr):
    """
    One-hot encoding kernel.
    Args:
        indices_ptr: Input indices
        output_ptr: Output one-hot tensor (n_indices, n_classes)
        n_classes: Number of classes
        n_indices: Number of indices
    """
    pid = tl.program_id(axis=0)
    idx = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = idx < n_indices
    
    # Load the index for this element
    indices = tl.load(indices_ptr + idx, mask=mask, other=0)
    
    # For each index, fill the corresponding row in output
    for i in range(BLOCK_SIZE):
        if pid * BLOCK_SIZE + i < n_indices:
            index_val = tl.load(indices_ptr + pid * BLOCK_SIZE + i)
            # Fill entire row for this index
            row_start = (pid * BLOCK_SIZE + i) * n_classes
            for j in range(n_classes):
                value = 1.0 if j == index_val else 0.0
                tl.store(output_ptr + row_start + j, value)


@triton.jit
def maximum_kernel(x_ptr, y_ptr, output_ptr, N, BLOCK_SIZE: tl.constexpr):
    """
    Maximum kernel.
    """
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = tl.maximum(x, y)
    tl.store(output_ptr + offsets, output, mask=mask)


# =============================================================================
# GPU OPERATIONS
# =============================================================================


@register_cuda("add")
def add(x, y):
    """
    Add with broadcasting support and optimizations.
    """
    if isinstance(y, CUDAStorage):
        # Compute broadcasted shape
        broadcast_shape = broadcast_shapes(x.shape, y.shape)
        
        # Broadcast both tensors to the same shape
        if x.shape != broadcast_shape:
            x = x.broadcast_to(broadcast_shape)
        if y.shape != broadcast_shape:
            y = y.broadcast_to(broadcast_shape)
        
        # Now both tensors have the same shape - use strided-aware kernel
        output = CUDAStorage(broadcast_shape, dtype=x.dtype)

        # Use strided add kernel for 1-4D, fallback to contiguous for >4D
        n_elements = x.size
        ndim = len(x.shape)

        if ndim <= 4:
            # Optimized strided path for 1-4D tensors
            x_shape = list(x.shape) + [1] * (4 - ndim)
            x_strides = list(x.strides) + [1] * (4 - ndim)
            y_shape = list(y.shape) + [1] * (4 - ndim)
            y_strides = list(y.strides) + [1] * (4 - ndim)
            out_shape = list(output.shape) + [1] * (4 - ndim)
            out_strides = list(output.strides) + [1] * (4 - ndim)

            grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

            strided_binary_kernel[grid](
                x, x_shape[0], x_strides[0], x_shape[1], x_strides[1], x_shape[2], x_strides[2], x_shape[3], x_strides[3],
                y, y_shape[0], y_strides[0], y_shape[1], y_strides[1], y_shape[2], y_strides[2], y_shape[3], y_strides[3],
                output, out_shape[0], out_strides[0], out_shape[1], out_strides[1], out_shape[2], out_strides[2], out_shape[3], out_strides[3],
                n_elements, ndim, op_type=0,  # for add
                BLOCK_SIZE=1024
            )
        else:
            # Fallback for >4D: use contiguous tensors
            x_cont = x.contiguous() if not x.is_contiguous() else x
            y_cont = y.contiguous() if not y.is_contiguous() else y

            grid = (triton.cdiv(n_elements, 1024),)
            add_kernel[grid](x_cont, y_cont, output, n_elements, BLOCK_SIZE=1024)
        
        return output
            
    else:
        # Scalar addition - support strided tensors
        output = CUDAStorage(x.shape, dtype=x.dtype)
        n_elements = x.size
        ndim = len(x.shape)

        if ndim <= 4:
            # Use strided scalar kernel for 1-4D
            x_shape = list(x.shape) + [1] * (4 - ndim)
            x_strides = list(x.strides) + [1] * (4 - ndim)
            out_shape = list(output.shape) + [1] * (4 - ndim)
            out_strides = list(output.strides) + [1] * (4 - ndim)

            grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
            strided_scalar_kernel[grid](
                x, x_shape[0], x_strides[0], x_shape[1], x_strides[1], x_shape[2], x_strides[2], x_shape[3], x_strides[3],
                y,  # scalar value
                output, out_shape[0], out_strides[0], out_shape[1], out_strides[1], out_shape[2], out_strides[2], out_shape[3], out_strides[3],
                n_elements, ndim, op_type=0,  # for add
                BLOCK_SIZE=1024
            )
        else:
            # Fallback for >4D
            if not x.is_contiguous():
                x = x.contiguous()
            grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]), )
            add_scalar_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)
    
    return output


@register_cuda("add_inplace")
def add_inplace(x, y):
    """
    In-place addition: x += y
    Modifies x directly without creating new tensor.
    """
    if isinstance(y, CUDAStorage):
        # Compute broadcasted shape
        broadcast_shape = broadcast_shapes(x.shape, y.shape)

        # For in-place operation, x must be able to accommodate the result
        if x.shape != broadcast_shape:
            raise RuntimeError(f"Cannot broadcast in-place: x.shape={x.shape} vs result.shape={broadcast_shape}")

        # Broadcast y if needed
        if y.shape != broadcast_shape:
            y = y.broadcast_to(broadcast_shape)

        # Ensure contiguous for efficiency
        if not x.is_contiguous():
            raise RuntimeError("In-place operations require contiguous tensors")
        if not y.is_contiguous():
            y = y.contiguous()

        n_elements = x.size

        # Use in-place add kernel (modifies x.storage directly)
        grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]), )
        add_inplace_kernel[grid](x, y, n_elements, BLOCK_SIZE=1024)

        return x  # Return original tensor object

    else:
        # Scalar in-place addition
        if not x.is_contiguous():
            raise RuntimeError("In-place operations require contiguous tensors")

        n_elements = x.size

        # Use scalar in-place add kernel
        grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]), )
        add_scalar_inplace_kernel[grid](x, y, n_elements, BLOCK_SIZE=1024)

    return x


@register_cuda("sub_inplace")
def sub_inplace(x, y):
    """
    In-place subtraction: x -= y
    Modifies x directly without creating new tensor.
    """
    if isinstance(y, CUDAStorage):
        # Compute broadcasted shape
        broadcast_shape = broadcast_shapes(x.shape, y.shape)

        # For in-place operation, x must be able to accommodate the result
        if x.shape != broadcast_shape:
            raise RuntimeError(f"Cannot broadcast in-place: x.shape={x.shape} vs result.shape={broadcast_shape}")

        # Broadcast y if needed
        if y.shape != broadcast_shape:
            y = y.broadcast_to(broadcast_shape)

        # Ensure contiguous for efficiency
        if not x.is_contiguous():
            raise RuntimeError("In-place operations require contiguous tensors")
        if not y.is_contiguous():
            y = y.contiguous()

        n_elements = x.size

        # Use in-place sub kernel (modifies x.storage directly)
        grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]), )
        sub_inplace_kernel[grid](x, y, n_elements, BLOCK_SIZE=1024)

        return x  # Return original tensor object

    else:
        # Scalar in-place subtraction
        if not x.is_contiguous():
            raise RuntimeError("In-place operations require contiguous tensors")

        n_elements = x.size

        # Use scalar in-place sub kernel
        grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]), )
        sub_scalar_inplace_kernel[grid](x, y, n_elements, BLOCK_SIZE=1024)

    return x


@register_cuda("mul_inplace")
def mul_inplace(x, y):
    """
    In-place multiplication: x *= y
    Modifies x directly without creating new tensor.
    """
    if isinstance(y, CUDAStorage):
        # Compute broadcasted shape
        broadcast_shape = broadcast_shapes(x.shape, y.shape)

        # For in-place operation, x must be able to accommodate the result
        if x.shape != broadcast_shape:
            raise RuntimeError(f"Cannot broadcast in-place: x.shape={x.shape} vs result.shape={broadcast_shape}")

        # Broadcast y if needed
        if y.shape != broadcast_shape:
            y = y.broadcast_to(broadcast_shape)

        # Ensure contiguous for efficiency
        if not x.is_contiguous():
            raise RuntimeError("In-place operations require contiguous tensors")
        if not y.is_contiguous():
            y = y.contiguous()

        n_elements = x.size

        # Use in-place mul kernel (modifies x.storage directly)
        grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]), )
        mul_inplace_kernel[grid](x, y, n_elements, BLOCK_SIZE=1024)

        return x  # Return original tensor object

    else:
        # Scalar in-place multiplication
        if not x.is_contiguous():
            raise RuntimeError("In-place operations require contiguous tensors")

        n_elements = x.size

        # Use scalar in-place mul kernel
        grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]), )
        mul_scalar_inplace_kernel[grid](x, y, n_elements, BLOCK_SIZE=1024)

    return x


@register_cuda("sub")
def sub(x, y):
    """
    Sub with broadcasting support.
    """
    if isinstance(y, CUDAStorage):
        # Compute broadcasted shape
        broadcast_shape = broadcast_shapes(x.shape, y.shape)
        
        # Broadcast both tensors to the same shape
        if x.shape != broadcast_shape:
            x = x.broadcast_to(broadcast_shape)
        if y.shape != broadcast_shape:
            y = y.broadcast_to(broadcast_shape)
        
        # Now both tensors have the same shape - use strided-aware kernel
        output = CUDAStorage(broadcast_shape, dtype=x.dtype)

        # Use strided sub kernel for 1-4D, fallback to contiguous for >4D
        n_elements = x.size
        ndim = len(x.shape)

        if ndim <= 4:
            # Optimized strided path for 1-4D tensors
            x_shape = list(x.shape) + [1] * (4 - ndim)
            x_strides = list(x.strides) + [1] * (4 - ndim)
            y_shape = list(y.shape) + [1] * (4 - ndim)
            y_strides = list(y.strides) + [1] * (4 - ndim)
            out_shape = list(output.shape) + [1] * (4 - ndim)
            out_strides = list(output.strides) + [1] * (4 - ndim)

            grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

            strided_binary_kernel[grid](
                x, x_shape[0], x_strides[0], x_shape[1], x_strides[1], x_shape[2], x_strides[2], x_shape[3], x_strides[3],
                y, y_shape[0], y_strides[0], y_shape[1], y_strides[1], y_shape[2], y_strides[2], y_shape[3], y_strides[3],
                output, out_shape[0], out_strides[0], out_shape[1], out_strides[1], out_shape[2], out_strides[2], out_shape[3], out_strides[3],
                n_elements, ndim, op_type=1,  # for sub
                BLOCK_SIZE=1024
            )
        else:
            # Fallback for >4D: use contiguous tensors
            x_cont = x.contiguous() if not x.is_contiguous() else x
            y_cont = y.contiguous() if not y.is_contiguous() else y

            grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
            sub_kernel[grid](x_cont, y_cont, output, n_elements, BLOCK_SIZE=1024)
        
        return output
    else:
        # Scalar subtraction: x - scalar = x + (-scalar)
        return add(x, -y)


def iadd(x, y):
    """
    Inplace add kernel.
    """
    if isinstance(y, CUDAStorage):
        # For in-place operations, shapes must be compatible
        if x.shape != y.shape:
            raise ValueError("In-place addition requires compatible shapes")
        output_shape = x.shape
    else:
        output_shape = x.shape
    
    if not x.is_contiguous():
        x = x.contiguous()

    if isinstance(y, CUDAStorage):
        if not y.is_contiguous():
            y = y.contiguous()

    n_elements = x.size

    if isinstance(y, CUDAStorage):
        # Use optimized autotune add_kernel for tensor-tensor addition
        grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK"]), )
        add_kernel[grid](x, y, x, n_elements)  # In-place operation
    else:
        # Use scalar add kernel for tensor-scalar addition
        grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]), )
        add_scalar_kernel[grid](x, y, x, n_elements, BLOCK_SIZE=1024)  # In-place operation

    return x


@register_cuda("mul")
def mul(x, y):
    """
    Mul with broadcasting support.
    """
    # Convert y to CUDAStorage if it's a scalar tensor (0-dimensional)
    if isinstance(y, CUDAStorage):
        # Handle both tensor-tensor and tensor-scalar_tensor multiplication
        # Compute broadcasted shape
        broadcast_shape = broadcast_shapes(x.shape, y.shape)
        
        # Broadcast both tensors to the same shape
        if x.shape != broadcast_shape:
            x = x.broadcast_to(broadcast_shape)
        if y.shape != broadcast_shape:
            y = y.broadcast_to(broadcast_shape)
        
        # Now both tensors have the same shape - use strided-aware kernel
        output = CUDAStorage(broadcast_shape, dtype=x.dtype)

        # Use strided mul kernel for 1-4D, fallback to contiguous for >4D
        n_elements = x.size
        ndim = len(x.shape)

        if ndim <= 4:
            # Optimized strided path for 1-4D tensors
            x_shape = list(x.shape) + [1] * (4 - ndim)
            x_strides = list(x.strides) + [1] * (4 - ndim)
            y_shape = list(y.shape) + [1] * (4 - ndim)
            y_strides = list(y.strides) + [1] * (4 - ndim)
            out_shape = list(output.shape) + [1] * (4 - ndim)
            out_strides = list(output.strides) + [1] * (4 - ndim)

            grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

            strided_binary_kernel[grid](
                x, x_shape[0], x_strides[0], x_shape[1], x_strides[1], x_shape[2], x_strides[2], x_shape[3], x_strides[3],
                y, y_shape[0], y_strides[0], y_shape[1], y_strides[1], y_shape[2], y_strides[2], y_shape[3], y_strides[3],
                output, out_shape[0], out_strides[0], out_shape[1], out_strides[1], out_shape[2], out_strides[2], out_shape[3], out_strides[3],
                n_elements, ndim, op_type=2,  # for mul
                BLOCK_SIZE=1024
            )
        else:
            # Fallback for >4D: use contiguous tensors with existing kernel
            x_cont = x.contiguous() if not x.is_contiguous() else x
            y_cont = y.contiguous() if not y.is_contiguous() else y

            grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

            mul_kernel[grid](
                x_cont, y_cont, output, n_elements,
                BLOCK_SIZE=1024
            )
        
        return output
    else:
        # Scalar multiplication - support strided tensors
        output = CUDAStorage(x.shape, dtype=x.dtype)
        n_elements = x.size
        ndim = len(x.shape)

        # Ensure scalar type compatibility
        if x.dtype == "float32":
            y = float(y)
        elif x.dtype == "float16" or x.dtype == "bfloat16":
            y = float(y)
        elif x.dtype in ["int32", "int64"]:
            y = int(y)

        if ndim <= 4:
            # Use strided scalar kernel for 1-4D
            x_shape = list(x.shape) + [1] * (4 - ndim)
            x_strides = list(x.strides) + [1] * (4 - ndim)
            out_shape = list(output.shape) + [1] * (4 - ndim)
            out_strides = list(output.strides) + [1] * (4 - ndim)

            grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
            strided_scalar_kernel[grid](
                x, x_shape[0], x_strides[0], x_shape[1], x_strides[1], x_shape[2], x_strides[2], x_shape[3], x_strides[3],
                y,  # scalar value
                output, out_shape[0], out_strides[0], out_shape[1], out_strides[1], out_shape[2], out_strides[2], out_shape[3], out_strides[3],
                n_elements, ndim, op_type=2,  # for mul
                BLOCK_SIZE=1024
            )
        else:
            # Fallback for >4D
            if not x.is_contiguous():
                x = x.contiguous()
            grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]), )
            mul_scalar_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)
    
    return output


@register_cuda("truediv")
@register_cuda("div")
def truediv(x, y):
    """
    True division with broadcasting support.
    """
    if isinstance(y, CUDAStorage):
        # Compute broadcasted shape
        broadcast_shape = broadcast_shapes(x.shape, y.shape)
        
        # Broadcast both tensors to the same shape
        if x.shape != broadcast_shape:
            x = x.broadcast_to(broadcast_shape)
        if y.shape != broadcast_shape:
            y = y.broadcast_to(broadcast_shape)
        
        # Now both tensors have the same shape - use strided-aware kernel
        output = CUDAStorage(broadcast_shape, dtype=x.dtype)

        # Use strided div kernel for 1-4D, fallback to contiguous for >4D
        n_elements = x.size
        ndim = len(x.shape)

        if ndim <= 4:
            # Optimized strided path for 1-4D tensors
            x_shape = list(x.shape) + [1] * (4 - ndim)
            x_strides = list(x.strides) + [1] * (4 - ndim)
            y_shape = list(y.shape) + [1] * (4 - ndim)
            y_strides = list(y.strides) + [1] * (4 - ndim)
            out_shape = list(output.shape) + [1] * (4 - ndim)
            out_strides = list(output.strides) + [1] * (4 - ndim)

            grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

            strided_binary_kernel[grid](
                x, x_shape[0], x_strides[0], x_shape[1], x_strides[1], x_shape[2], x_strides[2], x_shape[3], x_strides[3],
                y, y_shape[0], y_strides[0], y_shape[1], y_strides[1], y_shape[2], y_strides[2], y_shape[3], y_strides[3],
                output, out_shape[0], out_strides[0], out_shape[1], out_strides[1], out_shape[2], out_strides[2], out_shape[3], out_strides[3],
                n_elements, ndim, op_type=3,  # for div
                BLOCK_SIZE=1024
            )
        else:
            # Fallback for >4D: use contiguous tensors
            x_cont = x.contiguous() if not x.is_contiguous() else x
            y_cont = y.contiguous() if not y.is_contiguous() else y

            grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
            div_kernel[grid](x_cont, y_cont, output, n_elements, BLOCK_SIZE=1024)
        
        return output
    else:
        # Scalar division
        output = CUDAStorage(x.shape, dtype=x.dtype)
        if not x.is_contiguous():
            x = x.contiguous()
        
        n_elements = output.size
        
        # Use scalar div kernel
        grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]), )
        div_scalar_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)
    
    return output


@register_cuda("rsub")
def rsub(x, y):
    """
    Reverse subtraction (scalar - tensor).
    """
    output = CUDAStorage(x.shape, dtype=x.dtype)
    n_elements = output.size
    ndim = len(x.shape)

    if ndim <= 4:
        # Use strided scalar kernel for 1-4D
        x_shape = list(x.shape) + [1] * (4 - ndim)
        x_strides = list(x.strides) + [1] * (4 - ndim)
        out_shape = list(output.shape) + [1] * (4 - ndim)
        out_strides = list(output.strides) + [1] * (4 - ndim)

        grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

        strided_scalar_kernel[grid](
            x, x_shape[0], x_strides[0], x_shape[1], x_strides[1], x_shape[2], x_strides[2], x_shape[3], x_strides[3],
            y,
            output, out_shape[0], out_strides[0], out_shape[1], out_strides[1], out_shape[2], out_strides[2], out_shape[3], out_strides[3],
            n_elements, ndim, op_type=SCALAR_RSUB,
            BLOCK_SIZE=1024
        )
    else:
        # Fallback for >4D: use contiguous tensors
        if not x.is_contiguous():
            x = x.contiguous()

        # Use scalar rsub kernel
        grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]), )
        rsub_scalar_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)

    return output


@register_cuda("rpower")
def rpower(x, y):
    """
    Reverse power (scalar ** tensor).
    """
    output = CUDAStorage(x.shape, dtype=x.dtype)
    n_elements = output.size
    ndim = len(x.shape)

    if ndim <= 4:
        # Use strided scalar kernel for 1-4D
        x_shape = list(x.shape) + [1] * (4 - ndim)
        x_strides = list(x.strides) + [1] * (4 - ndim)
        out_shape = list(output.shape) + [1] * (4 - ndim)
        out_strides = list(output.strides) + [1] * (4 - ndim)

        grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

        strided_scalar_kernel[grid](
            x, x_shape[0], x_strides[0], x_shape[1], x_strides[1], x_shape[2], x_strides[2], x_shape[3], x_strides[3],
            y,
            output, out_shape[0], out_strides[0], out_shape[1], out_strides[1], out_shape[2], out_strides[2], out_shape[3], out_strides[3],
            n_elements, ndim, op_type=SCALAR_RPOWER,
            BLOCK_SIZE=1024
        )
    else:
        # Fallback for >4D: use contiguous tensors
        if not x.is_contiguous():
            x = x.contiguous()

        # Use scalar rpower kernel
        grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]), )
        rpower_scalar_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)

    return output


@register_cuda("rdiv")
def rtruediv(x, y):
    """
    Reverse true division (scalar / tensor).
    """
    output = CUDAStorage(x.shape, dtype=x.dtype)
    n_elements = output.size
    ndim = len(x.shape)

    if ndim <= 4:
        # Use strided scalar kernel for 1-4D
        x_shape = list(x.shape) + [1] * (4 - ndim)
        x_strides = list(x.strides) + [1] * (4 - ndim)
        out_shape = list(output.shape) + [1] * (4 - ndim)
        out_strides = list(output.strides) + [1] * (4 - ndim)

        grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

        strided_scalar_kernel[grid](
            x, x_shape[0], x_strides[0], x_shape[1], x_strides[1], x_shape[2], x_strides[2], x_shape[3], x_strides[3],
            y,
            output, out_shape[0], out_strides[0], out_shape[1], out_strides[1], out_shape[2], out_strides[2], out_shape[3], out_strides[3],
            n_elements, ndim, op_type=SCALAR_RDIV,
            BLOCK_SIZE=1024
        )
    else:
        # Fallback for >4D: use contiguous tensors
        if not x.is_contiguous():
            x = x.contiguous()

        # Use scalar rdiv kernel
        grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]), )
        rdiv_scalar_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)

    return output


@register_cuda("pow")
def pow(x, scalar):
    """
    Element-wise power operation.
    """
    # Handle case where x and scalar might be swapped (for rpower)
    if not hasattr(x, 'shape'):
        # x is a scalar, scalar is a tensor - this is rpower case: scalar^x
        base_scalar = x
        tensor = scalar
        
        if base_scalar == 0:
            result = CUDAStorage(tensor.shape, dtype=tensor.dtype)
            result.fill(0.0)
            return result
        elif base_scalar == 1:
            result = CUDAStorage(tensor.shape, dtype=tensor.dtype)
            result.fill(1.0)
            return result
        elif base_scalar < 0:
            # Negative base with tensor exponent - should be NaN for non-integer exponents
            result = CUDAStorage(tensor.shape, dtype=tensor.dtype)
            result.fill(float('nan'))
            return result
        else:
            # base^tensor = exp(tensor * log(base))
            log_base = np.log(base_scalar)
            scaled = mul(tensor, log_base)
            return exp(scaled)
    
    # Normal case: tensor^scalar
    if scalar == 0:
        result = CUDAStorage(x.shape, dtype=x.dtype)
        result.fill(1.0)
        return result
    elif scalar == 1:
        return x
    elif scalar == 2:
        return mul(x, x)  # x^2 = x * x
    elif int(scalar) == scalar and scalar >= 0:
        # Integer exponent, can handle negative bases
        if scalar == 3:
            return mul(mul(x, x), x)  # x^3 = x * x * x
        elif scalar == 4:
            x2 = mul(x, x)
            return mul(x2, x2)  # x^4 = (x^2)^2
        else:
            # Use repeated multiplication for small integer powers
            result = x
            for _ in range(int(scalar) - 1):
                result = mul(result, x)
            return result
    else:
        # Non-integer exponent - use exp(scalar * log(x))
        # This will naturally produce NaN for negative inputs
        log_x = log(x)
        scaled = mul(log_x, scalar)
        return exp(scaled)


@register_cuda("log")
def log(x):
    """
    Element-wise natural logarithm.
    """
    output = CUDAStorage(x.shape, dtype=x.dtype)

    # Use strided unary kernel for 1-4D, fallback to contiguous for >4D
    n_elements = x.size
    ndim = len(x.shape)

    if ndim <= 4:
        # Optimized strided path for 1-4D tensors
        x_shape = list(x.shape) + [1] * (4 - ndim)
        x_strides = list(x.strides) + [1] * (4 - ndim)
        out_shape = list(output.shape) + [1] * (4 - ndim)
        out_strides = list(output.strides) + [1] * (4 - ndim)

        grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

        strided_unary_kernel[grid](
            x, x_shape[0], x_strides[0], x_shape[1], x_strides[1], x_shape[2], x_strides[2], x_shape[3], x_strides[3],
            output, out_shape[0], out_strides[0], out_shape[1], out_strides[1], out_shape[2], out_strides[2], out_shape[3], out_strides[3],
            n_elements, ndim, op_type=1,  # for log
            BLOCK_SIZE=1024
        )
    else:
        # Fallback for >4D: use contiguous tensors
        x_cont = x.contiguous() if not x.is_contiguous() else x
        grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
        log_kernel[grid](x_cont, output, n_elements, BLOCK_SIZE=1024)

    return output


@register_cuda("copy")
def copy(dst, src):
    """
    Copy data from src to dst in-place.
    Handles both contiguous and non-contiguous tensors.

    Args:
        dst: Destination tensor (will be modified)
        src: Source tensor to copy from

    Returns:
        dst (modified in-place)
    """
    # Ensure shapes match
    if dst.shape != src.shape:
        raise RuntimeError(f"Shape mismatch in copy: dst.shape={dst.shape}, src.shape={src.shape}")

    # If both are contiguous, use simple copy kernel
    if dst.is_contiguous() and src.is_contiguous():
        n_elements = dst.size
        grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]), )
        copy_kernel[grid](src, dst, n_elements, BLOCK_SIZE=1024)
    else:
        # For non-contiguous tensors, we need element-wise copy
        # First make source contiguous if needed
        if not src.is_contiguous():
            src = src.contiguous()

        # If dst is non-contiguous, we need special handling
        if not dst.is_contiguous():
            # For non-contiguous dst, we can't do simple element-wise copy
            # Instead, make dst contiguous by copying data to new memory layout
            temp = CUDAStorage(dst.shape, dtype=dst.dtype)
            n_elements = dst.size
            grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]), )
            copy_kernel[grid](src, temp, n_elements, BLOCK_SIZE=1024)

            # Copy temp back to dst's memory location
            copy_kernel[grid](temp, dst, n_elements, BLOCK_SIZE=1024)
            # Update dst to be contiguous
            dst.strides = temp.strides
            dst._is_contiguous = True
        else:
            # dst is contiguous, just copy
            n_elements = dst.size
            grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]), )
            copy_kernel[grid](src, dst, n_elements, BLOCK_SIZE=1024)

    return dst


@register_cuda("exp")
def exp(x):
    """
    Element-wise exponential.
    """
    output = CUDAStorage(x.shape, dtype=x.dtype)

    # Use strided unary kernel for 1-4D, fallback to contiguous for >4D
    n_elements = x.size
    ndim = len(x.shape)

    if ndim <= 4:
        # Optimized strided path for 1-4D tensors
        x_shape = list(x.shape) + [1] * (4 - ndim)
        x_strides = list(x.strides) + [1] * (4 - ndim)
        out_shape = list(output.shape) + [1] * (4 - ndim)
        out_strides = list(output.strides) + [1] * (4 - ndim)

        grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

        strided_unary_kernel[grid](
            x, x_shape[0], x_strides[0], x_shape[1], x_strides[1], x_shape[2], x_strides[2], x_shape[3], x_strides[3],
            output, out_shape[0], out_strides[0], out_shape[1], out_strides[1], out_shape[2], out_strides[2], out_shape[3], out_strides[3],
            n_elements, ndim, op_type=0,  # for exp
            BLOCK_SIZE=1024
        )
    else:
        # Fallback for >4D: use contiguous tensors
        x_cont = x.contiguous() if not x.is_contiguous() else x
        grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
        exp_kernel[grid](x_cont, output, n_elements, BLOCK_SIZE=1024)

    return output


@register_cuda("sin")
def sin(x):
    """
    Element-wise sine.
    """
    output = CUDAStorage(x.shape, dtype=x.dtype)

    # Use strided unary kernel for 1-4D, fallback to contiguous for >4D
    n_elements = x.size
    ndim = len(x.shape)

    if ndim <= 4:
        # Optimized strided path for 1-4D tensors
        # Pad shapes and strides to 4D
        x_shape = list(x.shape) + [1] * (4 - ndim)
        x_strides = list(x.strides) + [1] * (4 - ndim)
        out_shape = list(output.shape) + [1] * (4 - ndim)
        out_strides = list(output.strides) + [1] * (4 - ndim)

        grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

        strided_unary_kernel[grid](
            x, x_shape[0], x_strides[0], x_shape[1], x_strides[1], x_shape[2], x_strides[2], x_shape[3], x_strides[3],
            output, out_shape[0], out_strides[0], out_shape[1], out_strides[1], out_shape[2], out_strides[2], out_shape[3], out_strides[3],
            n_elements, ndim, op_type=5,  # for sin
            BLOCK_SIZE=1024
        )
    else:
        # Fallback for >4D: use contiguous tensors
        x_cont = x.contiguous() if not x.is_contiguous() else x
        grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
        sin_kernel[grid](x_cont, output, n_elements, BLOCK_SIZE=1024)

    return output


@register_cuda("cos")
def cos(x):
    """
    Element-wise cosine.
    """
    output = CUDAStorage(x.shape, dtype=x.dtype)

    # Use strided unary kernel for 1-4D, fallback to contiguous for >4D
    n_elements = x.size
    ndim = len(x.shape)

    if ndim <= 4:
        # Optimized strided path for 1-4D tensors
        # Pad shapes and strides to 4D
        x_shape = list(x.shape) + [1] * (4 - ndim)
        x_strides = list(x.strides) + [1] * (4 - ndim)
        out_shape = list(output.shape) + [1] * (4 - ndim)
        out_strides = list(output.strides) + [1] * (4 - ndim)

        grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

        strided_unary_kernel[grid](
            x, x_shape[0], x_strides[0], x_shape[1], x_strides[1], x_shape[2], x_strides[2], x_shape[3], x_strides[3],
            output, out_shape[0], out_strides[0], out_shape[1], out_strides[1], out_shape[2], out_strides[2], out_shape[3], out_strides[3],
            n_elements, ndim, op_type=6,  # for cos
            BLOCK_SIZE=1024
        )
    else:
        # Fallback for >4D: use contiguous tensors
        x_cont = x.contiguous() if not x.is_contiguous() else x
        grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
        cos_kernel[grid](x_cont, output, n_elements, BLOCK_SIZE=1024)

    return output


@register_cuda("sqrt")
def sqrt(x):
    """
    Element-wise square root.
    """
    output = CUDAStorage(x.shape, dtype=x.dtype)

    # Use strided unary kernel for 1-4D, fallback to contiguous for >4D
    n_elements = x.size
    ndim = len(x.shape)

    if ndim <= 4:
        # Optimized strided path for 1-4D tensors
        # Pad shapes and strides to 4D
        x_shape = list(x.shape) + [1] * (4 - ndim)
        x_strides = list(x.strides) + [1] * (4 - ndim)
        out_shape = list(output.shape) + [1] * (4 - ndim)
        out_strides = list(output.strides) + [1] * (4 - ndim)

        grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

        strided_unary_kernel[grid](
            x, x_shape[0], x_strides[0], x_shape[1], x_strides[1], x_shape[2], x_strides[2], x_shape[3], x_strides[3],
            output, out_shape[0], out_strides[0], out_shape[1], out_strides[1], out_shape[2], out_strides[2], out_shape[3], out_strides[3],
            n_elements, ndim, op_type=2,  # for sqrt
            BLOCK_SIZE=1024
        )
    else:
        # Fallback for >4D: use contiguous tensors
        x_cont = x.contiguous() if not x.is_contiguous() else x
        grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
        sqrt_kernel[grid](x_cont, output, n_elements, BLOCK_SIZE=1024)

    return output


@triton.jit
def abs_kernel(x_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    """
    Absolute value kernel.
    """
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    output = tl.abs(x)
    tl.store(output_ptr + offsets, output, mask=mask)


@register_cuda("abs")
def abs(x):
    """
    Element-wise absolute value.
    """
    output = CUDAStorage(x.shape, dtype=x.dtype)

    # Use strided unary kernel for 1-4D, fallback to contiguous for >4D
    n_elements = x.size
    ndim = len(x.shape)

    if ndim <= 4:
        # Optimized strided path for 1-4D tensors
        # Pad shapes and strides to 4D
        x_shape = list(x.shape) + [1] * (4 - ndim)
        x_strides = list(x.strides) + [1] * (4 - ndim)
        out_shape = list(output.shape) + [1] * (4 - ndim)
        out_strides = list(output.strides) + [1] * (4 - ndim)

        grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

        strided_unary_kernel[grid](
            x, x_shape[0], x_strides[0], x_shape[1], x_strides[1], x_shape[2], x_strides[2], x_shape[3], x_strides[3],
            output, out_shape[0], out_strides[0], out_shape[1], out_strides[1], out_shape[2], out_strides[2], out_shape[3], out_strides[3],
            n_elements, ndim, op_type=3,  # for abs
            BLOCK_SIZE=1024
        )
    else:
        # Fallback for >4D: use contiguous tensors
        x_cont = x.contiguous() if not x.is_contiguous() else x
        grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
        abs_kernel[grid](x_cont, output, n_elements, BLOCK_SIZE=1024)

    return output


@triton.jit
def sign_kernel(x_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    """
    Sign function kernel.
    """
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    # Sign function: -1 if x < 0, 0 if x == 0, 1 if x > 0
    output = tl.where(x > 0, 1.0, tl.where(x < 0, -1.0, 0.0))
    tl.store(output_ptr + offsets, output, mask=mask)


@register_cuda("sign")
def sign(x):
    """
    Element-wise sign function.
    """
    output = CUDAStorage(x.shape, dtype=x.dtype)
    n_elements = x.size
    ndim = len(x.shape)

    if ndim <= 4:
        # Use strided special kernel for 1-4D
        x_shape = list(x.shape) + [1] * (4 - ndim)
        x_strides = list(x.strides) + [1] * (4 - ndim)
        out_shape = list(output.shape) + [1] * (4 - ndim)
        out_strides = list(output.strides) + [1] * (4 - ndim)

        grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

        strided_special_kernel[grid](
            x, x_shape[0], x_strides[0], x_shape[1], x_strides[1], x_shape[2], x_strides[2], x_shape[3], x_strides[3],
            output, out_shape[0], out_strides[0], out_shape[1], out_strides[1], out_shape[2], out_strides[2], out_shape[3], out_strides[3],
            n_elements, ndim,
            op_type=SPECIAL_SIGN, min_val=0.0, max_val=0.0, scalar=0.0,  # no scalar params needed for sign
            BLOCK_SIZE=1024
        )
    else:
        # Fallback for >4D: use contiguous tensors
        if not x.is_contiguous():
            x = x.contiguous()

        grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]), )
        sign_kernel[grid](x, output, n_elements, BLOCK_SIZE=1024)

    return output

@register_cuda("getitem")
def getitem(x, index):
    """Get item from CUDA tensor using indexing."""
    # Handle tuple index where elements might be CUDAStorage
    if isinstance(index, tuple):
        processed_index = []
        for idx in index:
            if hasattr(idx, 'shape'):  # It's a CUDAStorage tensor
                processed_index.append(idx)
            else:
                processed_index.append(idx)
        index = tuple(processed_index)
    result = x[index]  # CUDAStorage supports indexing directly
    return result

@register_cuda("setitem")
def setitem(x, index, value):
    """Set item in CUDA tensor using indexing."""
    x[index] = value
    return x


@triton.jit
def clamp_kernel(x_ptr, output_ptr, min_val, max_val, n_elements, BLOCK_SIZE: tl.constexpr):
    """
    Clamp kernel.
    """
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    
    # Apply clamp
    if min_val is not None:
        x = tl.where(x < min_val, min_val, x)
    if max_val is not None:
        x = tl.where(x > max_val, max_val, x)
    
    tl.store(output_ptr + offsets, x, mask=mask)


@register_cuda("clamp")
def clamp(x, min_val=None, max_val=None):
    """
    Element-wise clamp/clip operation.
    """
    output = CUDAStorage(x.shape, dtype=x.dtype)
    n_elements = x.size
    ndim = len(x.shape)

    if ndim <= 4:
        # Use strided special kernel for 1-4D
        x_shape = list(x.shape) + [1] * (4 - ndim)
        x_strides = list(x.strides) + [1] * (4 - ndim)
        out_shape = list(output.shape) + [1] * (4 - ndim)
        out_strides = list(output.strides) + [1] * (4 - ndim)

        grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

        strided_special_kernel[grid](
            x, x_shape[0], x_strides[0], x_shape[1], x_strides[1], x_shape[2], x_strides[2], x_shape[3], x_strides[3],
            output, out_shape[0], out_strides[0], out_shape[1], out_strides[1], out_shape[2], out_strides[2], out_shape[3], out_strides[3],
            n_elements, ndim,
            op_type=SPECIAL_CLAMP, min_val=min_val, max_val=max_val, scalar=0.0,  # scalar param not used for clamp
            BLOCK_SIZE=1024
        )
    else:
        # Fallback for >4D: use contiguous tensors
        if not x.is_contiguous():
            x = x.contiguous()

        grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]), )
        clamp_kernel[grid](x, output, min_val, max_val, n_elements, BLOCK_SIZE=1024)

    return output


@triton.jit
def comparison_kernel(x_ptr, y_val, output_ptr, n_elements, op_type, BLOCK_SIZE: tl.constexpr):
    """
    Element-wise comparison kernel.
    op_type: 0=greater_equal, 1=less_equal
    """
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    
    if op_type == 0:  # greater_equal
        result = x >= y_val
    else:  # less_equal
        result = x <= y_val
    
    # Convert bool to float
    result = tl.where(result, 1.0, 0.0)
    tl.store(output_ptr + offsets, result, mask=mask)


def greater_equal(x, y):
    """Element-wise greater than or equal comparison."""
    output = CUDAStorage(x.shape, dtype=x.dtype)
    if not x.is_contiguous():
        x = x.contiguous()
    
    n_elements = x.size
    y_val = float(y) if not isinstance(y, CUDAStorage) else y  # For scalar comparison
    
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]), )
    comparison_kernel[grid](x, y_val, output, n_elements, 0, BLOCK_SIZE=1024)
    
    return output


def less_equal(x, y):
    """Element-wise less than or equal comparison."""
    output = CUDAStorage(x.shape, dtype=x.dtype)
    if not x.is_contiguous():
        x = x.contiguous()
    
    n_elements = x.size
    y_val = float(y) if not isinstance(y, CUDAStorage) else y  # For scalar comparison
    
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]), )
    comparison_kernel[grid](x, y_val, output, n_elements, 1, BLOCK_SIZE=1024)
    
    return output


def ones(shape, device=None, dtype="float32"):
    """Create tensor filled with ones."""
    return CUDAStorage(shape, dtype=dtype).fill(1.0)


def zeros(shape, device=None, dtype="float32"):
    """Create tensor filled with zeros."""
    return CUDAStorage(shape, dtype=dtype).fill(0.0)


@register_cuda("maximum")
def maximum(x, y):
    """
    Element-wise maximum with broadcasting support.
    """
    if isinstance(y, CUDAStorage):
        # Compute broadcasted shape
        broadcast_shape = broadcast_shapes(x.shape, y.shape)
        
        # Broadcast both tensors to the same shape
        if x.shape != broadcast_shape:
            x = x.broadcast_to(broadcast_shape)
        if y.shape != broadcast_shape:
            y = y.broadcast_to(broadcast_shape)
        
        output = CUDAStorage(broadcast_shape, dtype=x.dtype)
        
        if not x.is_contiguous():
            x = x.contiguous()
        if not y.is_contiguous():
            y = y.contiguous()
        
        n_elements = output.size
        
        grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]), )
        maximum_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)
        
        return output
    else:
        # Scalar maximum
        output = CUDAStorage(x.shape, dtype=x.dtype)
        n_elements = output.size
        ndim = len(x.shape)

        if ndim <= 4:
            # Use strided special kernel for 1-4D
            x_shape = list(x.shape) + [1] * (4 - ndim)
            x_strides = list(x.strides) + [1] * (4 - ndim)
            out_shape = list(output.shape) + [1] * (4 - ndim)
            out_strides = list(output.strides) + [1] * (4 - ndim)

            grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

            strided_special_kernel[grid](
                x, x_shape[0], x_strides[0], x_shape[1], x_strides[1], x_shape[2], x_strides[2], x_shape[3], x_strides[3],
                output, out_shape[0], out_strides[0], out_shape[1], out_strides[1], out_shape[2], out_strides[2], out_shape[3], out_strides[3],
                n_elements, ndim,
                op_type=SPECIAL_MAXIMUM, min_val=0.0, max_val=0.0, scalar=y,  # y is the scalar param
                BLOCK_SIZE=1024
            )
        else:
            # Fallback for >4D: use contiguous tensors
            if not x.is_contiguous():
                x = x.contiguous()

            grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]), )
            maximum_scalar_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)
    
    return output


@register_cuda("gt")
def gt(x, y):
    """Greater than comparison with support for tensor-tensor and tensor-scalar."""
    if isinstance(y, (int, float)):
        # Scalar comparison
        output = CUDAStorage(x.shape, dtype="bool")
        n_elements = output.size
        ndim = len(x.shape)

        if ndim <= 4:
            # Use strided compare kernel for 1-4D
            x_shape = list(x.shape) + [1] * (4 - ndim)
            x_strides = list(x.strides) + [1] * (4 - ndim)
            out_shape = list(output.shape) + [1] * (4 - ndim)
            out_strides = list(output.strides) + [1] * (4 - ndim)

            grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

            strided_compare_scalar_kernel[grid](
                x, x_shape[0], x_strides[0], x_shape[1], x_strides[1], x_shape[2], x_strides[2], x_shape[3], x_strides[3],
                y,
                output, out_shape[0], out_strides[0], out_shape[1], out_strides[1], out_shape[2], out_strides[2], out_shape[3], out_strides[3],
                n_elements, ndim, COMPARE_GT,
                BLOCK_SIZE=1024
            )
        else:
            # Fallback for >4D: use contiguous tensors
            if not x.is_contiguous():
                x = x.contiguous()

            grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]), )
            gt_scalar_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)

        return output
    else:
        # Tensor comparison
        output = CUDAStorage(x.shape, dtype="bool")
        if not x.is_contiguous():
            x = x.contiguous()
        if not y.is_contiguous():
            y = y.contiguous()
        
        n_elements = output.size
        grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]), )
        gt_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)
        return output


@register_cuda("eq")
def eq(x, y):
    """Equality comparison with support for tensor-tensor and tensor-scalar."""
    if isinstance(y, (int, float)):
        # Scalar comparison
        output = CUDAStorage(x.shape, dtype="bool")
        n_elements = output.size
        ndim = len(x.shape)

        if ndim <= 4:
            # Use strided compare kernel for 1-4D
            x_shape = list(x.shape) + [1] * (4 - ndim)
            x_strides = list(x.strides) + [1] * (4 - ndim)
            out_shape = list(output.shape) + [1] * (4 - ndim)
            out_strides = list(output.strides) + [1] * (4 - ndim)

            grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

            strided_compare_scalar_kernel[grid](
                x, x_shape[0], x_strides[0], x_shape[1], x_strides[1], x_shape[2], x_strides[2], x_shape[3], x_strides[3],
                y,
                output, out_shape[0], out_strides[0], out_shape[1], out_strides[1], out_shape[2], out_strides[2], out_shape[3], out_strides[3],
                n_elements, ndim, COMPARE_EQ,
                BLOCK_SIZE=1024
            )
        else:
            # Fallback for >4D: use contiguous tensors
            if not x.is_contiguous():
                x = x.contiguous()

            grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]), )
            eq_scalar_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)

        return output
    else:
        # Tensor comparison
        output = CUDAStorage(x.shape, dtype="bool")
        if not x.is_contiguous():
            x = x.contiguous()
        if not y.is_contiguous():
            y = y.contiguous()
        
        n_elements = output.size
        grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]), )
        eq_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)
        return output


def mul_scalar_kernel_wrapper(x, scalar):
    """
    Wrapper for scalar multiplication kernel.
    """
    output = CUDAStorage(x.shape, dtype=x.dtype)
    if not x.is_contiguous():
        x = x.contiguous()
    
    n_elements = x.size
    
    # Ensure scalar type compatibility
    if x.dtype == "float32":
        scalar = float(scalar)
    elif x.dtype == "float16" or x.dtype == "bfloat16":
        scalar = float(scalar)
    elif x.dtype in ["int32", "int64"]:
        scalar = int(scalar)
    
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]), )
    mul_scalar_kernel[grid](x, scalar, output, n_elements, BLOCK_SIZE=1024)
    
    return output


def arange_gpu(start, end, step, dtype="float32"):
    """
    GPU implementation of arange using Triton kernel.
    """
    length = max(0, int((end - start) / step))
    if length == 0:
        return CUDAStorage((0,), dtype=dtype)
    
    output = CUDAStorage((length,), dtype=dtype)
    
    grid = lambda meta: (triton.cdiv(length, meta["BLOCK_SIZE"]), )
    arange_kernel[grid](output, float(start), float(step), length, BLOCK_SIZE=1024)
    
    return output

@register_cuda("arange")
def arange(start, end, step, dtype="float32"):
    """
    Alias for arange_gpu for consistency with CPU backend.
    """
    return arange_gpu(start, end, step, dtype)


@register_cuda("one_hot")
def one_hot(indices, n_classes, dtype="float32"):
    """
    GPU implementation of one-hot encoding using Triton kernel.

    Args:
        indices: Indices tensor (CUDAStorage)
        n_classes: Number of classes
        dtype: Data type string

    Returns:
        CUDAStorage: One-hot encoded tensor
    """
    if not indices.is_contiguous():
        indices = indices.contiguous()

    # Flatten indices for processing
    original_shape = indices.shape
    flat_indices = indices.reshape((-1,))
    n_indices = flat_indices.size

    # Create output tensor
    output_shape = original_shape + (n_classes,)
    output = CUDAStorage(output_shape, dtype=dtype)
    output.fill(0.0)

    # Launch kernel
    grid = lambda meta: (triton.cdiv(n_indices, meta["BLOCK_SIZE"]), )
    one_hot_kernel[grid](flat_indices, output, n_classes, n_indices, BLOCK_SIZE=256)

    return output


@triton.jit
def where_kernel(condition_ptr, x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    """Triton kernel for element-wise where operation (both x and y are tensors)."""
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    condition = tl.load(condition_ptr + offsets, mask=mask)
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)

    # Convert condition to boolean (non-zero is True)
    condition_bool = condition != 0.0
    result = tl.where(condition_bool, x, y)
    tl.store(output_ptr + offsets, result, mask=mask)


@triton.jit
def where_scalar_x_kernel(
    condition_ptr, x_scalar, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr
):
    """Triton kernel for where operation when x is a scalar."""
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    condition = tl.load(condition_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)

    # Convert condition to boolean (non-zero is True)
    condition_bool = condition != 0.0
    result = tl.where(condition_bool, x_scalar, y)
    tl.store(output_ptr + offsets, result, mask=mask)


@triton.jit
def where_scalar_y_kernel(
    condition_ptr, x_ptr, y_scalar, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr
):
    """Triton kernel for where operation when y is a scalar."""
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    condition = tl.load(condition_ptr + offsets, mask=mask)
    x = tl.load(x_ptr + offsets, mask=mask)

    # Convert condition to boolean (non-zero is True)
    condition_bool = condition != 0.0
    result = tl.where(condition_bool, x, y_scalar)
    tl.store(output_ptr + offsets, result, mask=mask)


@triton.jit
def where_scalar_both_kernel(
    condition_ptr, x_scalar, y_scalar, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr
):
    """Triton kernel for where operation when both x and y are scalars."""
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    condition = tl.load(condition_ptr + offsets, mask=mask)

    # Convert condition to boolean (non-zero is True)
    condition_bool = condition != 0.0
    result = tl.where(condition_bool, x_scalar, y_scalar)
    tl.store(output_ptr + offsets, result, mask=mask)


@register_cuda("where")
def where(condition, x, y):
    """
    Optimized element-wise selection of values from x or y based on condition.

    This implementation efficiently handles scalar values without materializing
    them into full-size tensors, significantly reducing memory usage.

    Args:
        condition: Boolean tensor
        x: Value(s) where condition is True (tensor or scalar)
        y: Value(s) where condition is False (tensor or scalar)

    Returns:
        Output tensor with same shape as condition

    Memory savings:
        - Scalar x or y: Saves full tensor allocation (e.g., 256MB per scalar)
        - Both scalars: Saves 2x tensor allocations
        - Reduces memory usage from 53GB to ~18GB for Qwen 0.5B model
    """
    # Ensure condition is contiguous
    if not condition.is_contiguous():
        condition = condition.contiguous()

    # Determine output shape and dtype
    if isinstance(y, CUDAStorage):
        output_shape = y.shape
        output_dtype = y.dtype
    elif isinstance(x, CUDAStorage):
        output_shape = x.shape
        output_dtype = x.dtype
    else:
        output_shape = condition.shape
        output_dtype = condition.dtype

    # Track whether inputs are scalars
    x_is_scalar = not isinstance(x, CUDAStorage)
    y_is_scalar = not isinstance(y, CUDAStorage)

    # Handle tensor inputs that need broadcasting or contiguous conversion
    if not x_is_scalar:
        if not x.is_contiguous():
            x = x.contiguous()
        if x.shape != output_shape:
            x = broadcast_to(x, output_shape)
            if not x.is_contiguous():
                x = x.contiguous()

    if not y_is_scalar:
        if not y.is_contiguous():
            y = y.contiguous()
        if y.shape != output_shape:
            y = broadcast_to(y, output_shape)
            if not y.is_contiguous():
                y = y.contiguous()

    # Broadcast condition if needed
    if condition.shape != output_shape:
        condition = broadcast_to(condition, output_shape)
        if not condition.is_contiguous():
            condition = condition.contiguous()

    n_elements = condition.size
    output = CUDAStorage(condition.shape, dtype=output_dtype)

    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

    # Choose the appropriate kernel based on input types
    # This optimization avoids materializing scalar values as full tensors
    if x_is_scalar and y_is_scalar:
        # Both are scalars - use scalar kernel
        x_val = float(x)
        y_val = float(y)
        where_scalar_both_kernel[grid](
            condition, x_val, y_val, output, n_elements, BLOCK_SIZE=1024
        )
    elif x_is_scalar:
        # Only x is scalar - use scalar_x kernel (e.g., float("-inf") in attention)
        # This saves ~256MB per call for seq_len=2048, batch_size=1
        x_val = float(x)
        where_scalar_x_kernel[grid](
            condition, x_val, y, output, n_elements, BLOCK_SIZE=1024
        )
    elif y_is_scalar:
        # Only y is scalar - use scalar_y kernel
        y_val = float(y)
        where_scalar_y_kernel[grid](
            condition, x, y_val, output, n_elements, BLOCK_SIZE=1024
        )
    else:
        # Both are tensors - use tensor kernel
        where_kernel[grid](
            condition, x, y, output, n_elements, BLOCK_SIZE=1024
        )

    return output


def zeros_like(x):
    """Create tensor of zeros with same shape and dtype as x."""
    output = CUDAStorage(x.shape, dtype=x.dtype)
    output.fill(0.0)
    return output


@triton.jit
def argmax_kernel(input_ptr, output_ptr, n_elements, stride_input, stride_output, BLOCK_SIZE: tl.constexpr):
    """Triton kernel for argmax along last dimension."""
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input values
    input_vals = tl.load(input_ptr + offsets * stride_input, mask=mask, other=float('-inf'))
    
    # Find argmax within the block
    max_val = tl.max(input_vals, axis=0)
    is_max = input_vals == max_val
    
    # Find first occurrence of max value
    indices = tl.arange(0, BLOCK_SIZE)
    masked_indices = tl.where(is_max, indices, n_elements)  # Use large number for non-max
    argmax_idx = tl.min(masked_indices, axis=0)
    
    # Store result
    if pid == 0:  # Only first block stores the result
        tl.store(output_ptr, block_start + argmax_idx)


@triton.jit
def argmin_kernel(input_ptr, output_ptr, n_elements, stride_input, stride_output, BLOCK_SIZE: tl.constexpr):
    """Triton kernel for argmin along last dimension."""
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input values
    input_vals = tl.load(input_ptr + offsets * stride_input, mask=mask, other=float('inf'))
    
    # Find argmin within the block
    min_val = tl.min(input_vals, axis=0)
    is_min = input_vals == min_val
    
    # Find first occurrence of min value
    indices = tl.arange(0, BLOCK_SIZE)
    masked_indices = tl.where(is_min, indices, n_elements)  # Use large number for non-min
    argmin_idx = tl.min(masked_indices, axis=0)
    
    # Store result
    if pid == 0:  # Only first block stores the result
        tl.store(output_ptr, block_start + argmin_idx)


def argmax(x, dim=None, keepdim=False):
    """Return indices of maximum values along dimension."""
    if not x.is_contiguous():
        x = x.contiguous()
    
    if dim is None:
        # Flatten and find global argmax
        flat_x = x.reshape((-1,))
        output = CUDAStorage((), dtype="int64")
        n_elements = flat_x.size
        
        grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]), )
        argmax_kernel[grid](flat_x, output, n_elements, 1, 1, BLOCK_SIZE=1024)
        
        if keepdim:
            # Reshape to maintain original shape with 1s
            output = output.reshape((1,) * len(x.shape))
        return output
    else:
        raise NotImplementedError("argmax with specific dimension is not yet implemented for CUDA backend")


def argmin(x, dim=None, keepdim=False):
    """Return indices of minimum values along dimension."""
    if not x.is_contiguous():
        x = x.contiguous()
    
    if dim is None:
        # Flatten and find global argmin
        flat_x = x.reshape((-1,))
        output = CUDAStorage((), dtype="int64")
        n_elements = flat_x.size
        
        grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]), )
        argmin_kernel[grid](flat_x, output, n_elements, 1, 1, BLOCK_SIZE=1024)
        
        if keepdim:
            # Reshape to maintain original shape with 1s
            output = output.reshape((1,) * len(x.shape))
        return output
    else:
        raise NotImplementedError("argmin with specific dimension is not yet implemented for CUDA backend")


@triton.jit
def isinf_kernel(input_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    """Triton kernel to check if elements are infinite"""
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input values
    x = tl.load(input_ptr + offsets, mask=mask)
    
    # Check for infinity: x > finite_max OR x < -finite_max
    finite_max = 3.4028235e+38  # Maximum finite float32 value
    is_pos_inf = x > finite_max
    is_neg_inf = x < -finite_max
    result = is_pos_inf | is_neg_inf
    
    # Store result as boolean (int8)
    tl.store(output_ptr + offsets, result.to(tl.int8), mask=mask)


@triton.jit
def isnan_kernel(input_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    """Triton kernel to check if elements are NaN"""
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input values
    x = tl.load(input_ptr + offsets, mask=mask)
    
    # Check for NaN: x != x (NaN property)
    result = x != x
    
    # Store result as boolean (int8)
    tl.store(output_ptr + offsets, result.to(tl.int8), mask=mask)


@triton.jit
def isfinite_kernel(input_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    """Triton kernel to check if elements are finite"""
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input values
    x = tl.load(input_ptr + offsets, mask=mask)
    
    # Check finite: not infinity and not NaN
    finite_max = 3.4028235e+38  # Maximum finite float32 value
    is_inf = (x > finite_max) | (x < -finite_max)
    is_nan = x != x
    result = ~(is_inf | is_nan)
    
    # Store result as boolean (int8)
    tl.store(output_ptr + offsets, result.to(tl.int8), mask=mask)


@register_cuda("isinf")
def isinf(x):
    """Tests each element to see if it is infinite"""
    output = CUDAStorage(x.shape, np.bool_)
    if not x.is_contiguous():
        x = x.contiguous()
    
    n_elements = x.size
    
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]), )
    isinf_kernel[grid](x, output, n_elements, BLOCK_SIZE=1024)
    
    return output


@register_cuda("isnan")
def isnan(x):
    """Tests each element to see if it is NaN"""
    output = CUDAStorage(x.shape, np.bool_)
    if not x.is_contiguous():
        x = x.contiguous()
    
    n_elements = x.size
    
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]), )
    isnan_kernel[grid](x, output, n_elements, BLOCK_SIZE=1024)
    
    return output


@register_cuda("isfinite")
def isfinite(x):
    """Tests each element to see if it is finite"""
    output = CUDAStorage(x.shape, np.bool_)
    if not x.is_contiguous():
        x = x.contiguous()
    
    n_elements = x.size
    
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]), )
    isfinite_kernel[grid](x, output, n_elements, BLOCK_SIZE=1024)
    
    return output


@triton.jit
def neg_kernel(x_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    """Negation kernel."""
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    x = tl.load(x_ptr + offsets, mask=mask)
    result = -x
    tl.store(output_ptr + offsets, result, mask=mask)


@register_cuda("neg")
def neg(x):
    """Negate tensor elements."""
    output = CUDAStorage(x.shape, x.dtype)

    # Use strided unary kernel for 1-4D, fallback to contiguous for >4D
    n_elements = x.size
    ndim = len(x.shape)

    if ndim <= 4:
        # Optimized strided path for 1-4D tensors
        # Pad shapes and strides to 4D
        x_shape = list(x.shape) + [1] * (4 - ndim)
        x_strides = list(x.strides) + [1] * (4 - ndim)
        out_shape = list(output.shape) + [1] * (4 - ndim)
        out_strides = list(output.strides) + [1] * (4 - ndim)

        grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

        strided_unary_kernel[grid](
            x, x_shape[0], x_strides[0], x_shape[1], x_strides[1], x_shape[2], x_strides[2], x_shape[3], x_strides[3],
            output, out_shape[0], out_strides[0], out_shape[1], out_strides[1], out_shape[2], out_strides[2], out_shape[3], out_strides[3],
            n_elements, ndim, op_type=4,  # for neg
            BLOCK_SIZE=1024
        )
    else:
        # Fallback for >4D: use contiguous tensors
        x_cont = x.contiguous() if not x.is_contiguous() else x
        grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
        neg_kernel[grid](x_cont, output, n_elements, BLOCK_SIZE=1024)

    return output


@triton.jit
def count_nonzero_kernel(x_ptr, count_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    """Count number of nonzero elements."""
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    nonzero_count = tl.sum((x != 0).to(tl.int32))

    # Atomic add to global counter
    if nonzero_count > 0:
        tl.atomic_add(count_ptr, nonzero_count)


@triton.jit
def fill_nonzero_1d_kernel(x_ptr, indices_ptr, counter_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    """Fill indices for 1D tensor.

    Note: This kernel processes elements sequentially within each block
    using a loop over individual elements to handle atomic operations.
    """
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE

    # Process each element in the block sequentially
    # This is necessary because we need atomic operations per element
    for i in range(BLOCK_SIZE):
        idx = block_start + i
        if idx < n_elements:
            val = tl.load(x_ptr + idx)
            if val != 0:
                # Atomically get output position and store index
                out_pos = tl.atomic_add(counter_ptr, 1)
                tl.store(indices_ptr + out_pos, idx.to(tl.int64))


@triton.jit
def flat_to_multi_index_kernel(
    flat_indices_ptr, output_ptr, shape_ptr,
    num_nonzero, ndim,
    BLOCK_SIZE: tl.constexpr
):
    """Convert flat indices to multi-dimensional coordinates.

    Args:
        flat_indices_ptr: Input flat indices (num_nonzero,)
        output_ptr: Output multi-dimensional indices (num_nonzero, ndim)
        shape_ptr: Shape of the original tensor (ndim,)
        num_nonzero: Number of nonzero elements
        ndim: Number of dimensions
    """
    pid = tl.program_id(0)
    idx = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = idx < num_nonzero

    # Load flat indices
    flat_idx = tl.load(flat_indices_ptr + idx, mask=mask, other=0)

    # Convert to multi-dimensional index
    # Process each dimension from last to first
    for dim in range(ndim - 1, -1, -1):
        dim_size = tl.load(shape_ptr + dim)
        coord = flat_idx % dim_size
        flat_idx = flat_idx // dim_size

        # Store coordinate
        output_offset = idx * ndim + dim
        tl.store(output_ptr + output_offset, coord, mask=mask)


@register_cuda("nonzero")
def nonzero(x, as_tuple=False):
    """
    Returns indices of nonzero elements using Triton kernels.

    Args:
        x: Input CUDAStorage
        as_tuple: If True, return tuple of 1D tensors. If False, return 2D tensor.

    Returns:
        Indices of nonzero elements
    """
    # Ensure contiguous
    if not x.is_contiguous():
        x = x.contiguous()

    n_elements = x.size
    ndim = len(x.shape)

    # Step 1: Count nonzero elements using atomic counter
    count = CUDAStorage((1,), dtype='int32')
    count.fill(0)  # Initialize to 0

    BLOCK_SIZE = 1024
    grid = lambda meta: (triton.cdiv(n_elements, BLOCK_SIZE),)
    count_nonzero_kernel[grid](x, count, n_elements, BLOCK_SIZE=BLOCK_SIZE)

    # Get count
    num_nonzero = int(count.to_numpy()[0])

    if num_nonzero == 0:
        # No nonzero elements
        if as_tuple:
            return tuple(CUDAStorage((0,), dtype='int64') for _ in range(ndim))
        else:
            return CUDAStorage((0, ndim), dtype='int64')

    # Step 2: Fill flat indices
    flat_indices = CUDAStorage((num_nonzero,), dtype='int64')
    count.fill(0)  # Reset counter

    fill_nonzero_1d_kernel[grid](x, flat_indices, count, n_elements, BLOCK_SIZE=BLOCK_SIZE)

    # Step 3: Convert flat indices to multi-dimensional if needed
    if ndim == 1:
        # 1D case: flat indices are the result
        if as_tuple:
            return (flat_indices,)
        else:
            # Reshape to (num_nonzero, 1)
            return flat_indices.view(num_nonzero, 1)
    else:
        # Multi-dimensional: convert flat indices to coordinates using Triton kernel
        # Create shape storage on GPU
        shape_storage = CUDAStorage((ndim,), dtype='int64')
        for i, dim_size in enumerate(x.shape):
            # Fill shape information
            temp = CUDAStorage((1,), dtype='int64')
            temp.fill(dim_size)
            # Copy to shape_storage[i] - need a kernel for this
            # TODO: Implement efficient way to set individual elements

        # Create output storage for multi-dimensional indices
        indices_2d = CUDAStorage((num_nonzero, ndim), dtype='int64')

        # Launch kernel to convert flat to multi-dimensional
        BLOCK_SIZE_CONVERT = 256
        grid_convert = lambda meta: (triton.cdiv(num_nonzero, BLOCK_SIZE_CONVERT),)
        flat_to_multi_index_kernel[grid_convert](
            flat_indices, indices_2d, shape_storage,
            num_nonzero, ndim,
            BLOCK_SIZE=BLOCK_SIZE_CONVERT
        )

        if as_tuple:
            # Extract each dimension column from indices_2d
            result = []
            for dim in range(ndim):
                # Create storage for this dimension
                dim_indices = CUDAStorage((num_nonzero,), dtype='int64')
                # Extract column - need strided copy kernel
                # TODO: Implement column extraction
                result.append(dim_indices)
            return tuple(result)
        else:
            return indices_2d


