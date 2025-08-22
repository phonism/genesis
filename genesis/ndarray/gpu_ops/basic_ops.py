"""
Basic arithmetic operations for GPU backend.
"""
import numpy as np
import triton
import triton.language as tl
from ..cuda_storage import CUDAStorage
from .utils import broadcast_shapes


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
    output = x + scalar
    tl.store(output_ptr + offsets, output, mask=mask)


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
        
        # Now both tensors have the same shape - use optimized kernel
        output = CUDAStorage(broadcast_shape, dtype=x.dtype)
        
        if not x.is_contiguous():
            x = x.contiguous()
        if not y.is_contiguous():
            y = y.contiguous()
        
        n_elements = output.size
        
        # Use fixed block size add_kernel
        grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]), )
        add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)
        
        return output
            
    else:
        # Scalar addition
        output = CUDAStorage(x.shape, dtype=x.dtype)
        if not x.is_contiguous():
            x = x.contiguous()
        
        n_elements = output.size
        
        # Use scalar add_kernel
        grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]), )
        add_scalar_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)
    
    return output


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
        
        # Now both tensors have the same shape - use simple sub kernel
        output = CUDAStorage(broadcast_shape, dtype=x.dtype)
        
        if not x.is_contiguous():
            x = x.contiguous()
        if not y.is_contiguous():
            y = y.contiguous()
        
        n_elements = output.size
        
        # Use sub kernel
        grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]), )
        sub_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)
        
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
        
        # Now both tensors have the same shape - use simple mul kernel
        output = CUDAStorage(broadcast_shape, dtype=x.dtype)
        
        if not x.is_contiguous():
            x = x.contiguous()
        if not y.is_contiguous():
            y = y.contiguous()
        
        n_elements = output.size
        
        # Use mul kernel
        grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]), )
        mul_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)
        
        return output
    else:
        # Scalar multiplication 
        output = CUDAStorage(x.shape, dtype=x.dtype)
        if not x.is_contiguous():
            x = x.contiguous()
        
        n_elements = output.size
        
        # Use scalar mul kernel
        # Ensure scalar type compatibility
        if x.dtype == "float32":
            y = float(y)
        elif x.dtype == "float16" or x.dtype == "bfloat16":
            y = float(y)
        elif x.dtype in ["int32", "int64"]:
            y = int(y)
        
        grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]), )
        mul_scalar_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)
    
    return output


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
        
        # Now both tensors have the same shape
        output = CUDAStorage(broadcast_shape, dtype=x.dtype)
        
        if not x.is_contiguous():
            x = x.contiguous()
        if not y.is_contiguous():
            y = y.contiguous()
        
        n_elements = output.size
        
        # Use div kernel
        grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]), )
        div_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)
        
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


def rtruediv(x, y):
    """
    Reverse true division (scalar / tensor).
    """
    output = CUDAStorage(x.shape, dtype=x.dtype)
    if not x.is_contiguous():
        x = x.contiguous()
    
    n_elements = output.size
    
    # Use scalar rdiv kernel  
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]), )
    rdiv_scalar_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)

    return output


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
            from ..cuda_storage import CUDAStorage
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
            scaled = mul_scalar_kernel_wrapper(tensor, log_base)
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
        scaled = mul_scalar_kernel_wrapper(log_x, scalar)
        return exp(scaled)


def log(x):
    """
    Element-wise natural logarithm.
    """
    output = CUDAStorage(x.shape, dtype=x.dtype)
    if not x.is_contiguous():
        x = x.contiguous()
    
    n_elements = x.size
    
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]), )
    log_kernel[grid](x, output, n_elements, BLOCK_SIZE=1024)
    
    return output


def exp(x):
    """
    Element-wise exponential.
    """
    output = CUDAStorage(x.shape, dtype=x.dtype)
    if not x.is_contiguous():
        x = x.contiguous()
    
    n_elements = x.size
    
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]), )
    exp_kernel[grid](x, output, n_elements, BLOCK_SIZE=1024)
    
    return output


def sin(x):
    """
    Element-wise sine.
    """
    output = CUDAStorage(x.shape, dtype=x.dtype)
    if not x.is_contiguous():
        x = x.contiguous()
    
    n_elements = x.size
    
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]), )
    sin_kernel[grid](x, output, n_elements, BLOCK_SIZE=1024)
    
    return output


def cos(x):
    """
    Element-wise cosine.
    """
    output = CUDAStorage(x.shape, dtype=x.dtype)
    if not x.is_contiguous():
        x = x.contiguous()
    
    n_elements = x.size
    
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]), )
    cos_kernel[grid](x, output, n_elements, BLOCK_SIZE=1024)
    
    return output


def sqrt(x):
    """
    Element-wise square root.
    """
    output = CUDAStorage(x.shape, dtype=x.dtype)
    if not x.is_contiguous():
        x = x.contiguous()
    
    n_elements = x.size
    
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]), )
    sqrt_kernel[grid](x, output, n_elements, BLOCK_SIZE=1024)
    
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


def abs(x):
    """
    Element-wise absolute value.
    """
    output = CUDAStorage(x.shape, dtype=x.dtype)
    if not x.is_contiguous():
        x = x.contiguous()
    
    n_elements = x.size
    
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]), )
    abs_kernel[grid](x, output, n_elements, BLOCK_SIZE=1024)
    
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


def sign(x):
    """
    Element-wise sign function.
    """
    output = CUDAStorage(x.shape, dtype=x.dtype)
    if not x.is_contiguous():
        x = x.contiguous()
    
    n_elements = x.size
    
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]), )
    sign_kernel[grid](x, output, n_elements, BLOCK_SIZE=1024)
    
    return output


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


def clamp(x, min_val=None, max_val=None):
    """
    Element-wise clamp/clip operation.
    """
    output = CUDAStorage(x.shape, dtype=x.dtype)
    if not x.is_contiguous():
        x = x.contiguous()
    
    n_elements = x.size
    
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
        if not x.is_contiguous():
            x = x.contiguous()
        
        n_elements = output.size
        
        grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]), )
        maximum_scalar_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)
    
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

def arange(start, end, step, dtype="float32"):
    """
    Alias for arange_gpu for consistency with CPU backend.
    """
    return arange_gpu(start, end, step, dtype)


def one_hot(n_classes, indices, dtype="float32"):
    """
    GPU implementation of one-hot encoding using Triton kernel.
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
    """Triton kernel for element-wise where operation."""
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


def where(condition, x, y):
    """Element-wise selection of values from x or y based on condition."""
    # Ensure inputs are contiguous
    if not condition.is_contiguous():
        condition = condition.contiguous()
    if not x.is_contiguous():
        x = x.contiguous()
    if not y.is_contiguous():
        y = y.contiguous()
    
    # All inputs should have the same shape
    if condition.shape != x.shape or condition.shape != y.shape:
        raise ValueError(f"Shape mismatch: condition {condition.shape}, x {x.shape}, y {y.shape}")
    
    n_elements = condition.size
    output = CUDAStorage(condition.shape, dtype=x.dtype)
    
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]), )
    where_kernel[grid](condition, x, y, output, n_elements, BLOCK_SIZE=1024)
    
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
        # Use PyTorch's argmax for specific dimensions (simpler for now)
        torch_result = torch.argmax(x.to_torch(), dim=dim, keepdim=keepdim)
        return CUDAStorage.from_torch(torch_result, dtype="int64")


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
        # Use PyTorch's argmin for specific dimensions (simpler for now)
        torch_result = torch.argmin(x.to_torch(), dim=dim, keepdim=keepdim)
        return CUDAStorage.from_torch(torch_result, dtype="int64")