"""
Basic arithmetic operations for GPU backend.
"""
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
            import numpy as np
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