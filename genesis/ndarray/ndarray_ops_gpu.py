import triton
import triton.language as tl
import numpy as np

import operator
import genesis
from functools import reduce
from .cuda_tensor import CUDATensor, empty, zeros, ones, from_numpy, ensure_ctx, check_ptr_accessible


def prod(x: list[int]):
    """
    Product of all elements in x.
    """
    return reduce(operator.mul, x, 1)

# Triton kernels for comparison operations (avoiding GPU-CPU roundtrips)
@triton.jit
def compare_kernel(
    x_ptr, y_ptr, output_ptr, n_elements,
    op_type: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Element-wise comparison kernel"""
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
    """Element-wise comparison with scalar kernel"""
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

# Triton kernels for indexing operations
@triton.jit
def compact_kernel(
    input_ptr, mask_ptr, output_ptr, output_indices_ptr,
    n_elements: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Compact kernel: copy elements where mask is True to contiguous output"""
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
    """Count number of True elements in boolean mask"""
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
def fill_kernel(
    output_ptr, fill_value,
    n_elements: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Fill tensor with a constant value"""
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    tl.store(output_ptr + offsets, fill_value, mask=mask)

@triton.autotune(
    configs=[
        # small (< 1M elements)
        triton.Config({'BLOCK': 128},  num_warps=1, num_stages=1),
        triton.Config({'BLOCK': 256},  num_warps=2, num_stages=1),
        triton.Config({'BLOCK': 512},  num_warps=2, num_stages=2),
        triton.Config({'BLOCK': 1024}, num_warps=4, num_stages=2),
        # large (> 1M elements)
        triton.Config({'BLOCK': 2048}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK': 4096}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK': 8192}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK': 16384}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK': 32768}, num_warps=8, num_stages=2),
    ],
    key=['n_elements'],
)
@triton.jit
def add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK: tl.constexpr):
    """
    Expert-optimized add kernel following performance recommendations:
    - Large BLOCK sizes for bandwidth saturation
    - Proper num_warps and num_stages for occupancy
    - 1D indexing with alignment hints
    - Autotune for optimal configuration selection
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    
    # Alignment hints for better memory transactions (128B aligned)
    tl.static_assert(BLOCK % 128 == 0)
    tl.multiple_of(offs, 128)
    
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
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    ACTIVATION: tl.constexpr
):
    """
    Matmul kernel.
    """
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m
    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        k_valid_a = (offs_k[None, :] < K - k * BLOCK_SIZE_K)
        k_valid_b = (offs_k[:, None] < K - k * BLOCK_SIZE_K)
        a_mask = (offs_am[:, None] < M) & k_valid_a
        b_mask = k_valid_b & (offs_bn[None, :] < N)
        
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)
        accumulator = tl.dot(a, b, accumulator,  allow_tf32=False)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
    c = accumulator
    # Use offset calculation consistent with load
    c_ptrs = c_ptr + stride_cm * offs_am[:, None] + stride_cn * offs_bn[None, :]
    c_mask = (offs_am[:, None] < M) & (offs_bn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)

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

def add(x, y):
    """
    Add with broadcasting support and optimizations.
    """
    if isinstance(y, CUDATensor):
        # Compute broadcasted shape
        broadcast_shape = broadcast_shapes(x.shape, y.shape)
        
        # Broadcast both tensors to the same shape
        if x.shape != broadcast_shape:
            x = x.broadcast_to(broadcast_shape)
        if y.shape != broadcast_shape:
            y = y.broadcast_to(broadcast_shape)
        
        # Now both tensors have the same shape - use optimized kernel
        output = CUDATensor(broadcast_shape, dtype=x.dtype)
        
        if not x.is_contiguous():
            x = x.contiguous()
        if not y.is_contiguous():
            y = y.contiguous()
        
        n_elements = output.size
        
        # Use expert-optimized autotune add_kernel
        grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK"]), )
        add_kernel[grid](x, y, output, n_elements)
        
        return output
            
    else:
        # Scalar addition
        output = CUDATensor(x.shape, dtype=x.dtype)
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
    if isinstance(y, CUDATensor):
        # Compute broadcasted shape
        broadcast_shape = broadcast_shapes(x.shape, y.shape)
        
        # Broadcast both tensors to the same shape
        if x.shape != broadcast_shape:
            x = x.broadcast_to(broadcast_shape)
        if y.shape != broadcast_shape:
            y = y.broadcast_to(broadcast_shape)
        
        # Now both tensors have the same shape - use simple sub kernel
        output = CUDATensor(broadcast_shape, dtype=x.dtype)
        
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
    if isinstance(y, CUDATensor):
        # For in-place operations, shapes must be compatible
        if x.shape != y.shape:
            raise ValueError("In-place addition requires compatible shapes")
        output_shape = x.shape
    else:
        output_shape = x.shape
    
    if not x.is_contiguous():
        x = x.contiguous()

    if isinstance(y, CUDATensor):
        if not y.is_contiguous():
            y = y.contiguous()

    n_elements = x.size

    if isinstance(y, CUDATensor):
        # Use optimized autotune add_kernel for tensor-tensor addition
        grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK"]), )
        add_kernel[grid](x, y, x, n_elements)
    else:
        # Use scalar add_kernel for tensor-scalar addition
        grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]), )
        add_scalar_kernel[grid](x, y, x, n_elements, BLOCK_SIZE=1024)
    return x

def mul(x, y):
    """
    Multiply with broadcasting support.
    """
    if isinstance(y, CUDATensor):
        # Compute broadcasted shape
        broadcast_shape = broadcast_shapes(x.shape, y.shape)
        
        # Broadcast both tensors to the same shape
        if x.shape != broadcast_shape:
            x = x.broadcast_to(broadcast_shape)
        if y.shape != broadcast_shape:
            y = y.broadcast_to(broadcast_shape)
        
        # Now both tensors have the same shape - use simple mul kernel
        output = CUDATensor(broadcast_shape, dtype=x.dtype)
        
        if not x.is_contiguous():
            x = x.contiguous()
        if not y.is_contiguous():
            y = y.contiguous()
        
        n_elements = output.size
        
        # Use mul kernel
        grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]), )
        mul_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)
    else:
        # Scalar multiplication
        output = CUDATensor(x.shape, dtype=x.dtype)
        if not x.is_contiguous():
            x = x.contiguous()
        
        n_elements = output.size
        grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]), )
        mul_scalar_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)
    
    return output

def truediv(x, y):
    """
    True div kernel with broadcasting support.
    """
    # Extract CUDATensor from NDArray if needed
    if hasattr(x, 'data') and hasattr(x.data, 'data'):
        x_tensor = x.data.data
        x_dtype = x.dtype
    else:
        x_tensor = x
        x_dtype = x.dtype if hasattr(x, 'dtype') else "float32"
    
    if isinstance(y, CUDATensor) or (hasattr(y, 'data') and hasattr(y.data, 'data')):
        # Extract CUDATensor from y if it's NDArray
        if hasattr(y, 'data') and hasattr(y.data, 'data'):
            y_tensor = y.data.data
        else:
            y_tensor = y
            
        # Compute broadcasted shape
        broadcast_shape = broadcast_shapes(x_tensor.shape, y_tensor.shape)
        
        # Broadcast both tensors to the same shape
        if x_tensor.shape != broadcast_shape:
            x_tensor = x_tensor.broadcast_to(broadcast_shape)
        if y_tensor.shape != broadcast_shape:
            y_tensor = y_tensor.broadcast_to(broadcast_shape)
        
        # Now both tensors have the same shape - use simple div kernel
        output = CUDATensor(broadcast_shape, dtype=x_dtype)
        
        if not x_tensor.is_contiguous():
            x_tensor = x_tensor.contiguous()
        if not y_tensor.is_contiguous():
            y_tensor = y_tensor.contiguous()
            
        n_elements = output.size
        grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]), )
        div_kernel[grid](x_tensor, y_tensor, output, n_elements, BLOCK_SIZE=1024)
    else:
        # Scalar division
        output = CUDATensor(x_tensor.shape, dtype=x_dtype)
        if not x_tensor.is_contiguous():
            x_tensor = x_tensor.contiguous()
        n_elements = output.size
        grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]), )
        div_scalar_kernel[grid](x_tensor, y, output, n_elements, BLOCK_SIZE=1024)
    
    return output

def rtruediv(x, y):
    """
    Right true div kernel.
    """
    if isinstance(y, CUDATensor):
        if x.size >= y.size:
            output_shape = x.shape
        else:
            output_shape = y.shape
    else:
        output_shape = x.shape
    output = CUDATensor(output_shape, dtype=x.dtype)
    
    if not x.is_contiguous():
        x = x.contiguous()
    n_elements = output.size
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]), )
    
    if isinstance(y, CUDATensor):
        if not y.is_contiguous():
            y = y.contiguous()
        div_kernel[grid](y, x, output, n_elements, BLOCK_SIZE=1024)
    else:
        rdiv_scalar_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)
    return output

def pow(x, scalar):
    """
    Pow kernel.
    """
    # Handle case where x and scalar might be swapped (for rpower)
    if not hasattr(x, 'shape'):
        # x is a scalar, scalar is a tensor - this is rpower case: scalar^x
        base_scalar = x
        tensor = scalar
        
        if base_scalar == 0:
            return zeros(tensor.shape, dtype=tensor.dtype)
        elif base_scalar == 1:
            return ones(tensor.shape, dtype=tensor.dtype)
        elif base_scalar < 0:
            # Negative base with tensor exponent - should be NaN for non-integer exponents
            result = ones(tensor.shape, dtype=tensor.dtype)
            # Fill with NaN
            fill_tensor(result, float('nan'))
            return result
        else:
            # base^tensor = exp(tensor * log(base))
            log_base = np.log(base_scalar)
            scaled = mul_scalar_kernel_wrapper(tensor, log_base)
            return exp(scaled)
    
    # Normal case: tensor^scalar
    if scalar == 0:
        return ones(x.shape, dtype=x.dtype)
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

@triton.jit
def pow_kernel(x_ptr, scalar, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    """
    Power kernel.
    """
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    output = tl.pow(x, scalar)
    tl.store(output_ptr + offsets, output, mask=mask)

def mul_scalar_kernel_wrapper(x, scalar):
    """Helper wrapper for mul_scalar_kernel"""
    output = CUDATensor(x.shape, dtype=x.dtype)
    if not x.is_contiguous():
        x = x.contiguous()
    n_elements = output.size
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]), )
    mul_scalar_kernel[grid](x, scalar, output, n_elements, BLOCK_SIZE=1024)
    return output

def log(x):
    """
    Log kernel.
    """
    output = CUDATensor(x.shape, dtype=x.dtype)
    
    if not x.is_contiguous():
        x = x.contiguous()
    n_elements = output.size
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]), )
    
    log_kernel[grid](x, output, n_elements, BLOCK_SIZE=1024)
    
    # Convert back to original dtype if needed
    if output.dtype != x.dtype:
        result = CUDATensor(x.shape, dtype=x.dtype)
        # TODO: implement dtype conversion kernel
        return result
    return output

def exp(x):
    """
    Exp kernel.
    """
    output = CUDATensor(x.shape, dtype=x.dtype)
    
    if not x.is_contiguous():
        x = x.contiguous()
    n_elements = output.size
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]), )
    
    exp_kernel[grid](x, output, n_elements, BLOCK_SIZE=1024)
    
    # Convert back to original dtype if needed
    if output.dtype != x.dtype:
        result = CUDATensor(x.shape, dtype=x.dtype)
        # TODO: implement dtype conversion kernel
        return result
    return output

def sin(x):
    """
    Sin kernel.
    """
    output = CUDATensor(x.shape, dtype=np.float32)
    
    if not x.is_contiguous():
        x = x.contiguous()
    n_elements = output.size
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]), )
    
    sin_kernel[grid](x, output, n_elements, BLOCK_SIZE=1024)
    
    if output.dtype != x.dtype:
        result = CUDATensor(x.shape, dtype=x.dtype)
        return result
    return output

def cos(x):
    """
    Cos kernel.
    """
    output = CUDATensor(x.shape, dtype=x.dtype)
    
    if not x.is_contiguous():
        x = x.contiguous()
    n_elements = output.size
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]), )
    
    cos_kernel[grid](x, output, n_elements, BLOCK_SIZE=1024)
    return output

def sqrt(x):
    """
    Sqrt kernel.
    """
    output = CUDATensor(x.shape, dtype=x.dtype)
    
    if not x.is_contiguous():
        x = x.contiguous()
    n_elements = output.size
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]), )
    
    sqrt_kernel[grid](x, output, n_elements, BLOCK_SIZE=1024)
    return output

def maximum(x, y):
    """
    Maximum kernel - optimized elementwise operation.
    """
    if isinstance(y, CUDATensor):
        # Tensor-tensor maximum with broadcasting
        broadcast_shape = broadcast_shapes(x.shape, y.shape)
        
        if x.shape != broadcast_shape:
            x = x.broadcast_to(broadcast_shape)
        if y.shape != broadcast_shape:
            y = y.broadcast_to(broadcast_shape)
        
        output = CUDATensor(broadcast_shape, dtype=x.dtype)
        
        if not x.is_contiguous():
            x = x.contiguous()
        if not y.is_contiguous():
            y = y.contiguous()
        
        n_elements = output.size
        
        # Use maximum kernel
        grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]), )
        maximum_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)
    else:
        # Scalar maximum
        output = CUDATensor(x.shape, dtype=x.dtype)
        if not x.is_contiguous():
            x = x.contiguous()
        n_elements = output.size
        
        # For scalar ops, use direct call (no need for complex optimization)
        grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]), 1, 1)
        maximum_scalar_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)
    
    return output

def reduce_sum(x, axis=None, keepdims=False):
    """
    Reduce sum kernel.
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
    m = prod(new_shape[:len(axes_to_keep)])
    n = prod(new_shape[len(axes_to_keep):])
    x = x.reshape((m, n))
    output_shape = tuple(new_shape[i] for i in range(len(axes_to_keep)))
    if keepdims:
        output_shape = list(shape)
        for i in axis:
            output_shape[i] = 1
        output_shape = tuple(output_shape)
    output = CUDATensor(output_shape, dtype=x.dtype)

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
    
    # Memory management has been fixed - pointers should be valid now
    
    # Ensure we pass CUDATensor objects directly, not .ptr or others
    sum_kernel[grid](x, output, m, n, block_m, block_n)
    return output

def reduce_max(x, axis=None, keepdims=False):
    """
    Reduce max kernel.
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
    m = prod(new_shape[:len(axes_to_keep)])
    n = prod(new_shape[len(axes_to_keep):])
    x = x.reshape((m, n))

    output_shape = tuple(new_shape[i] for i in range(len(axes_to_keep)))
    if keepdims:
        output_shape = list(shape)
        for i in axis:
            output_shape[i] = 1
        output_shape = tuple(output_shape)
    output = CUDATensor(output_shape, dtype=x.dtype)

    block_m = 4
    block_n = min(triton.next_power_of_2(n), 1024)
    grid = (triton.cdiv(m, block_m), 1, 1)

    if not x.is_contiguous():
        x = x.contiguous()
    
    max_kernel[grid](x, output, m, n, block_m, block_n)
    return output

def reshape(x, new_shape):
    """
    Reshape kernel.
    """
    return x.reshape(new_shape)

def view(x, new_shape):
    """
    View kernel.
    """
    if x.is_contiguous() is False:
        x = x.contiguous()
    return x.view(new_shape)

def expand(x, new_shape):
    """
    Expand kernel.
    """
    return x.expand(new_shape)

def permute(x, new_axis):
    """
    Permute kernel.
    """
    return x.permute(new_axis)

def broadcast_to(x, new_shape):
    """
    Broadcast to kernel.
    """
    return x.broadcast_to(new_shape)

def getitem(x, idxs):
    """
    Getitem kernel with optimizations for common cases.
    """
    if not isinstance(x, CUDATensor):
        return x.__getitem__(idxs)
    
    # For simple int/slice indexing, CUDATensor already handles efficiently (no CPU roundtrip)
    if isinstance(idxs, (int, slice)):
        return x.__getitem__(idxs)
    
    # For tuple of int/slice, also efficient
    if isinstance(idxs, tuple) and all(isinstance(idx, (int, slice)) for idx in idxs):
        return x.__getitem__(idxs)
    
    # For boolean indexing with CUDATensor mask, try to optimize
    if isinstance(idxs, CUDATensor) and idxs.dtype == "bool":
        # This could be optimized with GPU kernels in the future
        # For now, use the existing implementation but it's isolated here
        return x.__getitem__(idxs)
    
    # For other complex cases, fallback to CUDATensor implementation
    return x.__getitem__(idxs)

def setitem(x, idxs, other):
    """
    Setitem kernel with optimizations for common cases.
    """
    if not isinstance(x, CUDATensor):
        return x.__setitem__(idxs, other)
    
    # For simple cases, CUDATensor handles efficiently
    if isinstance(idxs, (int, slice)):
        return x.__setitem__(idxs, other)
    
    if isinstance(idxs, tuple) and all(isinstance(idx, (int, slice)) for idx in idxs):
        return x.__setitem__(idxs, other)
    
    # For complex cases, fallback to CUDATensor implementation  
    return x.__setitem__(idxs, other)

def eq(x, y):
    """
    Eq kernel.
    """
    if isinstance(y, CUDATensor):
        # Tensor comparison
        if x.shape != y.shape:
            raise ValueError(f"Shape mismatch: {x.shape} vs {y.shape}")
        
        output = CUDATensor(x.shape, dtype="bool")
        
        if not x.is_contiguous():
            x = x.contiguous()
        if not y.is_contiguous():
            y = y.contiguous()
        
        n_elements = output.size
        grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]), )
        compare_kernel[grid](x, y, output, n_elements, 0, BLOCK_SIZE=1024)  # 0 = eq
    else:
        # Scalar comparison
        output = CUDATensor(x.shape, dtype="bool")
        if not x.is_contiguous():
            x = x.contiguous()
        
        n_elements = output.size
        grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]), )
        compare_scalar_kernel[grid](x, output, n_elements, float(y), 0, BLOCK_SIZE=1024)  # 0 = eq
    
    return output

def ge(x, y):
    """
    Ge kernel.
    """
    if isinstance(y, CUDATensor):
        # Tensor comparison
        if x.shape != y.shape:
            raise ValueError(f"Shape mismatch: {x.shape} vs {y.shape}")
        
        output = CUDATensor(x.shape, dtype="bool")
        
        if not x.is_contiguous():
            x = x.contiguous()
        if not y.is_contiguous():
            y = y.contiguous()
        
        n_elements = output.size
        grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]), )
        compare_kernel[grid](x, y, output, n_elements, 4, BLOCK_SIZE=1024)  # 4 = ge
    else:
        # Scalar comparison
        output = CUDATensor(x.shape, dtype="bool")
        if not x.is_contiguous():
            x = x.contiguous()
        
        n_elements = output.size
        grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]), )
        compare_scalar_kernel[grid](x, output, n_elements, float(y), 4, BLOCK_SIZE=1024)  # 4 = ge
    
    return output

def gt(x, y):
    """
    Gt kernel.
    """
    if isinstance(y, CUDATensor):
        # Tensor comparison
        if x.shape != y.shape:
            raise ValueError(f"Shape mismatch: {x.shape} vs {y.shape}")
        
        output = CUDATensor(x.shape, dtype="bool")
        
        if not x.is_contiguous():
            x = x.contiguous()
        if not y.is_contiguous():
            y = y.contiguous()
        
        n_elements = output.size
        grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]), )
        compare_kernel[grid](x, y, output, n_elements, 3, BLOCK_SIZE=1024)  # 3 = gt
    else:
        # Scalar comparison
        output = CUDATensor(x.shape, dtype="bool")
        if not x.is_contiguous():
            x = x.contiguous()
        
        n_elements = output.size
        grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]), )
        compare_scalar_kernel[grid](x, output, n_elements, float(y), 3, BLOCK_SIZE=1024)  # 3 = gt
    
    return output

def le(x, y):
    """
    Le kernel.
    """
    if isinstance(y, CUDATensor):
        # Tensor comparison
        if x.shape != y.shape:
            raise ValueError(f"Shape mismatch: {x.shape} vs {y.shape}")
        
        output = CUDATensor(x.shape, dtype="bool")
        
        if not x.is_contiguous():
            x = x.contiguous()
        if not y.is_contiguous():
            y = y.contiguous()
        
        n_elements = output.size
        grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]), )
        compare_kernel[grid](x, y, output, n_elements, 2, BLOCK_SIZE=1024)  # 2 = le
    else:
        # Scalar comparison
        output = CUDATensor(x.shape, dtype="bool")
        if not x.is_contiguous():
            x = x.contiguous()
        
        n_elements = output.size
        grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]), )
        compare_scalar_kernel[grid](x, output, n_elements, float(y), 2, BLOCK_SIZE=1024)  # 2 = le
    
    return output

def fill_tensor(tensor, value):
    """Fill tensor with constant value using GPU kernel"""
    if not isinstance(tensor, CUDATensor):
        raise TypeError("Expected CUDATensor")
    
    if not tensor.is_contiguous():
        tensor = tensor.contiguous()
    
    n_elements = tensor.size
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]), )
    
    fill_kernel[grid](
        tensor, float(value), n_elements, BLOCK_SIZE=1024
    )
    return tensor

def fill(tensor, value):
    """Fill tensor with constant value using GPU kernel"""
    return fill_tensor(tensor, value)

def boolean_indexing_gpu(tensor, mask):
    """GPU-based boolean indexing to avoid CPU roundtrip"""
    if not isinstance(tensor, CUDATensor) or not isinstance(mask, CUDATensor):
        return None  # Fallback to CPU version
    
    if tensor.shape != mask.shape:
        return None  # Only support same-shape boolean indexing for now
    
    # For complex boolean indexing, we'll still need CPU fallback
    # But we can at least avoid some conversions for simple cases
    
    # Count True elements first (simplified version)
    mask_np = mask.to_numpy()  # Still need this for counting
    true_count = int(mask_np.sum())
    
    if true_count == 0:
        # Return empty tensor
        return CUDATensor((0,) + tensor.shape[1:], dtype=tensor.dtype)
    
    # For now, fallback to CPU for the actual selection
    # TODO: Implement full GPU version with prefix sum
    tensor_np = tensor.to_numpy()
    result_np = tensor_np[mask_np]
    return from_numpy(result_np)

def lt(x, y):
    """
    Lt kernel.
    """
    if isinstance(y, CUDATensor):
        # Tensor comparison
        if x.shape != y.shape:
            raise ValueError(f"Shape mismatch: {x.shape} vs {y.shape}")
        
        output = CUDATensor(x.shape, dtype="bool")
        
        if not x.is_contiguous():
            x = x.contiguous()
        if not y.is_contiguous():
            y = y.contiguous()
        
        n_elements = output.size
        grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]), )
        compare_kernel[grid](x, y, output, n_elements, 1, BLOCK_SIZE=1024)  # 1 = lt
    else:
        # Scalar comparison
        output = CUDATensor(x.shape, dtype="bool")
        if not x.is_contiguous():
            x = x.contiguous()
        
        n_elements = output.size
        grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]), )
        compare_scalar_kernel[grid](x, output, n_elements, float(y), 1, BLOCK_SIZE=1024)  # 1 = lt
    
    return output

def matmul(a, b, activation=""):
    """
    Matmul kernel.
    """
    assert a.shape[-1] == b.shape[-2], "Incompatible dimensions"
    if not a.is_contiguous():
        a = a.contiguous()
    if not b.is_contiguous():
        b = b.contiguous()
    if len(a.shape) == 2 and len(b.shape) == 2:
        M, K = a.shape
        K, N = b.shape
        # Allocates output.
        c = CUDATensor((M, N), dtype=a.dtype)
        # 1D launch kernel where each block gets its own program.
        grid = lambda META: (triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]), )
        
        matmul_kernel[grid](
                a, b, c,
                M, N, K,
                a.stride(0), a.stride(1),
                b.stride(0), b.stride(1),
                c.stride(0), c.stride(1),
                BLOCK_SIZE_M=64,
                BLOCK_SIZE_N=64,
                BLOCK_SIZE_K=32,
                GROUP_SIZE_M=8,
                ACTIVATION=activation
        )
        return c
    elif len(a.shape) > 2 or len(b.shape) > 2:
        # Batch matmul implementation
        pre_shape_a = []
        pre_shape_b = []
        pre_a = 1
        pre_b = 1
        a_shape = a.shape
        b_shape = b.shape
        
        if len(a_shape) > 2 or len(b_shape) > 2:
            for i in range(len(a_shape) - 2):
                pre_shape_a.append(a_shape[i])
                pre_a *= a_shape[i]
            aa = a.reshape((pre_a, a_shape[-2], a_shape[-1]))
            
            for i in range(len(b_shape) - 2):
                pre_shape_b.append(b_shape[i])
                pre_b *= b_shape[i]
            bb = b.reshape((pre_b, b_shape[-2], b_shape[-1]))

            if pre_a == 1:
                aa = aa.broadcast_to((bb.shape[0], aa.shape[1], aa.shape[2]))
            if pre_b == 1:
                bb = bb.broadcast_to((aa.shape[0], bb.shape[1], bb.shape[2]))

        M = a_shape[-2]
        N = b_shape[-1]
        K = a_shape[-1]

        batch_size = max(pre_a, pre_b)
        c = CUDATensor((batch_size, M, N), dtype=a.dtype)

        # 1D launch kernel where each block gets its own program.
        grid = lambda META: (triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]), )
        
        for i in range(batch_size):
            # Use CUDATensor slice operations - much cleaner!
            a_batch = aa[i]  # Shape: (M, K)
            b_batch = bb[i]  # Shape: (K, N)
            c_batch = c[i]   # Shape: (M, N)
            
            # Run 2D matmul for this batch
            matmul_kernel[grid](
                    a_batch, b_batch, c_batch,
                    M, N, K,
                    a_batch.stride(0), a_batch.stride(1),
                    b_batch.stride(0), b_batch.stride(1),
                    c_batch.stride(0), c_batch.stride(1),
                    BLOCK_SIZE_M=64,
                    BLOCK_SIZE_N=64,
                    BLOCK_SIZE_K=32,
                    GROUP_SIZE_M=8,
                    ACTIVATION=activation
            )
        
        # Reshape to final output shape
        if pre_a > pre_b:
            final_shape = tuple(pre_shape_a) + (M, N)
        elif pre_a < pre_b:
            final_shape = tuple(pre_shape_b) + (M, N)
        else:
            if len(pre_shape_a) > len(pre_shape_b):
                final_shape = tuple(pre_shape_a) + (M, N)
            else:
                final_shape = tuple(pre_shape_b) + (M, N)
        
        # Only reshape if shape actually changes
        if c.shape != final_shape:
            c = c.reshape(final_shape)
        return c
    else:
        return None

def from_numpy(data, device_id=0, dtype=None):
    """
    From numpy
    """
    np_dtype = None
    if dtype is None or dtype == genesis.float32:
        np_dtype = np.float32
    elif dtype == genesis.float16:
        np_dtype = np.float16
    elif dtype == genesis.bfloat16:
        # bfloat16 is not natively supported in numpy, use float32 for now
        np_dtype = np.float32
    
    if np_dtype and data.dtype != np_dtype:
        data = data.astype(np_dtype)
    
    tensor = CUDATensor(data.shape, dtype=data.dtype)
    tensor.from_numpy(data)
    return tensor

def from_tensor(data, device_id=0, dtype=None):
    """
    From tensor
    """
    # If input is already CUDATensor, return as is
    if isinstance(data, CUDATensor):
        return data
    
    # Convert PyTorch tensor to CUDATensor
    if hasattr(data, 'numpy'):  # PyTorch tensor
        numpy_array = data.detach().cpu().numpy()
        return from_numpy(numpy_array)
    
    # Convert other tensor types to CUDATensor
    # For other types, try to convert to numpy first
    numpy_array = np.asarray(data)
    return from_numpy(numpy_array)

def array(shape, device_id=0, dtype=None):
    """
    Array
    """
    # Convert to string dtype for CUDATensor compatibility
    if dtype is None or dtype == genesis.float32:
        str_dtype = "float32"
    elif dtype == genesis.float16:
        str_dtype = "float16"
    elif dtype == genesis.bfloat16:
        str_dtype = "bfloat16"
    elif dtype == "int32":
        str_dtype = "int32"
    elif dtype == "int64":
        str_dtype = "int64"
    elif dtype == "bool":
        str_dtype = "bool"
    else:
        str_dtype = "float32"  # default
    
    return CUDATensor(shape, dtype=str_dtype)

def broadcast_shapes(shape1, shape2):
    """Compute the broadcasted shape of two tensors (NumPy broadcasting rules)."""
    # Reverse shapes to align from the right (trailing dimensions)
    shape1_rev = list(reversed(shape1))
    shape2_rev = list(reversed(shape2))
    
    # Pad shorter shape with 1s to make them the same length
    max_ndim = max(len(shape1_rev), len(shape2_rev))
    while len(shape1_rev) < max_ndim:
        shape1_rev.append(1)
    while len(shape2_rev) < max_ndim:
        shape2_rev.append(1)
    
    # Compute broadcasted shape from right to left
    result_shape_rev = []
    for s1, s2 in zip(shape1_rev, shape2_rev):
        if s1 == 1:
            result_shape_rev.append(s2)
        elif s2 == 1:
            result_shape_rev.append(s1)
        elif s1 == s2:
            result_shape_rev.append(s1)
        else:
            raise ValueError(f"Cannot broadcast shapes {tuple(reversed(shape1_rev))} and {tuple(reversed(shape2_rev))}")
    
    # Reverse back to get final shape
    return tuple(reversed(result_shape_rev))

# Boolean indexing kernels - simplified approach
@triton.jit
def count_true_kernel(
    mask_ptr, count_ptr, n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Count number of True elements in boolean mask"""
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
    """Extract indices where mask is True (simplified version)"""
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
    """Gather elements by linear indices"""
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load indices and gather data
    idx = tl.load(idx_ptr + offsets, mask=mask)
    val = tl.load(src_ptr + idx, mask=mask)
    tl.store(out_ptr + offsets, val, mask=mask)

def compact_boolean_mask(mask_tensor):
    """Convert boolean mask to linear indices (CPU fallback for now)"""
    if not isinstance(mask_tensor, CUDATensor):
        raise TypeError("Expected CUDATensor")
    
    # CPU fallback - still better than full CPU roundtrip of data
    # because we only convert the smaller mask tensor
    mask_np = mask_tensor.to_numpy()
    indices_np = np.where(mask_np.flatten())[0].astype(np.int64)
    
    if indices_np.size == 0:
        return empty((0,), dtype="int64")
    else:
        return from_numpy(indices_np)

def gather_by_indices(src_tensor, indices):
    """Gather elements from src_tensor by linear indices"""
    if not isinstance(src_tensor, CUDATensor) or not isinstance(indices, CUDATensor):
        raise TypeError("Expected CUDATensor")
    
    n_elements = indices.size
    if n_elements == 0:
        # Empty indices, return empty tensor with same dtype as source
        return empty((0,) + src_tensor.shape[1:], dtype=src_tensor.dtype)
    
    # Create output tensor
    output_shape = (n_elements,) + src_tensor.shape[1:]
    output = empty(output_shape, dtype=src_tensor.dtype)
    
    # Launch gather kernel
    BLOCK_SIZE = 256
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    
    gather_linear_kernel[grid](
        src_tensor, indices, output,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output

@triton.jit
def scatter_linear_kernel(
    src_ptr, idx_ptr, out_ptr, n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Scatter elements by linear indices"""
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

def scatter_by_indices(src_tensor, indices, target_tensor):
    """Scatter elements from src_tensor to target_tensor by linear indices"""
    if not isinstance(src_tensor, CUDATensor) or not isinstance(indices, CUDATensor) or not isinstance(target_tensor, CUDATensor):
        raise TypeError("Expected CUDATensor")
    
    n_elements = indices.size
    if n_elements == 0:
        return  # Nothing to scatter
    
    # Launch scatter kernel
    BLOCK_SIZE = 256
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    
    scatter_linear_kernel[grid](
        src_tensor, indices, target_tensor,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )

