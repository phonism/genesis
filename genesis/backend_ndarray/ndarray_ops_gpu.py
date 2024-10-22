import torch
import triton
import triton.language as tl

import operator
from functools import reduce

def prod(x):
    """
    prod
    """
    return reduce(operator.mul, x, 1)

def get_cuda_autotune_config():
    return [
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=3,
                      num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=5,
                      num_warps=2),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=5,
                      num_warps=2),
        # Good config for fp8 inputs.
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=3,
                      num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=3,
                      num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4)
    ]

def get_autotune_config():
    return get_cuda_autotune_config()

@triton.jit
def add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    tl.store(output_ptr + offsets, output, mask=mask)

@triton.jit
def add_scalar_kernel(x_ptr, scalar, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    output = x + scalar
    tl.store(output_ptr + offsets, output, mask=mask)

@triton.jit
def mul_scalar_kernel(x_ptr, scalar, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    output = x * scalar
    tl.store(output_ptr + offsets, output, mask=mask)

@triton.jit
def mul_kernel(x_ptr, y_ptr, output_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x * y
    tl.store(output_ptr + offsets, output, mask=mask)

@triton.jit
def div_scalar_kernel(x_ptr, scalar, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    output = x / scalar
    tl.store(output_ptr + offsets, output, mask=mask)

@triton.jit
def div_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
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
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    output = tl.maximum(x, scalar)
    tl.store(output_ptr + offsets, output, mask=mask)

@triton.jit
def maximum_kernel(x_ptr, y_ptr, output_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = tl.maximum(x, y)
    tl.store(output_ptr + offsets, output, mask=mask)

@triton.jit
def log_kernel(x_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    output = tl.log(x)
    tl.store(output_ptr + offsets, output, mask=mask)

@triton.jit
def exp_kernel(x_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    output = tl.exp(x)
    tl.store(output_ptr + offsets, output, mask=mask)

@triton.jit
def cos_kernel(x_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    output = tl.cos(x)
    tl.store(output_ptr + offsets, output, mask=mask)

@triton.jit
def sin_kernel(x_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    output = tl.sin(x)
    tl.store(output_ptr + offsets, output, mask=mask)

@triton.jit
def sqrt_kernel(x_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    output = tl.sqrt(x)
    tl.store(output_ptr + offsets, output, mask=mask)

#@triton.autotune(
        #configs=get_autotune_config(),
        #key=['M', 'N', 'K'],
#)
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
        ACTIVATION: tl.constexpr):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        accumulator = tl.dot(a, b, accumulator,  allow_tf32=False)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
    c = accumulator
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)

@triton.jit
def sum_kernel(x_ptr, output_ptr, M, N, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    """
    pid_m = tl.program_id(axis=0)
    m_offset = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offset = tl.arange(0, BLOCK_N)
    offset = m_offset[:, None] * N + n_offset[None, :]
    m_mask = m_offset < M
    n_mask = n_offset < N
    mask = m_mask[:, None] & n_mask[None, :]
    inp = tl.load(x_ptr + offset, mask=mask, other=0)
    out = tl.sum(inp, axis=1)
    tl.store(output_ptr + m_offset, out, mask=m_mask)
    """
    pid_m = tl.program_id(axis=0)
    m_offset = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    m_mask = m_offset < M  # 计算行掩码
    out = tl.zeros((BLOCK_M,), dtype=tl.float32) 
    # 分块处理列数据
    for start in range(0, N, BLOCK_N):
        n_offset = start + tl.arange(0, BLOCK_N)
        offset = m_offset[:, None] * N + n_offset[None, :]
        n_mask = n_offset < N  # 计算列掩码
        mask = m_mask[:, None] & n_mask[None, :]  # 组合掩码
        inp = tl.load(x_ptr + offset, mask=mask, other=0)
        out += tl.sum(inp, axis=1)

    tl.store(output_ptr + m_offset, out, mask=m_mask)

@triton.jit
def max_kernel(x_ptr, output_ptr, M, N, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    pid_m = tl.program_id(axis=0)
    m_offset = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    m_mask = m_offset < M  # 行掩码
    
    # 初始化输出为最小的可能值
    out = tl.full((BLOCK_M,), -float('inf'), dtype=tl.float32)

    for start in range(0, N, BLOCK_N):
        n_offset = start + tl.arange(0, BLOCK_N)
        offset = m_offset[:, None] * N + n_offset[None, :]
        n_mask = n_offset < N  # 列掩码
        mask = m_mask[:, None] & n_mask[None, :]  # 组合掩码
        inp = tl.load(x_ptr + offset, mask=mask, other=-float('inf'))
        out = tl.maximum(out, tl.max(inp, axis=1))
                                                                                        
    tl.store(output_ptr + m_offset, out, mask=m_mask)


def add(x, y):
    output = torch.empty_like(x, device=torch.device("cuda"), dtype=x.dtype)
    assert x.is_cuda and output.is_cuda
    if x.is_contiguous() is False:
        x = x.contiguous()
    n_elements = output.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]), )
    if isinstance(y, torch.Tensor):
        assert y.is_cuda
        if y.is_contiguous() is False:
            y = y.contiguous()
        add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)
    else:
        add_scalar_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)
    return output

def mul(x, y):
    output = torch.empty_like(x, device=torch.device("cuda"), dtype=x.dtype)
    assert x.is_cuda and output.is_cuda
    if x.is_contiguous() is False:
        x = x.contiguous()
    n_elements = output.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]), )
    if isinstance(y, torch.Tensor):
        assert y.is_cuda
        if y.is_contiguous() is False:
            y = y.contiguous()
        mul_kernel[grid](x, y, output.data, n_elements, BLOCK_SIZE=1024)
    else:
        mul_scalar_kernel[grid](x, y, output.data, n_elements, BLOCK_SIZE=1024)
    return output

def truediv(x, y):
    output = torch.empty_like(x, dtype=x.dtype)
    assert x.is_cuda and output.is_cuda
    if x.is_contiguous() is False:
        x = x.contiguous()
    n_elements = output.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]), )
    if isinstance(y, torch.Tensor):
        if y.is_contiguous() is False:
            y = y.contiguous()
        div_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)
    else:
        div_scalar_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)
    return output

def pow(x, scalar):
    return x ** scalar

def log(x):
    output = torch.empty_like(x, dtype=x.dtype)
    assert x.is_cuda and output.is_cuda
    if x.is_contiguous() is False:
        x = x.contiguous()
    n_elements = output.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]), )
    log_kernel[grid](x, output, n_elements, BLOCK_SIZE=1024)
    return output

def exp(x):
    output = torch.empty_like(x, dtype=x.dtype)
    assert x.is_cuda and output.is_cuda
    if x.is_contiguous() is False:
        x = x.contiguous()
    n_elements = output.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]), )
    exp_kernel[grid](x, output, n_elements, BLOCK_SIZE=1024)
    return output

def sin(x):
    output = torch.empty_like(x, dtype=x.dtype)
    assert x.is_cuda and output.is_cuda
    if x.is_contiguous() is False:
        x = x.contiguous()
    n_elements = output.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]), )
    sin_kernel[grid](x, output, n_elements, BLOCK_SIZE=1024)
    return output

def cos(x):
    output = torch.empty_like(x, dtype=x.dtype)
    assert x.is_cuda and output.is_cuda
    if x.is_contiguous() is False:
        x = x.contiguous()
    n_elements = output.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]), )
    cos_kernel[grid](x, output, n_elements, BLOCK_SIZE=1024)
    return output

def sqrt(x):
    output = torch.empty_like(x, dtype=x.dtype)
    assert x.is_cuda and output.is_cuda
    if x.is_contiguous() is False:
        x = x.contiguous()
    n_elements = output.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]), )
    sqrt_kernel[grid](x, output, n_elements, BLOCK_SIZE=1024)
    return output

def maximum(x, y):
    output = torch.zeros(x.shape, device=torch.device("cuda"), dtype=x.dtype)
    assert x.is_cuda and output.is_cuda
    if x.is_contiguous() is False:
        x = x.contiguous()
    n_elements = output.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]), 1, 1)
    if isinstance(y, torch.Tensor):
        if y.is_contiguous() is False:
            y = y.contiguous()
        maximum_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)
    else:
        maximum_scalar_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)
    return output

def reduce_sum(x, axis=None, keepdims=False):
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
    output = torch.empty(output_shape, device=torch.device("cuda"), dtype=x.dtype)

    if x.is_contiguous() is False:
        x = x.contiguous()

    block_m = 4
    block_n = min(triton.next_power_of_2(n), 1024)
    grid = (triton.cdiv(m, block_m), 1, 1)
    sum_kernel[grid](x, output, m, n, block_m, block_n)
    return output

def reduce_max(x, axis=None, keepdims=False):
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
    output = torch.empty(output_shape, device=torch.device("cuda"), dtype=x.dtype)

    block_m = 4
    block_n = min(triton.next_power_of_2(n), 1024)
    grid = (triton.cdiv(m, block_m), 1, 1)

    if x.is_contiguous() is False:
        x = x.contiguous()
    max_kernel[grid](x, output, m, n, block_m, block_n)
    return output

def reshape(x, new_shape):
    return x.reshape(new_shape)

def permute(x, new_axis):
    return x.permute(new_axis)

def broadcast_to(x, new_shape):
    return x.broadcast_to(new_shape)

def getitem(x, idxs):
    return x.__getitem__(idxs)

def setitem(x, idxs, other):
    return x.__setitem__(idxs, other)

def eq(x, y):
    return x.__eq__(y)

def ge(x, y):
    return x.__ge__(y)

def matmul(a, b, activation=""):
    assert a.shape[-1] == b.shape[-2], "Incompatible dimensions"
    if a.is_contiguous() is False:
        a = a.contiguous()
    if b.is_contiguous() is False:
        b = b.contiguous()
    if len(a.shape) == 2 and len(b.shape) == 2:
        M, K = a.shape
        K, N = b.shape
        # Allocates output.
        c = torch.empty((M, N), device=torch.device("cuda"), dtype=a.dtype)
        # 1D launch kernel where each block gets its own program.
        grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']), )
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
            aa = a.reshape(pre_a, a_shape[-2], a_shape[-1])
            for i in range(len(b_shape) - 2):
                pre_shape_b.append(b_shape[i])
                pre_b *= b_shape[i]
            bb = b.data.reshape(pre_b, b_shape[-2], b_shape[-1])

            if pre_a == 1:
                aa = aa.broadcast_to(bb.shape[0], aa.shape[1], aa.shape[2])
            if pre_b == 1:
                bb = bb.broadcast_to(aa.shape[0], bb.shape[1], bb.shape[2])

        M = a_shape[-2]
        N = b_shape[-1]
        K = a_shape[-1]

        c = torch.empty((max(pre_a, pre_b), M, N), device=torch.device("cuda"), dtype=a.dtype)

        # 1D launch kernel where each block gets its own program.
        grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']), )
        for i in range(max(pre_a, pre_b)):
            matmul_kernel[grid](
                    aa[i], bb[i], c[i],
                    M, N, K,
                    a.stride(-2), a.stride(-1),
                    b.stride(-2), b.stride(-1),
                    c.stride(-2), c.stride(-1),
                    BLOCK_SIZE_M=64,
                    BLOCK_SIZE_N=64,
                    BLOCK_SIZE_K=32,
                    GROUP_SIZE_M=8,
                    ACTIVATION=activation
            )
        if pre_a > pre_b:
            c = c.reshape(tuple(pre_shape_a) + (M, N))
        else:
            c = c.reshape(tuple(pre_shape_b) + (M, N))
        return c
    else:
        return None


def from_numpy(data):
    return torch.from_numpy(data).cuda()

def from_tensor(data):
    return data.cuda()

def array(shape):
    return torch.empty(shape, device=torch.device("cuda"), dtype=torch.float32)
