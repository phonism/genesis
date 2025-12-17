"""
Matrix operations for GPU backend with optimized kernels.
"""
import os
import torch
import triton
import triton.language as tl
from genesis.backends.cuda import CUDAStorage
from ..dispatcher import register_cuda


# =============================================================================
# AUTOTUNE CONFIGURATION
# =============================================================================
# Set GENESIS_FAST_TEST=1 to use minimal autotune configs for faster testing
_FAST_TEST_MODE = os.environ.get("GENESIS_FAST_TEST", "0") == "1"

if _FAST_TEST_MODE:
    # Minimal config for fast testing - single balanced configuration
    _MATMUL_CONFIGS = [
        triton.Config(
            {"TILE_M": 64, "TILE_N": 64, "TILE_K": 32, "GROUP_M": 8},
            num_stages=4, num_warps=4
        ),
    ]
else:
    # Full autotune configs for production performance
    _MATMUL_CONFIGS = [
        triton.Config(
            {"TILE_M": 128, "TILE_N": 256, "TILE_K": 64, "GROUP_M": 8},
            num_stages=3, num_warps=8
        ),
        triton.Config(
            {"TILE_M": 64, "TILE_N": 256, "TILE_K": 32, "GROUP_M": 8},
            num_stages=4, num_warps=4
        ),
        triton.Config(
            {"TILE_M": 128, "TILE_N": 128, "TILE_K": 32, "GROUP_M": 8},
            num_stages=4, num_warps=4
        ),
        triton.Config(
            {"TILE_M": 128, "TILE_N": 64, "TILE_K": 32, "GROUP_M": 8},
            num_stages=4, num_warps=4
        ),
        triton.Config(
            {"TILE_M": 64, "TILE_N": 128, "TILE_K": 32, "GROUP_M": 8},
            num_stages=4, num_warps=4
        ),
        triton.Config(
            {"TILE_M": 128, "TILE_N": 32, "TILE_K": 32, "GROUP_M": 8},
            num_stages=4, num_warps=4
        ),
        triton.Config(
            {"TILE_M": 64, "TILE_N": 32, "TILE_K": 32, "GROUP_M": 8},
            num_stages=5, num_warps=2
        ),
        triton.Config(
            {"TILE_M": 32, "TILE_N": 64, "TILE_K": 32, "GROUP_M": 8},
            num_stages=5, num_warps=2
        ),
    ]


# =============================================================================
# OPTIMIZED TRITON KERNELS WITH AUTOTUNE
# =============================================================================

@triton.autotune(configs=_MATMUL_CONFIGS, key=["M", "N", "K"])
@triton.jit
def matmul_kernel(
    A,
    B,
    C,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    TILE_M: tl.constexpr,
    TILE_N: tl.constexpr,
    TILE_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    """
    Optimized matrix multiplication kernel with autotune.

    Uses 1D grid with tile grouping for better L2 cache utilization.
    """
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, TILE_M)
    num_pid_n = tl.cdiv(N, TILE_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # Use modulo for out-of-bounds safety (tiles at boundary)
    offs_am = (pid_m * TILE_M + tl.arange(0, TILE_M)) % M
    offs_bn = (pid_n * TILE_N + tl.arange(0, TILE_N)) % N
    offs_k = tl.arange(0, TILE_K)

    a_ptrs = A + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = B + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    accumulator = tl.zeros((TILE_M, TILE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, TILE_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * TILE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * TILE_K, other=0.0)
        accumulator = tl.dot(a, b, accumulator)
        a_ptrs += TILE_K * stride_ak
        b_ptrs += TILE_K * stride_bk

    # Cast accumulator to output dtype
    c = accumulator.to(C.dtype.element_ty)

    offs_cm = pid_m * TILE_M + tl.arange(0, TILE_M)
    offs_cn = pid_n * TILE_N + tl.arange(0, TILE_N)
    c_ptrs = C + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


@triton.autotune(configs=_MATMUL_CONFIGS, key=["M", "N", "K"])
@triton.jit
def bmm_kernel(
    A,
    B,
    O,
    M,
    N,
    K,
    stride_ab,
    stride_am,
    stride_ak,
    stride_bb,
    stride_bk,
    stride_bn,
    stride_ob,
    stride_om,
    stride_on,
    TILE_M: tl.constexpr,
    TILE_N: tl.constexpr,
    TILE_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    """
    Batch matrix multiplication kernel with autotune.

    Uses 1D grid per batch for better L2 cache utilization.
    """
    # Get batch index
    pid_b = tl.program_id(1)
    A += pid_b * stride_ab
    B += pid_b * stride_bb
    O += pid_b * stride_ob

    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, TILE_M)
    num_pid_n = tl.cdiv(N, TILE_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_am = (pid_m * TILE_M + tl.arange(0, TILE_M)) % M
    offs_bn = (pid_n * TILE_N + tl.arange(0, TILE_N)) % N
    offs_k = tl.arange(0, TILE_K)

    a_ptrs = A + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = B + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    accumulator = tl.zeros((TILE_M, TILE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, TILE_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * TILE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * TILE_K, other=0.0)
        accumulator = tl.dot(a, b, accumulator)
        a_ptrs += TILE_K * stride_ak
        b_ptrs += TILE_K * stride_bk

    # Cast accumulator to output dtype
    o = accumulator.to(O.dtype.element_ty)

    offs_om = pid_m * TILE_M + tl.arange(0, TILE_M)
    offs_on = pid_n * TILE_N + tl.arange(0, TILE_N)
    o_ptrs = O + stride_om * offs_om[:, None] + stride_on * offs_on[None, :]
    o_mask = (offs_om[:, None] < M) & (offs_on[None, :] < N)
    tl.store(o_ptrs, o, mask=o_mask)


# =============================================================================
# GPU OPERATIONS
# =============================================================================

@register_cuda("matmul")
def matmul(a, b, activation=""):
    """
    Optimized matrix multiplication operation with strided tensor support.

    Uses autotuned Triton kernels with 1D grid for better L2 cache utilization.
    Supports non-contiguous (strided) tensors to avoid unnecessary copies.
    """
    assert a.shape[-1] == b.shape[-2], "Incompatible dimensions"

    if len(a.shape) == 2 and len(b.shape) == 2:
        M, K = a.shape
        K2, N = b.shape
        assert K == K2, f"Incompatible dimensions: {K} != {K2}"

        # Allocate output
        c = CUDAStorage((M, N), dtype=a.dtype)

        # Use 1D grid with autotune - grid size computed by autotune lambda
        def grid(META):
            return (triton.cdiv(M, META["TILE_M"]) * triton.cdiv(N, META["TILE_N"]),)

        matmul_kernel[grid](
            a, b, c,
            M, N, K,
            a.stride(0), a.stride(1),
            b.stride(0), b.stride(1),
            c.stride(0), c.stride(1),
        )

        return c

    elif len(a.shape) > 2 or len(b.shape) > 2:
        # Handle batch matrix multiplication
        a_shape = a.shape
        b_shape = b.shape

        if len(a_shape) == 2 and len(b_shape) == 3:
            # Case: (M, K) @ (B, K, N) -> (B, M, N)
            M, K = a_shape
            B, K2, N = b_shape
            assert K == K2, f"Incompatible dimensions: {K} != {K2}"

            # Reshape to 2D: (M, K) @ (B*K, N) is not straightforward
            # Fall back to broadcast and batch processing
            a_expanded = a.unsqueeze(0).broadcast_to((B, M, K))
            a_2d = a_expanded.reshape(B * M, K)
            b_2d = b.reshape(B * K, N)

            # This case is more complex, fall through to general case below
            pass

        # Use 3D batch kernel for all batch cases
        # Handle broadcasting and complex batch scenarios
        pre_shape_a = []
        pre_shape_b = []
        pre_a = 1
        pre_b = 1

        if len(a_shape) > 2:
            for i in range(len(a_shape) - 2):
                pre_shape_a.append(a_shape[i])
                pre_a *= a_shape[i]
            aa = a.reshape((pre_a, a_shape[-2], a_shape[-1]))
        else:
            aa = a.unsqueeze(0)
            pre_a = 1

        if len(b_shape) > 2:
            for i in range(len(b_shape) - 2):
                pre_shape_b.append(b_shape[i])
                pre_b *= b_shape[i]
            bb = b.reshape((pre_b, b_shape[-2], b_shape[-1]))
        else:
            bb = b.unsqueeze(0)
            pre_b = 1

        # Broadcast if needed
        batch_size = max(pre_a, pre_b)
        if pre_a == 1 and batch_size > 1:
            aa = aa.broadcast_to((batch_size, aa.shape[1], aa.shape[2]))
        if pre_b == 1 and batch_size > 1:
            bb = bb.broadcast_to((batch_size, bb.shape[1], bb.shape[2]))

        M = a_shape[-2]
        K = a_shape[-1]
        N = b_shape[-1]

        # Allocate output
        c = CUDAStorage((batch_size, M, N), dtype=a.dtype)

        # Use 2D grid: (tiles, batch) with autotune
        def grid(META):
            return (
                triton.cdiv(M, META["TILE_M"]) * triton.cdiv(N, META["TILE_N"]),
                batch_size,
            )

        # Now supports strided tensors! No need for contiguous copies
        bmm_kernel[grid](
            aa, bb, c,
            M, N, K,
            aa.stride(0), aa.stride(1), aa.stride(2),
            bb.stride(0), bb.stride(1), bb.stride(2),
            c.stride(0), c.stride(1), c.stride(2),
        )

        # Reshape output
        output_shape = []
        if len(a_shape) > 2:
            output_shape.extend(pre_shape_a)
        elif len(b_shape) > 2:
            output_shape.extend(pre_shape_b)
        output_shape.extend([M, N])

        if len(output_shape) != len(c.shape):
            c = c.reshape(tuple(output_shape))

        return c
