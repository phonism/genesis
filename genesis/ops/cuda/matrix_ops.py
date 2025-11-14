"""
Matrix operations for GPU backend with optimized kernels.
"""
import torch
import triton
import triton.language as tl
from genesis.backends.cuda import CUDAStorage
from ..dispatcher import register_cuda


# =============================================================================
# OPTIMIZED TRITON KERNELS (flag_gems style)
# =============================================================================

@triton.heuristics({
    'DIVISIBLE_M': lambda args: args['M'] % args['TILE_M'] == 0,
    'DIVISIBLE_N': lambda args: args['N'] % args['TILE_N'] == 0,
    'DIVISIBLE_K': lambda args: args['K'] % args['TILE_K'] == 0,
})
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
    DIVISIBLE_M: tl.constexpr,
    DIVISIBLE_N: tl.constexpr,
    DIVISIBLE_K: tl.constexpr,
):
    """Optimized matrix multiplication kernel with better tiling strategy."""
    pidx = tl.program_id(0)
    pidy = tl.program_id(1)

    # Reorder CTAs for better L2 cache hit rate
    if GROUP_M == 1:
        pid_m, pid_n = pidx, pidy
    else:
        gridx = tl.num_programs(0)
        gridy = tl.num_programs(1)
        pid = pidx + pidy * gridx

        num_CTA_per_group = gridy * GROUP_M
        group_id = pid // num_CTA_per_group
        inner_group_id = pid % num_CTA_per_group
        GROUP_SIZE = tl.where(
            (group_id * GROUP_M + GROUP_M) > gridx, gridx % GROUP_M, GROUP_M
        )
        pid_m = group_id * GROUP_M + inner_group_id % GROUP_SIZE
        pid_n = inner_group_id // GROUP_SIZE

    # Compute offsets
    offs_m = pid_m * TILE_M + tl.arange(0, TILE_M)
    offs_n = pid_n * TILE_N + tl.arange(0, TILE_N)
    offs_k = tl.arange(0, TILE_K)

    # Create masks for boundary conditions
    if not DIVISIBLE_M:
        mask_m = offs_m < M
    if not DIVISIBLE_N:
        mask_n = offs_n < N

    # Compute pointers
    a_ptrs = A + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = B + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
    c_ptrs = C + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn

    # Main computation loop
    num_iters = tl.cdiv(K, TILE_K)
    accumulator = tl.zeros((TILE_M, TILE_N), dtype=tl.float32)

    for _ in range(num_iters):
        # Create masks for loading
        if DIVISIBLE_K:
            if DIVISIBLE_M:
                mask_a = None
            else:
                mask_a = mask_m[:, None]
            if DIVISIBLE_N:
                mask_b = None
            else:
                mask_b = mask_n[None, :]
        else:
            mask_k = offs_k < K
            if DIVISIBLE_M:
                mask_a = mask_k[None, :]
            else:
                mask_a = mask_m[:, None] & mask_k[None, :]
            if DIVISIBLE_N:
                mask_b = mask_k[:, None]
            else:
                mask_b = mask_k[:, None] & mask_n[None, :]

        # Load and compute
        if mask_a is None:
            a = tl.load(a_ptrs)
        else:
            a = tl.load(a_ptrs, mask=mask_a, other=0.0)

        if mask_b is None:
            b = tl.load(b_ptrs)
        else:
            b = tl.load(b_ptrs, mask=mask_b, other=0.0)

        # Update offsets and pointers
        offs_k += TILE_K
        a_ptrs += TILE_K * stride_ak
        b_ptrs += TILE_K * stride_bk

        # Accumulate
        accumulator += tl.dot(a, b, allow_tf32=False)

    # Store result
    if DIVISIBLE_M and DIVISIBLE_N:
        mask_c = None
    elif DIVISIBLE_M and not DIVISIBLE_N:
        mask_c = mask_n[None, :]
    elif not DIVISIBLE_M and DIVISIBLE_N:
        mask_c = mask_m[:, None]
    else:
        mask_c = mask_m[:, None] & mask_n[None, :]

    if mask_c is None:
        tl.store(c_ptrs, accumulator)
    else:
        tl.store(c_ptrs, accumulator, mask=mask_c)


@triton.heuristics({
    'DIVISIBLE_M': lambda args: args['M'] % args['TILE_M'] == 0,
    'DIVISIBLE_N': lambda args: args['N'] % args['TILE_N'] == 0,
    'DIVISIBLE_K': lambda args: args['K'] % args['TILE_K'] == 0,
})
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
    DIVISIBLE_M: tl.constexpr,
    DIVISIBLE_N: tl.constexpr,
    DIVISIBLE_K: tl.constexpr,
):
    """Batch matrix multiplication kernel with strided tensor support."""
    # Get batch index
    pid_b = tl.program_id(2)
    A += pid_b * stride_ab
    B += pid_b * stride_bb
    O += pid_b * stride_ob

    pidx = tl.program_id(0)
    pidy = tl.program_id(1)

    # Reorder CTAs
    if GROUP_M == 1:
        pid_m, pid_n = pidx, pidy
    else:
        gridx = tl.num_programs(0)
        gridy = tl.num_programs(1)
        pid = pidx + pidy * gridx

        num_CTA_per_group = gridy * GROUP_M
        group_id = pid // num_CTA_per_group
        inner_group_id = pid % num_CTA_per_group
        GROUP_SIZE = tl.where(
            (group_id * GROUP_M + GROUP_M) > gridx, gridx % GROUP_M, GROUP_M
        )
        pid_m = group_id * GROUP_M + inner_group_id % GROUP_SIZE
        pid_n = inner_group_id // GROUP_SIZE

    offs_m = pid_m * TILE_M + tl.arange(0, TILE_M)
    offs_n = pid_n * TILE_N + tl.arange(0, TILE_N)
    offs_k = tl.arange(0, TILE_K)

    if not DIVISIBLE_M:
        mask_m = offs_m < M
    if not DIVISIBLE_N:
        mask_n = offs_n < N

    a_ptrs = A + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = B + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
    o_ptrs = O + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on

    num_iters = tl.cdiv(K, TILE_K)
    o = tl.zeros((TILE_M, TILE_N), dtype=tl.float32)

    for _ in range(num_iters):
        if DIVISIBLE_K:
            if DIVISIBLE_M:
                mask_a = None
            else:
                mask_a = mask_m[:, None]
            if DIVISIBLE_N:
                mask_b = None
            else:
                mask_b = mask_n[None, :]
        else:
            mask_k = offs_k < K
            if DIVISIBLE_M:
                mask_a = mask_k[None, :]
            else:
                mask_a = mask_m[:, None] & mask_k[None, :]
            if DIVISIBLE_N:
                mask_b = mask_k[:, None]
            else:
                mask_b = mask_k[:, None] & mask_n[None, :]

        if mask_a is None:
            a = tl.load(a_ptrs)
        else:
            a = tl.load(a_ptrs, mask=mask_a, other=0.0)

        if mask_b is None:
            b = tl.load(b_ptrs)
        else:
            b = tl.load(b_ptrs, mask=mask_b, other=0.0)

        offs_k += TILE_K
        a_ptrs += TILE_K * stride_ak
        b_ptrs += TILE_K * stride_bk

        o += tl.dot(a, b, allow_tf32=False)

    if DIVISIBLE_M and DIVISIBLE_N:
        mask_c = None
    elif DIVISIBLE_M and not DIVISIBLE_N:
        mask_c = mask_n[None, :]
    elif not DIVISIBLE_M and DIVISIBLE_N:
        mask_c = mask_m[:, None]
    else:
        mask_c = mask_m[:, None] & mask_n[None, :]

    if mask_c is None:
        tl.store(o_ptrs, o)
    else:
        tl.store(o_ptrs, o, mask=mask_c)


# =============================================================================
# GPU OPERATIONS
# =============================================================================

def get_matmul_config(M, N, K):
    """
    Select optimal configuration based on matrix dimensions.
    Based on FlagGems and empirical testing.
    """
    # For very large output matrices (like LM Head)
    if M * N > 100000000:  # >100M elements
        if N > 100000:  # Vocabulary size (LM Head case)
            # Special config for LM Head (respecting shared memory limits)
            # Shared memory = (TILE_M + TILE_N) * TILE_K * 4 bytes * num_stages
            # Must be < 166KB
            return {'TILE_M': 128, 'TILE_N': 128, 'TILE_K': 32, 'GROUP_M': 2}, 4, 3
        else:
            return {'TILE_M': 128, 'TILE_N': 128, 'TILE_K': 32, 'GROUP_M': 2}, 4, 3

    # For large matrices
    elif M * N > 10000000:  # >10M elements
        return {'TILE_M': 128, 'TILE_N': 64, 'TILE_K': 32, 'GROUP_M': 2}, 4, 2

    # For medium matrices
    elif M * N > 1000000:  # >1M elements
        return {'TILE_M': 64, 'TILE_N': 64, 'TILE_K': 32, 'GROUP_M': 2}, 4, 2

    # For small matrices
    else:
        return {'TILE_M': 32, 'TILE_N': 32, 'TILE_K': 32, 'GROUP_M': 1}, 4, 2

@register_cuda("matmul")
def matmul(a, b, activation=""):
    """
    Optimized matrix multiplication operation with strided tensor support.

    Supports non-contiguous (strided) tensors to avoid unnecessary copies.
    """
    assert a.shape[-1] == b.shape[-2], "Incompatible dimensions"

    # Removed forced contiguous - kernels handle strides directly
    # This avoids 11GB of wasted copies in backward pass!

    if len(a.shape) == 2 and len(b.shape) == 2:
        M, K = a.shape
        K2, N = b.shape
        assert K == K2, f"Incompatible dimensions: {K} != {K2}"

        # Get optimal configuration
        config, num_warps, num_stages = get_matmul_config(M, N, K)

        # Allocate output
        c = CUDAStorage((M, N), dtype=a.dtype)

        # Launch kernel with selected configuration
        grid = (
            triton.cdiv(M, config['TILE_M']),
            triton.cdiv(N, config['TILE_N']),
        )

        matmul_kernel[grid](
            a, b, c,
            M, N, K,
            a.stride(0), a.stride(1),
            b.stride(0), b.stride(1),
            c.stride(0), c.stride(1),
            TILE_M=config['TILE_M'],
            TILE_N=config['TILE_N'],
            TILE_K=config['TILE_K'],
            GROUP_M=config['GROUP_M'],
            num_warps=num_warps,
            num_stages=num_stages,
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
            # Removed forced contiguous - kernels handle strides directly
            # This avoids 11GB of wasted copies in backward pass!
            aa = a.reshape((pre_a, a_shape[-2], a_shape[-1]))
        else:
            aa = a.unsqueeze(0)
            pre_a = 1

        if len(b_shape) > 2:
            for i in range(len(b_shape) - 2):
                pre_shape_b.append(b_shape[i])
                pre_b *= b_shape[i]
            # Removed forced contiguous - kernels handle strides directly
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

        # Get optimal configuration for batch matmul
        config, num_warps, num_stages = get_matmul_config(M, N, K)

        # Allocate output
        c = CUDAStorage((batch_size, M, N), dtype=a.dtype)

        # Launch batch kernel with selected configuration
        grid = (
            triton.cdiv(M, config['TILE_M']),
            triton.cdiv(N, config['TILE_N']),
            batch_size,
        )

        # Now supports strided tensors! No need for contiguous copies
        bmm_kernel[grid](
            aa, bb, c,
            M, N, K,
            aa.stride(0), aa.stride(1), aa.stride(2),
            bb.stride(0), bb.stride(1), bb.stride(2),
            c.stride(0), c.stride(1), c.stride(2),
            TILE_M=config['TILE_M'],
            TILE_N=config['TILE_N'],
            TILE_K=config['TILE_K'],
            GROUP_M=config['GROUP_M'],
            num_warps=num_warps,
            num_stages=num_stages,
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
