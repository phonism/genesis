"""
Matrix operations for GPU backend.
"""
import triton
import triton.language as tl
from ..cuda_storage import CUDAStorage


# =============================================================================
# TRITON KERNELS
# =============================================================================

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
    Matrix multiplication kernel.
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


# =============================================================================
# GPU OPERATIONS
# =============================================================================


def matmul(a, b, activation=""):
    """
    Matrix multiplication operation.
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
        c = CUDAStorage((M, N), dtype=a.dtype)
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
        c = CUDAStorage((batch_size, M, N), dtype=a.dtype)

        # 1D launch kernel where each block gets its own program.
        grid = lambda META: (triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]), )
        
        for i in range(batch_size):
            # Use CUDAStorage slice operations - much cleaner!
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

        # Reshape output to match expected dimensions
        output_shape = []
        if len(a_shape) > 2:
            output_shape.extend(pre_shape_a)
        elif len(b_shape) > 2:
            output_shape.extend(pre_shape_b)
        output_shape.extend([M, N])
        
        if len(output_shape) != len(c.shape):
            c = c.reshape(tuple(output_shape))

        return c