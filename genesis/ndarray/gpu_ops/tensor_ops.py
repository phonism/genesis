"""
Additional tensor operations for GPU backend.
"""
import triton
import triton.language as tl
from ..cuda_storage import CUDAStorage


# =============================================================================
# TRITON KERNELS
# =============================================================================

@triton.jit
def triu_kernel(input_ptr, output_ptr, M, N, k, BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr):
    """
    Upper triangle kernel.
    """
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    
    mask_m = offs_m < M
    mask_n = offs_n < N
    
    # Load input
    input_ptrs = input_ptr + offs_m[:, None] * N + offs_n[None, :]
    input_vals = tl.load(input_ptrs, mask=mask_m[:, None] & mask_n[None, :], other=0.0)
    
    # Apply upper triangle condition: keep if j >= i + k
    condition = offs_n[None, :] >= offs_m[:, None] + k
    output_vals = tl.where(condition, input_vals, 0.0)
    
    # Store output
    output_ptrs = output_ptr + offs_m[:, None] * N + offs_n[None, :]
    tl.store(output_ptrs, output_vals, mask=mask_m[:, None] & mask_n[None, :])


@triton.jit
def dtype_convert_kernel(input_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    """
    Data type conversion kernel.
    """
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load and convert (Triton handles type conversion automatically)
    input_vals = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    tl.store(output_ptr + offsets, input_vals, mask=mask)


# =============================================================================
# GPU OPERATIONS
# =============================================================================

def triu(x, k=0):
    """
    Upper triangle of tensor using Triton kernel.
    """
    if len(x.shape) != 2:
        raise ValueError("triu only supports 2D tensors")
    
    M, N = x.shape
    output = CUDAStorage(x.shape, dtype=x.dtype)
    
    if not x.is_contiguous():
        x = x.contiguous()
    
    BLOCK_SIZE_M = 32
    BLOCK_SIZE_N = 32
    grid = (triton.cdiv(M, BLOCK_SIZE_M), triton.cdiv(N, BLOCK_SIZE_N))
    
    triu_kernel[grid](
        x, output, M, N, k,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N
    )
    
    return output


def split(x, cnt, dim=None):
    """
    Split tensor along dimension using CUDAStorage slicing.
    
    Args:
        x: Input tensor
        cnt: Either an integer (number of equal splits) or a list of sizes for each split
        dim: Dimension to split along
    """
    if dim is None:
        dim = -1
    
    # Normalize dimension
    if dim < 0:
        dim = len(x.shape) + dim
    
    dim_size = x.shape[dim]
    
    # Handle both integer (number of splits) and list (sizes of splits)
    if isinstance(cnt, int):
        # Equal splits
        split_size = dim_size // cnt
        remainder = dim_size % cnt
        sizes = [split_size + (1 if i < remainder else 0) for i in range(cnt)]
    else:
        # List of sizes
        sizes = cnt
        # Verify sizes sum to dimension size
        if sum(sizes) != dim_size:
            raise ValueError(f"Split sizes {sizes} don't sum to dimension size {dim_size}")
    
    # Create splits using CUDAStorage slicing
    result = []
    start = 0
    
    for size in sizes:
        if size == 0:
            continue
        
        # Create slice indices
        indices = [slice(None)] * len(x.shape)
        indices[dim] = slice(start, start + size)
        
        # Use CUDAStorage's __getitem__ to create the split
        split_tensor = x[tuple(indices)]
        result.append(split_tensor)
        start += size
    
    return result


def squeeze(x, dim=None):
    """
    Remove dimensions of size 1 using CUDAStorage.
    """
    return x.squeeze(dim)


def unsqueeze(x, dim):
    """
    Add a dimension of size 1 using CUDAStorage.
    """
    return x.unsqueeze(dim)


def to_dtype(x, dtype):
    """
    Convert tensor to specified dtype using Triton kernel.
    """
    if x.dtype == dtype:
        return x
    
    output = CUDAStorage(x.shape, dtype=dtype)
    
    if not x.is_contiguous():
        x = x.contiguous()
    
    n_elements = x.size
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]), )
    
    dtype_convert_kernel[grid](x, output, n_elements, BLOCK_SIZE=1024)
    
    return output