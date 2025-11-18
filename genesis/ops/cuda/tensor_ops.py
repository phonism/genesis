"""
Additional tensor operations for GPU backend.
"""
import triton
import triton.language as tl
from genesis.backends.cuda import CUDAStorage
from .reduction_ops import reduce_max
from ..dispatcher import register_cuda


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
def tril_kernel(input_ptr, output_ptr, M, N, k, BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr):
    """
    Lower triangle kernel.
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

    # Apply lower triangle condition: keep if j <= i + k
    condition = offs_n[None, :] <= offs_m[:, None] + k
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
@register_cuda("triu")
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


@register_cuda("tril")
def tril(x, k=0):
    """
    Lower triangle of tensor using Triton kernel.

    Args:
        x: Input 2D tensor
        k: Diagonal offset (0 for main diagonal, positive for above, negative for below)

    Returns:
        Tensor with elements above the k-th diagonal zeroed
    """
    if len(x.shape) != 2:
        raise ValueError("tril only supports 2D tensors")

    M, N = x.shape
    output = CUDAStorage(x.shape, dtype=x.dtype)

    if not x.is_contiguous():
        x = x.contiguous()

    BLOCK_SIZE_M = 32
    BLOCK_SIZE_N = 32
    grid = (triton.cdiv(M, BLOCK_SIZE_M), triton.cdiv(N, BLOCK_SIZE_N))

    tril_kernel[grid](
        x, output, M, N, k,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N
    )

    return output


@register_cuda("split")
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


@register_cuda("squeeze")
def squeeze(x, dim=None):
    """
    Remove dimensions of size 1 using CUDAStorage.
    """
    return x.squeeze(dim)

@register_cuda("unsqueeze")
def unsqueeze(x, dim):
    """
    Add a dimension of size 1 using CUDAStorage.
    """
    return x.unsqueeze(dim)


@register_cuda("to_dtype")
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

@register_cuda("argsort")
def argsort(x, dim=-1, descending=False):
    """
    Sort indices along a dimension using iterative approach like topk.
    """
    if dim < 0:
        dim = len(x.shape) + dim
    
    # Handle different dimensions by transposing
    if dim != len(x.shape) - 1:
        # Transpose to make target dim the last one
        perm = list(range(len(x.shape)))
        perm[dim], perm[-1] = perm[-1], perm[dim]
        x_transposed = x.permute(perm)
        
        # Sort and transpose back
        sorted_indices = argsort(x_transposed, dim=-1, descending=descending)
        
        # Transpose indices back
        return sorted_indices.permute(perm)
    
    # Work with last dimension
    batch_shape = x.shape[:-1]
    N = x.shape[-1]
    M = 1
    for s in batch_shape:
        M *= s
    
    # Reshape to 2D and ensure contiguous
    x_2d = x.reshape((M, N)).contiguous()
    
    # Define grid and block size
    BLOCK_SIZE = triton.next_power_of_2(N)
    grid = (M,)
    
    # Handle ascending by negating values (find_max_kernel finds max, so negate for min)
    if not descending:
        # Create negation kernel to handle -x_2d
        negate_kernel[grid](x_2d, M, N, BLOCK_SIZE=BLOCK_SIZE)
    
    # Create output tensor for indices
    indices_shape = list(batch_shape) + [N]
    indices_output = CUDAStorage(indices_shape, dtype="int64")
    
    # Create mask tensor using CUDAStorage
    mask_shape = [M, N]
    mask = CUDAStorage(mask_shape, dtype="float32")
    
    # Initialize mask to zeros using CUDA kernel
    mask.fill_(0.0)
    
    # Call kernel N times to find all elements iteratively (like topk)
    for i in range(N):
        # Create temporary storage for current iteration
        temp_value = CUDAStorage((M,), dtype=x.dtype)
        temp_index = CUDAStorage((M,), dtype="int64")
        
        find_max_kernel[grid](
            x_2d,
            temp_value,
            temp_index,
            mask,
            M, N,
            BLOCK_SIZE=BLOCK_SIZE
        )
        
        # Copy results to final output manually to avoid slicing issues
        copy_kernel = get_copy_kernel()
        copy_grid = (M,)
        copy_kernel[copy_grid](
            temp_value,
            temp_value,  # Not used for argsort, just placeholder
            temp_index, 
            indices_output,
            M, N, i,  # Use N instead of k for argsort
            BLOCK_SIZE=triton.next_power_of_2(M)
        )
    
    # Handle ascending case by negating back
    if not descending:
        # Use kernel to negate the original values back (cleanup)
        output_M = M
        output_N = N
        output_grid = (output_M,)
        negate_kernel[output_grid](x_2d, output_M, output_N, 
                                  BLOCK_SIZE=triton.next_power_of_2(output_N))
    
    return indices_output


@triton.jit
def find_max_kernel(
    input_ptr,
    output_value_ptr,
    output_index_ptr,
    mask_ptr,
    M, N,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Find max value and index in each row, masking already found elements.
    Pure Triton implementation that avoids complex indexing.
    """
    row_id = tl.program_id(0)
    
    if row_id >= M:
        return
    
    # Load data and mask
    col_offsets = tl.arange(0, BLOCK_SIZE)
    col_mask = col_offsets < N
    
    data_ptr = input_ptr + row_id * N
    mask_row_ptr = mask_ptr + row_id * N
    
    values = tl.load(data_ptr + col_offsets, mask=col_mask, other=float('-inf'))
    mask_vals = tl.load(mask_row_ptr + col_offsets, mask=col_mask, other=0.0)
    
    # Set masked positions to negative infinity
    masked_values = tl.where(mask_vals == 0.0, values, float('-inf'))
    
    # Find maximum value
    max_val = tl.max(masked_values, axis=0)
    
    # Find first index equal to max value
    is_max = (masked_values == max_val) & col_mask
    # Trick: set non-max positions to BLOCK_SIZE, then take minimum
    idx_candidates = tl.where(is_max, col_offsets, BLOCK_SIZE)
    min_idx = tl.min(idx_candidates, axis=0)
    
    # Store results
    tl.store(output_value_ptr + row_id, max_val)
    tl.store(output_index_ptr + row_id, min_idx)
    
    # Update mask: mark found element as used
    if min_idx < N:
        tl.store(mask_row_ptr + min_idx, 1.0)


@triton.jit
def negate_kernel(
    data_ptr,
    M, N,
    BLOCK_SIZE: tl.constexpr
):
    """Negate all elements in the tensor in-place."""
    row_id = tl.program_id(0)
    
    if row_id >= M:
        return
    
    # Load and negate row data
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < N
    
    row_ptr = data_ptr + row_id * N
    values = tl.load(row_ptr + col_offsets, mask=mask, other=0.0)
    negated_values = -values
    
    tl.store(row_ptr + col_offsets, negated_values, mask=mask)


@triton.jit
def copy_to_output_kernel(
    temp_values_ptr,
    output_values_ptr,
    temp_indices_ptr,
    output_indices_ptr,
    M, k, col_idx,
    BLOCK_SIZE: tl.constexpr
):
    """Copy temporary results to output arrays at specific column."""
    row_id = tl.program_id(0)
    
    if row_id >= M:
        return
    
    # Copy value
    temp_val = tl.load(temp_values_ptr + row_id)
    output_val_ptr = output_values_ptr + row_id * k + col_idx
    tl.store(output_val_ptr, temp_val)
    
    # Copy index
    temp_idx = tl.load(temp_indices_ptr + row_id)
    output_idx_ptr = output_indices_ptr + row_id * k + col_idx
    tl.store(output_idx_ptr, temp_idx)


def get_copy_kernel():
    """Return the copy kernel function."""
    return copy_to_output_kernel


@register_cuda("topk")
def topk(x, k, dim=-1, largest=True, sorted=True):
    """
    Top-k values and indices using pure Triton implementation.
    Uses iterative max-finding approach that avoids Triton limitations.
    """
    if dim < 0:
        dim = len(x.shape) + dim
    
    # Handle different dimensions by transposing
    if dim != len(x.shape) - 1:
        perm = list(range(len(x.shape)))
        perm[dim], perm[-1] = perm[-1], perm[dim]
        x_transposed = x.permute(perm)
        
        values, indices = topk(x_transposed, k, dim=-1, largest=largest, sorted=sorted)
        
        # Transpose back
        values = values.permute(perm)
        indices = indices.permute(perm)
        return values, indices
    
    # Work with last dimension
    batch_shape = x.shape[:-1]
    N = x.shape[-1]
    M = 1
    for s in batch_shape:
        M *= s
    
    if k > N:
        k = N
    
    # Reshape to 2D and ensure contiguous
    x_2d = x.reshape((M, N)).contiguous()
    
    # Define grid and block size
    BLOCK_SIZE = triton.next_power_of_2(N)
    grid = (M,)
    
    # Handle smallest values by negating using a kernel
    if not largest:
        # Create negation kernel to handle -x_2d
        negate_kernel[grid](x_2d, M, N, BLOCK_SIZE=BLOCK_SIZE)
    
    # Create output tensors using CUDAStorage
    values_shape = list(batch_shape) + [k]
    indices_shape = values_shape
    
    values_output = CUDAStorage(values_shape, dtype=x.dtype)
    indices_output = CUDAStorage(indices_shape, dtype="int64")
    
    # Create mask tensor using CUDAStorage
    mask_shape = [M, N]
    mask = CUDAStorage(mask_shape, dtype="float32")
    
    # Initialize mask to zeros using CUDA kernel
    mask.fill_(0.0)
    
    # Call kernel k times to find top-k elements iteratively
    for i in range(k):
        # Create temporary storage for current iteration
        temp_value = CUDAStorage((M,), dtype=x.dtype)
        temp_index = CUDAStorage((M,), dtype="int64")
        
        find_max_kernel[grid](
            x_2d,
            temp_value,
            temp_index,
            mask,
            M, N,
            BLOCK_SIZE=BLOCK_SIZE
        )
        
        # Copy results to final output manually to avoid slicing issues
        copy_kernel = get_copy_kernel()
        copy_grid = (M,)
        copy_kernel[copy_grid](
            temp_value,
            values_output,
            temp_index, 
            indices_output,
            M, k, i,
            BLOCK_SIZE=triton.next_power_of_2(M)
        )
    
    # Handle non-largest case by negating back
    if not largest:
        # Use kernel to negate the output values
        output_M = M
        output_N = k
        output_grid = (output_M,)
        negate_kernel[output_grid](values_output, output_M, output_N, 
                                  BLOCK_SIZE=triton.next_power_of_2(output_N))
    
    return values_output, indices_output


@triton.jit
def bincount_kernel(
    input_ptr, weights_ptr, output_ptr,
    N, num_classes, has_weights,
    BLOCK_SIZE: tl.constexpr
):
    """
    Simple bincount kernel - each thread processes one element.
    """
    # Each thread processes exactly one element  
    tid = tl.program_id(0) 
    
    if tid >= N:
        return
    
    # Load single element
    bin_idx = tl.load(input_ptr + tid)
    
    # Check bounds
    if bin_idx < 0 or bin_idx >= num_classes:
        return
    
    # Convert to int64 to match output tensor type
    bin_idx_64 = bin_idx.to(tl.int64)
    
    # Atomic add to histogram
    if not has_weights:
        # Use 1 as int64 (Triton should infer type from pointer)
        tl.atomic_add(output_ptr + bin_idx_64, 1)
    else:
        # Load weight and add it (keep as float)
        weight = tl.load(weights_ptr + tid)
        tl.atomic_add(output_ptr + bin_idx_64, weight)


@register_cuda("bincount")
def bincount(x, weights=None, minlength=0):
    """
    Count occurrences using Triton atomic operations.
    """
    if len(x.shape) != 1:
        raise ValueError("bincount only supports 1D tensors")

    if not x.is_contiguous():
        x = x.contiguous()

    N = x.shape[0]

    # Ensure input is int64 and contiguous
    if x.dtype_obj.name != "int64":
        raise ValueError(f"bincount input must be int64, got {x.dtype_obj.name}")


    # Find maximum value to determine output size
    if x.numel() > 0:
        max_tensor = reduce_max(x, axis=None, keepdims=False)
        max_val = int(max_tensor.to_numpy().item())
    else:
        max_val = 0
    num_classes = max(max_val + 1, minlength)

    # Sanity check - if num_classes is unreasonably large, something is wrong
    MAX_REASONABLE_CLASSES = 100_000_000  # 100M classes should be more than enough
    if num_classes > MAX_REASONABLE_CLASSES:
        raise ValueError(
            f"bincount: computed num_classes={num_classes} is unreasonably large. "
            f"Input max value: {max_val}, minlength: {minlength}. "
            f"This likely indicates corrupted input data."
        )

    # Create output tensor - int64 for unweighted, float32 for weighted
    output_dtype = "float32" if weights is not None else "int64"
    output = CUDAStorage((num_classes,), dtype=output_dtype)
    
    # Initialize to zero
    output.fill_(0)
    
    has_weights = weights is not None
    if has_weights:
        weights_ptr = weights if weights.is_contiguous() else weights.contiguous()
    else:
        # Create a dummy tensor for weights_ptr when not using weights
        weights_ptr = CUDAStorage((1,), dtype="float32")
    
    BLOCK_SIZE = 1  # Each thread handles one element
    grid = (N,)  # Launch N threads, one per element
    
    bincount_kernel[grid](
        x, weights_ptr, output,
        N, num_classes, has_weights,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return output
