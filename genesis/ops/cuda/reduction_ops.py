"""
Optimized reduction operations for GPU backend.
Inspired by FlagGems for better performance.
"""
import triton
import triton.language as tl
from functools import reduce as functools_reduce
import operator
import math
import genesis
from genesis.backends.cuda import CUDAStorage
from .basic_ops import zeros, add
from genesis.ops.dispatcher import register_cuda


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def calculate_tile_size(N, max_tile=1024, threshold=32):
    """
    Calculate optimal tile size for reduction.

    Args:
        N: Dimension size to tile
        max_tile: Maximum tile size
        threshold: Threshold below which we don't clamp

    Returns:
        Optimal tile size as power of 2
    """
    power_of_2 = triton.next_power_of_2(N)
    return min(max_tile, power_of_2) if N > threshold else power_of_2


def calculate_adaptive_block_size(n_elements):
    """
    Calculate adaptive block size based on element count.

    Uses sqrt-based heuristic inspired by FlagGems for balanced
    work distribution in two-stage reductions.

    Args:
        n_elements: Total number of elements

    Returns:
        Block size as power of 2, clamped to [32, 1024]
    """
    size = int(math.sqrt(n_elements))
    clamped = max(32, min(1024, size))
    return triton.next_power_of_2(clamped)


def normalize_axis(axis, ndim):
    """
    Normalize axis parameter to tuple of positive indices.

    Args:
        axis: Single int, tuple of ints, or None
        ndim: Number of dimensions in tensor

    Returns:
        Tuple of normalized positive axis indices, or None if axis is None
    """
    if axis is None:
        return None
    if isinstance(axis, int):
        axis = (axis,)
    return tuple(ax % ndim for ax in axis)


class ReductionKernels:
    """
    Container for kernel functions used in a specific reduction operation.

    Attributes:
        atomic_kernel: Kernel for small tensors using atomic operations
        stage1_kernel: First stage kernel for large two-stage reductions
        stage2_kernel: Second stage kernel for large two-stage reductions
        inner_kernel: Kernel for innermost dimension reductions
        fill_value: Value to fill empty tensors with
        name: Name of the reduction operation (for debugging)
    """
    def __init__(self, atomic_kernel, stage1_kernel, stage2_kernel, inner_kernel, fill_value, name):
        self.atomic_kernel = atomic_kernel
        self.stage1_kernel = stage1_kernel
        self.stage2_kernel = stage2_kernel
        self.inner_kernel = inner_kernel
        self.fill_value = fill_value
        self.name = name


# =============================================================================
# OPTIMIZED TRITON KERNELS - GENERIC IMPLEMENTATIONS
# =============================================================================

@triton.jit
def generic_reduction_stage1(
    inp_ptr,
    mid_ptr,
    M,
    BLOCK_SIZE: tl.constexpr,
    OP: tl.constexpr,  # "sum", "max", or "min"
):
    """
    Generic stage 1 kernel for two-stage reduction.
    Computes partial reductions (sum/max/min) for each block.

    Args:
        OP: Operation type - "sum", "max", or "min"
    """
    # Use float32 accumulation for fp16/bf16 to avoid precision loss
    if inp_ptr.dtype.element_ty == tl.float16:
        acc_dtype = tl.float32
    elif inp_ptr.dtype.element_ty == tl.bfloat16:
        acc_dtype = tl.float32
    else:
        acc_dtype = inp_ptr.dtype.element_ty

    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < M

    # Set initial value based on operation
    if OP == "sum":
        init_val = 0.0
    elif OP == "max":
        init_val = -float("inf")
    else:  # min
        init_val = float("inf")

    # Load and accumulate in higher precision
    vals = tl.load(inp_ptr + offset, mask=mask, other=init_val).to(acc_dtype)

    # Perform reduction based on operation type
    if OP == "sum":
        result = tl.sum(vals)
    elif OP == "max":
        result = tl.max(vals)
    else:  # min
        result = tl.min(vals)

    # Store partial result
    tl.store(mid_ptr + pid, result.to(inp_ptr.dtype.element_ty))


@triton.jit
def generic_reduction_stage2(
    mid_ptr,
    out_ptr,
    mid_size,
    BLOCK_SIZE: tl.constexpr,
    OP: tl.constexpr,  # "sum", "max", or "min"
):
    """
    Generic stage 2 kernel for two-stage reduction.
    Reduces all partial results to final output.

    Args:
        OP: Operation type - "sum", "max", or "min"
    """
    # Use float32 accumulation for fp16/bf16 to avoid precision loss
    if mid_ptr.dtype.element_ty == tl.float16:
        acc_dtype = tl.float32
    elif mid_ptr.dtype.element_ty == tl.bfloat16:
        acc_dtype = tl.float32
    else:
        acc_dtype = mid_ptr.dtype.element_ty

    offset = tl.arange(0, BLOCK_SIZE)
    mask = offset < mid_size

    # Set initial value based on operation
    if OP == "sum":
        init_val = 0.0
    elif OP == "max":
        init_val = -float("inf")
    else:  # min
        init_val = float("inf")

    vals = tl.load(mid_ptr + offset, mask=mask, other=init_val).to(acc_dtype)

    # Perform reduction based on operation type
    if OP == "sum":
        result = tl.sum(vals)
    elif OP == "max":
        result = tl.max(vals)
    else:  # min
        result = tl.min(vals)

    tl.store(out_ptr, result.to(mid_ptr.dtype.element_ty))


@triton.jit
def generic_reduction_inner(
    output_ptr,
    input_ptr,
    M,
    N,
    TILE_N: tl.constexpr,
    OP: tl.constexpr,  # "sum", "max", or "min"
):
    """
    Generic kernel for reducing the innermost dimension.
    Each thread block reduces one row.

    Args:
        OP: Operation type - "sum", "max", or "min"
    """
    # Use float32 accumulation for fp16/bf16 to avoid precision loss
    if input_ptr.dtype.element_ty == tl.float16:
        acc_dtype = tl.float32
    elif input_ptr.dtype.element_ty == tl.bfloat16:
        acc_dtype = tl.float32
    else:
        acc_dtype = input_ptr.dtype.element_ty

    pid_m = tl.program_id(0)

    # Set initial value based on operation
    if OP == "sum":
        init_val = 0.0
        acc = tl.zeros([], dtype=acc_dtype)
    elif OP == "max":
        init_val = -float("inf")
        acc = tl.full([], -float("inf"), dtype=acc_dtype)
    else:  # min
        init_val = float("inf")
        acc = tl.full([], float("inf"), dtype=acc_dtype)

    # Process row in chunks of TILE_N
    for start_n in range(0, N, TILE_N):
        n_offsets = start_n + tl.arange(0, TILE_N)
        inp_offsets = pid_m * N + n_offsets
        mask = n_offsets < N

        # Load chunk
        chunk = tl.load(input_ptr + inp_offsets, mask=mask, other=init_val).to(acc_dtype)

        # Accumulate based on operation type
        if OP == "sum":
            acc += tl.sum(chunk)
        elif OP == "max":
            acc = tl.maximum(acc, tl.max(chunk))
        else:  # min
            acc = tl.minimum(acc, tl.min(chunk))

    # Store result
    tl.store(output_ptr + pid_m, acc.to(input_ptr.dtype.element_ty))


@triton.jit
def generic_reduction_atomic(
    x_ptr,
    output_ptr,
    N,
    BLOCK_SIZE: tl.constexpr,
    OP: tl.constexpr,  # "sum", "max", or "min"
):
    """
    Generic atomic reduction kernel for small tensors.

    Args:
        OP: Operation type - "sum", "max", or "min"
    """
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N

    # Set initial value based on operation
    if OP == "sum":
        init_val = 0.0
    elif OP == "max":
        init_val = -float("inf")
    else:  # min
        init_val = float("inf")

    values = tl.load(x_ptr + offsets, mask=mask, other=init_val)

    # Perform block reduction
    if OP == "sum":
        block_result = tl.sum(values)
        tl.atomic_add(output_ptr, block_result)
    elif OP == "max":
        block_result = tl.max(values)
        tl.atomic_max(output_ptr, block_result)
    else:  # min
        block_result = tl.min(values)
        tl.atomic_min(output_ptr, block_result)


# =============================================================================
# SUM KERNELS (wrappers around generic implementations)
# =============================================================================

@triton.jit
def sum_kernel_stage1(
    inp_ptr,
    mid_ptr,
    M,
    BLOCK_SIZE: tl.constexpr,
):
    """Stage 1 of two-stage sum reduction."""
    generic_reduction_stage1(inp_ptr, mid_ptr, M, BLOCK_SIZE, "sum")


@triton.jit
def sum_kernel_stage2(
    mid_ptr,
    out_ptr,
    mid_size,
    BLOCK_SIZE: tl.constexpr
):
    """Stage 2 of two-stage sum reduction."""
    generic_reduction_stage2(mid_ptr, out_ptr, mid_size, BLOCK_SIZE, "sum")


@triton.jit
def sum_kernel_inner(
    output_ptr,
    input_ptr,
    M,
    N,
    TILE_N: tl.constexpr,
):
    """Inner kernel for sum reduction."""
    generic_reduction_inner(output_ptr, input_ptr, M, N, TILE_N, "sum")


@triton.jit
def sum_kernel_axis0_vectorized(
    output_ptr,
    input_ptr,
    M,
    N,
    TILE_N: tl.constexpr,
):
    """
    Vectorized kernel for axis=0 reduction with small M.
    Input shape: (M, N), Output shape: (N,)
    Each program processes TILE_N output elements.
    """
    # Use float32 accumulation for fp16/bf16 to avoid precision loss
    if input_ptr.dtype.element_ty == tl.float16:
        acc_dtype = tl.float32
    elif input_ptr.dtype.element_ty == tl.bfloat16:
        acc_dtype = tl.float32
    else:
        acc_dtype = input_ptr.dtype.element_ty

    pid = tl.program_id(0)
    n_start = pid * TILE_N
    n_offsets = n_start + tl.arange(0, TILE_N)
    n_mask = n_offsets < N

    # Accumulator for TILE_N output elements
    acc = tl.zeros([TILE_N], dtype=acc_dtype)

    # Sum over M dimension (should be small, like 2, 4, 8, etc.)
    for m in range(M):
        # Load row m
        offsets = m * N + n_offsets
        vals = tl.load(input_ptr + offsets, mask=n_mask, other=0.0).to(acc_dtype)
        acc += vals

    # Store results
    tl.store(output_ptr + n_offsets, acc.to(input_ptr.dtype.element_ty), mask=n_mask)


@triton.jit
def sum_kernel_atomic(
    x_ptr,
    output_ptr,
    N,
    BLOCK_SIZE: tl.constexpr
):
    """Atomic kernel for full tensor sum."""
    generic_reduction_atomic(x_ptr, output_ptr, N, BLOCK_SIZE, "sum")


# =============================================================================
# GPU OPERATIONS
# =============================================================================

def _reduce_generic(x, axis, keepdims, kernels):
    """
    Generic reduction implementation for sum/max/min operations.

    Args:
        x: Input storage object
        axis: Axis or axes to reduce along (None for full reduction)
        keepdims: Whether to keep reduced dimensions
        kernels: ReductionKernels object containing kernel functions and config

    Returns:
        Reduced storage object
    """
    shape = x.shape
    ndim = len(shape)

    # Ensure contiguous
    if not x.is_contiguous():
        x = x.contiguous()

    # Full tensor reduction
    if axis is None:
        n_elements = x.size
        output_shape = (1,) if keepdims else ()
        output = CUDAStorage(output_shape, dtype=x.dtype)

        if n_elements == 0:
            output.fill(kernels.fill_value)
            return output

        # Adaptive strategy based on size
        if n_elements <= 4096:
            # Small tensor: single kernel with atomic
            output.fill(kernels.fill_value)
            block_size = min(1024, triton.next_power_of_2(n_elements))
            grid = (triton.cdiv(n_elements, block_size),)
            kernels.atomic_kernel[grid](x, output, n_elements, block_size)
        else:
            # Large tensor: two-stage reduction
            block_size = calculate_adaptive_block_size(n_elements)
            mid_size = triton.cdiv(n_elements, block_size)

            # Stage 1: partial reductions
            mid = CUDAStorage((mid_size,), dtype=x.dtype)
            grid1 = (mid_size,)
            kernels.stage1_kernel[grid1](x, mid, n_elements, block_size)

            # Stage 2: final reduction
            block_mid = triton.next_power_of_2(min(2048, mid_size))
            grid2 = (1,)
            kernels.stage2_kernel[grid2](mid, output, mid_size, block_mid)

        return output

    # Normalize axis
    axis = normalize_axis(axis, ndim)

    # Single axis reduction
    if len(axis) == 1:
        ax = axis[0]

        # Calculate output shape
        output_shape = list(shape)
        if keepdims:
            output_shape[ax] = 1
        else:
            output_shape.pop(ax)
        output_shape = tuple(output_shape)

        # Optimize for different axis positions
        if ax == ndim - 1:
            # Innermost dimension reduction (most common in backward pass)
            M = functools_reduce(operator.mul, shape[:-1], 1)
            N = shape[-1]

            output = CUDAStorage((M,), dtype=x.dtype)
            x_flat = x.reshape((M, N))

            # Adaptive tile size
            tile_n = calculate_tile_size(N, max_tile=2048)

            grid = (M,)
            kernels.inner_kernel[grid](output, x_flat, M, N, tile_n)

            return output.reshape(output_shape)

        elif ax == 0 and kernels.name == "sum":
            # First dimension reduction (sum only - max/min use permute fallback)
            M = shape[0]
            rest = functools_reduce(operator.mul, shape[1:], 1)

            output = CUDAStorage(shape[1:] if not keepdims else output_shape, dtype=x.dtype)

            # Reshape to 2D
            x_2d = x.reshape((M, rest))
            out_flat = output.reshape((rest,))

            # For small M, use vectorized kernel that processes multiple outputs per program
            if M <= 16:
                # Vectorized approach: each program handles TILE_N output elements
                tile_n = min(2048, triton.next_power_of_2(min(rest, 1024)))
                grid = (triton.cdiv(rest, tile_n),)
                sum_kernel_axis0_vectorized[grid](out_flat, x_2d, M, rest, tile_n)
            else:
                # For larger M, fall back to transpose + inner kernel
                # IMPORTANT: Must call contiguous() after permute to ensure correct memory layout
                x_transposed = x_2d.permute((1, 0)).contiguous()
                tile_n = calculate_tile_size(M, max_tile=1024, threshold=0)
                grid = (rest,)
                kernels.inner_kernel[grid](out_flat, x_transposed, rest, M, tile_n)

            return output

        else:
            # Middle dimension reduction OR ax==0 for max/min - need permute
            axes_to_keep = tuple(i for i in range(ndim) if i != ax)
            new_order = axes_to_keep + (ax,)
            x = x.permute(new_order)

            M = functools_reduce(operator.mul, [shape[i] for i in axes_to_keep], 1)
            N = shape[ax]

            x_2d = x.reshape((M, N))
            temp_output = CUDAStorage((M,), dtype=x.dtype)

            # Use inner kernel after permute
            tile_n = calculate_tile_size(N, max_tile=1024)

            grid = (M,)
            kernels.inner_kernel[grid](temp_output, x_2d, M, N, tile_n)

            return temp_output.reshape(output_shape)

    # Multi-axis reduction
    axes_to_keep = tuple(i for i in range(ndim) if i not in axis)

    if not axes_to_keep:
        # Reducing all dimensions
        return _reduce_generic(x, axis=None, keepdims=keepdims, kernels=kernels)

    # Permute axes to group reduction dimensions
    new_order = axes_to_keep + axis
    x = x.permute(new_order)

    # Calculate dimensions
    keep_size = functools_reduce(operator.mul, [shape[i] for i in axes_to_keep], 1)
    reduce_size = functools_reduce(operator.mul, [shape[i] for i in axis], 1)

    # Reshape to 2D
    x_2d = x.reshape((keep_size, reduce_size))

    # Perform reduction
    temp_output = CUDAStorage((keep_size,), dtype=x.dtype)

    # Use optimized inner kernel
    tile_n = calculate_tile_size(reduce_size, max_tile=1024)

    grid = (keep_size,)
    kernels.inner_kernel[grid](temp_output, x_2d, keep_size, reduce_size, tile_n)

    # Reshape to final output
    if keepdims:
        final_shape = list(shape)
        for i in axis:
            final_shape[i] = 1
        return temp_output.reshape(tuple(final_shape))
    else:
        final_shape = tuple(shape[i] for i in axes_to_keep)
        return temp_output.reshape(final_shape)


@register_cuda("sum")
def reduce_sum(x, axis=None, keepdims=False):
    """
    Optimized reduce sum operation.
    Uses adaptive strategies based on tensor shape and reduction pattern.
    """
    # Handle bool tensors (sum-specific)
    if hasattr(x, 'dtype'):
        if x.dtype == genesis.bool:
            zeros_int64 = zeros(x.shape, dtype=genesis.int64)
            x = add(zeros_int64, x)

    # Create kernel configuration
    kernels = ReductionKernels(
        atomic_kernel=sum_kernel_atomic,
        stage1_kernel=sum_kernel_stage1,
        stage2_kernel=sum_kernel_stage2,
        inner_kernel=sum_kernel_inner,
        fill_value=0.0,
        name="sum"
    )

    # Delegate to generic implementation
    return _reduce_generic(x, axis, keepdims, kernels)


@register_cuda("sum_to_shape")
def sum_to_shape(x, target_shape):
    """
    Optimized sum_to_shape for backward operations.
    Efficiently reduces tensor to match target shape.
    """
    if not x.is_contiguous():
        x = x.contiguous()

    x_shape = x.shape
    target_shape = tuple(target_shape)

    # If shapes match, return as is
    if x_shape == target_shape:
        return x

    # Handle dimension mismatch
    if len(x_shape) > len(target_shape):
        # Sum leading dimensions
        leading_dims = tuple(range(len(x_shape) - len(target_shape)))
        x = reduce_sum(x, axis=leading_dims, keepdims=False)
        x_shape = x.shape

    # If shapes match now, return
    if x_shape == target_shape:
        return x

    # Find axes to sum (where target is 1 but current is not)
    axes_to_sum = []
    for i, (dim, target_dim) in enumerate(zip(x_shape, target_shape)):
        if target_dim == 1 and dim != 1:
            axes_to_sum.append(i)

    if not axes_to_sum:
        return x

    # Perform reduction with keepdims=True
    return reduce_sum(x, axis=tuple(axes_to_sum), keepdims=True)


# =============================================================================
# MAX KERNELS (wrappers around generic implementations)
# =============================================================================

@triton.jit
def max_kernel_stage1(
    inp_ptr,
    mid_ptr,
    M,
    BLOCK_SIZE: tl.constexpr,
):
    """Stage 1 of two-stage max reduction."""
    generic_reduction_stage1(inp_ptr, mid_ptr, M, BLOCK_SIZE, "max")


@triton.jit
def max_kernel_stage2(
    mid_ptr,
    out_ptr,
    mid_size,
    BLOCK_SIZE: tl.constexpr
):
    """Stage 2 of two-stage max reduction."""
    generic_reduction_stage2(mid_ptr, out_ptr, mid_size, BLOCK_SIZE, "max")


@triton.jit
def max_kernel_inner(
    output_ptr,
    input_ptr,
    M,
    N,
    TILE_N: tl.constexpr,
):
    """Inner kernel for max reduction."""
    generic_reduction_inner(output_ptr, input_ptr, M, N, TILE_N, "max")


@triton.jit
def max_kernel_atomic(
    x_ptr,
    output_ptr,
    N,
    BLOCK_SIZE: tl.constexpr
):
    """Atomic kernel for full tensor max."""
    generic_reduction_atomic(x_ptr, output_ptr, N, BLOCK_SIZE, "max")


@register_cuda("max")
def reduce_max(x, axis=None, keepdims=False):
    """
    Optimized reduce max operation.
    """
    # Create kernel configuration
    kernels = ReductionKernels(
        atomic_kernel=max_kernel_atomic,
        stage1_kernel=max_kernel_stage1,
        stage2_kernel=max_kernel_stage2,
        inner_kernel=max_kernel_inner,
        fill_value=-float('inf'),
        name="max"
    )

    # Delegate to generic implementation
    return _reduce_generic(x, axis, keepdims, kernels)

# =============================================================================
# MIN KERNELS (wrappers around generic implementations)
# =============================================================================

@triton.jit
def min_kernel_stage1(
    inp_ptr,
    mid_ptr,
    M,
    BLOCK_SIZE: tl.constexpr,
):
    """Stage 1 of two-stage min reduction."""
    generic_reduction_stage1(inp_ptr, mid_ptr, M, BLOCK_SIZE, "min")


@triton.jit
def min_kernel_stage2(
    mid_ptr,
    out_ptr,
    mid_size,
    BLOCK_SIZE: tl.constexpr
):
    """Stage 2 of two-stage min reduction."""
    generic_reduction_stage2(mid_ptr, out_ptr, mid_size, BLOCK_SIZE, "min")


@triton.jit
def min_kernel_inner(
    output_ptr,
    input_ptr,
    M,
    N,
    TILE_N: tl.constexpr,
):
    """Inner kernel for min reduction."""
    generic_reduction_inner(output_ptr, input_ptr, M, N, TILE_N, "min")


@triton.jit
def min_kernel_atomic(
    x_ptr,
    output_ptr,
    N,
    BLOCK_SIZE: tl.constexpr
):
    """Atomic kernel for full tensor min."""
    generic_reduction_atomic(x_ptr, output_ptr, N, BLOCK_SIZE, "min")


@register_cuda("min")
def reduce_min(x, axis=None, keepdims=False):
    """Optimized reduce min operation."""
    # Create kernel configuration
    kernels = ReductionKernels(
        atomic_kernel=min_kernel_atomic,
        stage1_kernel=min_kernel_stage1,
        stage2_kernel=min_kernel_stage2,
        inner_kernel=min_kernel_inner,
        fill_value=float('inf'),
        name="min"
    )

    # Delegate to generic implementation
    return _reduce_generic(x, axis, keepdims, kernels)
