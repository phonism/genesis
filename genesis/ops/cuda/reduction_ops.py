"""
Optimized reduction operations for GPU backend.
Inspired by FlagGems for better performance.
"""
import triton
import triton.language as tl
from functools import reduce as functools_reduce
import operator
import math
from genesis.backends.cuda import CUDAStorage
from .basic_ops import zeros, add
from genesis.ops.dispatcher import register_cuda


# =============================================================================
# OPTIMIZED TRITON KERNELS
# =============================================================================

@triton.jit
def sum_kernel_stage1(
    inp_ptr,
    mid_ptr,
    M,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Stage 1 of two-stage reduction: compute partial sums.
    Each block processes BLOCK_SIZE elements.
    """
    # Use float32 for accumulation with fp16/bf16 inputs
    if inp_ptr.dtype.element_ty == tl.float16:
        acc_dtype = tl.float32
    elif inp_ptr.dtype.element_ty == tl.bfloat16:
        acc_dtype = tl.float32
    else:
        acc_dtype = inp_ptr.dtype.element_ty

    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < M

    # Load and accumulate in higher precision
    vals = tl.load(inp_ptr + offset, mask=mask, other=0.0).to(acc_dtype)
    sum_val = tl.sum(vals)

    # Store partial sum
    tl.store(mid_ptr + pid, sum_val.to(inp_ptr.dtype.element_ty))


@triton.jit
def sum_kernel_stage2(
    mid_ptr,
    out_ptr,
    mid_size,
    BLOCK_SIZE: tl.constexpr
):
    """
    Stage 2 of two-stage reduction: sum all partial results.
    """
    # Use float32 for accumulation
    if mid_ptr.dtype.element_ty == tl.float16:
        acc_dtype = tl.float32
    elif mid_ptr.dtype.element_ty == tl.bfloat16:
        acc_dtype = tl.float32
    else:
        acc_dtype = mid_ptr.dtype.element_ty

    offset = tl.arange(0, BLOCK_SIZE)
    mask = offset < mid_size

    vals = tl.load(mid_ptr + offset, mask=mask, other=0.0).to(acc_dtype)
    sum_val = tl.sum(vals)

    tl.store(out_ptr, sum_val.to(mid_ptr.dtype.element_ty))


@triton.jit
def sum_kernel_inner(
    output_ptr,
    input_ptr,
    M,
    N,
    TILE_N: tl.constexpr,
):
    """
    Optimized kernel for reducing the innermost dimension.
    Each thread block reduces one row.
    """
    # Use float32 accumulation for fp16/bf16
    if input_ptr.dtype.element_ty == tl.float16:
        acc_dtype = tl.float32
    elif input_ptr.dtype.element_ty == tl.bfloat16:
        acc_dtype = tl.float32
    else:
        acc_dtype = input_ptr.dtype.element_ty

    pid_m = tl.program_id(0)

    # Accumulator for this row
    acc = tl.zeros([], dtype=acc_dtype)

    # Process row in chunks of TILE_N
    for start_n in range(0, N, TILE_N):
        n_offsets = start_n + tl.arange(0, TILE_N)
        inp_offsets = pid_m * N + n_offsets
        mask = n_offsets < N

        # Load chunk and accumulate
        chunk = tl.load(input_ptr + inp_offsets, mask=mask, other=0.0).to(acc_dtype)
        acc += tl.sum(chunk)

    # Store result
    tl.store(output_ptr + pid_m, acc.to(input_ptr.dtype.element_ty))


@triton.jit
def sum_kernel_outer(
    output_ptr,
    input_ptr,
    M,
    N,
    K,
    TILE_M: tl.constexpr,
    TILE_K: tl.constexpr,
):
    """
    Optimized kernel for reducing outer dimensions.
    Each thread block handles a portion of the output.
    """
    # Use float32 accumulation
    if input_ptr.dtype.element_ty == tl.float16:
        acc_dtype = tl.float32
    elif input_ptr.dtype.element_ty == tl.bfloat16:
        acc_dtype = tl.float32
    else:
        acc_dtype = input_ptr.dtype.element_ty

    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)

    n = pid_n
    k_start = pid_k * TILE_K
    k_offsets = k_start + tl.arange(0, TILE_K)
    k_mask = k_offsets < K

    # Accumulator for this output position
    acc = tl.zeros([TILE_K], dtype=acc_dtype)

    # Sum over M dimension
    for m_start in range(0, M, TILE_M):
        m_offsets = m_start + tl.arange(0, TILE_M)[:, None]
        m_mask = m_offsets < M

        # Compute input offsets: [M, N, K] layout
        inp_offsets = m_offsets * N * K + n * K + k_offsets[None, :]
        mask = m_mask & k_mask[None, :]

        # Load and accumulate
        vals = tl.load(input_ptr + inp_offsets, mask=mask, other=0.0).to(acc_dtype)
        acc += tl.sum(vals, axis=0)

    # Store results
    out_offsets = n * K + k_offsets
    tl.store(output_ptr + out_offsets, acc.to(input_ptr.dtype.element_ty), mask=k_mask)


@triton.jit
def sum_kernel_atomic(
    x_ptr,
    output_ptr,
    N,
    BLOCK_SIZE: tl.constexpr
):
    """
    Simple atomic reduction kernel for full tensor sum.
    """
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N

    # Load and sum this block
    values = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    block_sum = tl.sum(values)

    # Atomic add to output
    tl.atomic_add(output_ptr, block_sum)


# =============================================================================
# GPU OPERATIONS
# =============================================================================

@register_cuda("sum")
def reduce_sum(x, axis=None, keepdims=False):
    """
    Optimized reduce sum operation.
    Uses adaptive strategies based on tensor shape and reduction pattern.
    """
    # Handle bool tensors
    if hasattr(x, 'dtype'):
        import genesis
        if x.dtype == genesis.bool:
            zeros_int64 = zeros(x.shape, dtype=genesis.int64)
            x = add(zeros_int64, x)

    # x is a Storage object from the dispatcher
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
            output.fill(0.0)
            return output

        # Adaptive strategy based on size
        if n_elements <= 4096:
            # Small tensor: single kernel with atomic
            output.fill(0.0)
            block_size = min(1024, triton.next_power_of_2(n_elements))
            grid = (triton.cdiv(n_elements, block_size),)
            sum_kernel_atomic[grid](x, output, n_elements, block_size)
        else:
            # Large tensor: two-stage reduction
            # Adaptive block size (inspired by FlagGems)
            block_size = triton.next_power_of_2(min(1024, max(32, int(math.sqrt(n_elements)))))
            mid_size = triton.cdiv(n_elements, block_size)

            # Stage 1: partial sums
            mid = CUDAStorage((mid_size,), dtype=x.dtype)
            grid1 = (mid_size,)
            sum_kernel_stage1[grid1](x, mid, n_elements, block_size)

            # Stage 2: final reduction
            block_mid = triton.next_power_of_2(min(2048, mid_size))
            grid2 = (1,)
            sum_kernel_stage2[grid2](mid, output, mid_size, block_mid)

        return output

    # Normalize axis
    if isinstance(axis, int):
        axis = (axis,)
    axis = tuple(ax % ndim for ax in axis)

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
            tile_n = min(2048, triton.next_power_of_2(N)) if N > 32 else triton.next_power_of_2(N)

            grid = (M,)
            sum_kernel_inner[grid](output, x_flat, M, N, tile_n)

            return output.reshape(output_shape)

        elif ax == 0:
            # First dimension reduction
            M = shape[0]
            rest = functools_reduce(operator.mul, shape[1:], 1)

            output = CUDAStorage(shape[1:] if not keepdims else output_shape, dtype=x.dtype)

            # Reshape for efficient reduction
            x_2d = x.reshape((M, rest))
            out_flat = output.reshape((rest,))

            # Process each output element
            tile_m = min(256, triton.next_power_of_2(M))
            tile_k = min(128, triton.next_power_of_2(min(rest, 128)))

            grid = (rest, triton.cdiv(1, 1))
            sum_kernel_outer[grid](out_flat, x_2d, M, rest, 1, tile_m, tile_k)

            return output

        else:
            # Middle dimension reduction - need permute
            axes_to_keep = tuple(i for i in range(ndim) if i != ax)
            new_order = axes_to_keep + (ax,)
            x = x.permute(new_order)

            M = functools_reduce(operator.mul, [shape[i] for i in axes_to_keep], 1)
            N = shape[ax]

            x_2d = x.reshape((M, N))
            temp_output = CUDAStorage((M,), dtype=x.dtype)

            # Use inner kernel after permute
            tile_n = min(1024, triton.next_power_of_2(N)) if N > 32 else triton.next_power_of_2(N)

            grid = (M,)
            sum_kernel_inner[grid](temp_output, x_2d, M, N, tile_n)

            return temp_output.reshape(output_shape)

    # Multi-axis reduction
    axes_to_keep = tuple(i for i in range(ndim) if i not in axis)

    if not axes_to_keep:
        # Reducing all dimensions
        return reduce_sum(x, axis=None, keepdims=keepdims)

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
    tile_n = min(1024, triton.next_power_of_2(reduce_size)) if reduce_size > 32 else triton.next_power_of_2(reduce_size)

    grid = (keep_size,)
    sum_kernel_inner[grid](temp_output, x_2d, keep_size, reduce_size, tile_n)

    # Reshape to final output
    if keepdims:
        final_shape = list(shape)
        for i in axis:
            final_shape[i] = 1
        return temp_output.reshape(tuple(final_shape))
    else:
        final_shape = tuple(shape[i] for i in axes_to_keep)
        return temp_output.reshape(final_shape)


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
# MAX REDUCTION KERNELS
# =============================================================================

@triton.jit
def max_kernel_stage1(
    inp_ptr,
    mid_ptr,
    M,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Stage 1 of two-stage max reduction.
    """
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < M

    vals = tl.load(inp_ptr + offset, mask=mask, other=-float("inf"))
    max_val = tl.max(vals)

    tl.store(mid_ptr + pid, max_val)


@triton.jit
def max_kernel_stage2(
    mid_ptr,
    out_ptr,
    mid_size,
    BLOCK_SIZE: tl.constexpr
):
    """
    Stage 2 of two-stage max reduction.
    """
    offset = tl.arange(0, BLOCK_SIZE)
    mask = offset < mid_size

    vals = tl.load(mid_ptr + offset, mask=mask, other=-float("inf"))
    max_val = tl.max(vals)

    tl.store(out_ptr, max_val)


@triton.jit
def max_kernel_inner(
    output_ptr,
    input_ptr,
    M,
    N,
    TILE_N: tl.constexpr,
):
    """
    Max reduction for innermost dimension.
    """
    pid_m = tl.program_id(0)

    # Initialize with -inf
    max_val = tl.full([], -float("inf"), dtype=tl.float32)

    # Process row in chunks
    for start_n in range(0, N, TILE_N):
        n_offsets = start_n + tl.arange(0, TILE_N)
        inp_offsets = pid_m * N + n_offsets
        mask = n_offsets < N

        chunk = tl.load(input_ptr + inp_offsets, mask=mask, other=-float("inf"))
        max_val = tl.maximum(max_val, tl.max(chunk))

    tl.store(output_ptr + pid_m, max_val)


@triton.jit
def max_kernel_atomic(
    x_ptr,
    output_ptr,
    N,
    BLOCK_SIZE: tl.constexpr
):
    """
    Atomic max reduction kernel.
    """
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N

    values = tl.load(x_ptr + offsets, mask=mask, other=-float("inf"))
    block_max = tl.max(values)

    tl.atomic_max(output_ptr, block_max)


@register_cuda("max")
def reduce_max(x, axis=None, keepdims=False):
    """
    Optimized reduce max operation.
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
            output.fill(-float('inf'))
            return output

        # Adaptive strategy
        if n_elements <= 4096:
            # Small tensor: single kernel
            output.fill(-float('inf'))
            block_size = min(1024, triton.next_power_of_2(n_elements))
            grid = (triton.cdiv(n_elements, block_size),)
            max_kernel_atomic[grid](x, output, n_elements, block_size)
        else:
            # Large tensor: two-stage
            block_size = triton.next_power_of_2(min(1024, max(32, int(math.sqrt(n_elements)))))
            mid_size = triton.cdiv(n_elements, block_size)

            # Stage 1
            mid = CUDAStorage((mid_size,), dtype=x.dtype)
            grid1 = (mid_size,)
            max_kernel_stage1[grid1](x, mid, n_elements, block_size)

            # Stage 2
            block_mid = triton.next_power_of_2(min(2048, mid_size))
            grid2 = (1,)
            max_kernel_stage2[grid2](mid, output, mid_size, block_mid)

        return output

    # Normalize axis
    if isinstance(axis, int):
        axis = (axis,)
    axis = tuple(ax % ndim for ax in axis)

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

        # Innermost dimension
        if ax == ndim - 1:
            M = functools_reduce(operator.mul, shape[:-1], 1)
            N = shape[-1]

            output = CUDAStorage((M,), dtype=x.dtype)
            x_flat = x.reshape((M, N))

            tile_n = min(2048, triton.next_power_of_2(N)) if N > 32 else triton.next_power_of_2(N)

            grid = (M,)
            max_kernel_inner[grid](output, x_flat, M, N, tile_n)

            return output.reshape(output_shape)

        # For other axes, fallback to permute + inner reduction
        else:
            axes_to_keep = tuple(i for i in range(ndim) if i != ax)
            new_order = axes_to_keep + (ax,)
            x = x.permute(new_order)

            M = functools_reduce(operator.mul, [shape[i] for i in axes_to_keep], 1)
            N = shape[ax]

            x_2d = x.reshape((M, N))
            temp_output = CUDAStorage((M,), dtype=x.dtype)

            tile_n = min(1024, triton.next_power_of_2(N)) if N > 32 else triton.next_power_of_2(N)

            grid = (M,)
            max_kernel_inner[grid](temp_output, x_2d, M, N, tile_n)

            return temp_output.reshape(output_shape)

    # Multi-axis reduction - similar strategy as sum
    axes_to_keep = tuple(i for i in range(ndim) if i not in axis)

    if not axes_to_keep:
        return reduce_max(x, axis=None, keepdims=keepdims)

    # Permute and reshape
    new_order = axes_to_keep + axis
    x = x.permute(new_order)

    keep_size = functools_reduce(operator.mul, [shape[i] for i in axes_to_keep], 1)
    reduce_size = functools_reduce(operator.mul, [shape[i] for i in axis], 1)

    x_2d = x.reshape((keep_size, reduce_size))
    temp_output = CUDAStorage((keep_size,), dtype=x.dtype)

    tile_n = min(1024, triton.next_power_of_2(reduce_size)) if reduce_size > 32 else triton.next_power_of_2(reduce_size)

    grid = (keep_size,)
    max_kernel_inner[grid](temp_output, x_2d, keep_size, reduce_size, tile_n)

    # Reshape to final output
    if keepdims:
        final_shape = list(shape)
        for i in axis:
            final_shape[i] = 1
        return temp_output.reshape(tuple(final_shape))
    else:
        final_shape = tuple(shape[i] for i in axes_to_keep)
        return temp_output.reshape(final_shape)