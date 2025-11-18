"""Fused gradient checking and unscaling kernels for AMP training."""

import triton
import triton.language as tl


@triton.jit
def fused_unscale_and_check_kernel(
    grad_ptr,
    output_ptr,
    inv_scale,
    grad_norm_sq_ptr,
    size,
    BLOCK_SIZE: tl.constexpr,
    CONVERT_TO_FP32: tl.constexpr,
):
    """Fused kernel that unscales gradient and computes squared norm in one pass.

    This kernel performs three operations in a single pass:
    1. Convert gradient to FP32 (if needed) for accurate accumulation
    2. Compute squared L2 norm contribution
    3. Unscale gradient and write back

    Args:
        grad_ptr: Input gradient tensor pointer
        output_ptr: Output unscaled gradient tensor pointer
        inv_scale: Inverse scale factor (1.0 / scale)
        grad_norm_sq_ptr: Output pointer for squared norm (single value, atomic add)
        size: Total number of elements
        BLOCK_SIZE: Number of elements per block
        CONVERT_TO_FP32: Whether to convert FP16 gradients to FP32 for accumulation
    """
    # Get block offset
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < size

    # Load gradient
    grad = tl.load(grad_ptr + offset, mask=mask, other=0.0)

    # Convert to FP32 for accurate norm computation (if needed)
    if CONVERT_TO_FP32:
        grad_fp32 = grad.to(tl.float32)
    else:
        grad_fp32 = grad

    # Compute squared norm contribution for this block
    grad_sq = grad_fp32 * grad_fp32
    block_norm_sq = tl.sum(grad_sq)

    # Atomic add to global norm (this is safe and efficient in Triton)
    if pid == 0 or tl.program_id(0) < tl.num_programs(0):
        tl.atomic_add(grad_norm_sq_ptr, block_norm_sq)

    # Unscale gradient
    unscaled_grad = grad * inv_scale

    # Store unscaled gradient
    tl.store(output_ptr + offset, unscaled_grad, mask=mask)


@triton.jit
def check_finite_kernel(
    norm_sq_ptr,
    found_inf_ptr,
):
    """Check if squared norm is finite (single-threaded kernel).

    Args:
        norm_sq_ptr: Pointer to squared norm value
        found_inf_ptr: Output pointer for inf/nan flag (0 = finite, 1 = inf/nan)
    """
    # This is a single-value kernel, only first thread does the check
    if tl.program_id(0) == 0:
        norm_sq = tl.load(norm_sq_ptr)

        # Check if finite: not (inf or nan)
        is_finite = tl.isfinite(norm_sq)

        # Store result: 1 if found inf/nan, 0 if finite
        found_inf = tl.where(is_finite, 0.0, 1.0)
        tl.store(found_inf_ptr, found_inf)
