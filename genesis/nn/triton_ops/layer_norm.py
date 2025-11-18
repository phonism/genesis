"""Fused LayerNorm implementation using Triton for high performance.

This implementation fuses the entire LayerNorm operation (mean, variance, normalize, scale, shift)
into a single kernel, significantly reducing memory bandwidth and kernel launch overhead.
"""

from ...function import Function
from ...tensor import Tensor
from ...amp import AMPPolicy
import genesis
import triton
import triton.language as tl


@triton.jit
def _layer_norm_forward_kernel(
    x_ptr,  # Input pointer
    y_ptr,  # Output pointer
    weight_ptr,  # Weight pointer (gamma)
    bias_ptr,  # Bias pointer (beta)
    mean_ptr,  # Mean pointer (for backward)
    rstd_ptr,  # Reciprocal std pointer (for backward)
    N,  # Normalized dimension size
    eps,  # Epsilon for numerical stability
    BLOCK_SIZE: tl.constexpr,  # Block size (must be power of 2)
):
    """
    Fused LayerNorm forward kernel.

    Each program processes one row (last dimension).
    Computes: y = (x - mean(x)) / sqrt(var(x) + eps) * weight + bias
    """
    # Get row index
    row_idx = tl.program_id(0)

    # Compute row offset
    row_start = row_idx * N

    # Load entire row
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N
    x = tl.load(x_ptr + row_start + cols, mask=mask, other=0.0).to(tl.float32)

    # Compute mean
    mean = tl.sum(x, axis=0) / N

    # Compute variance
    x_centered = x - mean
    var = tl.sum(x_centered * x_centered, axis=0) / N

    # Compute reciprocal standard deviation
    rstd = 1.0 / tl.sqrt(var + eps)

    # Normalize
    x_normed = x_centered * rstd

    # Load weight and bias
    weight = tl.load(weight_ptr + cols, mask=mask, other=0.0)
    bias = tl.load(bias_ptr + cols, mask=mask, other=0.0)

    # Scale and shift
    y = x_normed * weight + bias

    # Store output
    tl.store(y_ptr + row_start + cols, y, mask=mask)

    # Store mean and rstd for backward (only first thread)
    tl.store(mean_ptr + row_idx, mean)
    tl.store(rstd_ptr + row_idx, rstd)


@triton.jit
def _layer_norm_backward_kernel(
    dy_ptr,  # Gradient of output
    x_ptr,  # Input (from forward)
    weight_ptr,  # Weight
    mean_ptr,  # Mean (from forward)
    rstd_ptr,  # Reciprocal std (from forward)
    dx_ptr,  # Gradient of input (output)
    dweight_ptr,  # Gradient of weight (output, accumulated)
    dbias_ptr,  # Gradient of bias (output, accumulated)
    N,  # Normalized dimension size
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused LayerNorm backward kernel.

    Computes gradients for input, weight, and bias.
    """
    # Get row index
    row_idx = tl.program_id(0)
    row_start = row_idx * N

    # Load data
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N

    dy = tl.load(dy_ptr + row_start + cols, mask=mask, other=0.0).to(tl.float32)
    x = tl.load(x_ptr + row_start + cols, mask=mask, other=0.0).to(tl.float32)
    weight = tl.load(weight_ptr + cols, mask=mask, other=0.0)
    mean = tl.load(mean_ptr + row_idx)
    rstd = tl.load(rstd_ptr + row_idx)

    # Compute x_hat (normalized x)
    x_centered = x - mean
    x_hat = x_centered * rstd

    # Gradient w.r.t. weight and bias (sum across batch)
    dweight = dy * x_hat
    dbias = dy

    # Atomic add for weight and bias gradients (accumulate across all rows)
    tl.atomic_add(dweight_ptr + cols, dweight, mask=mask)
    tl.atomic_add(dbias_ptr + cols, dbias, mask=mask)

    # Gradient w.r.t. input
    # Using efficient formulation: dx = (dy * weight - mean(dy * weight) - x_hat * mean(dy * weight * x_hat)) * rstd
    dy_weight = dy * weight

    # Mean of dy_weight
    mean_dy_weight = tl.sum(dy_weight, axis=0) / N

    # Mean of dy_weight * x_hat
    mean_dy_weight_xhat = tl.sum(dy_weight * x_hat, axis=0) / N

    # Compute dx
    dx = (dy_weight - mean_dy_weight - x_hat * mean_dy_weight_xhat) * rstd

    # Store dx
    tl.store(dx_ptr + row_start + cols, dx, mask=mask)


class FusedLayerNormFunction(Function):
    """Fused LayerNorm function with optimized Triton kernels."""

    # Use PRESERVE to allow AMP to handle dtype conversions naturally
    # The kernel itself is dtype-agnostic and works well in both FP16 and FP32
    amp_policy = AMPPolicy.PRESERVE

    @staticmethod
    def forward(ctx, x, weight, bias, eps=1e-5):
        """
        Fused LayerNorm forward pass.

        Args:
            x: Input tensor of shape (..., N)
            weight: Weight tensor of shape (N,)
            bias: Bias tensor of shape (N,)
            eps: Epsilon for numerical stability

        Returns:
            Normalized output tensor
        """
        # Get shape info
        original_shape = x.shape
        N = original_shape[-1]

        # Flatten to 2D: (batch * seq, hidden)
        x_2d = x.reshape(-1, N)
        num_rows = x_2d.shape[0]

        # Ensure contiguity
        x_2d = x_2d.contiguous()
        weight = weight.contiguous()
        bias = bias.contiguous()

        # Create output tensors
        y = genesis.empty_like(x_2d)
        mean = genesis.empty((num_rows,), device=x.device, dtype=genesis.float32)
        rstd = genesis.empty((num_rows,), device=x.device, dtype=genesis.float32)

        # Determine block size (must be power of 2 and >= N)
        BLOCK_SIZE = triton.next_power_of_2(N)

        # Launch kernel
        grid = (num_rows,)
        _layer_norm_forward_kernel[grid](
            x_2d, y, weight, bias, mean, rstd,
            N, eps, BLOCK_SIZE
        )

        # Reshape output back to original shape
        y = y.reshape(original_shape)

        # Save for backward
        ctx.save_for_backward(x_2d, weight, mean, rstd)
        ctx.N = N
        ctx.BLOCK_SIZE = BLOCK_SIZE
        ctx.original_shape = original_shape

        return y

    @staticmethod
    def backward(ctx, dy):
        """
        Fused LayerNorm backward pass.

        Args:
            dy: Gradient of output

        Returns:
            Tuple of (dx, dweight, dbias, None)
        """
        x_2d, weight, mean, rstd = ctx.saved_tensors
        N = ctx.N
        BLOCK_SIZE = ctx.BLOCK_SIZE
        original_shape = ctx.original_shape

        # Flatten dy to 2D
        dy_2d = dy.reshape(-1, N).contiguous()
        num_rows = dy_2d.shape[0]

        # Create output tensors
        dx = genesis.empty_like(x_2d)
        dweight = genesis.zeros((N,), device=weight.device, dtype=weight.dtype)
        dbias = genesis.zeros((N,), device=weight.device, dtype=weight.dtype)

        # Launch kernel
        grid = (num_rows,)
        _layer_norm_backward_kernel[grid](
            dy_2d, x_2d, weight, mean, rstd,
            dx, dweight, dbias,
            N, BLOCK_SIZE
        )

        # Reshape dx back to original shape
        dx = dx.reshape(original_shape)

        return dx, dweight, dbias, None


def fused_layer_norm(x, weight, bias, eps=1e-5):
    """
    Apply fused layer normalization.

    Normalizes the last dimension of input using a fused Triton kernel for optimal performance.
    This is significantly faster than the decomposed implementation, especially for FP16.

    Args:
        x: Input tensor of shape (..., N)
        weight: Scale parameter of shape (N,)
        bias: Shift parameter of shape (N,)
        eps: Small constant for numerical stability (default: 1e-5)

    Returns:
        Normalized tensor of same shape as input

    Example:
        >>> x = genesis.randn(2, 1024, 768)
        >>> weight = genesis.ones(768)
        >>> bias = genesis.zeros(768)
        >>> y = fused_layer_norm(x, weight, bias)
    """
    return FusedLayerNormFunction.apply(x, weight, bias, eps)
