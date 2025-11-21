"""
Fused RMSNorm implementation using Triton.

Combines all RMSNorm operations into a single kernel for better performance.
"""
import triton
import triton.language as tl
from ...function import Function
from ...tensor import Tensor
from ...init import empty_like, zeros_like


@triton.jit
def _fused_rmsnorm_kernel(
    x_ptr,
    weight_ptr,
    output_ptr,
    stride_x_row,
    stride_out_row,
    n_cols,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused RMSNorm kernel.

    Computes: output = (x / sqrt(mean(x^2) + eps)) * weight

    Each program processes one row (one sequence position or batch element).
    """
    # Get row index
    row = tl.program_id(0)

    # Compute row pointers
    x_row_ptr = x_ptr + row * stride_x_row
    out_row_ptr = output_ptr + row * stride_out_row

    # Process row in blocks
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    # Load input values
    x = tl.load(x_row_ptr + col_offsets, mask=mask, other=0.0)

    # Load weight (broadcasted across batch/sequence)
    weight = tl.load(weight_ptr + col_offsets, mask=mask, other=1.0)

    # Compute RMS: sqrt(mean(x^2))
    x_square = x * x
    x_square_sum = tl.sum(x_square)  # Sum across all columns in this row
    rms = tl.sqrt(x_square_sum / n_cols + eps)

    # Normalize and scale
    output = (x / rms) * weight

    # Store output
    tl.store(out_row_ptr + col_offsets, output, mask=mask)


@triton.jit
def _fused_rmsnorm_backward_kernel(
    grad_output_ptr,
    x_ptr,
    weight_ptr,
    grad_x_ptr,
    grad_weight_ptr,
    stride_row,
    stride_grad_row,
    n_cols,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused RMSNorm backward kernel.

    Computes gradients for both x and weight in a single pass.
    """
    row = tl.program_id(0)

    # Compute row pointers
    grad_out_row_ptr = grad_output_ptr + row * stride_grad_row
    x_row_ptr = x_ptr + row * stride_row
    grad_x_row_ptr = grad_x_ptr + row * stride_row

    # Column offsets
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    # Load values
    grad_out = tl.load(grad_out_row_ptr + col_offsets, mask=mask, other=0.0)
    x = tl.load(x_row_ptr + col_offsets, mask=mask, other=0.0)
    weight = tl.load(weight_ptr + col_offsets, mask=mask, other=1.0)

    # Recompute forward pass values
    x_square = x * x
    x_square_sum = tl.sum(x_square)  # Sum across all columns in this row
    rms = tl.sqrt(x_square_sum / n_cols + eps)
    x_normalized = x / rms

    # Gradient computation
    # d_loss/d_x = d_loss/d_output * d_output/d_x
    # where output = (x / rms) * weight

    # Part 1: gradient through weight multiplication
    grad_x_normalized = grad_out * weight

    # Part 2: gradient through normalization (x / rms)
    # This involves the chain rule through the RMS computation
    grad_rms = tl.sum(grad_x_normalized * (-x / (rms * rms)))  # Scalar gradient
    grad_x_square_sum = grad_rms * 0.5 / (rms * n_cols)
    grad_x_from_rms = 2.0 * x * grad_x_square_sum
    grad_x_from_norm = grad_x_normalized / rms

    grad_x = grad_x_from_norm + grad_x_from_rms

    # Gradient for weight: sum over batch/sequence dimension
    grad_weight = grad_out * x_normalized

    # Store gradients
    tl.store(grad_x_row_ptr + col_offsets, grad_x, mask=mask)

    # Atomic add for weight gradient (accumulated across rows)
    tl.atomic_add(grad_weight_ptr + col_offsets, grad_weight, mask=mask)


class FusedRMSNormFunction(Function):
    """
    Autograd function for fused RMSNorm.
    """

    @staticmethod
    def forward(ctx, x: Tensor, weight: Tensor, eps: float) -> Tensor:
        """
        Forward pass of fused RMSNorm.

        Args:
            x: Input tensor of shape (batch, seq_len, hidden_dim) or (batch*seq_len, hidden_dim)
            weight: Weight tensor of shape (hidden_dim,)
            eps: Epsilon for numerical stability

        Returns:
            Normalized tensor of same shape as x
        """
        # Flatten to 2D if needed
        original_shape = x.shape
        if len(x.shape) > 2:
            x = x.reshape(-1, x.shape[-1])

        n_rows, n_cols = x.shape

        # Allocate output
        output = empty_like(x)

        # Choose block size (must be power of 2 and >= n_cols)
        BLOCK_SIZE = triton.next_power_of_2(n_cols)
        if BLOCK_SIZE < 128:
            BLOCK_SIZE = 128

        # Ensure x is contiguous (weight should already be contiguous as a Parameter)
        x = x.contiguous()

        # Launch kernel
        grid = (n_rows,)
        _fused_rmsnorm_kernel[grid](
            x,
            weight,
            output,
            x.stride[0] if len(x.stride) > 1 else n_cols,
            output.stride[0] if len(output.stride) > 1 else n_cols,
            n_cols,
            eps,
            BLOCK_SIZE=BLOCK_SIZE,
        )

        # Reshape back to original shape
        if len(original_shape) > 2:
            output = output.reshape(original_shape)
            x = x.reshape(original_shape)

        # Save for backward
        ctx.save_for_backward(x, weight)
        ctx.eps = eps
        ctx.n_cols = n_cols

        return output

    @staticmethod
    def backward(ctx, grad_output: Tensor) -> tuple:
        """
        Backward pass of fused RMSNorm.

        Returns:
            Tuple of (grad_x, grad_weight, None) for (x, weight, eps)
        """
        x, weight = ctx.saved_tensors
        eps = ctx.eps
        n_cols = ctx.n_cols

        # Flatten to 2D if needed
        original_shape = grad_output.shape
        if len(grad_output.shape) > 2:
            grad_output = grad_output.reshape(-1, grad_output.shape[-1])
            x = x.reshape(-1, x.shape[-1])

        n_rows = grad_output.shape[0]

        # Allocate gradients
        grad_x = zeros_like(x)
        grad_weight = zeros_like(weight)

        # Choose block size
        BLOCK_SIZE = triton.next_power_of_2(n_cols)
        if BLOCK_SIZE < 128:
            BLOCK_SIZE = 128

        # Ensure contiguity (weight should already be contiguous)
        grad_output = grad_output.contiguous()
        x = x.contiguous()

        # Launch backward kernel
        grid = (n_rows,)
        _fused_rmsnorm_backward_kernel[grid](
            grad_output,
            x,
            weight,
            grad_x,
            grad_weight,
            x.stride[0] if len(x.stride) > 1 else n_cols,
            grad_output.stride[0] if len(grad_output.stride) > 1 else n_cols,
            n_cols,
            eps,
            BLOCK_SIZE=BLOCK_SIZE,
        )

        # Reshape back
        if len(original_shape) > 2:
            grad_x = grad_x.reshape(original_shape)

        return grad_x, grad_weight, None


def fused_rmsnorm(x: Tensor, weight: Tensor, eps: float = 1e-6) -> Tensor:
    """
    Fused RMSNorm operation.

    Computes: output = (x / sqrt(mean(x^2) + eps)) * weight

    This fused implementation is much faster than the decomposed version as it:
    1. Reduces kernel launches from 7 to 1
    2. Reduces memory traffic by keeping intermediate values in registers
    3. Improves cache locality

    Args:
        x: Input tensor of shape (..., hidden_dim)
        weight: Weight parameter of shape (hidden_dim,)
        eps: Small constant for numerical stability

    Returns:
        Normalized tensor of same shape as x

    Example:
        >>> x = genesis.randn(4, 2048, 1024, device="cuda")
        >>> weight = genesis.ones(1024, device="cuda")
        >>> output = fused_rmsnorm(x, weight, eps=1e-6)
        >>> output.shape
        (4, 2048, 1024)
    """
    return FusedRMSNormFunction.apply(x, weight, eps)

