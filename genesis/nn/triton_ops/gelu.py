"""Optimized GELU activation using Triton."""

from ...function import Function
from ...tensor import Tensor

import genesis
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 256}),
        triton.Config({"BLOCK_SIZE": 512}),
        triton.Config({"BLOCK_SIZE": 1024}),
        triton.Config({"BLOCK_SIZE": 2048}),
        triton.Config({"BLOCK_SIZE": 4096}),
    ],
    key=["size"],
)
@triton.jit
def _gelu_forward(x_ptr, y_ptr, size, BLOCK_SIZE: tl.constexpr):
    """
    Optimized GELU forward kernel with autotune.

    GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))

    Args:
        x_ptr: Input tensor pointer
        y_ptr: Output tensor pointer
        size: Total tensor size
        BLOCK_SIZE: Block size (auto-tuned)
    """
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < size

    # Load input data
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)

    # Constants for GELU approximation
    sqrt_2_over_pi = 0.7978845608028654  # sqrt(2/π)
    coeff = 0.044715

    # Compute GELU: 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
    x_cubed = x * x * x
    inner = x + coeff * x_cubed
    inner_scaled = sqrt_2_over_pi * inner

    # Compute tanh with numerical stability
    # For large positive x: tanh(x) ≈ 1
    # For large negative x: tanh(x) ≈ -1
    # Use: tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
    #              = (1 - exp(-2x)) / (1 + exp(-2x))  [for x > 0, more stable]
    #              = (exp(2x) - 1) / (exp(2x) + 1)    [for x < 0, more stable]
    abs_x = tl.abs(inner_scaled)
    exp_2x = tl.exp(-2.0 * abs_x)
    tanh_abs = (1.0 - exp_2x) / (1.0 + exp_2x)
    # Apply sign
    tanh_result = tl.where(inner_scaled >= 0.0, tanh_abs, -tanh_abs)

    y = 0.5 * x * (1.0 + tanh_result)

    # Store results
    tl.store(y_ptr + offsets, y, mask=mask)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 256}),
        triton.Config({"BLOCK_SIZE": 512}),
        triton.Config({"BLOCK_SIZE": 1024}),
        triton.Config({"BLOCK_SIZE": 2048}),
        triton.Config({"BLOCK_SIZE": 4096}),
    ],
    key=["size"],
)
@triton.jit
def _gelu_backward(x_ptr, dy_ptr, dx_ptr, size, BLOCK_SIZE: tl.constexpr):
    """
    Optimized GELU backward kernel with autotune.

    d/dx GELU(x) = 0.5 * (1 + tanh(z)) + 0.5 * x * sech²(z) * dz/dx
    where z = sqrt(2/π) * (x + 0.044715 * x³)
    and dz/dx = sqrt(2/π) * (1 + 3 * 0.044715 * x²)

    Args:
        x_ptr: Input tensor pointer (from forward pass)
        dy_ptr: Input gradient tensor pointer
        dx_ptr: Output gradient tensor pointer
        size: Total tensor size
        BLOCK_SIZE: Block size (auto-tuned)
    """
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < size

    # Load input and gradients
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    dy = tl.load(dy_ptr + offsets, mask=mask, other=0.0)

    # Constants
    sqrt_2_over_pi = 0.7978845608028654
    coeff = 0.044715

    # Recompute forward pass components
    x_squared = x * x
    x_cubed = x * x_squared
    inner = x + coeff * x_cubed
    inner_scaled = sqrt_2_over_pi * inner

    # Compute tanh with numerical stability (same as forward)
    abs_x = tl.abs(inner_scaled)
    exp_2x = tl.exp(-2.0 * abs_x)
    tanh_abs = (1.0 - exp_2x) / (1.0 + exp_2x)
    tanh_result = tl.where(inner_scaled >= 0.0, tanh_abs, -tanh_abs)

    # Compute sech²(z) = 1 - tanh²(z)
    tanh_squared = tanh_result * tanh_result
    sech_squared = 1.0 - tanh_squared

    # Compute dz/dx = sqrt(2/π) * (1 + 3 * 0.044715 * x²)
    inner_deriv = 1.0 + 3.0 * coeff * x_squared
    inner_deriv_scaled = sqrt_2_over_pi * inner_deriv

    # GELU gradient: 0.5 * (1 + tanh(z)) + 0.5 * x * sech²(z) * dz/dx
    term1 = 0.5 * (1.0 + tanh_result)
    term2 = 0.5 * x * sech_squared * inner_deriv_scaled
    gelu_grad = term1 + term2

    # Multiply by upstream gradient
    dx = dy * gelu_grad

    # Store result gradients
    tl.store(dx_ptr + offsets, dx, mask=mask)


class GELUFunction(Function):
    @staticmethod
    def forward(ctx, x):
        """
        Optimized GELU forward using Triton.
        """
        # Use Genesis tensors directly
        device = x.device

        # Create output tensor using Genesis
        y = genesis.empty_like(x)

        # Ensure contiguity for optimal memory access
        x_contiguous = x.contiguous()

        # Grid function for autotune
        grid = lambda meta: (triton.cdiv(x.numel(), meta["BLOCK_SIZE"]),)

        # Call optimized Triton kernel
        _gelu_forward[grid](
            x_contiguous, y, x.numel()
        )

        ctx.save_for_backward(x_contiguous)
        return y

    @staticmethod
    def backward(ctx, dy):
        x, = ctx.saved_tensors

        # Use Genesis tensors directly
        device = dy.device

        # Create output tensor using Genesis
        dx = genesis.empty_like(dy)

        # Ensure contiguity
        dy_contiguous = dy.contiguous()

        # Grid function for autotune
        grid = lambda meta: (triton.cdiv(dy.numel(), meta["BLOCK_SIZE"]),)

        # Pass Genesis tensors directly to Triton kernel
        _gelu_backward[grid](
            x, dy_contiguous, dx, dy.numel()
        )

        return (dx,)


def gelu(x):
    """Apply GELU (Gaussian Error Linear Unit) activation function.

    GELU(x) = x * Φ(x) where Φ is the cumulative distribution function
    of the standard normal distribution.

    This implementation uses the tanh approximation for efficiency:
    GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))

    Args:
        x: Input tensor

    Returns:
        Tensor: Output tensor with GELU activation applied
    """
    return GELUFunction.apply(x)
