"""Optimized SiLU (Swish) activation using Triton.

SiLU activation: f(x) = x * sigmoid(x) = x / (1 + exp(-x))

This implementation preserves input dtype (FP16/FP32) for optimal performance
in mixed precision training, following standard framework behavior.
"""

from ...function import Function
from ...tensor import Tensor
from ...amp import AMPPolicy

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
def _silu_forward_kernel(x_ptr, y_ptr, size, BLOCK_SIZE: tl.constexpr):
    """
    Optimized SiLU forward kernel.

    Computes: y = x * sigmoid(x) = x / (1 + exp(-x))

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

    # Load input and convert to FP32 for computation
    x_load = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    x = x_load.to(tl.float32)

    # Compute sigmoid with numerical stability
    # For x >= 0: sigmoid(x) = 1 / (1 + exp(-x))
    # For x < 0:  sigmoid(x) = exp(x) / (1 + exp(x))
    # This avoids overflow in exp() computation

    # Use stable sigmoid computation
    neg_abs_x = -tl.abs(x)
    exp_neg_abs = tl.exp(neg_abs_x)
    # sigmoid = 1 / (1 + exp(-|x|)) for x >= 0
    # sigmoid = exp(-|x|) / (1 + exp(-|x|)) for x < 0
    sigmoid = tl.where(
        x >= 0.0,
        1.0 / (1.0 + exp_neg_abs),
        exp_neg_abs / (1.0 + exp_neg_abs)
    )

    # SiLU = x * sigmoid(x)
    y = x * sigmoid

    # Convert back to original dtype and store
    y_store = y.to(x_load.dtype)
    tl.store(y_ptr + offsets, y_store, mask=mask)


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
def _silu_backward_kernel(
    grad_output_ptr, x_ptr, grad_input_ptr, size, BLOCK_SIZE: tl.constexpr
):
    """
    Optimized SiLU backward kernel.

    Computes: grad_input = grad_output * (sigmoid(x) * (1 + x * (1 - sigmoid(x))))
              = grad_output * sigmoid(x) * (1 + x - x * sigmoid(x))

    Args:
        grad_output_ptr: Gradient from upstream
        x_ptr: Original input tensor
        grad_input_ptr: Output gradient tensor
        size: Total tensor size
        BLOCK_SIZE: Block size (auto-tuned)
    """
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < size

    # Load data and convert to FP32 for computation
    grad_output_load = tl.load(grad_output_ptr + offsets, mask=mask, other=0.0)
    x_load = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    grad_output = grad_output_load.to(tl.float32)
    x = x_load.to(tl.float32)

    # Compute sigmoid with numerical stability (same as forward)
    neg_abs_x = -tl.abs(x)
    exp_neg_abs = tl.exp(neg_abs_x)
    sigmoid = tl.where(
        x >= 0.0,
        1.0 / (1.0 + exp_neg_abs),
        exp_neg_abs / (1.0 + exp_neg_abs)
    )

    # Derivative: sigmoid(x) * (1 + x - x * sigmoid(x))
    #           = sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x))
    silu_grad = sigmoid * (1.0 + x * (1.0 - sigmoid))

    # Chain rule
    grad_input = grad_output * silu_grad

    # Convert back to original dtype and store
    grad_input_store = grad_input.to(grad_output_load.dtype)
    tl.store(grad_input_ptr + offsets, grad_input_store, mask=mask)


class SiLUFunction(Function):
    """
    SiLU activation function with Triton-optimized kernels.

    Preserves input dtype for efficient mixed precision training.
    """

    # SiLU is numerically stable in FP16, inherits input dtype
    amp_policy = AMPPolicy.PRESERVE

    @staticmethod
    def forward(ctx, x: Tensor) -> Tensor:
        """
        Forward pass: SiLU(x) = x * sigmoid(x).

        Args:
            ctx: Context for saving tensors
            x: Input tensor

        Returns:
            Output tensor with same dtype as input
        """
        # Save input for backward (preserves dtype)
        ctx.save_for_backward(x)

        # Create output tensor (same dtype as input)
        y = genesis.empty_like(x)

        # Get device and ensure contiguous
        device = x.device
        x_contiguous = x.contiguous()

        # Launch kernel
        size = x.numel()
        grid = lambda meta: (triton.cdiv(size, meta["BLOCK_SIZE"]),)

        _silu_forward_kernel[grid](
            x_contiguous, y, size
        )

        y.requires_grad = x.requires_grad
        return y

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        """
        Backward pass: compute gradient w.r.t. input.

        Args:
            ctx: Context with saved tensors
            grad_output: Gradient from upstream

        Returns:
            Gradient w.r.t. input (same dtype as grad_output)
        """
        (x,) = ctx.saved_tensors

        # Create gradient tensor (same dtype as grad_output)
        grad_input = genesis.empty_like(x)

        # Ensure contiguous
        grad_output_contiguous = grad_output.contiguous()
        x_contiguous = x.contiguous()

        # Launch kernel
        size = x.numel()
        grid = lambda meta: (triton.cdiv(size, meta["BLOCK_SIZE"]),)

        _silu_backward_kernel[grid](
            grad_output_contiguous, x_contiguous, grad_input, size
        )

        grad_input.requires_grad = False
        return (grad_input,)


def silu(x: Tensor) -> Tensor:
    """
    Apply SiLU (Swish) activation function.

    SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x))

    This function preserves input dtype (FP16/FP32/BF16) for optimal
    performance in mixed precision training.

    Args:
        x: Input tensor

    Returns:
        Output tensor with same dtype as input

    Example:
        >>> x = genesis.randn(10, 10, dtype=genesis.float16, device='cuda')
        >>> y = silu(x)
        >>> y.dtype  # float16 - dtype preserved
    """
    if x.device.type.name == "CUDA" and genesis.use_triton:
        return SiLUFunction.apply(x)
    else:
        # CPU fallback: x * sigmoid(x)
        # Numerically stable sigmoid
        import genesis.nn.functional as F
        neg_x = -x
        # For numerical stability
        sigmoid = 1.0 / (1.0 + F.exp(neg_x))
        return x * sigmoid
