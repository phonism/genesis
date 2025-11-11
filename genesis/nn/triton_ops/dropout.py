from ...function import Function
from ...tensor import Tensor

import genesis
import triton
import triton.language as tl
import time
import os

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
def _dropout_forward(x_ptr, mask_ptr, y_ptr, prob, size, seed, BLOCK_SIZE: tl.constexpr):
    """
    Optimized dropout forward kernel with autotune and proper random seeding.
    
    Args:
        x_ptr: Input tensor pointer
        mask_ptr: Output mask tensor pointer  
        y_ptr: Output tensor pointer
        prob: Dropout probability
        size: Total tensor size
        seed: Random seed for this operation
        BLOCK_SIZE: Block size (auto-tuned)
    """
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < size
    
    # Generate random values for dropout mask
    # Use global seed with offsets - Triton handles uniqueness per offset
    random_values = tl.rand(seed, offsets)
    dropout_mask = random_values > prob
    
    # Load input data
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Apply dropout: keep elements with probability (1-prob), scale by 1/(1-prob)
    scale = 1.0 / (1.0 - prob)
    y = tl.where(dropout_mask, x * scale, 0.0)
    
    # Store results
    tl.store(y_ptr + offsets, y, mask=mask)
    tl.store(mask_ptr + offsets, dropout_mask.to(tl.int32), mask=mask)

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
def _dropout_backward(dy_ptr, mask_ptr, dx_ptr, prob, size, BLOCK_SIZE: tl.constexpr):
    """
    Optimized dropout backward kernel with autotune.
    
    Args:
        dy_ptr: Input gradient tensor pointer
        mask_ptr: Dropout mask tensor pointer
        dx_ptr: Output gradient tensor pointer  
        prob: Dropout probability
        size: Total tensor size
        BLOCK_SIZE: Block size (auto-tuned)
    """
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < size
    
    # Load gradients and dropout mask
    dy = tl.load(dy_ptr + offsets, mask=mask, other=0.0)
    dropout_mask = tl.load(mask_ptr + offsets, mask=mask, other=0.0)
    
    # Apply dropout to gradients with same scale as forward
    scale = 1.0 / (1.0 - prob)
    dx = dy * dropout_mask * scale
    
    # Store result gradients
    tl.store(dx_ptr + offsets, dx, mask=mask)

class DropoutFunction(Function):
    @staticmethod
    def forward(ctx, x, prob):
        """
        Optimized dropout forward with autotune and Genesis-style RNG.
        """
        ctx.prob = prob
        
        # Use Genesis tensors directly
        device = x.device
        
        # Create output and mask tensors using Genesis
        y = genesis.empty_like(x)
        mask = genesis.empty(x.shape, dtype='int32', device=device)
        
        # Ensure contiguity for optimal memory access
        x_contiguous = x.contiguous()

        # Use unified RNG system for reproducibility
        seed = genesis.random.default_generator().next_seed()
        
        # Grid function for autotune - block size will be auto-selected
        grid = lambda meta: (triton.cdiv(x.numel(), meta["BLOCK_SIZE"]),)
        
        # Call optimized Triton kernel (no .data needed for Genesis tensors)
        _dropout_forward[grid](
            x_contiguous, mask, y, prob, x.numel(), seed
        )
        
        ctx.save_for_backward(mask)
        return y

    @staticmethod
    def backward(ctx, dy):
        prob = ctx.prob
        mask, = ctx.saved_tensors
        
        # Use Genesis tensors directly
        device = dy.device
        
        # Create output tensor using Genesis
        dx = genesis.empty_like(dy)

        # Ensure contiguity
        dy_contiguous = dy.contiguous()

        # Grid function for autotune - BLOCK_SIZE auto-selected
        grid = lambda meta: (triton.cdiv(dy.numel(), meta["BLOCK_SIZE"]),)

        # Pass Genesis tensors directly to Triton kernel (no manual BLOCK_SIZE)
        _dropout_backward[grid](
            dy_contiguous, mask, dx, prob, dy.numel()
        )

        return dx, None

def dropout(x, p=0.5, training=True, inplace=False):
    """Apply dropout to input tensor.

    Standard dropout regularization with configurable dropout rate.

    Args:
        x: Input tensor
        p: Probability of an element to be zeroed (default: 0.5)
        training: Apply dropout if True (default: True)
        inplace: Whether to modify input in-place (default: False, not implemented)

    Returns:
        Tensor: Output tensor with dropout applied
    """
    if not training or p == 0:
        return x
    return DropoutFunction.apply(x, p)
