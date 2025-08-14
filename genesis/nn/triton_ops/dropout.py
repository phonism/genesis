from ...autograd import Function, NDArray, Tensor
from ...backend import array_api, NDArray

import genesis
import torch
import triton
import triton.language as tl

@triton.jit
def _dropout_forward(x_ptr, mask_ptr, y_ptr, prob, size, BLOCK_SIZE: tl.constexpr):
    idx = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = tl.rand(idx, size) > prob
    x = tl.load(x_ptr + idx, mask=idx < size, other=0.0)
    y = x * mask / (1.0 - prob)
    tl.store(y_ptr + idx, y, mask=idx < size)
    tl.store(mask_ptr + idx, mask, mask=idx < size)

@triton.jit
def _dropout_backward(dy_ptr, mask_ptr, dx_ptr, prob, size, BLOCK_SIZE: tl.constexpr):
    idx = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    dy = tl.load(dy_ptr + idx, mask=idx < size, other=0.0)
    mask = tl.load(mask_ptr + idx, mask=idx < size, other=0.0)
    dx = dy * mask / (1.0 - prob)
    tl.store(dx_ptr + idx, dx, mask=idx < size)

class DropoutFunction(Function):
    @staticmethod
    def forward(ctx, x, prob):
        ctx.prob = prob
        
        # Use Genesis tensors directly, just like FusedLayerNorm
        device = x.device
        
        # Create output and mask tensors using Genesis
        y = genesis.empty_like(x)
        mask = genesis.empty(x.shape, dtype='int32', device=device)
        
        # Ensure contiguity
        x_contiguous = x.contiguous()
        
        BLOCK_SIZE = 1024
        grid = lambda meta: (triton.cdiv(x.numel(), meta["BLOCK_SIZE"]),)
        
        # Pass Genesis tensors directly to Triton kernel
        _dropout_forward[grid](
            x_contiguous, mask, y, prob, x.numel(), 
            BLOCK_SIZE=BLOCK_SIZE
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
        
        BLOCK_SIZE = 1024
        grid = lambda meta: (triton.cdiv(dy.numel(), meta["BLOCK_SIZE"]),)
        
        # Pass Genesis tensors directly to Triton kernel
        _dropout_backward[grid](
            dy_contiguous, mask, dx, prob, dy.numel(), 
            BLOCK_SIZE=BLOCK_SIZE
        )
        
        return dx, None

def dropout(x, prob):
    return DropoutFunction.apply(x, prob)
