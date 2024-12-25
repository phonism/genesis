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
        tx = x.data.data
        if tx.is_contiguous() is False:
            tx = tx.contiguous()
        y = torch.empty_like(tx)
        mask = torch.empty_like(tx, dtype=torch.int32)
        
        BLOCK_SIZE = 1024
        grid = lambda meta: (triton.cdiv(tx.numel(), meta["BLOCK_SIZE"]),)
        
        _dropout_forward[grid](tx, mask, y, prob, tx.numel(), BLOCK_SIZE=BLOCK_SIZE)
        
        ctx.save_for_backward(mask)
        return Tensor(y, device=x.device, dtype=x.dtype)

    @staticmethod
    def backward(ctx, dy):
        prob = ctx.prob
        mask, = ctx.saved_tensors
        tdy = dy.data.data
        if tdy.is_contiguous() is False:
            tdy = tdy.contiguous()
        dx = torch.empty_like(tdy)
        
        BLOCK_SIZE = 1024
        grid = lambda meta: (triton.cdiv(tdy.numel(), meta["BLOCK_SIZE"]),)
        
        _dropout_backward[grid](tdy, mask, dx, prob, tdy.numel(), BLOCK_SIZE=BLOCK_SIZE)
        
        return Tensor(dx, device=dy.device, requires_grad=False, dtype=dy.dtype), None

def dropout(x, prob):
    return TritonDropoutFunction.apply(x, prob)
