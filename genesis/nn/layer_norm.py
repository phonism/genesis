"""Layer Normalization implementation using Triton kernels."""
# Global operator table.
from numbers import Number
from typing import Optional, List
import numpy
import genesis
import triton
import triton.language as tl
from ..autograd import Function, NDArray, Tensor
from genesis import init
from ..backend import array_api, NDArray

@triton.jit
def _layer_norm_fwd_kernel(X, Y, W, B, Mean, Rstd, stride, N, eps, BLOCK_SIZE: tl.constexpr,):
    # Map the program id to the row of X and Y it should compute.
    row = tl.program_id(0)
    Y += row * stride
    X += row * stride
    # Compute mean
    mean = 0
    _mean = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        a = tl.load(X + cols, mask=cols < N, other=0.).to(tl.float32)
        _mean += a
    mean = tl.sum(_mean, axis=0) / N
    # Compute variance
    _var = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        x = tl.load(X + cols, mask=cols < N, other=0.).to(tl.float32)
        x = tl.where(cols < N, x - mean, 0.)
        _var += x * x
    var = tl.sum(_var, axis=0) / N
    rstd = 1 / tl.sqrt(var + eps)
    # Write mean / rstd
    tl.store(Mean + row, mean)
    tl.store(Rstd + row, rstd)
    # Normalize and apply linear transformation
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        w = tl.load(W + cols, mask=mask)
        b = tl.load(B + cols, mask=mask)
        x = tl.load(X + cols, mask=mask, other=0.).to(tl.float32)
        x_hat = (x - mean) * rstd
        y = x_hat * w + b
        # Write output
        tl.store(Y + cols, y, mask=mask)

@triton.jit
def _layer_norm_bwd_dx_kernel(DX, DY, DW, DB, X, W, Mean, Rstd, Lock, stride, 
        N, GROUP_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr):
    # Map the program id to the elements of X, DX, and DY it should compute.
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE_N)
    mask = cols < N
    X += row * stride
    DY += row * stride
    DX += row * stride
    # Offset locks and weights/biases gradient pointer for parallel reduction
    lock_id = row % GROUP_SIZE_M
    Lock += lock_id
    Count = Lock + GROUP_SIZE_M
    DW = DW + lock_id * N + cols
    DB = DB + lock_id * N + cols
    # Load data to SRAM
    x = tl.load(X + cols, mask=mask, other=0).to(tl.float32)
    dy = tl.load(DY + cols, mask=mask, other=0).to(tl.float32)
    w = tl.load(W + cols, mask=mask).to(tl.float32)
    mean = tl.load(Mean + row)
    rstd = tl.load(Rstd + row)
    # Compute dx
    xhat = (x - mean) * rstd
    wdy = w * dy
    xhat = tl.where(mask, xhat, 0.)
    wdy = tl.where(mask, wdy, 0.)
    c1 = tl.sum(xhat * wdy, axis=0) / N
    c2 = tl.sum(wdy, axis=0) / N
    dx = (wdy - (xhat * c1 + c2)) * rstd
    # Write dx
    tl.store(DX + cols, dx, mask=mask)
    # Accumulate partial sums for dw/db
    partial_dw = (dy * xhat).to(w.dtype)
    partial_db = (dy).to(w.dtype)
    while tl.atomic_cas(Lock, 0, 1) == 1:
        pass
    count = tl.load(Count)
    # First store doesn't accumulate
    if count == 0:
        tl.atomic_xchg(Count, 1)
    else:
        partial_dw += tl.load(DW, mask=mask)
        partial_db += tl.load(DB, mask=mask)
    tl.store(DW, partial_dw, mask=mask)
    tl.store(DB, partial_db, mask=mask)
    # Release the lock
    tl.atomic_xchg(Lock, 0)

@triton.jit
def _layer_norm_bwd_dwdb_kernel(DW, DB, FINAL_DW, FINAL_DB, M, N, 
        BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr):
    # Map the program id to the elements of DW and DB it should compute.
    pid = tl.program_id(0)
    cols = pid * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    dw = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    db = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    # Iterate through the rows of DW and DB to sum the partial sums.
    for i in range(0, M, BLOCK_SIZE_M):
        rows = i + tl.arange(0, BLOCK_SIZE_M)
        mask = (rows[:, None] < M) & (cols[None, :] < N)
        offs = rows[:, None] * N + cols[None, :]
        dw += tl.load(DW + offs, mask=mask, other=0.)
        db += tl.load(DB + offs, mask=mask, other=0.)
    # Write the final sum to the output.
    sum_dw = tl.sum(dw, axis=0)
    sum_db = tl.sum(db, axis=0)
    tl.store(FINAL_DW + cols, sum_dw, mask=cols < N)
    tl.store(FINAL_DB + cols, sum_db, mask=cols < N)

class FusedLayerNormFunction(Function):
    @staticmethod
    def forward(ctx, x, weight, bias, eps=1e-6):
        # Use Genesis tensors directly, just like PyTorch!
        device = x.device
        
        # Use Genesis's empty_like (newly added function)
        y = genesis.empty_like(x)
        
        # Get reshaped tensor, keep Genesis Tensor type
        x_reshaped = x.reshape(-1, x.shape[-1])
        M, N = x_reshaped.shape
        
        # Create mean and rstd tensors
        mean = genesis.empty((M,), dtype='float32', device=device)  
        rstd = genesis.empty((M,), dtype='float32', device=device)
        
        # Calculate kernel parameters
        MAX_FUSED_SIZE = 65536 // x.element_size()  # Use new method
        BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(N))
        
        # Pass Genesis tensors directly to Triton kernel!
        _layer_norm_fwd_kernel[(M,)](
            x_reshaped, y, weight, bias, mean, rstd,
            x_reshaped.stride(0), N, eps,  # Use new method
            BLOCK_SIZE=BLOCK_SIZE, num_warps=8
        )
        
        ctx.save_for_backward(x, weight, bias, mean, rstd)
        ctx.BLOCK_SIZE = BLOCK_SIZE
        ctx.num_warps = 8
        return y

    @staticmethod
    def backward(ctx, out_grad):
        # Can also use Genesis tensors directly
        x, w, b, m, v = ctx.saved_tensors
        
        # heuristics for amount of parallel reduction stream for DW/DB
        N = w.shape[0]
        GROUP_SIZE_M = 64
        if N <= 8192: 
            GROUP_SIZE_M = 96
        if N <= 4096: 
            GROUP_SIZE_M = 128
        if N <= 1024: 
            GROUP_SIZE_M = 256
            
        # Pass directly to backward kernels, inherit requires_grad=False for gradient computation
        dx = genesis.empty_like(x, requires_grad=False)
        dw = genesis.empty_like(w, requires_grad=False)
        db = genesis.empty_like(b, requires_grad=False)
        
        # Create temporary buffers for parallel reduction
        # Use safer way to create buffers, avoid mul_scalar issues
        from genesis.ndarray.cuda_tensor import zeros
        locks = zeros((2 * GROUP_SIZE_M,), "int32")
        _dw = zeros((GROUP_SIZE_M, N), x.dtype)
        _db = zeros((GROUP_SIZE_M, N), x.dtype)
        
        # Get reshaped dimensions
        x_reshaped = x.reshape(-1, x.shape[-1])
        M, N = x_reshaped.shape
        
        # Launch backward dx kernel
        _layer_norm_bwd_dx_kernel[(M,)](
            dx, out_grad, _dw, _db, x, w, m, v, locks,
            x_reshaped.stride(0), N, 
            BLOCK_SIZE_N=ctx.BLOCK_SIZE,
            GROUP_SIZE_M=GROUP_SIZE_M,
            num_warps=ctx.num_warps)
        
        # Launch backward dw/db kernel
        grid = (triton.cdiv(N, 128),)
        _layer_norm_bwd_dwdb_kernel[grid](
            _dw, _db, dw, db, min(GROUP_SIZE_M, M), N,
            BLOCK_SIZE_M=32,
            BLOCK_SIZE_N=128)
        
        return dx, dw, db

def fused_layer_norm(x, weight, bias, eps=1e-6):
    return FusedLayerNormFunction.apply(x, weight, bias, eps=eps)
