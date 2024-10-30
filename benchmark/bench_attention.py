import sys
import time
import numpy as np
import triton
import torch
sys.path.append("../")
import genesis
import genesis.nn as nn

import triton
import triton.language as tl

def benchmark(B, S, H, D, provider):
    _A = np.random.randn(B, S, H * D).astype(np.float32)
    A = genesis.Tensor(_A, device=genesis.cuda())
    TA = torch.Tensor(_A).cuda()
    TA.requires_grad = True

    ma = genesis.nn.MultiheadAttention(H * D, H)
    ma.cuda()
    fma = genesis.nn.FusedMultiheadAttention(H * D, H)
    fma.cuda()
    tma = torch.nn.MultiheadAttention(H * D, H, bias=False, batch_first=True)
    tma.cuda()
    M = torch.triu(-float("inf") * torch.ones(S, S), 1).cuda()

    cache_size = 256 * 1024 * 1024
    cache = torch.empty(int(cache_size // 4), dtype=torch.int, device=torch.device("cuda"))

    # warmup
    for i in range(10):
        if provider.lower() == "torch               ":
            tma(TA, TA, TA, attn_mask=M)
        if provider.lower() == "genesis_triton      ":
            ma(A)
        if provider.lower() == "genesis_fused_triton":
            fma(A)
        
    start_time = time.time()
    all_time = 0
    for i in range(1010):
        cache.zero_()
        start_time = time.time()
        if provider.lower() == "torch               ":
            tma(TA, TA, TA, attn_mask=M)
        if provider.lower() == "genesis_triton      ":
            ma(A)
        if provider.lower() == "genesis_fused_triton":
            fma(A)
        all_time += time.time() - start_time
    torch.cuda.synchronize()
    print(provider, "cost_time:", all_time)

benchmark(1, 512, 8, 64, "torch               ")
benchmark(1, 512, 8, 64, "genesis_triton      ")
benchmark(1, 512, 8, 64, "genesis_fused_triton")
