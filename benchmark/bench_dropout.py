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

def benchmark(M, N, provider):
    _A = np.random.randn(M, N).astype(np.float32)
    A = genesis.Tensor(_A, device=genesis.cuda())
    TA = torch.Tensor(_A).cuda()
    TA.requires_grad = True

    ln = genesis.nn.Dropout(0.1)
    ln.cuda()
    tln = torch.nn.Dropout(0.1)
    tln.cuda()

    cache_size = 256 * 1024 * 1024
    cache = torch.empty(int(cache_size // 4), dtype=torch.int, device=torch.device("cuda"))

    # warmup
    for i in range(10):
        if provider.lower() == "torch               ":
            tln(TA)
        if provider.lower() == "genesis_triton      ":
            ln(A)
        
    start_time = time.time()
    all_time = 0
    for i in range(1010):
        cache.zero_()
        start_time = time.time()
        if provider.lower() == "torch               ":
            tln(TA)
        if provider.lower() == "genesis_triton      ":
            ln(A)
        all_time += time.time() - start_time
    torch.cuda.synchronize()
    print(provider, "cost_time:", all_time)

benchmark(1280, 748, "torch               ")
benchmark(1280, 748, "genesis_triton      ")

