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

class TorchLayerNorm(torch.nn.Module):
    def __init__(self, num_features, eps=1e-5):
        super(TorchLayerNorm, self).__init__()
        self.gamma = torch.nn.Parameter(torch.ones(num_features))
        self.beta = torch.nn.Parameter(torch.zeros(num_features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        x_normalized = (x - mean) / torch.sqrt(var + self.eps)
        out = self.gamma * x_normalized + self.beta
        return out

def benchmark(M, N, provider):
    _A = np.random.randn(M, N).astype(np.float32)
    A = genesis.Tensor(_A, device=genesis.cuda())
    TA = torch.Tensor(_A).cuda()
    TA.requires_grad = True

    ln = genesis.nn.LayerNorm(N)
    ln.cuda()
    fln = genesis.nn.FusedLayerNorm(N)
    fln.cuda()
    tln = TorchLayerNorm(N)
    tln.cuda()
    tfln = torch.nn.LayerNorm(N)
    tfln.cuda()
    rln = genesis.nn.RMSNorm(N)
    rln.cuda()

    cache_size = 256 * 1024 * 1024
    cache = torch.empty(int(cache_size // 4), dtype=torch.int, device=torch.device("cuda"))

    # warmup
    for i in range(10):
        if provider.lower() == "torch               ":
            tln(TA)
        if provider.lower() == "fused_torch         ":
            tfln(TA)
        if provider.lower() == "genesis_triton      ":
            ln(A)
        if provider.lower() == "genesis_fused_triton":
            fln(A)
        if provider.lower() == "rms_triton":
            rln(A)
        
    start_time = time.time()
    all_time = 0
    for i in range(1010):
        cache.zero_()
        start_time = time.time()
        if provider.lower() == "torch               ":
            tln(TA)
        if provider.lower() == "fused_torch         ":
            tfln(TA)
        if provider.lower() == "genesis_triton      ":
            ln(A)
        if provider.lower() == "genesis_fused_triton":
            fln(A)
        if provider.lower() == "rms_triton":
            rln(A)
        all_time += time.time() - start_time
    torch.cuda.synchronize()
    print(provider, "cost_time:", all_time)

benchmark(1280, 748, "torch               ")
benchmark(1280, 748, "fused_torch         ")
benchmark(1280, 748, "genesis_triton      ")
benchmark(1280, 748, "genesis_fused_triton")

