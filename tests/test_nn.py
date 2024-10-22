import sys
sys.path.append('./')
import itertools
import numpy as np
import pytest
import torch
import genesis.nn.functional as F

import genesis

_DEVICES = [
        genesis.cpu(),
        pytest.param(
                genesis.cuda(), 
                marks=pytest.mark.skipif(not genesis.cuda().enabled(), reason="No GPU"))]


SOFTMAX_SHAPES = [
        (8, 64),
        (8, 32, 64),
        (8, 2048, 64),
        (8, 32, 2048),
]
@pytest.mark.parametrize("shape", SOFTMAX_SHAPES)
@pytest.mark.parametrize("device", _DEVICES, ids=["cpu", "cuda"])
def test_softmax(shape, device):
    _A = np.random.randn(*shape).astype(np.float32)
    A = genesis.Tensor(_A, device=device)
    TA = torch.Tensor(_A)
    TA.requires_grad = True
    C = genesis.nn.Softmax(dim=-1)(A)
    TC = torch.nn.Softmax(dim=-1)(TA)
    np.testing.assert_allclose(TC.detach().numpy(), C.detach().numpy(), atol=1e-5, rtol=1e-5)

    C.sum().backward()
    TC.sum().backward()
    np.testing.assert_allclose(TA.grad.numpy(), A.grad.numpy(), atol=1e-5, rtol=1e-5)


NN_BASE_SHAPES = [
        (8, 64),
        (32, 32),
        (2048, 2048),
]
@pytest.mark.parametrize("shape", NN_BASE_SHAPES)
@pytest.mark.parametrize("device", _DEVICES, ids=["cpu", "cuda"])
def test_batchnorm1d(shape, device):
    _A = np.random.randn(*shape).astype(np.float32)
    A = genesis.Tensor(_A, device=device)
    TA = torch.Tensor(_A)
    TA.requires_grad = True
    norm = genesis.nn.BatchNorm1d(shape[1])
    if device == genesis.cuda():
        norm.cuda()
    C = norm(A)
    TC = torch.nn.BatchNorm1d(shape[1])(TA)
    np.testing.assert_allclose(TC.detach().numpy(), C.detach().numpy(), atol=1e-5, rtol=1e-5)

    C.sum().backward()
    TC.sum().backward()
    np.testing.assert_allclose(TA.grad.numpy(), A.grad.numpy(), atol=1e-5, rtol=1e-5)

@pytest.mark.parametrize("shape", SOFTMAX_SHAPES)
@pytest.mark.parametrize("device", _DEVICES, ids=["cpu", "cuda"])
def test_layernorm(shape, device):
    _A = np.random.randn(*shape).astype(np.float32)
    A = genesis.Tensor(_A, device=device)
    TA = torch.Tensor(_A)
    TA.requires_grad = True
    norm = genesis.nn.LayerNorm(shape[-1])
    if device == genesis.cuda():
        norm.cuda()
    C = norm(A)
    TC = torch.nn.LayerNorm(shape[-1])(TA)
    np.testing.assert_allclose(TC.detach().numpy(), C.detach().numpy(), atol=1e-5, rtol=1e-5)

    C.sum().backward()
    TC.sum().backward()
    np.testing.assert_allclose(TA.grad.numpy(), A.grad.numpy(), atol=1e-5, rtol=1e-5)

@pytest.mark.parametrize("shape", SOFTMAX_SHAPES)
@pytest.mark.parametrize("device", _DEVICES, ids=["cpu", "cuda"])
def test_fusedlayernorm(shape, device):
    if device == genesis.cpu():
        pytest.skip("Skipping CPU tests, only testing CUDA")
    _A = np.random.randn(*shape).astype(np.float32)
    A = genesis.Tensor(_A, device=device)
    TA = torch.Tensor(_A)
    TA.requires_grad = True
    norm = genesis.nn.FusedLayerNorm(shape[-1])
    if device == genesis.cuda():
        norm.cuda()
    C = norm(A)
    TC = torch.nn.LayerNorm(shape[-1])(TA)
    np.testing.assert_allclose(TC.detach().numpy(), C.detach().numpy(), atol=1e-5, rtol=1e-5)

    C.sum().backward()
    TC.sum().backward()
    np.testing.assert_allclose(TA.grad.numpy(), A.grad.numpy(), atol=1e-5, rtol=1e-5)

@pytest.mark.parametrize("shape", NN_BASE_SHAPES)
@pytest.mark.parametrize("device", _DEVICES, ids=["cpu", "cuda"])
def test_relu(shape, device):
    _A = np.random.randn(*shape).astype(np.float32)
    A = genesis.Tensor(_A, device=device)
    TA = torch.Tensor(_A)
    TA.requires_grad = True
    C = genesis.nn.ReLU()(A)
    TC = torch.nn.ReLU()(TA)
    np.testing.assert_allclose(TC.detach().numpy(), C.detach().numpy(), atol=1e-5, rtol=1e-5)

    C.sum().backward()
    TC.sum().backward()
    np.testing.assert_allclose(TA.grad.numpy(), A.grad.numpy(), atol=1e-5, rtol=1e-5)

ATTENTION_SHAPES = [
    (8, 32, 64),
    (8, 100, 32),
    (8, 100, 2048),
]
@pytest.mark.parametrize("shape", ATTENTION_SHAPES)
@pytest.mark.parametrize("device", _DEVICES, ids=["cpu", "cuda"])
def test_onehead_attention(shape, device):
    _A = np.random.randn(*shape).astype(np.float32)
    A = genesis.Tensor(_A, device=device)
    TA = torch.Tensor(_A)
    TA.requires_grad = True

    attn = genesis.nn.MultiheadAttention(shape[2], 1)

    torch_attn = torch.nn.MultiheadAttention(shape[2], 1, bias=False, batch_first=True)
    attn.w_qkv = genesis.nn.Parameter(torch_attn.in_proj_weight.detach().T.numpy())
    attn.w_out = genesis.nn.Parameter(torch_attn.out_proj.weight.detach().numpy().T)
    M = torch.triu(-float("inf") * torch.ones(shape[1], shape[1]), 1)

    if device == genesis.cuda():
        attn.cuda()
    genesis_out = attn(A)
    torch_out = torch_attn(TA, TA, TA, attn_mask=M)

    np.testing.assert_allclose(
            genesis_out[0].detach().numpy(), 
            torch_out[0].detach().numpy(), 
            atol=1e-5, rtol=1e-5)

    genesis_out[0].sum().backward()
    torch_out[0].sum().backward()
    np.testing.assert_allclose(TA.grad.numpy(), A.grad.numpy(), atol=1e-5, rtol=1e-5)


ATTENTION_SHAPES = [
    (8, 32, 64),
    (8, 100, 32),
    (8, 100, 2048),
]
@pytest.mark.parametrize("shape", ATTENTION_SHAPES)
@pytest.mark.parametrize("device", _DEVICES, ids=["cpu", "cuda"])
def test_multihead_attention(shape, device):
    _A = np.random.randn(*shape).astype(np.float32)
    A = genesis.Tensor(_A, device=device)
    TA = torch.Tensor(_A)
    TA.requires_grad = True

    attn = genesis.nn.MultiheadAttention(dim=shape[2], heads=4)

    torch_attn = torch.nn.MultiheadAttention(shape[2], 4, bias=False, batch_first=True)
    attn.w_qkv = genesis.nn.Parameter(torch_attn.in_proj_weight.detach().T.numpy())
    attn.w_out = genesis.nn.Parameter(torch_attn.out_proj.weight.detach().numpy().T)
    M = torch.triu(-float("inf") * torch.ones(shape[1], shape[1]), 1)

    if device == genesis.cuda():
        attn.cuda()
    genesis_out = attn(A)
    torch_out = torch_attn(TA, TA, TA, attn_mask=M)

    np.testing.assert_allclose(
            genesis_out[0].detach().numpy(), 
            torch_out[0].detach().numpy(), 
            atol=1e-5, rtol=1e-5)

    genesis_out[0].sum().backward()
    torch_out[0].sum().backward()
    np.testing.assert_allclose(TA.grad.numpy(), A.grad.numpy(), atol=1e-5, rtol=1e-5)

ATTENTION_SHAPES = [
    (8, 32, 16 * 64),
]
@pytest.mark.parametrize("shape", ATTENTION_SHAPES)
@pytest.mark.parametrize("device", _DEVICES, ids=["cpu", "cuda"])
def test_fused_multihead_attention(shape, device):
    if device == genesis.cpu():
        pytest.skip("Skipping CPU tests, only testing CUDA")
    _A = np.random.randn(*shape).astype(np.float32)
    A = genesis.Tensor(_A, device=device)
    TA = torch.Tensor(_A)
    TA.requires_grad = True
    heads = 16

    attn = genesis.nn.FusedMultiheadAttention(dim=shape[2], heads=heads)

    torch_attn = torch.nn.MultiheadAttention(shape[2], heads, bias=False, batch_first=True)
    attn.w_qkv = genesis.nn.Parameter(torch_attn.in_proj_weight.detach().T.numpy())
    attn.w_out = genesis.nn.Parameter(torch_attn.out_proj.weight.detach().numpy().T)
    M = torch.triu(-float("inf") * torch.ones(shape[1], shape[1]), 1)

    if device == genesis.cuda():
        attn.cuda()
    genesis_out = attn(A)
    torch_out = torch_attn(TA, TA, TA, attn_mask=M)

    np.testing.assert_allclose(
            genesis_out[0].detach().numpy(), 
            torch_out[0].detach().numpy(), 
            atol=1e-2, rtol=1e-2)

    genesis_out[0].sum().backward()
    torch_out[0].sum().backward()
    np.testing.assert_allclose(TA.grad.numpy(), A.grad.numpy(), atol=1e-2, rtol=1e-2)

@pytest.mark.parametrize("device", _DEVICES, ids=["cpu", "cuda"])
def test_embedding(device):
    num_embeddings = 1000
    embedding_dim = 32

    _A = np.array([[5, 6], [3, 4]]).astype(np.int64)
    A = genesis.Tensor(_A, device=device)
    TA = torch.LongTensor(_A)

    embed = genesis.nn.Embedding(num_embeddings, embedding_dim)
    torch_embed = torch.nn.Embedding(num_embeddings, embedding_dim)
    embed.weight = genesis.nn.Parameter(torch_embed.weight.detach().numpy())

    if device == genesis.cuda():
        embed.cuda()

    genesis_out = embed(A)
    torch_out = torch_embed(TA)

    np.testing.assert_allclose(
            genesis_out.detach().numpy(), 
            torch_out.detach().numpy(), 
            atol=1e-5, rtol=1e-5)

    genesis_out.sum().backward()
    torch_out.sum().backward()
    np.testing.assert_allclose(embed.weight.detach().numpy(), torch_embed.weight.detach().numpy(), atol=1e-5, rtol=1e-5)

class RotaryEmbedding(torch.nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float().to(device) / dim))
        self.register_buffer("inv_freq", inv_freq)

        # Build here to make `torch.jit.trace` work.
        self.max_seq_len_cached = max_position_embeddings
        t = torch.arange(self.max_seq_len_cached, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :], persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :], persistent=False)

    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        # This `if` block is unlikely to be run after we build sin/cos in `__init__`. Keep the logic here just in case.
        return (
            self.cos_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
            self.sin_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
        )


@pytest.mark.parametrize("device", _DEVICES, ids=["cpu", "cuda"])
def test_rotary_embedding(device):
    torch_rotary_embed = RotaryEmbedding(64, 32)
    rotary_embed = genesis.nn.RotaryEmbedding(64, 32)

    np.testing.assert_allclose(
            torch_rotary_embed.cos_cached.detach().numpy(), 
            rotary_embed.cos_cached.detach().numpy(), atol=1e-5, rtol=1e-5)

    np.testing.assert_allclose(
            torch_rotary_embed.sin_cached.detach().numpy(), 
            rotary_embed.sin_cached.detach().numpy(), atol=1e-5, rtol=1e-5)

    _A = np.random.randn(2, 3, 4, 5).astype(np.float32)
    A = genesis.Tensor(_A, device=device)
    TA = torch.Tensor(_A)
    TA.requires_grad = True
    torch_res = torch_rotary_embed(TA)
    res = rotary_embed(A)
    np.testing.assert_allclose(
            torch_res[0].detach().numpy(), 
            res[0].detach().numpy(), atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(
            torch_res[1].detach().numpy(), 
            res[1].detach().numpy(), atol=1e-5, rtol=1e-5)

@pytest.mark.parametrize("shape", SOFTMAX_SHAPES)
@pytest.mark.parametrize("device", _DEVICES, ids=["cpu", "cuda"])
def test_silu(shape, device):
    _A = np.random.randn(*shape).astype(np.float32)
    A = genesis.Tensor(_A, device=device)
    TA = torch.Tensor(_A)
    TA.requires_grad = True
    C = genesis.nn.SiLU()(A)
    TC = torch.nn.SiLU()(TA)
    np.testing.assert_allclose(TC.detach().numpy(), C.detach().numpy(), atol=1e-5, rtol=1e-5)

    C.sum().backward()
    TC.sum().backward()
    np.testing.assert_allclose(TA.grad.numpy(), A.grad.numpy(), atol=1e-5, rtol=1e-5)


if __name__ == "__main__":
    pytest.main()
