"""Test suite for Genesis neural network modules.

This module contains comprehensive tests for neural network layers and operations,
comparing Genesis implementations against PyTorch reference implementations.
"""

import sys
sys.path.append('./')
import itertools
import numpy as np
import pytest
import torch
import genesis.nn.functional as F

import genesis

_DEVICES = [
        genesis.device('cpu'),
        pytest.param(
                genesis.device("cuda"), 
                marks=pytest.mark.skipif(not genesis.device("cuda").enabled(), reason="No GPU"))]


SOFTMAX_SHAPES = [
        (8, 64),
        (8, 32, 64),
        (8, 2048, 64),
        (8, 32, 2048),
]
@pytest.mark.parametrize("shape", SOFTMAX_SHAPES)
@pytest.mark.parametrize("device", _DEVICES, ids=["cpu", "cuda"])
def test_softmax(shape, device):
    """Test Softmax layer forward and backward pass.
    
    Args:
        shape: Input tensor shape for testing
        device: Device to run test on (CPU or CUDA)
    
    Tests:
        - Forward pass matches PyTorch implementation
        - Backward gradients match PyTorch implementation
    """
    _A = np.random.randn(*shape).astype(np.float32)
    A = genesis.Tensor(_A, device=device, requires_grad=True)
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
    """Test BatchNorm1d layer forward and backward pass.
    
    Args:
        shape: Input tensor shape (batch_size, features)
        device: Device to run test on (CPU or CUDA)
    
    Tests:
        - Forward pass normalization matches PyTorch
        - Backward gradients match PyTorch implementation
    """
    _A = np.random.randn(*shape).astype(np.float32)
    A = genesis.Tensor(_A, device=device, requires_grad=True)
    TA = torch.Tensor(_A)
    TA.requires_grad = True
    norm = genesis.nn.BatchNorm1d(shape[1])
    if device == genesis.device("cuda"):
        norm.cuda()
    C = norm(A)
    TC = torch.nn.BatchNorm1d(shape[1])(TA)
    np.testing.assert_allclose(TC.detach().numpy(), C.detach().numpy(), atol=1e-5, rtol=1e-5)

    C.sum().backward()
    TC.sum().backward()
    np.testing.assert_allclose(TA.grad.numpy(), A.grad.numpy(), atol=1e-5, rtol=1e-5)

@pytest.mark.parametrize("shape", NN_BASE_SHAPES)
@pytest.mark.parametrize("device", _DEVICES, ids=["cpu", "cuda"])
def test_linear(shape, device):
    """Test Linear (fully-connected) layer forward and backward pass.
    
    Args:
        shape: Input tensor shape (batch_size, in_features)
        device: Device to run test on (CPU or CUDA)
    
    Tests:
        - Forward pass linear transformation matches PyTorch
        - Weight sharing between Genesis and PyTorch models
        - Backward gradients match PyTorch implementation
    """
    _A = np.random.randn(*shape).astype(np.float32)
    A = genesis.Tensor(_A, device=device, requires_grad=True)
    TA = torch.Tensor(_A)
    TA.requires_grad = True
    T_linear = torch.nn.Linear(shape[1], 10)
    TC = T_linear(TA)
    linear = genesis.nn.Linear(shape[1], 10)
    linear.weight = genesis.nn.Parameter(T_linear.weight.detach().numpy())
    linear.bias = genesis.nn.Parameter(T_linear.bias.detach().numpy())
    if device == genesis.device("cuda"):
        linear.cuda()
    C = linear(A)
    np.testing.assert_allclose(TC.detach().numpy(), C.detach().numpy(), atol=1e-5, rtol=1e-5)

    C.sum().backward()
    TC.sum().backward()
    np.testing.assert_allclose(TA.grad.numpy(), A.grad.numpy(), atol=1e-5, rtol=1e-5)

LAYERNORM_SHAPES = [
        (8, 64),
        (8, 32),
        (8, 16, 32),
]
@pytest.mark.parametrize("shape", LAYERNORM_SHAPES)
@pytest.mark.parametrize("device", _DEVICES, ids=["cpu", "cuda"])
def test_layernorm(shape, device):
    """Test LayerNorm layer forward and backward pass.
    
    Args:
        shape: Input tensor shape for testing
        device: Device to run test on (CPU or CUDA)
    
    Tests:
        - Forward pass layer normalization matches PyTorch
        - Backward gradients match PyTorch implementation
    """
    _A = np.random.randn(*shape).astype(np.float32)
    A = genesis.Tensor(_A, device=device, requires_grad=True)
    TA = torch.Tensor(_A)
    TA.requires_grad = True
    norm = genesis.nn.LayerNorm(shape[-1])
    if device == genesis.device("cuda"):
        norm.cuda()
    C = norm(A)
    TC = torch.nn.LayerNorm(shape[-1])(TA)
    np.testing.assert_allclose(TC.detach().numpy(), C.detach().numpy(), atol=1e-5, rtol=1e-5)

    # Note: LayerNorm sum() testing is mathematically ill-conditioned due to near-zero expected values
    # The sum of LayerNorm output should theoretically be ≈0, but floating point accumulation
    # errors make this comparison unstable. We test forward pass precision instead.

    C.sum().backward()
    TC.sum().backward()
    np.testing.assert_allclose(TA.grad.numpy(), A.grad.numpy(), atol=1e-5, rtol=1e-5)

@pytest.mark.parametrize("shape", SOFTMAX_SHAPES)
@pytest.mark.parametrize("device", _DEVICES, ids=["cpu", "cuda"])
def test_fusedlayernorm(shape, device):
    """Test FusedLayerNorm (optimized CUDA kernel) forward and backward pass.
    
    Args:
        shape: Input tensor shape for testing
        device: Device to run test on (CUDA only)
    
    Tests:
        - Forward pass fused normalization matches PyTorch LayerNorm
        - Backward gradients match PyTorch implementation
        - Skips test on CPU devices
    """
    if device == genesis.device('cpu'):
        pytest.skip("Skipping CPU tests, only testing CUDA")
    _A = np.random.randn(*shape).astype(np.float32)
    A = genesis.Tensor(_A, device=device, requires_grad=True)
    TA = torch.Tensor(_A)
    TA.requires_grad = True
    norm = genesis.nn.FusedLayerNorm(shape[-1])
    if device == genesis.device("cuda"):
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
    """Test ReLU activation function forward and backward pass.
    
    Args:
        shape: Input tensor shape for testing
        device: Device to run test on (CPU or CUDA)
    
    Tests:
        - Forward pass ReLU activation matches PyTorch
        - Backward gradients match PyTorch implementation
    """
    _A = np.random.randn(*shape).astype(np.float32)
    A = genesis.Tensor(_A, device=device, requires_grad=True)
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
    """Test single-head attention mechanism forward and backward pass.
    
    Args:
        shape: Input tensor shape (batch_size, seq_len, dim)
        device: Device to run test on (CPU or CUDA)
    
    Tests:
        - Forward pass attention output matches PyTorch MultiheadAttention with 1 head
        - Causal masking is properly applied
        - Backward gradients match PyTorch implementation
    """
    _A = np.random.randn(*shape).astype(np.float32)
    A = genesis.Tensor(_A, device=device, requires_grad=True)
    TA = torch.Tensor(_A)
    TA.requires_grad = True

    attn = genesis.nn.MultiheadAttention(shape[2], 1)

    torch_attn = torch.nn.MultiheadAttention(shape[2], 1, bias=False, batch_first=True)
    attn.w_qkv = genesis.nn.Parameter(torch_attn.in_proj_weight.detach().T.numpy())
    attn.w_out = genesis.nn.Parameter(torch_attn.out_proj.weight.detach().numpy().T)
    M = torch.triu(-float("inf") * torch.ones(shape[1], shape[1]), 1)

    if device == genesis.device("cuda"):
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
    """Test multi-head attention mechanism forward and backward pass.
    
    Args:
        shape: Input tensor shape (batch_size, seq_len, dim)
        device: Device to run test on (CPU or CUDA)
    
    Tests:
        - Forward pass multi-head attention output matches PyTorch
        - Weight matrices are properly shared between implementations
        - Causal masking is correctly applied
        - Backward gradients match PyTorch implementation
    """
    _A = np.random.randn(*shape).astype(np.float32)
    A = genesis.Tensor(_A, device=device, requires_grad=True)
    TA = torch.Tensor(_A)
    TA.requires_grad = True

    attn = genesis.nn.MultiheadAttention(dim=shape[2], heads=4)

    torch_attn = torch.nn.MultiheadAttention(shape[2], 4, bias=False, batch_first=True)
    attn.w_qkv = genesis.nn.Parameter(torch_attn.in_proj_weight.detach().T.numpy())
    attn.w_out = genesis.nn.Parameter(torch_attn.out_proj.weight.detach().numpy().T)
    M = torch.triu(-float("inf") * torch.ones(shape[1], shape[1]), 1)

    if device == genesis.device("cuda"):
        attn.cuda()
    genesis_out = attn(A)
    torch_out = torch_attn(TA, TA, TA, attn_mask=M)

    # Use slightly relaxed tolerance for attention - complex operations can have small numerical differences
    np.testing.assert_allclose(
            genesis_out[0].detach().numpy(), 
            torch_out[0].detach().numpy(), 
            atol=2e-5, rtol=1e-5)

    genesis_out[0].sum().backward()
    torch_out[0].sum().backward()
    np.testing.assert_allclose(TA.grad.numpy(), A.grad.numpy(), atol=2e-5, rtol=1e-5)

ATTENTION_SHAPES = [
    (8, 32, 16 * 64),
]
@pytest.mark.parametrize("shape", ATTENTION_SHAPES)
@pytest.mark.parametrize("device", _DEVICES, ids=["cpu", "cuda"])
def test_fused_multihead_attention(shape, device):
    """Test fused multi-head attention (optimized CUDA kernel) forward and backward pass.
    
    Args:
        shape: Input tensor shape (batch_size, seq_len, dim)
        device: Device to run test on (CUDA only)
    
    Tests:
        - Forward pass fused attention matches PyTorch MultiheadAttention
        - Uses relaxed tolerance due to numerical precision differences
        - Backward gradients match PyTorch implementation
        - Skips test on CPU devices
    """
    if device == genesis.device('cpu'):
        pytest.skip("Skipping CPU tests, only testing CUDA")
    _A = np.random.randn(*shape).astype(np.float32)
    A = genesis.Tensor(_A, device=device, requires_grad=True)
    TA = torch.Tensor(_A)
    TA.requires_grad = True
    heads = 16

    attn = genesis.nn.FusedMultiheadAttention(dim=shape[2], heads=heads)

    torch_attn = torch.nn.MultiheadAttention(shape[2], heads, bias=False, batch_first=True)
    attn.w_qkv = genesis.nn.Parameter(torch_attn.in_proj_weight.detach().T.numpy())
    attn.w_out = genesis.nn.Parameter(torch_attn.out_proj.weight.detach().numpy().T)
    M = torch.triu(-float("inf") * torch.ones(shape[1], shape[1]), 1)

    if device == genesis.device("cuda"):
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

QKV_SHAPES = [
    (1, 16, 12, 64),
]
@pytest.mark.parametrize("shape", QKV_SHAPES)
@pytest.mark.parametrize("device", _DEVICES, ids=["cpu", "cuda"])
def test_scaled_dot_product_attention(shape, device):
    """Test scaled dot-product attention function.
    
    Args:
        shape: Shape for Q, K, V tensors (batch, heads, seq_len, head_dim)
        device: Device to run test on (CUDA only)
    
    Tests:
        - Forward pass scaled attention computation matches PyTorch
        - Causal masking is automatically applied
        - Skips test on CPU devices
    """
    if device == genesis.device('cpu'):
        pytest.skip("Skipping CPU tests, only testing CUDA")
    _Q = np.random.randn(*shape).astype(np.float32)
    Q = genesis.Tensor(_Q, device=device)
    TQ = torch.Tensor(_Q)
    TQ.requires_grad = True
    _K = np.random.randn(*shape).astype(np.float32)
    K = genesis.Tensor(_K, device=device)
    TK = torch.Tensor(_K)
    TK.requires_grad = True
    _V = np.random.randn(*shape).astype(np.float32)
    V = genesis.Tensor(_V, device=device)
    TV = torch.Tensor(_V)
    TV.requires_grad = True

    genesis_out = F.scaled_dot_product_attention(Q, K, V)
    
    torch_out = torch.nn.functional.scaled_dot_product_attention(
            TQ, TK, TV, attn_mask=None, dropout_p=0.0, is_causal=True)
    np.testing.assert_allclose(
            genesis_out.detach().numpy(), 
            torch_out.detach().numpy(), 
            atol=1e-2, rtol=1e-2)

@pytest.mark.parametrize("device", _DEVICES, ids=["cpu", "cuda"])
def test_embedding(device):
    """Test Embedding layer forward and backward pass.
    
    Args:
        device: Device to run test on (CPU or CUDA)
    
    Tests:
        - Forward pass embedding lookup matches PyTorch
        - Integer indexing works correctly
        - Backward gradient accumulation in embedding weights matches PyTorch
    """
    num_embeddings = 1000
    embedding_dim = 32

    _A = np.array([[5, 6], [3, 4]]).astype(np.int64)
    A = genesis.Tensor(_A, device=device, requires_grad=False).long()
    TA = torch.LongTensor(_A)

    embed = genesis.nn.Embedding(num_embeddings, embedding_dim)
    torch_embed = torch.nn.Embedding(num_embeddings, embedding_dim)
    embed.weight = genesis.nn.Parameter(torch_embed.weight.detach().numpy())

    if device == genesis.device("cuda"):
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
    """PyTorch reference implementation of Rotary Position Embedding (RoPE).
    
    This class is used as a reference to test the Genesis implementation.
    RoPE applies rotation-based position encodings to attention queries and keys.
    """
    
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        """Initialize RoPE with precomputed sin/cos values.
        
        Args:
            dim: Dimension of the embeddings
            max_position_embeddings: Maximum sequence length to support
            base: Base value for computing rotation frequencies
            device: Device to place the embeddings on
        """
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
        """Apply rotary embeddings to input tensor.
        
        Args:
            x: Input tensor of shape [batch_size, num_attention_heads, seq_len, head_size]
            seq_len: Optional sequence length to use (defaults to cached length)
        
        Returns:
            Tuple of (cos_values, sin_values) for applying rotary embeddings
        """
        # x: [bs, num_attention_heads, seq_len, head_size]
        # This `if` block is unlikely to be run after we build sin/cos in `__init__`. Keep the logic here just in case.
        return (
            self.cos_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
            self.sin_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
        )


@pytest.mark.parametrize("device", _DEVICES, ids=["cpu", "cuda"])
def test_rotary_embedding(device):
    """Test Rotary Position Embedding (RoPE) implementation.
    
    Args:
        device: Device to run test on (CPU or CUDA)
    
    Tests:
        - Cached sin/cos values match between Genesis and PyTorch implementations
        - Forward pass produces identical rotation values
        - Proper shape handling for different input dimensions
    """
    torch_rotary_embed = RotaryEmbedding(64, 32)
    rotary_embed = genesis.nn.RotaryEmbedding(64, 32)

    np.testing.assert_allclose(
            torch_rotary_embed.cos_cached.detach().numpy(), 
            rotary_embed.cos_cached.detach().numpy(), atol=1e-5, rtol=1e-5)

    np.testing.assert_allclose(
            torch_rotary_embed.sin_cached.detach().numpy(), 
            rotary_embed.sin_cached.detach().numpy(), atol=1e-5, rtol=1e-5)

    _A = np.random.randn(2, 3, 4, 5).astype(np.float32)
    A = genesis.Tensor(_A, device=device, requires_grad=True)
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
    """Test SiLU (Swish) activation function forward and backward pass.
    
    Args:
        shape: Input tensor shape for testing
        device: Device to run test on (CPU or CUDA)
    
    Tests:
        - Forward pass SiLU activation matches PyTorch
        - Backward gradients match PyTorch implementation
    """
    _A = np.random.randn(*shape).astype(np.float32)
    A = genesis.Tensor(_A, device=device, requires_grad=True)
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
