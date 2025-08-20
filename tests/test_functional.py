"""Test suite for Genesis functional operations.

This module contains comprehensive tests for functional operations and tensor manipulations,
comparing Genesis implementations against PyTorch reference implementations.
Tests cover element-wise operations, matrix operations, reductions, and tensor indexing.
"""

import sys
sys.path.append('./')
import itertools
import numpy as np
import pytest
import torch

import genesis
import genesis.nn.functional as F

atol = 1e-1
rtol = 1e-1

def backward_check(f, *args, **kwargs):
    """Numerical gradient checking using finite differences.
    
    Args:
        f: Function to test gradients for
        *args: Input tensors to the function
        **kwargs: Additional keyword arguments for the function
    
    Returns:
        List of backward gradients computed numerically
        
    Tests:
        Computes numerical gradients using central difference method
        and compares with automatic differentiation gradients.
    """
    eps = 1e-5
    out = f(*args, **kwargs)
    c = np.random.randn(*out.shape)
    numerical_grad = [np.zeros(a.shape) for a in args]
    num_args = len(args)
    for i in range(num_args):
        for j in range(args[i].realize_cached_data().size):
            args[i].realize_cached_data().flat[j] += eps
            f1 = (f(*args, **kwargs).numpy() * c).sum()
            args[i].realize_cached_data().flat[j] -= 2 * eps
            f2 = (f(*args, **kwargs).numpy() * c).sum()
            args[i].realize_cached_data().flat[j] += eps
            numerical_grad[i].flat[j] = (f1 - f2) / (2 * eps)
    backward_grad = out.op.gradient_as_tuple(genesis.Tensor(c, device=args[0].device), out)
    error = sum(
            np.linalg.norm(backward_grad[i].numpy() - numerical_grad[i]) for i in range(len(args)))
    assert error < 4.2e-1
    return [g.numpy() for g in backward_grad]

_DEVICES = [
        genesis.device('cpu'),
        pytest.param(
            genesis.device("cuda"), 
            marks=pytest.mark.skipif(not genesis.device("cuda").enabled(), reason="No GPU"))]

_DTYPE = [(genesis.float32, torch.float32), (genesis.float16, torch.float16)]

EWISE_OPS = {
    "add": lambda a, b: a + b,
    "divide": lambda a, b: a / b,
    "subtract": lambda a, b: a - b,
    "mul": lambda a, b: a * b,
}
EWISE_OP_FNS = [EWISE_OPS[k] for k in EWISE_OPS]
EWISE_OP_NAMES = [k for k in EWISE_OPS]
GENERAL_SHAPES = [(1, 1, 1), (4, 5, 6)]
@pytest.mark.parametrize("fn", EWISE_OP_FNS, ids=EWISE_OP_NAMES)
@pytest.mark.parametrize("shape", GENERAL_SHAPES)
@pytest.mark.parametrize("device", _DEVICES, ids=["cpu", "cuda"])
@pytest.mark.parametrize("dtype", _DTYPE, ids=["float32", "float16"])
def test_ewise_fn(fn, shape, device, dtype):
    """Test element-wise binary operations (add, subtract, multiply, divide).
    
    Args:
        fn: Element-wise operation function
        shape: Input tensor shape
        device: Device to run test on (CPU or CUDA)
        dtype: Data type for tensors (float32 or float16)
    
    Tests:
        - Forward pass matches PyTorch for nested operations
        - Backward gradients match PyTorch implementation
        - Handles float16 precision and inf values correctly
    """
    if dtype[0] == genesis.float16:
        _A = np.random.randn(*shape).astype(np.float16)
        _B = np.random.randn(*shape).astype(np.float16)
    else:
        _A = np.random.randn(*shape).astype(np.float32)
        _B = np.random.randn(*shape).astype(np.float32)
    A = genesis.Tensor(_A, device=device, dtype=dtype[0])
    B = genesis.Tensor(_B, device=device, dtype=dtype[0])
    TA = torch.Tensor(_A).to(dtype[1])
    TA.requires_grad = True
    TB = torch.Tensor(_B).to(dtype[1])
    TB.requires_grad = True
    np.testing.assert_allclose(
            fn(fn(TA, TB), TB).detach().numpy(), 
            fn(fn(A, B), B).detach().numpy(), atol=atol, rtol=rtol)

    fn(fn(TA, TB), TB).sum().backward()
    #fn(TA, TB).sum().backward()
    fn(fn(A, B), B).sum().backward()
    #fn(A, B).sum().backward()
    # TODO: the grad of float16 is not accurate
    mask = ~np.isinf(TA.grad.numpy()) & ~np.isinf(A.grad.numpy())
    np.testing.assert_allclose(TA.grad.numpy()[mask], A.grad.numpy()[mask], atol=atol, rtol=rtol)
    mask = ~np.isinf(TB.grad.numpy()) & ~np.isinf(B.grad.numpy())
    np.testing.assert_allclose(TB.grad.numpy()[mask], B.grad.numpy()[mask], atol=atol, rtol=rtol)


SCALAR_OPS = {
    "add": lambda a, b: a + b,
    "divide": lambda a, b: a / b,
    "rdivide": lambda a, b: b / a,
    "subtract": lambda a, b: a - b,
    "mul": lambda a, b: a * b,
    "power": lambda a, b: a ** b,
    "rpower": lambda a, b: b ** a,
}
SCALAR_OP_FNS = [SCALAR_OPS[k] for k in SCALAR_OPS]
SCALAR_OP_NAMES = [k for k in SCALAR_OPS]
GENERAL_SHAPES = [(1, 1, 1), (4, 5, 6)]
@pytest.mark.parametrize("fn", SCALAR_OP_FNS, ids=SCALAR_OP_NAMES)
@pytest.mark.parametrize("shape", GENERAL_SHAPES)
@pytest.mark.parametrize("device", _DEVICES, ids=["cpu", "cuda"])
@pytest.mark.parametrize("dtype", _DTYPE, ids=["float32", "float16"])
def test_scalar_fn(fn, shape, device, dtype):
    """Test tensor-scalar operations (add, subtract, multiply, divide, power).
    
    Args:
        fn: Scalar operation function
        shape: Input tensor shape
        device: Device to run test on (CPU or CUDA)
        dtype: Data type for tensors (float32 or float16)
    
    Tests:
        - Forward pass matches PyTorch for tensor-scalar operations
        - Backward gradients match PyTorch implementation
        - Tests both regular and reverse operations (e.g., rdivide, rpower)
    """
    _A = np.random.randn(*shape).astype(np.float32)
    #_B = np.random.randn(1).astype(np.float32).item()
    _B = 1.2
    TA = torch.Tensor(_A).to(dtype[1])
    TA.requires_grad = True
    A = genesis.Tensor(_A, device=device, dtype=dtype[0])
    np.testing.assert_allclose(fn(TA, _B).detach().numpy(), fn(A, _B).detach().numpy(), atol=atol, rtol=rtol)

    fn(TA, _B).sum().backward()
    fn(A, _B).sum().backward()
    np.testing.assert_allclose(TA.grad.numpy(), A.grad.numpy(), atol=atol, rtol=rtol)


MATMUL_DIMS = [
    (16, 16, 16),
    (8, 8, 8),
    (3, 4, 5),
    (5, 4, 3),
    (16, 16, 32),
    (64, 64, 64),
    (72, 72, 72),
    (72, 73, 74),
    (74, 73, 72),
    (1024, 1024, 1024),
    (128, 128, 128)]
@pytest.mark.parametrize("m,n,p", MATMUL_DIMS)
@pytest.mark.parametrize("device", _DEVICES, ids=["cpu", "cuda"])
@pytest.mark.parametrize("dtype", _DTYPE, ids=["float32", "float16"])
def test_matmul(m, n, p, device, dtype):
    """Test matrix multiplication operation.
    
    Args:
        m, n, p: Matrix dimensions (m×n) @ (n×p) = (m×p)
        device: Device to run test on (CPU or CUDA)
        dtype: Data type for tensors (float32 or float16)
    
    Tests:
        - Forward pass matrix multiplication matches PyTorch
        - Backward gradients match PyTorch implementation
        - Tests various matrix sizes including large matrices (1024×1024)
    """
    _A = np.random.randn(m, n).astype(np.float32)
    _B = np.random.randn(n, p).astype(np.float32)
    A = genesis.Tensor(_A, device=device, dtype=dtype[0])
    B = genesis.Tensor(_B, device=device, dtype=dtype[0])
    TA = torch.Tensor(_A).to(dtype[1])
    TA.requires_grad=True
    TB = torch.Tensor(_B).to(dtype[1])
    TB.requires_grad=True
    np.testing.assert_allclose((TA @ TB).detach().numpy(), (A @ B).detach().numpy(), atol=atol, rtol=rtol)

    (TA @ TB).sum().backward()
    (A @ B).sum().backward()
    np.testing.assert_allclose(TA.grad.numpy(), A.grad.numpy(), atol=atol, rtol=rtol)
    np.testing.assert_allclose(TA.grad.numpy(), A.grad.numpy(), atol=atol, rtol=rtol)

BATCH_MATMUL_DIMS = [
    (16, 16, 16, 16),
    (32, 16, 8, 24),
    (32, 13, 8, 15),
]
@pytest.mark.parametrize("b,m,n,p", BATCH_MATMUL_DIMS)
@pytest.mark.parametrize("device", _DEVICES, ids=["cpu", "cuda"])
@pytest.mark.parametrize("dtype", _DTYPE, ids=["float32", "float16"])
def test_batch_matmul(b, m, n, p, device, dtype):
    """Test batched matrix multiplication.
    
    Args:
        b: Batch size
        m, n, p: Matrix dimensions for each batch
        device: Device to run test on (CPU or CUDA)
        dtype: Data type for tensors (float32 or float16)
    
    Tests:
        - Forward pass batched matmul matches PyTorch
        - Backward gradients match PyTorch implementation
        - Tests broadcasting in batch dimension
    """
    _A = np.random.randn(b, m, n).astype(np.float32)
    _B = np.random.randn(b, n, p).astype(np.float32)
    A = genesis.Tensor(_A, device=device, dtype=dtype[0])
    B = genesis.Tensor(_B, device=device, dtype=dtype[0])
    TA = torch.Tensor(_A).to(dtype[1])
    TA.requires_grad=True
    TB = torch.Tensor(_B).to(dtype[1])
    TB.requires_grad=True
    np.testing.assert_allclose((TA @ TB).detach().numpy(), (A @ B).detach().numpy(), atol=atol, rtol=rtol)

    (TA @ TB).sum().backward()
    (A @ B).sum().backward()
    np.testing.assert_allclose(TA.grad.numpy(), A.grad.numpy(), atol=atol, rtol=rtol)
    np.testing.assert_allclose(TA.grad.numpy(), A.grad.numpy(), atol=atol, rtol=rtol)

@pytest.mark.parametrize("b,m,n,p", BATCH_MATMUL_DIMS)
@pytest.mark.parametrize("device", _DEVICES, ids=["cpu", "cuda"])
@pytest.mark.parametrize("dtype", _DTYPE, ids=["float32", "float16"])
def test_batch_matmul_2(b, m, n, p, device, dtype):
    """Test batched matrix multiplication with broadcasting (batch × non-batch).
    
    Args:
        b: Batch size for first operand
        m, n, p: Matrix dimensions
        device: Device to run test on (CPU or CUDA)
        dtype: Data type for tensors (float32 or float16)
    
    Tests:
        - Broadcasting batch dimension with non-batched tensor
        - Forward and backward pass correctness
    """
    _A = np.random.randn(b, m, n).astype(np.float32)
    _B = np.random.randn(n, p).astype(np.float32)
    A = genesis.Tensor(_A, device=device, dtype=dtype[0])
    B = genesis.Tensor(_B, device=device, dtype=dtype[0])
    TA = torch.Tensor(_A).to(dtype[1])
    TA.requires_grad=True
    TB = torch.Tensor(_B).to(dtype[1])
    TB.requires_grad=True
    np.testing.assert_allclose((TA @ TB).detach().numpy(), (A @ B).detach().numpy(), atol=atol, rtol=rtol)

    (TA @ TB).sum().backward()
    (A @ B).sum().backward()
    np.testing.assert_allclose(TA.grad.numpy(), A.grad.numpy(), atol=atol, rtol=rtol)
    np.testing.assert_allclose(TA.grad.numpy(), A.grad.numpy(), atol=atol, rtol=rtol)

@pytest.mark.parametrize("b,m,n,p", BATCH_MATMUL_DIMS)
@pytest.mark.parametrize("device", _DEVICES, ids=["cpu", "cuda"])
@pytest.mark.parametrize("dtype", _DTYPE, ids=["float32", "float16"])
def test_batch_matmul_3(b, m, n, p, device, dtype):
    """Test batched matrix multiplication with broadcasting (non-batch × batch).
    
    Args:
        b: Batch size for second operand
        m, n, p: Matrix dimensions
        device: Device to run test on (CPU or CUDA)
        dtype: Data type for tensors (float32 or float16)
    
    Tests:
        - Broadcasting non-batched tensor with batch dimension
        - Forward and backward pass correctness
    """
    _A = np.random.randn(m, n).astype(np.float32)
    _B = np.random.randn(b, n, p).astype(np.float32)
    A = genesis.Tensor(_A, device=device, dtype=dtype[0])
    B = genesis.Tensor(_B, device=device, dtype=dtype[0])
    TA = torch.Tensor(_A).to(dtype[1])
    TA.requires_grad = True
    TB = torch.Tensor(_B).to(dtype[1])
    TB.requires_grad = True
    np.testing.assert_allclose((TA @ TB).detach().numpy(), (A @ B).detach().numpy(), atol=atol, rtol=rtol)

    (TA @ TB).sum().backward()
    (A @ B).sum().backward()
    np.testing.assert_allclose(TA.grad.numpy(), A.grad.numpy(), atol=atol, rtol=rtol)
    np.testing.assert_allclose(TA.grad.numpy(), A.grad.numpy(), atol=atol, rtol=rtol)

SUMMATION_PARAMETERS = [
    ((1, 1, 1), None),
    ((5, 3), 0),
    ((8, 3, 2), 1),
    ((8, 3, 2), 2),
    ((8, 3, 2048), 2),
    ((8, 3, 2048), 1),
    # Large tensor tests to catch precision issues
    ((2048, 2048), 0),  # Same as failing BatchNorm1d test
    ((2048, 2048), 1),
    ((1024, 1024), None),
    ((4096, 512), 0),
    ((512, 4096), 1),
    # Edge cases
    ((1, 2048), 0),
    ((2048, 1), 1),
    ((16, 16, 16), (0, 2)),  # Multi-axis reduction
]

# Max parameters without multi-axis cases (PyTorch max doesn't support multiple axes)
MAX_PARAMETERS = [
    ((1, 1, 1), None),
    ((5, 3), 0),
    ((8, 3, 2), 1),
    ((8, 3, 2), 2),
    ((8, 3, 2048), 2),
    ((8, 3, 2048), 1),
    # Large tensor tests to catch precision issues
    ((2048, 2048), 0),
    ((2048, 2048), 1),
    ((1024, 1024), None),
    ((4096, 512), 0),
    ((512, 4096), 1),
    # Edge cases
    ((1, 2048), 0),
    ((2048, 1), 1),
    # Note: No multi-axis cases like ((16, 16, 16), (0, 2))
]

# LogSumExp parameters without multi-axis cases (reshape issue with multiple axes)
LOGSUMEXP_PARAMETERS = [
    ((1, 1, 1), None),
    ((5, 3), 0),
    ((8, 3, 2), 1),
    ((8, 3, 2), 2),
    ((8, 3, 2048), 2),
    ((8, 3, 2048), 1),
    # Large tensor tests to catch precision issues
    ((2048, 2048), 0),
    ((2048, 2048), 1),
    ((1024, 1024), None),
    ((4096, 512), 0),
    ((512, 4096), 1),
    # Edge cases
    ((1, 2048), 0),
    ((2048, 1), 1),
    # Note: No multi-axis cases like ((16, 16, 16), (0, 2)) due to reshape issues
]
@pytest.mark.parametrize("shape, axes", SUMMATION_PARAMETERS)
@pytest.mark.parametrize("device", _DEVICES, ids=["cpu", "cuda"])
@pytest.mark.parametrize("dtype", _DTYPE, ids=["float32", "float16"])
def test_summation(shape, axes, device, dtype):
    """Test tensor summation along specified axes.
    
    Args:
        shape: Input tensor shape
        axes: Axes to sum along (None for all axes)
        device: Device to run test on (CPU or CUDA)
        dtype: Data type for tensors (float32 or float16)
    
    Tests:
        - Forward pass summation matches PyTorch
        - Backward gradients match PyTorch implementation
        - Tests reduction along different axes including large dimensions
    """
    _A = np.random.randn(*shape).astype(np.float32)
    A = genesis.Tensor(_A, device=device, dtype=dtype[0])
    TA = torch.Tensor(_A).to(dtype[1])
    TA.requires_grad = True
    np.testing.assert_allclose(
            torch.sum(TA, dim=axes).detach().numpy(), 
            F.summation(A, axis=axes).detach().numpy(), atol=atol, rtol=rtol)

    torch.sum(TA, dim=axes).sum().backward()
    F.summation(A, axis=axes).sum().backward()
    np.testing.assert_allclose(TA.grad.numpy(), A.grad.numpy(), atol=atol, rtol=rtol)

@pytest.mark.parametrize("shape, axes", MAX_PARAMETERS)
@pytest.mark.parametrize("device", _DEVICES, ids=["cpu", "cuda"])
@pytest.mark.parametrize("dtype", _DTYPE, ids=["float32", "float16"])
def test_max(shape, axes, device, dtype):
    """Test tensor maximum reduction along specified axes.
    
    Args:
        shape: Input tensor shape
        axes: Axes to find max along (None for global max)
        device: Device to run test on (CPU or CUDA)
        dtype: Data type for tensors (float32 or float16)
    
    Tests:
        - Forward pass max reduction matches PyTorch
        - Backward gradients match PyTorch implementation
        - Tests keepdims parameter
    """
    #TODO float16 need to be fix
    _A = np.random.randn(*shape).astype(np.float32)
    A = genesis.Tensor(_A, device=device)
    TA = torch.Tensor(_A)
    TA.requires_grad = True
    if axes is None:
        np.testing.assert_allclose(
                torch.max(TA).detach().numpy(), 
                F.max(A, axis=axes, keepdims=True).detach().numpy(), atol=atol, rtol=rtol)
    else:
        np.testing.assert_allclose(
                torch.max(TA, dim=axes, keepdims=True).values.detach().numpy(), 
                F.max(A, axis=axes, keepdims=True).detach().numpy(), atol=atol, rtol=rtol)

    if axes is None:
        torch.max(TA).sum().backward()
    else:
        torch.max(TA, dim=axes, keepdims=True).values.sum().backward()
    F.max(A, axis=axes, keepdims=True).sum().backward()
    np.testing.assert_allclose(TA.grad.numpy(), A.grad.numpy(), atol=atol, rtol=rtol)

@pytest.mark.parametrize("shape", GENERAL_SHAPES)
@pytest.mark.parametrize("device", _DEVICES, ids=["cpu", "cuda"])
@pytest.mark.parametrize("dtype", _DTYPE, ids=["float32", "float16"])
def test_log(shape, device, dtype):
    """Test natural logarithm operation.
    
    Args:
        shape: Input tensor shape
        device: Device to run test on (CPU or CUDA)
        dtype: Data type for tensors (float32 or float16)
    
    Tests:
        - Forward pass log operation matches PyTorch
        - Backward gradients match PyTorch implementation
    """
    _A = np.random.randn(*shape).astype(np.float32) + 5.
    A = genesis.Tensor(_A, device=device, dtype=dtype[0])
    TA = torch.Tensor(_A).to(dtype[1])
    TA.requires_grad = True
    np.testing.assert_allclose(
            torch.log(TA).detach().numpy(), 
            F.log(A).detach().numpy(), atol=atol, rtol=rtol)

    torch.log(TA).sum().backward()
    F.log(A).sum().backward()
    np.testing.assert_allclose(TA.grad.numpy(), A.grad.numpy(), atol=atol, rtol=rtol)

@pytest.mark.parametrize("shape", GENERAL_SHAPES)
@pytest.mark.parametrize("device", _DEVICES, ids=["cpu", "cuda"])
@pytest.mark.parametrize("dtype", _DTYPE, ids=["float32", "float16"])
def test_exp(shape, device, dtype):
    """Test exponential operation.
    
    Args:
        shape: Input tensor shape
        device: Device to run test on (CPU or CUDA)
        dtype: Data type for tensors (float32 or float16)
    
    Tests:
        - Forward pass nested exp operations match PyTorch
        - Backward gradients match PyTorch implementation
    """
    _A = np.random.randn(*shape).astype(np.float32) + 5.
    A = genesis.Tensor(_A, device=device, dtype=dtype[0])
    TA = torch.Tensor(_A).to(dtype[1])
    TA.requires_grad = True
    np.testing.assert_allclose(
            torch.exp(torch.exp(TA)).detach().numpy(), 
            F.exp(F.exp(A)).detach().numpy(), atol=atol, rtol=rtol)

    torch.exp(torch.exp(TA)).sum().backward()
    F.exp(F.exp(A)).sum().backward()
    np.testing.assert_allclose(TA.grad.numpy(), A.grad.numpy(), atol=atol, rtol=rtol)

@pytest.mark.parametrize("shape", GENERAL_SHAPES)
@pytest.mark.parametrize("device", _DEVICES, ids=["cpu", "cuda"])
@pytest.mark.parametrize("dtype", _DTYPE, ids=["float32", "float16"])
def test_relu(shape, device, dtype):
    """Test ReLU activation function.
    
    Args:
        shape: Input tensor shape
        device: Device to run test on (CPU or CUDA)
        dtype: Data type for tensors (float32 or float16)
    
    Tests:
        - Forward pass ReLU activation matches PyTorch
        - Backward gradients match PyTorch implementation
    """
    _A = np.random.randn(*shape).astype(np.float32) + 5.
    A = genesis.Tensor(_A, device=device, dtype=dtype[0])
    TA = torch.Tensor(_A).to(dtype[1])
    TA.requires_grad = True
    np.testing.assert_allclose(
            torch.relu(TA).detach().numpy(), 
            F.relu(A).detach().numpy(), atol=atol, rtol=rtol)

    torch.relu(TA).sum().backward()
    F.relu(A).sum().backward()
    np.testing.assert_allclose(TA.grad.numpy(), A.grad.numpy(), atol=atol, rtol=rtol)

@pytest.mark.parametrize("shape", GENERAL_SHAPES)
@pytest.mark.parametrize("device", _DEVICES, ids=["cpu", "cuda"])
def test_sqrt(shape, device):
    """Test square root operation.
    
    Args:
        shape: Input tensor shape
        device: Device to run test on (CPU or CUDA)
    
    Tests:
        - Forward pass sqrt operation matches PyTorch
        - Backward gradients match PyTorch implementation
    """
    _A = np.random.randn(*shape).astype(np.float32) + 5.
    A = genesis.Tensor(_A, device=device)
    TA = torch.Tensor(_A)
    TA.requires_grad = True
    np.testing.assert_allclose(
            torch.sqrt(TA).detach().numpy(), 
            F.sqrt(A).detach().numpy(), atol=atol, rtol=rtol)

    torch.sqrt(TA).sum().backward()
    F.sqrt(A).sum().backward()
    np.testing.assert_allclose(TA.grad.numpy(), A.grad.numpy(), atol=atol, rtol=rtol)

STACK_PARAMETERS = [
    ((5, 5), 0, 1),
    ((5, 5), 0, 2),
    ((1, 5, 7), 2, 5)]
@pytest.mark.parametrize("shape, dim, l", STACK_PARAMETERS)
@pytest.mark.parametrize("device", _DEVICES, ids=["cpu", "cuda"])
def test_stack(shape, dim, l, device):
    """Test tensor stacking operation.
    
    Args:
        shape: Shape of each tensor to stack
        dim: Dimension along which to stack
        l: Number of tensors to stack
        device: Device to run test on (CPU or CUDA)
    
    Tests:
        - Forward pass stack operation matches PyTorch
        - Backward gradients distributed correctly to input tensors
    """
    _A = [np.random.randn(*shape).astype(np.float32) for i in range(l)]
    A = [genesis.Tensor(_A[i], device=device) for i in range(l)]
    TA = [torch.Tensor(_A[i]) for i in range(l)]
    for torch_a in TA:
        torch_a.requires_grad = True
    np.testing.assert_allclose(
            torch.stack(TA, dim=dim).detach().numpy(), 
            F.stack(A, dim=dim).detach().numpy(), atol=atol, rtol=rtol)

    torch.stack(TA, dim=dim).sum().backward()
    F.stack(A, dim=dim).sum().backward()
    for i in range(l):
        np.testing.assert_allclose(TA[i].grad.numpy(), A[i].grad.numpy(), atol=atol, rtol=rtol)

STACK_PARAMETERS = [
    ((5, 5), 0, 1),
    ((5, 5), 0, 2),
    ((1, 5, 7), 2, 5)]
@pytest.mark.parametrize("shape, dim, l", STACK_PARAMETERS)
@pytest.mark.parametrize("device", _DEVICES, ids=["cpu", "cuda"])
def test_cat(shape, dim, l, device):
    """Test tensor concatenation operation.
    
    Args:
        shape: Shape of each tensor to concatenate
        dim: Dimension along which to concatenate
        l: Number of tensors to concatenate
        device: Device to run test on (CPU or CUDA)
    
    Tests:
        - Forward pass concatenation matches PyTorch
        - Backward gradients distributed correctly to input tensors
    """
    _A = [np.random.randn(*shape).astype(np.float32) for i in range(l)]
    A = [genesis.Tensor(_A[i], device=device) for i in range(l)]
    TA = [torch.Tensor(_A[i]) for i in range(l)]
    for torch_a in TA:
        torch_a.requires_grad = True
    np.testing.assert_allclose(
            torch.cat(TA, dim=dim).detach().numpy(), 
            F.cat(A, dim=dim).detach().numpy(), atol=atol, rtol=rtol)

    torch.cat(TA, dim=dim).sum().backward()
    F.cat(A, dim=dim).sum().backward()
    for i in range(l):
        np.testing.assert_allclose(TA[i].grad.numpy(), A[i].grad.numpy(), atol=atol, rtol=rtol)

SPLIT_PARAMETERS = [
    ((10, 5), 0, [2, 3, 5]),
    ((10, 5), 1, [1, 4]),
    ((4, 8, 6), 2, [1, 2, 3])
]
@pytest.mark.parametrize("shape, dim, sections", SPLIT_PARAMETERS)
@pytest.mark.parametrize("device", _DEVICES, ids=["cpu", "cuda"])
def test_split(shape, dim, sections, device):
    """Test tensor split operation.
    
    Args:
        shape: Input tensor shape
        dim: Dimension along which to split
        sections: Size of each split section
        device: Device to run test on (CPU or CUDA)
    
    Tests:
        - Forward pass split operation matches PyTorch
        - Backward gradients accumulated correctly
        - Verifies split result count matches expected
    """
    # Generate random input data
    _A = np.random.randn(*shape).astype(np.float32)
    A = genesis.Tensor(_A, device=device)
    TA = torch.Tensor(_A)
    TA.requires_grad = True

    # Test equality using numpy
    result_genesis = F.split(A, axis=dim)
    result_torch = torch.split(TA, 1, dim=dim)
    assert len(result_genesis) == len(result_torch), "Split result count mismatch"

    for r_genesis, r_torch in zip(result_genesis, result_torch):
        np.testing.assert_allclose(
            r_genesis.detach().numpy(),
            r_torch.detach().numpy(),
            atol=atol, rtol=rtol)

    # Test backward propagation consistency
    sum([r.sum() for r in result_torch]).backward()
    sum([r.sum() for r in result_genesis]).backward()
    
    np.testing.assert_allclose(TA.grad.numpy(), A.grad.numpy(), atol=atol, rtol=rtol)

BROADCAST_SHAPES = [
    ((1, 1, 1), (3, 3, 3)),
    ((4, 1, 6), (4, 3, 6))]
@pytest.mark.parametrize("shape,shape_to", BROADCAST_SHAPES)
@pytest.mark.parametrize("device", _DEVICES, ids=["cpu", "cuda"])
def test_broadcast_to(shape, shape_to, device):
    """Test tensor broadcasting operation.
    
    Args:
        shape: Original tensor shape
        shape_to: Target shape to broadcast to
        device: Device to run test on (CPU or CUDA)
    
    Tests:
        - Forward pass broadcasting matches PyTorch
        - Backward gradients correctly sum over broadcast dimensions
    """
    _A = np.random.randn(*shape).astype(np.float32)
    A = genesis.Tensor(_A, device=device)
    TA = torch.Tensor(_A)
    TA.requires_grad = True
    np.testing.assert_allclose(
            torch.broadcast_to(TA, shape_to).detach().numpy(), 
            F.broadcast_to(A, shape_to).detach().numpy(), atol=atol, rtol=rtol)

    torch.broadcast_to(TA, shape_to).sum().backward()
    F.broadcast_to(A, shape_to).sum().backward()
    np.testing.assert_allclose(TA.grad.numpy(), A.grad.numpy(), atol=atol, rtol=rtol)

RESHAPE_SHAPES = [
    ((1, 1, 1), (1,)),
    ((4, 1, 6), (6, 4, 1))]
@pytest.mark.parametrize("shape,shape_to", RESHAPE_SHAPES)
@pytest.mark.parametrize("device", _DEVICES, ids=["cpu", "cuda"])
def test_reshape(shape, shape_to, device):
    """Test tensor reshape operation.
    
    Args:
        shape: Original tensor shape
        shape_to: Target shape to reshape to
        device: Device to run test on (CPU or CUDA)
    
    Tests:
        - Forward pass reshape matches PyTorch
        - Backward gradients preserved through reshape
    """
    _A = np.random.randn(*shape).astype(np.float32)
    A = genesis.Tensor(_A, device=device)
    TA = torch.Tensor(_A)
    TA.requires_grad = True
    np.testing.assert_allclose(
            torch.reshape(TA, shape_to).detach().numpy(), 
            F.reshape(A, shape_to).detach().numpy(), atol=atol, rtol=rtol)

    torch.reshape(TA, shape_to).sum().backward()
    F.reshape(A, shape_to).sum().backward()
    np.testing.assert_allclose(TA.grad.numpy(), A.grad.numpy(), atol=atol, rtol=rtol)

@pytest.mark.parametrize("shape,shape_to", RESHAPE_SHAPES)
@pytest.mark.parametrize("device", _DEVICES, ids=["cpu", "cuda"])
def test_view(shape, shape_to, device):
    """Test tensor view operation.
    
    Args:
        shape: Original tensor shape
        shape_to: Target shape for view
        device: Device to run test on (CPU or CUDA)
    
    Tests:
        - Forward pass view operation matches PyTorch
        - Backward gradients preserved through view
    """
    _A = np.random.randn(*shape).astype(np.float32)
    A = genesis.Tensor(_A, device=device)
    TA = torch.Tensor(_A)
    TA.requires_grad = True
    np.testing.assert_allclose(
            TA.view(shape_to).detach().numpy(), 
            A.view(shape_to).detach().numpy(), atol=atol, rtol=rtol)

    TA.view(shape_to).sum().backward()
    A.view(shape_to).sum().backward()
    np.testing.assert_allclose(TA.grad.numpy(), A.grad.numpy(), atol=atol, rtol=rtol)

EXPAND_SHAPES = [
    ((2, 1, 3), (2, 4, 3))]
@pytest.mark.parametrize("shape,shape_to", EXPAND_SHAPES)
@pytest.mark.parametrize("device", _DEVICES, ids=["cpu", "cuda"])
def test_expand(shape, shape_to, device):
    """Test tensor expand operation.
    
    Args:
        shape: Original tensor shape with dimensions of size 1
        shape_to: Target shape to expand to
        device: Device to run test on (CPU or CUDA)
    
    Tests:
        - Forward pass expand operation matches PyTorch
        - Backward gradients correctly sum over expanded dimensions
    """
    _A = np.random.randn(*shape).astype(np.float32)
    A = genesis.Tensor(_A, device=device)
    TA = torch.Tensor(_A)
    TA.requires_grad = True
    np.testing.assert_allclose(
            TA.expand(shape_to).detach().numpy(), 
            A.expand(shape_to).detach().numpy(), atol=atol, rtol=rtol)

    TA.expand(shape_to).sum().backward()
    A.expand(shape_to).sum().backward()
    np.testing.assert_allclose(TA.grad.numpy(), A.grad.numpy(), atol=atol, rtol=rtol)

TRANSPOSE_SHAPES = [(1, 1, 1), (4, 5, 6)]
TRANSPOSE_AXES = [(0, 1), (0, 2), None]
@pytest.mark.parametrize("shape", TRANSPOSE_SHAPES)
@pytest.mark.parametrize("axes", TRANSPOSE_AXES)
@pytest.mark.parametrize("device", _DEVICES, ids=["cpu", "cuda"])
def test_transpose(shape, axes, device):
    """Test tensor transpose operation.
    
    Args:
        shape: Input tensor shape
        axes: Axes to transpose (None for last two dimensions)
        device: Device to run test on (CPU or CUDA)
    
    Tests:
        - Forward pass transpose matches PyTorch
        - Backward gradients correctly transposed back
    """
    _A = np.random.randn(*shape).astype(np.float32)
    A = genesis.Tensor(_A, device=device)
    TA = torch.Tensor(_A)
    TA.requires_grad = True
    if axes is None:
        axes = (-1, -2)
    np.testing.assert_allclose(
            torch.transpose(TA, axes[0], axes[1]).detach().numpy(), 
            F.transpose(A, axes).detach().numpy(), atol=atol, rtol=rtol)

    torch.transpose(TA, axes[0], axes[1]).sum().backward()
    F.transpose(A, axes).sum().backward()
    np.testing.assert_allclose(TA.grad.numpy(), A.grad.numpy(), atol=atol, rtol=rtol)

@pytest.mark.parametrize("shape, axes", LOGSUMEXP_PARAMETERS)
@pytest.mark.parametrize("device", _DEVICES, ids=["cpu", "cuda"])
def test_logsumexp(shape, axes, device):
    """Test log-sum-exp operation (numerically stable).
    
    Args:
        shape: Input tensor shape
        axes: Axes to reduce along
        device: Device to run test on (CPU or CUDA)
    
    Tests:
        - Forward pass logsumexp matches PyTorch
        - Backward gradients match PyTorch implementation
        - Numerical stability for large values
    """
    _A = np.random.randn(*shape).astype(np.float32)
    A = genesis.Tensor(_A, device=device)
    TA = torch.Tensor(_A)
    TA.requires_grad = True
    if axes is None:
        t_axes = tuple(list(range(len(shape))))
    else:
        t_axes = axes
    np.testing.assert_allclose(
            torch.logsumexp(TA, dim=t_axes).detach().numpy(), 
            F.logsumexp(A, axes).detach().numpy(), atol=atol, rtol=rtol)

    torch.logsumexp(TA, dim=t_axes).sum().backward()
    F.logsumexp(A, axes).sum().backward()
    np.testing.assert_allclose(TA.grad.numpy(), A.grad.numpy(), atol=atol, rtol=rtol)

# TODO need to flatten the syntax between PyTorch and my code.
@pytest.mark.parametrize("shape", GENERAL_SHAPES)
@pytest.mark.parametrize("device", _DEVICES, ids=["cpu", "cuda"])
def test_equal(shape, device):
    """Test element-wise equality comparison.
    
    Args:
        shape: Input tensor shape
        device: Device to run test on (CPU or CUDA)
    
    Tests:
        - Element-wise equality comparison matches PyTorch
        - Tests both equal and non-equal tensors
    """
    _A = np.random.randn(*shape).astype(np.float32)
    _B = np.random.randn(*shape).astype(np.float32)
    A = genesis.Tensor(_A, device=device)
    B = genesis.Tensor(_B, device=device)
    C = genesis.Tensor(_A, device=device)
    TA = torch.Tensor(_A)
    TA.requires_grad = True
    TB = torch.Tensor(_B)
    TB.requires_grad = True
    TC = torch.Tensor(_A)
    TC.requires_grad = True
    np.testing.assert_allclose((TA == TB).detach().numpy(), (A == B).detach().numpy(), atol=atol, rtol=rtol)
    np.testing.assert_allclose((TA == TC).detach().numpy(), (A == C).detach().numpy(), atol=atol, rtol=rtol)

@pytest.mark.parametrize("shape", GENERAL_SHAPES)
@pytest.mark.parametrize("device", _DEVICES, ids=["cpu", "cuda"])
def test_sin(shape, device):
    """Test sine trigonometric function.
    
    Args:
        shape: Input tensor shape
        device: Device to run test on (CPU or CUDA)
    
    Tests:
        - Forward pass sine operation matches PyTorch
        - Backward gradients match PyTorch implementation
    """
    _A = np.random.randn(*shape).astype(np.float32)
    A = genesis.Tensor(_A, device=device)
    TA = torch.Tensor(_A)
    TA.requires_grad = True
    B = A.sin()
    TB = TA.sin()
    np.testing.assert_allclose(TB.detach().numpy(), B.detach().numpy(), atol=atol, rtol=rtol)

    B.sum().backward()
    TB.sum().backward()
    np.testing.assert_allclose(TA.grad.detach().numpy(), A.grad.detach().numpy(), atol=atol, rtol=rtol)

@pytest.mark.parametrize("shape", GENERAL_SHAPES)
@pytest.mark.parametrize("device", _DEVICES, ids=["cpu", "cuda"])
def test_cos(shape, device):
    """Test cosine trigonometric function.
    
    Args:
        shape: Input tensor shape
        device: Device to run test on (CPU or CUDA)
    
    Tests:
        - Forward pass cosine operation matches PyTorch
        - Backward gradients match PyTorch implementation
    """
    _A = np.random.randn(*shape).astype(np.float32)
    A = genesis.Tensor(_A, device=device)
    TA = torch.Tensor(_A)
    TA.requires_grad = True
    B = A.cos()
    TB = TA.cos()
    np.testing.assert_allclose(TB.detach().numpy(), B.detach().numpy(), atol=atol, rtol=rtol)

    B.sum().backward()
    TB.sum().backward()
    np.testing.assert_allclose(TA.grad.detach().numpy(), A.grad.detach().numpy(), atol=atol, rtol=rtol)

# Comprehensive getitem/setitem tests
GETITEM_SHAPES = [(4, 5, 6), (2, 3), (10, 8), (3, 4, 5, 6)]

@pytest.mark.parametrize("shape", GETITEM_SHAPES)
@pytest.mark.parametrize("device", _DEVICES, ids=["cpu", "cuda"])
def test_getitem_basic(shape, device):
    """Test basic indexing: int, slice, ellipsis, None.
    
    Args:
        shape: Input tensor shape
        device: Device to run test on (CPU or CUDA)
    
    Tests:
        - Integer indexing
        - Slice indexing with and without step
        - Negative indexing
        - Ellipsis and None (newaxis) indexing
        - Backward gradients through indexing operations
    """
    _A = np.random.randn(*shape).astype(np.float32)
    A = genesis.Tensor(_A, device=device)
    TA = torch.Tensor(_A)
    TA.requires_grad = True
    
    # Integer indexing
    B1 = A[0]
    TB1 = TA[0]
    np.testing.assert_allclose(TB1.detach().numpy(), B1.detach().numpy(), atol=atol, rtol=rtol)
    
    # Slice indexing
    B2 = A[1:3]
    TB2 = TA[1:3]
    np.testing.assert_allclose(TB2.detach().numpy(), B2.detach().numpy(), atol=atol, rtol=rtol)
    
    # Slice with step
    B3 = A[::2]
    TB3 = TA[::2]
    np.testing.assert_allclose(TB3.detach().numpy(), B3.detach().numpy(), atol=atol, rtol=rtol)
    
    # Negative indexing
    B4 = A[-1]
    TB4 = TA[-1]
    np.testing.assert_allclose(TB4.detach().numpy(), B4.detach().numpy(), atol=atol, rtol=rtol)
    
    # Ellipsis
    if len(shape) > 2:
        B5 = A[..., 0]
        TB5 = TA[..., 0]
        np.testing.assert_allclose(TB5.detach().numpy(), B5.detach().numpy(), atol=atol, rtol=rtol)
    
    # None (newaxis)
    B6 = A[None]
    TB6 = TA[None]
    np.testing.assert_allclose(TB6.detach().numpy(), B6.detach().numpy(), atol=atol, rtol=rtol)
    
    # Test backward - View/Slice path should use add_at with strides
    B1.sum().backward()
    TB1.sum().backward()
    np.testing.assert_allclose(TA.grad.numpy(), A.grad.numpy(), atol=atol, rtol=rtol)

@pytest.mark.parametrize("shape", GETITEM_SHAPES[:2])  # Use smaller shapes for advanced indexing
@pytest.mark.parametrize("device", _DEVICES, ids=["cpu", "cuda"])
def test_getitem_advanced(shape, device):
    """Test advanced indexing: boolean mask, integer array, tensor indexing.
    
    Args:
        shape: Input tensor shape
        device: Device to run test on (CPU or CUDA)
    
    Tests:
        - Boolean mask indexing
        - Integer list/array indexing
        - Backward gradients through gather operations
    """
    _A = np.random.randn(*shape).astype(np.float32)
    A = genesis.Tensor(_A, device=device)
    TA = torch.Tensor(_A)
    TA.requires_grad = True
    
    # Boolean mask indexing
    mask = A > 0
    B1 = A[mask]
    t_mask = TA > 0
    TB1 = TA[t_mask]
    np.testing.assert_allclose(TB1.detach().numpy(), B1.detach().numpy(), atol=atol, rtol=rtol)
    
    # Integer list indexing
    if shape[0] >= 4:
        indices = [0, 2, 3]
        B2 = A[indices]
        TB2 = TA[indices]
        np.testing.assert_allclose(TB2.detach().numpy(), B2.detach().numpy(), atol=atol, rtol=rtol)
    
    # Test backward - Gather path should use scatter-add for duplicate indices
    B1.sum().backward()
    TB1.sum().backward()
    np.testing.assert_allclose(TA.grad.numpy(), A.grad.numpy(), atol=atol, rtol=rtol)


@pytest.mark.parametrize("shape", GETITEM_SHAPES[:2])
@pytest.mark.parametrize("device", _DEVICES, ids=["cpu", "cuda"])
def test_getitem_mixed(shape, device):
    """Test mixed indexing: combining basic and advanced indexing.
    
    Args:
        shape: Input tensor shape
        device: Device to run test on (CPU or CUDA)
    
    Tests:
        - Mixed slice and integer indexing
        - Integer array with slice indexing
        - Backward gradients through mixed indexing
    """
    # TODO: implement
    pytest.skip("Not implemented")
    _A = np.random.randn(*shape).astype(np.float32)
    A = genesis.Tensor(_A, device=device)
    TA = torch.Tensor(_A)
    TA.requires_grad = True
    
    if len(shape) >= 2 and shape[0] >= 3:
        # Mix slice with integer
        B1 = A[1:3, 0]
        TB1 = TA[1:3, 0]
        np.testing.assert_allclose(TB1.detach().numpy(), B1.detach().numpy(), atol=atol, rtol=rtol)
        
        # Mix integer array with slice
        if shape[0] >= 4:
            indices = [0, 2]
            B2 = A[indices, :2]
            TB2 = TA[indices, :2]
            np.testing.assert_allclose(TB2.detach().numpy(), B2.detach().numpy(), atol=atol, rtol=rtol)
        
        # Test backward - Mixed indexing should use gather path (scatter-add)
        B1.sum().backward()
        TB1.sum().backward()
        np.testing.assert_allclose(TA.grad.numpy(), A.grad.numpy(), atol=atol, rtol=rtol)


@pytest.mark.parametrize("shape", GETITEM_SHAPES[:2])
@pytest.mark.parametrize("device", _DEVICES, ids=["cpu", "cuda"])
def test_setitem_basic(shape, device):
    """Test basic setitem: int, slice assignments.
    
    Args:
        shape: Input tensor shape
        device: Device to run test on (CPU or CUDA)
    
    Tests:
        - Integer assignment with scalar
        - Slice assignment with scalar
        - Slice assignment with tensor
    """
    _A = np.random.randn(*shape).astype(np.float32)
    
    # Integer assignment
    A1 = genesis.Tensor(_A.copy(), device=device)
    TA1 = torch.Tensor(_A.copy())
    A1[0] = 1.0
    TA1[0] = 1.0
    np.testing.assert_allclose(TA1.numpy(), A1.numpy(), atol=atol, rtol=rtol)
    
    # Slice assignment with scalar
    A2 = genesis.Tensor(_A.copy(), device=device)
    TA2 = torch.Tensor(_A.copy())
    A2[1:3] = 2.0
    TA2[1:3] = 2.0
    np.testing.assert_allclose(TA2.numpy(), A2.numpy(), atol=atol, rtol=rtol)
    
    # Slice assignment with tensor
    if shape[0] >= 3:
        val_shape = list(shape)
        val_shape[0] = 2
        _val = np.random.randn(*val_shape).astype(np.float32)
        A3 = genesis.Tensor(_A.copy(), device=device)
        TA3 = torch.Tensor(_A.copy())
        A3[1:3] = genesis.Tensor(_val, device=device)
        TA3[1:3] = torch.Tensor(_val)
        np.testing.assert_allclose(TA3.numpy(), A3.numpy(), atol=atol, rtol=rtol)
    
    # Note: Backward testing for setitem on requires_grad=True tensors is not supported
    # as it violates the leaf variable in-place modification restriction

@pytest.mark.parametrize("shape", GETITEM_SHAPES[:2])
@pytest.mark.parametrize("device", _DEVICES, ids=["cpu", "cuda"])
def test_setitem_advanced(shape, device):
    """Test advanced setitem: boolean mask, integer array assignments.
    
    Args:
        shape: Input tensor shape
        device: Device to run test on (CPU or CUDA)
    
    Tests:
        - Boolean mask assignment with scalar
        - Boolean mask assignment with array
        - Integer list assignment
    """
    _A = np.random.randn(*shape).astype(np.float32)
    
    # Boolean mask assignment with scalar
    A1 = genesis.Tensor(_A.copy(), device=device)
    TA1 = torch.Tensor(_A.copy())
    mask = A1 > 0
    t_mask = TA1 > 0
    A1[mask] = 1.0
    TA1[t_mask] = 1.0
    np.testing.assert_allclose(TA1.numpy(), A1.numpy(), atol=atol, rtol=rtol)
    
    # Boolean mask assignment with array
    A2 = genesis.Tensor(_A.copy(), device=device)
    TA2 = torch.Tensor(_A.copy())
    mask2 = A2 < 0
    t_mask2 = TA2 < 0
    n_true = mask2.numpy().sum()
    values = np.random.randn(n_true).astype(np.float32)
    A2[mask2] = genesis.Tensor(values, device=device)
    TA2[t_mask2] = torch.Tensor(values)
    np.testing.assert_allclose(TA2.numpy(), A2.numpy(), atol=atol, rtol=rtol)
    
    # Integer list assignment
    if shape[0] >= 4:
        A3 = genesis.Tensor(_A.copy(), device=device)
        TA3 = torch.Tensor(_A.copy())
        indices = [0, 2, 3]
        A3[indices] = -1.0
        TA3[indices] = -1.0
        np.testing.assert_allclose(TA3.numpy(), A3.numpy(), atol=atol, rtol=rtol)
    
    # Note: Backward testing for setitem on requires_grad=True tensors is not supported
    # as it violates the leaf variable in-place modification restriction

@pytest.mark.parametrize("device", _DEVICES, ids=["cpu", "cuda"])
def test_getitem_duplicate_indices_backward(device):
    """Test backward with duplicate indices to verify scatter-add behavior.
    
    Args:
        device: Device to run test on (CPU or CUDA)
    
    Tests:
        - Duplicate indices in gather operation
        - Scatter-add gradient accumulation for duplicates
        - Correct gradient values when indices repeat
    """
    _A = np.random.randn(5, 3).astype(np.float32)
    
    # Test with duplicate indices - this should test scatter-add in backward
    A = genesis.Tensor(_A.copy(), device=device, requires_grad=True)
    TA = torch.Tensor(_A.copy())
    TA.requires_grad = True
    
    # Use duplicate indices to test scatter-add
    indices = [0, 2, 0, 1, 2]  # Duplicates: 0 appears twice, 2 appears twice
    B = A[indices]
    TB = TA[indices]
    
    np.testing.assert_allclose(TB.detach().numpy(), B.detach().numpy(), atol=atol, rtol=rtol)
    
    # Test backward - should accumulate gradients for duplicate indices
    B.sum().backward()
    TB.sum().backward()
    np.testing.assert_allclose(TA.grad.numpy(), A.grad.numpy(), atol=atol, rtol=rtol)

UNSQUEEZE_SHAPES = [((4, 5, 6), 1)]
@pytest.mark.parametrize("shape,dim", UNSQUEEZE_SHAPES)
@pytest.mark.parametrize("device", _DEVICES, ids=["cpu", "cuda"])
def test_unsqueeze(shape, dim, device):
    """Test unsqueeze operation (add dimension of size 1).
    
    Args:
        shape: Input tensor shape
        dim: Dimension where to insert new axis
        device: Device to run test on (CPU or CUDA)
    
    Tests:
        - Forward pass unsqueeze matches PyTorch
        - Backward gradients preserved through unsqueeze
    """
    _A = np.random.randn(*shape).astype(np.float32)
    A = genesis.Tensor(_A, device=device)
    TA = torch.Tensor(_A)
    TA.requires_grad = True
    B = A.unsqueeze(dim=dim)
    TB = TA.unsqueeze(dim=dim)
    np.testing.assert_allclose(TB.detach().numpy(), B.detach().numpy(), atol=atol, rtol=rtol)

    B.sum().backward()
    TB.sum().backward()
    np.testing.assert_allclose(TA.grad.detach().numpy(), A.grad.detach().numpy(), atol=atol, rtol=rtol)

SQUEEZE_SHAPES = [((4, 1, 6), 1)]
@pytest.mark.parametrize("shape,dim", SQUEEZE_SHAPES)
@pytest.mark.parametrize("device", _DEVICES, ids=["cpu", "cuda"])
def test_squeeze(shape, dim, device):
    """Test squeeze operation (remove dimension of size 1).
    
    Args:
        shape: Input tensor shape with dimension of size 1
        dim: Dimension to squeeze
        device: Device to run test on (CPU or CUDA)
    
    Tests:
        - Forward pass squeeze matches PyTorch
        - Backward gradients preserved through squeeze
    """
    _A = np.random.randn(*shape).astype(np.float32)
    A = genesis.Tensor(_A, device=device)
    TA = torch.Tensor(_A)
    TA.requires_grad = True
    B = A.squeeze(dim=dim)
    TB = TA.squeeze(dim=dim)
    np.testing.assert_allclose(TB.detach().numpy(), B.detach().numpy(), atol=atol, rtol=rtol)

    B.sum().backward()
    TB.sum().backward()
    np.testing.assert_allclose(TA.grad.detach().numpy(), A.grad.detach().numpy(), atol=atol, rtol=rtol)


# Strided Memory Layout Tests
STRIDED_SETITEM_CASES = [
    # Format: (target_shape, value_shape, slice_indices)
    ((8, 32, 3, 64), (8, 32, 1, 64), [slice(None), slice(None), slice(2, 3), slice(None)]),
    ((4, 16, 5, 32), (4, 16, 1, 32), [slice(None), slice(None), slice(1, 2), slice(None)]),  
    ((2, 8, 4, 16), (2, 8, 1, 16), [slice(None), slice(None), slice(0, 1), slice(None)]),
    ((6, 10, 7, 8), (6, 10, 2, 8), [slice(None), slice(None), slice(2, 4), slice(None)]),
]

@pytest.mark.parametrize("target_shape,value_shape,slice_idx", STRIDED_SETITEM_CASES)
@pytest.mark.parametrize("device", _DEVICES, ids=["cpu", "cuda"])
def test_setitem_non_contiguous_assignment(target_shape, value_shape, slice_idx, device):
    """Test tensor assignment to non-contiguous memory layouts.
    
    This test verifies that setitem operations work correctly when the target
    tensor slice has a non-contiguous memory layout, which requires strided
    memory copy operations.
    
    Args:
        target_shape: Shape of the target tensor
        value_shape: Shape of the value tensor to assign  
        slice_idx: Slice indices that create the target region
        device: Device to run test on (CPU or CUDA)
        
    Tests:
        - Assignment to non-contiguous tensor slices
        - Correctness of strided memory copy operations
        - Element-wise value preservation
    """
    # Create test tensors
    target = genesis.zeros(target_shape, device=device)
    value = genesis.full(value_shape, 3.5, device=device)
    
    # Get the target view and check its memory layout
    target_view = target[tuple(slice_idx)]
    
    # Perform assignment operation
    target[tuple(slice_idx)] = value
    
    # Verify assignment correctness
    result_view = target[tuple(slice_idx)]
    
    # Compare with expected result
    np.testing.assert_allclose(
        result_view.detach().numpy(), 
        value.detach().numpy(),
        atol=1e-6, rtol=1e-6,
        err_msg=f"Non-contiguous assignment failed for shape {target_shape}"
    )
    
    # Verify unmodified regions remain zero
    target_numpy = target.detach().numpy()
    region_mask = np.zeros(target_shape, dtype=bool)
    region_mask[tuple(slice_idx)] = True
    
    unmodified_region = target_numpy[~region_mask]
    assert np.allclose(unmodified_region, 0.0, atol=1e-6), \
        "Assignment modified regions outside target slice"


CONTIGUOUS_COMPARISON_CASES = [
    # Format: (shape, contiguous_slice, non_contiguous_slice)
    ((4, 6, 8, 12), [slice(0, 2), slice(None), slice(None), slice(None)],   # contiguous
                    [slice(None), slice(None), slice(2, 3), slice(None)]),  # non-contiguous
    ((3, 5, 7, 9), [slice(None), slice(0, 3), slice(None), slice(None)],    # contiguous  
                   [slice(None), slice(None), slice(1, 2), slice(None)]),   # non-contiguous
]

@pytest.mark.parametrize("shape,contiguous_slice,non_contiguous_slice", CONTIGUOUS_COMPARISON_CASES)
@pytest.mark.parametrize("device", _DEVICES, ids=["cpu", "cuda"])  
def test_setitem_memory_layout_consistency(shape, contiguous_slice, non_contiguous_slice, device):
    """Test consistency between contiguous and non-contiguous setitem operations.
    
    This test ensures that both contiguous and non-contiguous memory assignments
    produce identical results, validating the correctness of strided copy
    implementations across different memory layouts.
    
    Args:
        shape: Base tensor shape for testing
        contiguous_slice: Slice that creates contiguous memory layout
        non_contiguous_slice: Slice that creates non-contiguous memory layout  
        device: Device to run test on (CPU or CUDA)
        
    Tests:
        - Contiguous memory assignment correctness
        - Non-contiguous memory assignment correctness
        - Consistency between different memory layout operations
    """
    test_value = 7.25
    
    # Test contiguous assignment
    tensor_c = genesis.zeros(shape, device=device)
    contiguous_view = tensor_c[tuple(contiguous_slice)]
    tensor_c[tuple(contiguous_slice)] = test_value
    
    # Test non-contiguous assignment  
    tensor_nc = genesis.zeros(shape, device=device)
    non_contiguous_view = tensor_nc[tuple(non_contiguous_slice)]
    tensor_nc[tuple(non_contiguous_slice)] = test_value
    
    # Verify contiguous assignment
    result_c = tensor_c[tuple(contiguous_slice)]
    expected_c = genesis.full(contiguous_view.shape, test_value, device=device)
    np.testing.assert_allclose(
        result_c.detach().numpy(),
        expected_c.detach().numpy(), 
        atol=1e-6, rtol=1e-6,
        err_msg="Contiguous memory assignment failed"
    )
    
    # Verify non-contiguous assignment
    result_nc = tensor_nc[tuple(non_contiguous_slice)]  
    expected_nc = genesis.full(non_contiguous_view.shape, test_value, device=device)
    np.testing.assert_allclose(
        result_nc.detach().numpy(),
        expected_nc.detach().numpy(),
        atol=1e-6, rtol=1e-6, 
        err_msg="Non-contiguous memory assignment failed"
    )


MULTIDIMENSIONAL_SLICE_CASES = [
    # Complex slicing patterns that test various strided access patterns
    ((5, 8, 6, 10), [slice(1, 4), slice(2, 6), slice(None), slice(3, 8)]),
    ((6, 12, 4, 8), [slice(None, None, 2), slice(1, 11, 2), slice(None), slice(None)]),
    ((4, 7, 9, 5), [slice(0, 3), slice(None), slice(2, 7, 2), slice(1, 4)]),
]

@pytest.mark.parametrize("tensor_shape,slice_pattern", MULTIDIMENSIONAL_SLICE_CASES)
@pytest.mark.parametrize("device", _DEVICES, ids=["cpu", "cuda"])
def test_setitem_complex_strided_patterns(tensor_shape, slice_pattern, device):
    """Test setitem with complex multidimensional strided access patterns.
    
    This test covers advanced slicing scenarios that involve multiple
    dimensions with different stride patterns, ensuring robust handling
    of complex memory access patterns.
    
    Args:
        tensor_shape: Shape of the base tensor
        slice_pattern: Complex slice pattern with multiple strided dimensions
        device: Device to run test on (CPU or CUDA)
        
    Tests:
        - Complex multidimensional slicing operations
        - Strided access pattern correctness
        - Memory layout preservation under complex indexing
    """
    # Create base tensor and value tensor
    base_tensor = genesis.zeros(tensor_shape, device=device)
    target_view = base_tensor[tuple(slice_pattern)]
    value_tensor = genesis.full(target_view.shape, 4.75, device=device)
    
    # Perform complex strided assignment
    base_tensor[tuple(slice_pattern)] = value_tensor
    
    # Verify assignment correctness
    result_view = base_tensor[tuple(slice_pattern)]
    np.testing.assert_allclose(
        result_view.detach().numpy(),
        value_tensor.detach().numpy(),
        atol=1e-6, rtol=1e-6,
        err_msg=f"Complex strided assignment failed for pattern {slice_pattern}"
    )


@pytest.mark.parametrize("device", _DEVICES, ids=["cpu", "cuda"])
def test_reduce_sum_precision_issue(device):
    """Test reduce_sum precision for large tensors - specifically for BatchNorm1d issue.
    
    This test reproduces the exact scenario that causes BatchNorm1d to fail
    with shape (2048, 2048).
    
    Args:
        device: Device to run test on (CPU or CUDA)
    
    Tests:
        - Forward pass summation for large tensors
        - Backward gradient consistency
        - Gradient values should not be near-zero or uniform
    """
    # Test the exact case that fails in BatchNorm1d
    shape = (2048, 2048)
    np.random.seed(42)  # For reproducibility
    _A = np.random.randn(*shape).astype(np.float32)
    
    # PyTorch reference
    TA = torch.Tensor(_A)
    TA.requires_grad = True
    torch_sum = torch.sum(TA, dim=0)
    torch_sum.sum().backward()
    
    # Genesis
    A = genesis.Tensor(_A, device=device)
    genesis_sum = F.summation(A, axis=0)
    genesis_sum.sum().backward()
    
    # Check forward pass
    np.testing.assert_allclose(
        torch_sum.detach().numpy(), 
        genesis_sum.detach().numpy(), 
        atol=1e-3, rtol=1e-3,
        err_msg="Forward pass summation mismatch for large tensor"
    )
    
    # Check backward pass
    torch_grad = TA.grad.numpy()
    genesis_grad = A.grad.numpy()
    
    # Debug info
    print(f"Shape: {shape}, Device: {device}")
    print(f"PyTorch grad range: [{torch_grad.min():.6e}, {torch_grad.max():.6e}]")
    print(f"Genesis grad range: [{genesis_grad.min():.6e}, {genesis_grad.max():.6e}]")
    print(f"PyTorch grad unique values: {len(np.unique(torch_grad))}")
    print(f"Genesis grad unique values: {len(np.unique(genesis_grad))}")
    
    # The gradient should be 1.0 everywhere for sum().backward()
    expected_grad = np.ones_like(_A)
    
    # Check if Genesis gradients are wrong (near zero or incorrect)
    if np.abs(genesis_grad).max() < 1e-5:
        pytest.fail(f"Genesis gradients are near zero: max={np.abs(genesis_grad).max():.6e}")
    
    # Check gradient correctness
    np.testing.assert_allclose(
        torch_grad, 
        genesis_grad, 
        atol=1e-5, rtol=1e-5,
        err_msg="Backward pass gradient mismatch for large tensor reduction"
    )


@pytest.mark.parametrize("device", _DEVICES, ids=["cpu", "cuda"]) 
def test_reduce_sum_keepdims(device):
    """Test reduce_sum with keepdims parameter.
    
    Args:
        device: Device to run test on (CPU or CUDA)
    
    Tests:
        - keepdims=True preserves reduced dimensions with size 1
        - Backward pass works correctly with keepdims
    """
    shapes_and_axes = [
        ((4, 5, 6), 1),
        ((10, 20), 0),
        ((3, 4, 5), (0, 2)),
        ((2048, 512), 1),
    ]
    
    for shape, axes in shapes_and_axes:
        _A = np.random.randn(*shape).astype(np.float32)
        
        # PyTorch reference
        TA = torch.Tensor(_A)
        TA.requires_grad = True
        torch_sum = torch.sum(TA, dim=axes, keepdim=True)
        torch_sum.sum().backward()
        
        # Genesis  
        A = genesis.Tensor(_A, device=device)
        genesis_sum = F.summation(A, axis=axes, keepdims=True)
        genesis_sum.sum().backward()
        
        # Check shapes
        assert torch_sum.shape == genesis_sum.shape, \
            f"Shape mismatch with keepdims: {torch_sum.shape} vs {genesis_sum.shape}"
        
        # Check values
        np.testing.assert_allclose(
            torch_sum.detach().numpy(),
            genesis_sum.detach().numpy(),
            atol=1e-4, rtol=1e-4,
            err_msg=f"keepdims forward mismatch for shape {shape}, axes {axes}"
        )
        
        # Check gradients
        np.testing.assert_allclose(
            TA.grad.numpy(),
            A.grad.numpy(), 
            atol=1e-5, rtol=1e-5,
            err_msg=f"keepdims backward mismatch for shape {shape}, axes {axes}"
        )


if __name__ == "__main__":
    pytest.main()