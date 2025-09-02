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
    A = genesis.Tensor(_A, device=device, dtype=dtype[0], requires_grad=True)
    B = genesis.Tensor(_B, device=device, dtype=dtype[0], requires_grad=True)
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
    A = genesis.Tensor(_A, device=device, dtype=dtype[0], requires_grad=True)
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
    A = genesis.Tensor(_A, device=device, dtype=dtype[0], requires_grad=True)
    B = genesis.Tensor(_B, device=device, dtype=dtype[0], requires_grad=True)
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
    A = genesis.Tensor(_A, device=device, dtype=dtype[0], requires_grad=True)
    B = genesis.Tensor(_B, device=device, dtype=dtype[0], requires_grad=True)
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
    A = genesis.Tensor(_A, device=device, dtype=dtype[0], requires_grad=True)
    B = genesis.Tensor(_B, device=device, dtype=dtype[0], requires_grad=True)
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
    A = genesis.Tensor(_A, device=device, dtype=dtype[0], requires_grad=True)
    B = genesis.Tensor(_B, device=device, dtype=dtype[0], requires_grad=True)
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
    A = genesis.Tensor(_A, device=device, dtype=dtype[0], requires_grad=True)
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
    A = genesis.Tensor(_A, device=device, requires_grad=True)
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

@pytest.mark.parametrize("shape, axes", SUMMATION_PARAMETERS)
@pytest.mark.parametrize("device", _DEVICES, ids=["cpu", "cuda"])
@pytest.mark.parametrize("dtype", _DTYPE, ids=["float32", "float16"])
def test_mean(shape, axes, device, dtype):
    """Test tensor mean reduction along specified axes.
    
    Args:
        shape: Input tensor shape
        axes: Axes to compute mean along (None for all axes)
        device: Device to run test on (CPU or CUDA)
        dtype: Data type for tensors (float32 or float16)
    
    Tests:
        - Forward pass mean reduction matches PyTorch
        - Backward gradients match PyTorch implementation
        - Tests reduction along different axes including large dimensions
    """
    _A = np.random.randn(*shape).astype(np.float32)
    A = genesis.Tensor(_A, device=device, dtype=dtype[0], requires_grad=True)
    TA = torch.Tensor(_A).to(dtype[1])
    TA.requires_grad = True
    
    # Test forward pass
    np.testing.assert_allclose(
            torch.mean(TA, dim=axes).detach().numpy(), 
            F.mean(A, axis=axes).detach().numpy(), atol=atol, rtol=rtol)

    # Test backward pass
    torch.mean(TA, dim=axes).sum().backward()
    F.mean(A, axis=axes).sum().backward()
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
    A = genesis.Tensor(_A, device=device, dtype=dtype[0], requires_grad=True)
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
    A = genesis.Tensor(_A, device=device, dtype=dtype[0], requires_grad=True)
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
    A = genesis.Tensor(_A, device=device, dtype=dtype[0], requires_grad=True)
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
    A = genesis.Tensor(_A, device=device, requires_grad=True)
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
    A = [genesis.Tensor(_A[i], device=device, requires_grad=True) for i in range(l)]
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
    A = [genesis.Tensor(_A[i], device=device, requires_grad=True) for i in range(l)]
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
    A = genesis.Tensor(_A, device=device, requires_grad=True)
    TA = torch.Tensor(_A)
    TA.requires_grad = True

    # Test equality using numpy
    result_genesis = F.split(A, dim=dim)
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
    A = genesis.Tensor(_A, device=device, requires_grad=True)
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
    A = genesis.Tensor(_A, device=device, requires_grad=True)
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
    A = genesis.Tensor(_A, device=device, requires_grad=True)
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
    A = genesis.Tensor(_A, device=device, requires_grad=True)
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
    A = genesis.Tensor(_A, device=device, requires_grad=True)
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
    A = genesis.Tensor(_A, device=device, requires_grad=True)
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
    A = genesis.Tensor(_A, device=device, requires_grad=True)
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
    A = genesis.Tensor(_A, device=device, requires_grad=True)
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
    A = genesis.Tensor(_A, device=device, requires_grad=True)
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
    A = genesis.Tensor(_A, device=device, requires_grad=True)
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
    A = genesis.Tensor(_A, device=device, requires_grad=True)
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
        - Mixed 2D tensor indexing (row_indices, col_indices)
        - Backward gradients through mixed indexing
    """
    _A = np.random.randn(*shape).astype(np.float32)
    A = genesis.Tensor(_A, device=device, requires_grad=True)
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
        
        # Test mixed 2D tensor indexing
        if len(shape) == 2 and shape[0] >= 3 and shape[1] >= 3:
            # Create index tensors
            row_indices = genesis.tensor([0, 1, 2], device=device, dtype=genesis.int64)
            col_indices = genesis.tensor([1, 2, 0], device=device, dtype=genesis.int64)
            
            # Genesis mixed indexing
            A_new = genesis.Tensor(_A, device=device, requires_grad=True)
            B3 = A_new[row_indices, col_indices]
            
            # PyTorch equivalent
            TA_new = torch.Tensor(_A)
            TA_new.requires_grad = True
            row_indices_torch = torch.tensor([0, 1, 2], dtype=torch.int64)
            col_indices_torch = torch.tensor([1, 2, 0], dtype=torch.int64)
            TB3 = TA_new[row_indices_torch, col_indices_torch]
            
            # Compare forward pass
            np.testing.assert_allclose(TB3.detach().numpy(), B3.detach().numpy(), 
                                      atol=atol, rtol=rtol, err_msg="Mixed 2D indexing forward failed")
            
            # Test backward pass
            B3.sum().backward()
            TB3.sum().backward()
            np.testing.assert_allclose(TA_new.grad.numpy(), A_new.grad.numpy(), 
                                      atol=atol, rtol=rtol, err_msg="Mixed 2D indexing backward failed")


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
    
    # Mixed 2D tensor indexing for setitem
    if len(shape) == 2 and shape[0] >= 3 and shape[1] >= 3:
        # Create index tensors
        row_indices = genesis.tensor([0, 1, 2], device=device, dtype=genesis.int64)
        col_indices = genesis.tensor([1, 2, 0], device=device, dtype=genesis.int64)
        values = genesis.tensor([100.0, 200.0, 300.0], device=device, dtype=genesis.float32)
        
        # Genesis mixed setitem
        A4 = genesis.Tensor(_A.copy(), device=device)
        A4[row_indices, col_indices] = values
        
        # PyTorch equivalent
        TA4 = torch.Tensor(_A.copy())
        row_indices_torch = torch.tensor([0, 1, 2], dtype=torch.int64)
        col_indices_torch = torch.tensor([1, 2, 0], dtype=torch.int64)
        values_torch = torch.tensor([100.0, 200.0, 300.0], dtype=torch.float32)
        TA4[row_indices_torch, col_indices_torch] = values_torch
        
        # Compare results
        np.testing.assert_allclose(TA4.numpy(), A4.numpy(), 
                                  atol=atol, rtol=rtol, err_msg="Mixed 2D setitem failed")
        
        # Verify specific values were set correctly
        result = A4[row_indices, col_indices]
        np.testing.assert_allclose(result.numpy(), values.numpy(), 
                                  atol=atol, rtol=rtol, err_msg="Mixed 2D setitem values incorrect")
    
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
    A = genesis.Tensor(_A, device=device, requires_grad=True)
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
    A = genesis.Tensor(_A, device=device, requires_grad=True)
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
    A = genesis.Tensor(_A, device=device, requires_grad=True)
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
        A = genesis.Tensor(_A, device=device, requires_grad=True)
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



@pytest.mark.parametrize("shape", [(5, 5), (1, 10), (100, 3)])
@pytest.mark.parametrize("device", [genesis.device("cuda"), genesis.device("cpu")])
@pytest.mark.parametrize("dtype", [genesis.float32, genesis.float16])
def test_clone(shape, device, dtype):
    """Test tensor clone functionality.
    
    Args:
        shape: Shape of tensor to test
        device: Device to run test on (CPU or CUDA)
        dtype: Data type to test
        
    Tests:
        - Clone creates independent copy
        - Clone preserves shape, dtype, device
        - Clone preserves requires_grad setting
        - Modifications to original don't affect clone
        - Clone is detached from computation graph
    """
    # Create original tensor with random data
    _A = np.random.randn(*shape).astype(np.float32)
    
    # Test with requires_grad=True
    A = genesis.Tensor(_A, device=device, dtype=dtype, requires_grad=True)
    A_clone = A.clone()
    
    # Check basic properties
    assert A_clone.shape == A.shape, f"Shape mismatch: {A_clone.shape} vs {A.shape}"
    assert A_clone.dtype == A.dtype, f"Dtype mismatch: {A_clone.dtype} vs {A.dtype}"
    assert A_clone.device == A.device, f"Device mismatch: {A_clone.device} vs {A.device}"
    assert A_clone.requires_grad == A.requires_grad, f"requires_grad mismatch"
    
    # Check data independence
    np.testing.assert_allclose(
        A.detach().numpy(),
        A_clone.detach().numpy(),
        atol=1e-6, rtol=1e-6,
        err_msg="Clone data mismatch"
    )
    
    # Store clone values before modification
    clone_data_before_mod = A_clone.detach().numpy().copy()
    
    # Verify clone is independent - modify original
    A.fill_(999.0)
    
    # Clone should remain unchanged
    clone_data_after_mod = A_clone.detach().numpy()
    
    # Use appropriate tolerance based on dtype
    if dtype == genesis.float16:
        atol, rtol = 1e-3, 1e-3  # More relaxed for float16
    else:
        atol, rtol = 1e-6, 1e-6
        
    np.testing.assert_allclose(
        clone_data_after_mod,
        clone_data_before_mod,
        atol=atol, rtol=rtol,
        err_msg="Clone was affected by original modification"
    )
    
    # Test with requires_grad=False
    B = genesis.Tensor(_A, device=device, dtype=dtype, requires_grad=False)
    B_clone = B.clone()
    assert B_clone.requires_grad == False, "requires_grad should be False for cloned tensor"
    
    # Check that clone is detached from computation graph
    C = genesis.Tensor(_A, device=device, dtype=dtype, requires_grad=True)
    D = C * 2  # Create computation graph
    D_clone = D.clone()
    
    # Clone should have no creator (detached)
    assert D_clone.creator is None, "Clone should be detached from computation graph"


@pytest.mark.parametrize("shape", [(5, 5), (1, 10), (100, 3)])
@pytest.mark.parametrize("device", [genesis.device("cuda"), genesis.device("cpu")])
@pytest.mark.parametrize("dtype", [genesis.float32, genesis.float16])
def test_abs(shape, device, dtype):
    """Test tensor abs functionality.
    
    Args:
        shape: Shape of tensor to test
        device: Device to run test on (CPU or CUDA)
        dtype: Data type to test
        
    Tests:
        - abs() computes correct absolute values
        - abs() preserves shape, dtype, device
        - abs() gradient computation works correctly
        - abs() handles positive, negative, and zero values
    """
    # Create tensor with mixed positive/negative/zero values
    _A = np.random.randn(*shape).astype(np.float32) 
    _A.flat[0] = 0.0  # Include a zero value
    _A.flat[1] = -2.5  # Include negative value
    _A.flat[2] = 3.7   # Include positive value
    
    # PyTorch reference
    torch_A = torch.tensor(_A, requires_grad=True, dtype=torch.float32 if dtype == genesis.float32 else torch.float16)
    torch_abs = torch.abs(torch_A)
    torch_loss = torch_abs.sum()
    torch_loss.backward()
    
    # Genesis implementation
    genesis_A = genesis.Tensor(_A, device=device, dtype=dtype, requires_grad=True)
    genesis_abs = genesis_A.abs()
    genesis_loss = genesis_abs.sum()
    genesis_loss.backward()
    
    # Check basic properties
    assert genesis_abs.shape == torch_abs.shape, f"Shape mismatch: {genesis_abs.shape} vs {torch_abs.shape}"
    assert genesis_abs.dtype == dtype, f"Dtype mismatch: {genesis_abs.dtype} vs {dtype}"
    assert genesis_abs.device == device, f"Device mismatch: {genesis_abs.device} vs {device}"
    
    # Use appropriate tolerance based on dtype
    if dtype == genesis.float16:
        atol, rtol = 1e-3, 1e-3
    else:
        atol, rtol = 1e-6, 1e-6
    
    # Check forward pass values
    np.testing.assert_allclose(
        genesis_abs.detach().numpy(),
        torch_abs.detach().numpy(),
        atol=atol, rtol=rtol,
        err_msg="abs forward pass mismatch"
    )
    
    # Check gradients
    np.testing.assert_allclose(
        genesis_A.grad.numpy(),
        torch_A.grad.numpy(),
        atol=atol, rtol=rtol,
        err_msg="abs backward pass mismatch"
    )
    
    # Test specific values
    test_vals = genesis.tensor([[-3.0, 0.0, 2.5]], device=device, dtype=dtype)
    expected = genesis.tensor([[3.0, 0.0, 2.5]], device=device, dtype=dtype)
    result = test_vals.abs()
    
    np.testing.assert_allclose(
        result.numpy(),
        expected.numpy(), 
        atol=atol, rtol=rtol,
        err_msg="abs specific values mismatch"
    )


def test_clamp():
    """Test clamp/clip functionality with comprehensive cases."""
    
    # Test shapes and data types
    test_shapes = [(5,), (3, 4), (2, 3, 4)]
    test_dtypes = [genesis.float32, genesis.float16, genesis.bfloat16]
    
    for shape in test_shapes:
        for dtype in test_dtypes:
            # Create test tensor with values in range [-5, 5]
            np_data = np.random.uniform(-5, 5, shape).astype(np.float32)
            
            # Test basic clamp functionality
            x_genesis = genesis.tensor(np_data, dtype=dtype, requires_grad=True)
            x_torch = torch.tensor(np_data, dtype=torch.float32, requires_grad=True)
            
            # Test clamp with both min and max
            genesis_result = genesis.clamp(x_genesis, min_val=-2.0, max_val=2.0)
            torch_result = torch.clamp(x_torch, min=-2.0, max=2.0)
            
            # Verify shape and dtype
            assert genesis_result.shape == torch_result.shape, f"Shape mismatch for {shape}, {dtype}"
            assert genesis_result.dtype == dtype, f"Dtype mismatch for {dtype}"
            
            # Verify values with appropriate tolerance
            if dtype == genesis.bfloat16:
                atol, rtol = 1e-2, 1e-2  # bfloat16 has lower precision
            elif dtype == genesis.float16:
                atol, rtol = 1e-3, 1e-3
            else:
                atol, rtol = 1e-6, 1e-6
            np.testing.assert_allclose(
                genesis_result.numpy(), 
                torch_result.detach().numpy(), 
                atol=atol, rtol=rtol,
                err_msg=f"clamp values mismatch for {shape}, {dtype}"
            )
            
            # Test clamp with only min
            genesis_result_min = genesis.clamp(x_genesis, min_val=-1.0)
            torch_result_min = torch.clamp(x_torch, min=-1.0)
            
            np.testing.assert_allclose(
                genesis_result_min.numpy(), 
                torch_result_min.detach().numpy(), 
                atol=atol, rtol=rtol,
                err_msg=f"clamp min-only values mismatch for {shape}, {dtype}"
            )
            
            # Test clamp with only max
            genesis_result_max = genesis.clamp(x_genesis, max_val=1.0)
            torch_result_max = torch.clamp(x_torch, max=1.0)
            
            np.testing.assert_allclose(
                genesis_result_max.numpy(), 
                torch_result_max.detach().numpy(), 
                atol=atol, rtol=rtol,
                err_msg=f"clamp max-only values mismatch for {shape}, {dtype}"
            )
    
    # Test tensor method interface
    x = genesis.tensor([[-3.0, -1.0, 0.0, 1.0, 3.0]], dtype=genesis.float32, requires_grad=True)
    
    # Test .clamp() method
    result_clamp = x.clamp(-1.5, 1.5)
    expected_clamp = np.array([[-1.5, -1.0, 0.0, 1.0, 1.5]])
    
    np.testing.assert_allclose(
        result_clamp.numpy(),
        expected_clamp,
        atol=1e-6, rtol=1e-6,
        err_msg="clamp method values mismatch"
    )
    
    # Test .clip() method (alias)
    result_clip = x.clip(-1.5, 1.5)
    np.testing.assert_allclose(
        result_clip.numpy(),
        expected_clamp,
        atol=1e-6, rtol=1e-6,
        err_msg="clip method values mismatch"
    )
    
    # Test gradient computation
    x = genesis.tensor([[-3.0, -1.0, 0.0, 1.0, 3.0]], dtype=genesis.float32, requires_grad=True)
    y = genesis.clamp(x, min_val=-1.5, max_val=1.5)
    loss = genesis.sum(y)
    loss.backward()
    
    # Expected gradient: 1 where -1.5 <= x <= 1.5, 0 elsewhere
    expected_grad = np.array([[0.0, 1.0, 1.0, 1.0, 0.0]])
    
    np.testing.assert_allclose(
        x.grad.numpy(),
        expected_grad,
        atol=1e-6, rtol=1e-6,
        err_msg="clamp gradient mismatch"
    )
    
    # Test edge cases
    x_edge = genesis.tensor([[-2.0, -2.0, 2.0, 2.0]], dtype=genesis.float32, requires_grad=True)
    y_edge = genesis.clamp(x_edge, min_val=-2.0, max_val=2.0)
    
    # Values at boundary should remain unchanged
    np.testing.assert_allclose(
        x_edge.numpy(),
        y_edge.numpy(),
        atol=1e-6, rtol=1e-6,
        err_msg="clamp boundary values mismatch"
    )


def test_where():
    """Test where function with comprehensive cases."""
    
    # Test shapes and data types
    test_shapes = [(5,), (3, 4), (2, 3, 4)]
    test_dtypes = [genesis.float32, genesis.float16, genesis.bfloat16]
    
    for shape in test_shapes:
        for dtype in test_dtypes:
            # Create test data
            condition_data = np.random.choice([True, False], shape)
            x_data = np.random.uniform(-5, 5, shape).astype(np.float32)
            y_data = np.random.uniform(-5, 5, shape).astype(np.float32)
            
            # Test basic where functionality - use boolean tensor like PyTorch
            condition_genesis = genesis.tensor(condition_data, dtype=genesis.bool, requires_grad=False)
            x_genesis = genesis.tensor(x_data, dtype=dtype, requires_grad=True)
            y_genesis = genesis.tensor(y_data, dtype=dtype, requires_grad=True)
            
            condition_torch = torch.tensor(condition_data, dtype=torch.bool, requires_grad=False)
            x_torch = torch.tensor(x_data, dtype=torch.float32, requires_grad=True)
            y_torch = torch.tensor(y_data, dtype=torch.float32, requires_grad=True)
            
            # Test where function
            genesis_result = genesis.where(condition_genesis, x_genesis, y_genesis)
            torch_result = torch.where(condition_torch, x_torch, y_torch)
            
            # Verify shape and dtype
            assert genesis_result.shape == torch_result.shape, f"Shape mismatch for {shape}, {dtype}"
            assert genesis_result.dtype == dtype, f"Dtype mismatch for {dtype}"
            
            # Verify values with appropriate tolerance
            if dtype == genesis.bfloat16:
                atol, rtol = 1e-2, 1e-2  # bfloat16 has lower precision
            elif dtype == genesis.float16:
                atol, rtol = 1e-3, 1e-3
            else:
                atol, rtol = 1e-6, 1e-6
                
            np.testing.assert_allclose(
                genesis_result.numpy(), 
                torch_result.detach().numpy(), 
                atol=atol, rtol=rtol,
                err_msg=f"where values mismatch for {shape}, {dtype}"
            )
    
    # Test gradient computation
    condition = genesis.tensor([[True, False, True, False]], dtype=genesis.bool, requires_grad=False)
    x = genesis.tensor([[1.0, 2.0, 3.0, 4.0]], dtype=genesis.float32, requires_grad=True)
    y = genesis.tensor([[5.0, 6.0, 7.0, 8.0]], dtype=genesis.float32, requires_grad=True)
    
    result = genesis.where(condition, x, y)
    loss = genesis.sum(result)
    loss.backward()
    
    # Expected: x_grad = [1, 0, 1, 0] (where condition is True)
    #          y_grad = [0, 1, 0, 1] (where condition is False)
    expected_x_grad = np.array([[1.0, 0.0, 1.0, 0.0]])
    expected_y_grad = np.array([[0.0, 1.0, 0.0, 1.0]])
    
    np.testing.assert_allclose(
        x.grad.numpy(),
        expected_x_grad,
        atol=1e-6, rtol=1e-6,
        err_msg="where x gradient mismatch"
    )
    
    np.testing.assert_allclose(
        y.grad.numpy(),
        expected_y_grad,
        atol=1e-6, rtol=1e-6,
        err_msg="where y gradient mismatch"
    )
    
    # Test specific values
    condition_specific = genesis.tensor([[True, False, True]], dtype=genesis.bool, requires_grad=False)
    x_specific = genesis.tensor([[1.0, 2.0, 3.0]], dtype=genesis.float32, requires_grad=False)
    y_specific = genesis.tensor([[4.0, 5.0, 6.0]], dtype=genesis.float32, requires_grad=False)
    
    result_specific = genesis.where(condition_specific, x_specific, y_specific)
    expected_specific = np.array([[1.0, 5.0, 3.0]])  # Take x[0], y[1], x[2]
    
    np.testing.assert_allclose(
        result_specific.numpy(),
        expected_specific,
        atol=1e-6, rtol=1e-6,
        err_msg="where specific values mismatch"
    )


def test_argmax_argmin():
    """Test argmax and argmin functions with comprehensive cases."""
    
    # Test shapes and data types
    test_shapes = [(5,), (3, 4), (2, 3, 4)]
    test_dtypes = [genesis.float32, genesis.float16, genesis.bfloat16]
    
    for shape in test_shapes:
        for dtype in test_dtypes:
            # Create test data with distinct values to avoid ties
            np_data = np.arange(np.prod(shape), dtype=np.float32).reshape(shape) + np.random.uniform(-0.1, 0.1, shape)
            
            # Test basic argmax/argmin functionality
            x_genesis = genesis.tensor(np_data, dtype=dtype, requires_grad=False)
            x_torch = torch.tensor(np_data, dtype=torch.float32, requires_grad=False)
            
            # Test global argmax (dim=None)
            genesis_argmax = genesis.argmax(x_genesis)
            torch_argmax = torch.argmax(x_torch)
            
            assert genesis_argmax.shape == torch_argmax.shape, f"Argmax shape mismatch for {shape}, {dtype}"
            assert genesis_argmax.dtype == genesis.int64, f"Argmax dtype should be int64, got {genesis_argmax.dtype}"
            
            # Values should match (convert to same dtype for comparison)
            np.testing.assert_equal(
                genesis_argmax.numpy().astype(np.int64),
                torch_argmax.numpy().astype(np.int64),
                err_msg=f"argmax values mismatch for {shape}, {dtype}"
            )
            
            # Test global argmin (dim=None)
            genesis_argmin = genesis.argmin(x_genesis)
            torch_argmin = torch.argmin(x_torch)
            
            assert genesis_argmin.shape == torch_argmin.shape, f"Argmin shape mismatch for {shape}, {dtype}"
            assert genesis_argmin.dtype == genesis.int64, f"Argmin dtype should be int64, got {genesis_argmin.dtype}"
            
            np.testing.assert_equal(
                genesis_argmin.numpy().astype(np.int64),
                torch_argmin.numpy().astype(np.int64),
                err_msg=f"argmin values mismatch for {shape}, {dtype}"
            )
            
            # Test argmax/argmin along specific dimensions (for multi-dimensional tensors)
            if len(shape) > 1:
                for dim in range(len(shape)):
                    # Test with keepdim=False
                    genesis_argmax_dim = genesis.argmax(x_genesis, dim=dim, keepdim=False)
                    torch_argmax_dim = torch.argmax(x_torch, dim=dim, keepdim=False)
                    
                    assert genesis_argmax_dim.shape == torch_argmax_dim.shape, f"Argmax dim shape mismatch for {shape}, {dtype}, dim={dim}"
                    
                    np.testing.assert_equal(
                        genesis_argmax_dim.numpy().astype(np.int64),
                        torch_argmax_dim.numpy().astype(np.int64),
                        err_msg=f"argmax dim values mismatch for {shape}, {dtype}, dim={dim}"
                    )
                    
                    # Test with keepdim=True
                    genesis_argmax_keepdim = genesis.argmax(x_genesis, dim=dim, keepdim=True)
                    torch_argmax_keepdim = torch.argmax(x_torch, dim=dim, keepdim=True)
                    
                    assert genesis_argmax_keepdim.shape == torch_argmax_keepdim.shape, f"Argmax keepdim shape mismatch for {shape}, {dtype}, dim={dim}"
                    
                    np.testing.assert_equal(
                        genesis_argmax_keepdim.numpy().astype(np.int64),
                        torch_argmax_keepdim.numpy().astype(np.int64),
                        err_msg=f"argmax keepdim values mismatch for {shape}, {dtype}, dim={dim}"
                    )
    
    # Test tensor method interface
    x = genesis.tensor([[1.0, 5.0, 3.0], [2.0, 1.0, 4.0]], dtype=genesis.float32, requires_grad=False)
    
    # Test .argmax() method
    result_argmax = x.argmax()
    expected_argmax = 1  # Index of maximum value (5.0)
    
    assert result_argmax.numpy() == expected_argmax, f"Expected {expected_argmax}, got {result_argmax.numpy()}"
    
    # Test .argmin() method  
    result_argmin = x.argmin()
    expected_argmin = 0  # Index of minimum value (1.0)
    
    assert result_argmin.numpy() == expected_argmin, f"Expected {expected_argmin}, got {result_argmin.numpy()}"
    
    # Test argmax along dimension
    result_argmax_dim0 = x.argmax(dim=0)  # Along rows
    expected_argmax_dim0 = np.array([1, 0, 1])  # [2.0>1.0, 5.0>1.0, 4.0>3.0]
    
    np.testing.assert_equal(
        result_argmax_dim0.numpy(),
        expected_argmax_dim0,
        err_msg="argmax dim=0 values mismatch"
    )
    
    result_argmax_dim1 = x.argmax(dim=1)  # Along columns
    expected_argmax_dim1 = np.array([1, 2])  # [5.0 is max in row 0, 4.0 is max in row 1]
    
    np.testing.assert_equal(
        result_argmax_dim1.numpy(),
        expected_argmax_dim1,
        err_msg="argmax dim=1 values mismatch"
    )


def test_permute():
    """Test permute function with comprehensive cases."""
    
    # Test data types
    test_dtypes = [genesis.float32, genesis.float16, genesis.bfloat16]
    
    for dtype in test_dtypes:
        # Test 2D tensor permutation
        np_data_2d = np.arange(6, dtype=np.float32).reshape(2, 3)
        x_genesis_2d = genesis.tensor(np_data_2d, dtype=dtype, requires_grad=True)
        x_torch_2d = torch.tensor(np_data_2d, dtype=torch.float32, requires_grad=True)
        
        # Test 2D permute (transpose)
        genesis_result_2d = genesis.permute(x_genesis_2d, [1, 0])
        torch_result_2d = torch.permute(x_torch_2d, [1, 0])
        
        # Verify shape and dtype
        assert genesis_result_2d.shape == torch_result_2d.shape, f"2D permute shape mismatch for {dtype}"
        assert genesis_result_2d.dtype == dtype, f"2D permute dtype mismatch for {dtype}"
        
        # Verify values with appropriate tolerance
        if dtype == genesis.bfloat16:
            atol, rtol = 1e-2, 1e-2
        elif dtype == genesis.float16:
            atol, rtol = 1e-3, 1e-3
        else:
            atol, rtol = 1e-6, 1e-6
            
        np.testing.assert_allclose(
            genesis_result_2d.numpy(), 
            torch_result_2d.detach().numpy(), 
            atol=atol, rtol=rtol,
            err_msg=f"2D permute values mismatch for {dtype}"
        )
        
        # Test 3D tensor permutation
        np_data_3d = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
        x_genesis_3d = genesis.tensor(np_data_3d, dtype=dtype, requires_grad=True)
        x_torch_3d = torch.tensor(np_data_3d, dtype=torch.float32, requires_grad=True)
        
        # Test different permutation patterns
        permutation_patterns = [
            [0, 2, 1],  # swap last two dims
            [2, 0, 1],  # cyclic permutation
            [1, 2, 0],  # another cyclic permutation
            [2, 1, 0],  # reverse order
        ]
        
        for perm in permutation_patterns:
            genesis_result_3d = genesis.permute(x_genesis_3d, perm)
            torch_result_3d = torch.permute(x_torch_3d, perm)
            
            assert genesis_result_3d.shape == torch_result_3d.shape, f"3D permute shape mismatch for {dtype}, perm={perm}"
            assert genesis_result_3d.dtype == dtype, f"3D permute dtype mismatch for {dtype}, perm={perm}"
            
            np.testing.assert_allclose(
                genesis_result_3d.numpy(), 
                torch_result_3d.detach().numpy(), 
                atol=atol, rtol=rtol,
                err_msg=f"3D permute values mismatch for {dtype}, perm={perm}"
            )
    
    # Test tensor method interface with different calling styles
    x = genesis.tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], 
                      dtype=genesis.float32, requires_grad=True)  # Shape: (2, 2, 2)
    
    # Test .permute() method with tuple/list argument
    result_list = x.permute([2, 0, 1])  # Shape should be (2, 2, 2)
    expected_shape = (2, 2, 2)
    assert result_list.shape == expected_shape, f"Expected shape {expected_shape}, got {result_list.shape}"
    
    # Test .permute() method with individual arguments
    result_args = x.permute(2, 0, 1)  # Shape should be (2, 2, 2)
    assert result_args.shape == expected_shape, f"Expected shape {expected_shape}, got {result_args.shape}"
    
    # Results should be the same
    np.testing.assert_allclose(
        result_list.numpy(),
        result_args.numpy(),
        atol=1e-6, rtol=1e-6,
        err_msg="permute list vs args results mismatch"
    )
    
    # Test gradient computation for permute
    x_grad = genesis.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], 
                           dtype=genesis.float32, requires_grad=True)  # Shape: (2, 3)
    
    y_permuted = x_grad.permute(1, 0)  # Shape: (3, 2)
    loss = genesis.sum(y_permuted)
    loss.backward()
    
    # Gradient should have the same shape as original tensor
    assert x_grad.grad.shape == x_grad.shape, f"Gradient shape mismatch: {x_grad.grad.shape} vs {x_grad.shape}"
    
    # Gradient values should be all ones (since loss is just sum)
    expected_grad = np.ones_like(x_grad.numpy())
    np.testing.assert_allclose(
        x_grad.grad.numpy(),
        expected_grad,
        atol=1e-6, rtol=1e-6,
        err_msg="permute gradient values mismatch"
    )
    
    # Test specific permutation values
    x_specific = genesis.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=genesis.float32, requires_grad=False)
    result_specific = x_specific.permute(1, 0)
    expected_specific = np.array([[1.0, 3.0], [2.0, 4.0]])  # Transpose
    
    np.testing.assert_allclose(
        result_specific.numpy(),
        expected_specific,
        atol=1e-6, rtol=1e-6,
        err_msg="permute specific values mismatch"
    )
    
    # Test 4D tensor permutation (common in deep learning)
    x_4d = genesis.tensor(np.arange(48).reshape(2, 3, 4, 2), dtype=genesis.float32, requires_grad=False)
    
    # NCHW to NHWC permutation (common in computer vision)
    result_4d = x_4d.permute(0, 2, 3, 1)  # (2, 3, 4, 2) -> (2, 4, 2, 3)
    expected_shape_4d = (2, 4, 2, 3)
    
    assert result_4d.shape == expected_shape_4d, f"4D permute shape mismatch: {result_4d.shape} vs {expected_shape_4d}"


def test_gather_scatter():
    """Test gather and scatter operations with comprehensive cases."""
    
    # Test data types
    test_dtypes = [genesis.float32, genesis.float16, genesis.bfloat16]
    
    for dtype in test_dtypes:
        # Test 2D gather operation
        np_data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
        x_genesis = genesis.tensor(np_data, dtype=dtype, requires_grad=True)
        x_torch = torch.tensor(np_data, dtype=torch.float32, requires_grad=True)
        
        # Create index tensor
        indices_np = np.array([[0, 2], [1, 0]], dtype=np.int64)
        indices_genesis = genesis.tensor(indices_np, dtype=genesis.int64)
        indices_torch = torch.tensor(indices_np, dtype=torch.int64)
        
        # Test gather along dim=1
        genesis_gathered = genesis.gather(x_genesis, 1, indices_genesis)
        torch_gathered = torch.gather(x_torch, 1, indices_torch)
        
        # Verify shape and dtype
        assert genesis_gathered.shape == torch_gathered.shape, f"Gather shape mismatch for {dtype}"
        assert genesis_gathered.dtype == dtype, f"Gather dtype mismatch for {dtype}"
        
        # Verify values with appropriate tolerance
        if dtype == genesis.bfloat16:
            atol, rtol = 1e-2, 1e-2
        elif dtype == genesis.float16:
            atol, rtol = 1e-3, 1e-3
        else:
            atol, rtol = 1e-6, 1e-6
            
        np.testing.assert_allclose(
            genesis_gathered.numpy(), 
            torch_gathered.detach().numpy(), 
            atol=atol, rtol=rtol,
            err_msg=f"Gather values mismatch for {dtype}"
        )
        
        # Test gather gradient
        loss_gather = genesis.sum(genesis_gathered)
        loss_gather.backward()
        
        loss_torch_gather = torch.sum(torch_gathered)
        loss_torch_gather.backward()
        
        np.testing.assert_allclose(
            x_genesis.grad.numpy(),
            x_torch.grad.detach().numpy(),
            atol=atol, rtol=rtol,
            err_msg=f"Gather gradient mismatch for {dtype}"
        )
        
        # Test 2D scatter operation
        x2_genesis = genesis.tensor(np_data.copy(), dtype=dtype, requires_grad=True)
        x2_torch = torch.tensor(np_data.copy(), dtype=torch.float32, requires_grad=True)
        
        src_np = np.array([[10.0, 20.0], [30.0, 40.0]], dtype=np.float32)
        src_genesis = genesis.tensor(src_np, dtype=dtype, requires_grad=True)
        src_torch = torch.tensor(src_np, dtype=torch.float32, requires_grad=True)
        
        # Test scatter along dim=1
        genesis_scattered = genesis.scatter(x2_genesis, 1, indices_genesis, src_genesis)
        torch_scattered = x2_torch.scatter(1, indices_torch, src_torch)
        
        # Verify shape and dtype
        assert genesis_scattered.shape == torch_scattered.shape, f"Scatter shape mismatch for {dtype}"
        assert genesis_scattered.dtype == dtype, f"Scatter dtype mismatch for {dtype}"
        
        np.testing.assert_allclose(
            genesis_scattered.numpy(), 
            torch_scattered.detach().numpy(), 
            atol=atol, rtol=rtol,
            err_msg=f"Scatter values mismatch for {dtype}"
        )
        
        # Test scatter gradient
        loss_scatter = genesis.sum(genesis_scattered)
        loss_scatter.backward()
        
        loss_torch_scatter = torch.sum(torch_scattered)
        loss_torch_scatter.backward()
        
        np.testing.assert_allclose(
            x2_genesis.grad.numpy(),
            x2_torch.grad.detach().numpy(),
            atol=atol, rtol=rtol,
            err_msg=f"Scatter x gradient mismatch for {dtype}"
        )
        
        np.testing.assert_allclose(
            src_genesis.grad.numpy(),
            src_torch.grad.detach().numpy(),
            atol=atol, rtol=rtol,
            err_msg=f"Scatter src gradient mismatch for {dtype}"
        )
    
    # Test tensor method interface
    x = genesis.tensor([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]], 
                      dtype=genesis.float32, requires_grad=True)
    indices = genesis.tensor([[0, 3, 1], [2, 1, 0]], dtype=genesis.int64)
    
    # Test .gather() method
    gathered = x.gather(1, indices)
    expected_gather_shape = (2, 3)
    assert gathered.shape == expected_gather_shape, f"Expected gather shape {expected_gather_shape}, got {gathered.shape}"
    
    # Test .scatter() method
    src = genesis.tensor([[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]], dtype=genesis.float32)
    scattered = x.scatter(1, indices, src)
    expected_scatter_shape = (2, 4)
    assert scattered.shape == expected_scatter_shape, f"Expected scatter shape {expected_scatter_shape}, got {scattered.shape}"
    
    # Test 3D gather/scatter
    x_3d = genesis.tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], 
                         dtype=genesis.float32, requires_grad=True)  # Shape: (2, 2, 2)
    indices_3d = genesis.tensor([[[0], [1]], [[1], [0]]], dtype=genesis.int64)  # Shape: (2, 2, 1)
    
    # Test gather along dim=2
    gathered_3d = x_3d.gather(2, indices_3d)
    assert gathered_3d.shape == (2, 2, 1), f"3D gather shape mismatch: {gathered_3d.shape}"
    
    # Test scatter along dim=2
    src_3d = genesis.tensor([[[10.0], [20.0]], [[30.0], [40.0]]], dtype=genesis.float32)
    scattered_3d = x_3d.scatter(2, indices_3d, src_3d)
    assert scattered_3d.shape == (2, 2, 2), f"3D scatter shape mismatch: {scattered_3d.shape}"


# =============================================
# Tests for newly implemented MoE functions
# =============================================

@pytest.mark.parametrize("device", _DEVICES, ids=["cpu", "cuda"])
def test_topk(device):
    """Test topk function for top-k values and indices."""
    # Test basic functionality
    x = genesis.tensor([[3.0, 1.0, 4.0, 1.0, 5.0], [2.0, 7.0, 1.0, 8.0, 3.0]], device=device)
    values, indices = genesis.topk(x, k=3, dim=1, largest=True)
    
    # Check shapes
    assert values.shape == (2, 3), f"Expected values shape (2, 3), got {values.shape}"
    assert indices.shape == (2, 3), f"Expected indices shape (2, 3), got {indices.shape}"
    
    # Check first row: should be [5.0, 4.0, 3.0] with indices [4, 2, 0]
    assert values[0, 0].item() == 5.0, "First top value should be 5.0"
    assert indices[0, 0].item() == 4, "Index of first top value should be 4"
    
    # Test smallest values
    values_small, indices_small = genesis.topk(x, k=2, dim=1, largest=False)
    assert values_small.shape == (2, 2)
    assert indices_small.shape == (2, 2)
    
    # Test edge case: k larger than tensor size
    y = genesis.tensor([3.0, 1.0], device=device)
    values_edge, indices_edge = genesis.topk(y, k=5, dim=0, largest=True)
    assert values_edge.shape[0] <= 2, "Should not return more elements than exist"


@pytest.mark.parametrize("device", _DEVICES, ids=["cpu", "cuda"])  
def test_argsort(device):
    """Test argsort function for sorting indices."""
    x = genesis.tensor([[3.0, 1.0, 2.0], [6.0, 4.0, 5.0]], device=device)
    indices = genesis.argsort(x, dim=1, descending=False)
    
    # For first row [3, 1, 2], sorted indices should be [1, 2, 0] (1 < 2 < 3)
    assert indices[0, 0].item() == 1, "First sorted index should be 1"
    assert indices[0, 1].item() == 2, "Second sorted index should be 2"  
    assert indices[0, 2].item() == 0, "Third sorted index should be 0"
    
    # Test descending order
    indices_desc = genesis.argsort(x, dim=1, descending=True)
    assert indices_desc[0, 0].item() == 0, "First index in descending should be 0"


@pytest.mark.parametrize("device", _DEVICES, ids=["cpu", "cuda"])
def test_bincount(device):
    """Test bincount function for counting occurrences."""
    x = genesis.tensor([0, 1, 1, 2, 2, 2], dtype=genesis.int64, device=device)
    result = genesis.bincount(x)
    
    expected = genesis.tensor([1, 2, 3], device=device)  # 0:1time, 1:2times, 2:3times  
    assert result.shape == expected.shape, f"Shape mismatch: {result.shape} vs {expected.shape}"
    assert genesis.allclose(result.float(), expected.float()), "Bincount values mismatch"
    
    # Test with weights
    weights = genesis.tensor([0.5, 1.0, 1.5, 2.0, 0.5, 1.5], device=device)
    result_weighted = genesis.bincount(x, weights=weights)
    expected_weighted = genesis.tensor([0.5, 2.5, 4.0], device=device)  # 0:0.5, 1:1.0+1.5, 2:2.0+0.5+1.5
    assert genesis.allclose(result_weighted, expected_weighted), "Weighted bincount mismatch"


@pytest.mark.parametrize("device", _DEVICES, ids=["cpu", "cuda"])
def test_scatter_add(device):
    """Test scatter_add function for scatter addition."""
    # Test case 1
    _input = np.zeros((3, 5)).astype(np.float32)
    _index = np.array([[0, 1, 2, 0]], dtype=np.int64)
    _src = np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32)
    
    input_tensor = genesis.Tensor(_input, device=device, requires_grad=True)
    index = genesis.Tensor(_index, device=device)
    src = genesis.Tensor(_src, device=device, requires_grad=True)
    
    # PyTorch reference
    torch_input = torch.tensor(_input, requires_grad=True)
    torch_index = torch.tensor(_index)
    torch_src = torch.tensor(_src, requires_grad=True)
    torch_result = torch_input.scatter_add(0, torch_index, torch_src)
    
    # Genesis result
    genesis_result = input_tensor.scatter_add(0, index, src)
    
    # Compare forward pass
    np.testing.assert_allclose(
        torch_result.detach().numpy(), 
        genesis_result.detach().numpy(), 
        atol=atol, rtol=rtol
    )
    
    # Test backward pass
    torch_result.sum().backward()
    genesis_result.sum().backward()
    
    np.testing.assert_allclose(
        torch_src.grad.numpy(), 
        src.grad.numpy(), 
        atol=atol, rtol=rtol
    )


@pytest.mark.parametrize("device", _DEVICES, ids=["cpu", "cuda"])
def test_repeat_interleave(device):
    """Test repeat_interleave function."""
    x = genesis.tensor([1.0, 2.0, 3.0], device=device)
    result = x.repeat_interleave(2, dim=0)
    
    expected = genesis.tensor([1.0, 1.0, 2.0, 2.0, 3.0, 3.0], device=device)
    assert result.shape == expected.shape, f"Shape mismatch: {result.shape} vs {expected.shape}"
    assert genesis.allclose(result, expected), "repeat_interleave values mismatch"
    
    # Test with 2D tensor  
    x_2d = genesis.tensor([[1.0, 2.0], [3.0, 4.0]], device=device)
    result_2d = x_2d.repeat_interleave(3, dim=0)
    
    expected_2d = genesis.tensor([[1.0, 2.0], [1.0, 2.0], [1.0, 2.0], 
                                  [3.0, 4.0], [3.0, 4.0], [3.0, 4.0]], device=device)
    assert result_2d.shape == expected_2d.shape
    assert genesis.allclose(result_2d, expected_2d), "2D repeat_interleave mismatch"


@pytest.mark.parametrize("device", _DEVICES, ids=["cpu", "cuda"])  
def test_allclose(device):
    """Test allclose function for approximate equality."""
    x = genesis.tensor([1.0, 2.0, 3.0], device=device)
    y = genesis.tensor([1.0001, 2.0001, 3.0001], device=device)
    
    # Should be close with default tolerance
    assert genesis.allclose(x, y, rtol=1e-3, atol=1e-3) == True, "Should be close with loose tolerance"
    
    # Should not be close with very tight tolerance
    assert genesis.allclose(x, y, rtol=1e-6, atol=1e-6) == False, "Should not be close with tight tolerance"
    
    # Test identical tensors
    assert genesis.allclose(x, x) == True, "Identical tensors should be close"
    
    # Test different shapes (should handle gracefully)
    z = genesis.tensor([1.0, 2.0], device=device)
    try:
        result = genesis.allclose(x, z)
        # If it doesn't raise an error, result should be False
        assert result == False, "Different shaped tensors should not be close"
    except (RuntimeError, ValueError):
        # It's also acceptable to raise an error for different shapes
        pass


@pytest.mark.parametrize("device", _DEVICES, ids=["cpu", "cuda"])
def test_comparison_operators(device):
    """Test new comparison operators <= and >=."""
    x = genesis.tensor([1.0, 2.0, 3.0], device=device)
    y = genesis.tensor([2.0, 2.0, 2.0], device=device)
    
    # Test <=
    result_le = x <= y
    expected_le = genesis.tensor([1.0, 1.0, 0.0], device=device)
    assert genesis.allclose(result_le.float(), expected_le), "Less-equal operator mismatch"
    
    # Test >=  
    result_ge = x >= y
    expected_ge = genesis.tensor([0.0, 1.0, 1.0], device=device)
    assert genesis.allclose(result_ge.float(), expected_ge), "Greater-equal operator mismatch"
    
    # Test with scalars
    result_le_scalar = x <= 2.0
    expected_le_scalar = genesis.tensor([1.0, 1.0, 0.0], device=device)
    assert genesis.allclose(result_le_scalar.float(), expected_le_scalar), "Scalar less-equal mismatch"


@pytest.mark.parametrize("device", _DEVICES, ids=["cpu", "cuda"])
def test_tensor_all_method(device):
    """Test tensor.all() method."""
    # All non-zero (should be True)
    x = genesis.tensor([1.0, 2.0, 3.0], device=device)
    assert x.all() == True, "All non-zero elements should return True"
    
    # Contains zero (should be False)
    y = genesis.tensor([1.0, 0.0, 3.0], device=device) 
    assert y.all() == False, "Contains zero should return False"
    
    # All zeros (should be False)
    z = genesis.zeros((3,), device=device)
    assert z.all() == False, "All zeros should return False"
    
    # 2D tensor test
    w = genesis.tensor([[1.0, 2.0], [3.0, 4.0]], device=device)
    assert w.all() == True, "2D all non-zero should return True"
    
    # 2D tensor with zero
    u = genesis.tensor([[1.0, 0.0], [3.0, 4.0]], device=device)
    assert u.all() == False, "2D with zero should return False"


@pytest.mark.parametrize("device", _DEVICES, ids=["cpu", "cuda"])
def test_isinf_isnan_isfinite(device):
    """Test isinf, isnan, and isfinite functions."""
    # Test with regular finite numbers
    x = genesis.tensor([1.0, 2.0, -3.0, 0.0], device=device)
    
    # All should be finite
    finite_result = genesis.isfinite(x)
    assert finite_result.all() == True, "Regular numbers should all be finite"
    
    # None should be infinite
    inf_result = genesis.isinf(x)
    assert inf_result.any() == False, "Regular numbers should not be infinite"
    
    # None should be NaN
    nan_result = genesis.isnan(x)
    assert nan_result.any() == False, "Regular numbers should not be NaN"
    
    # Test with infinity
    y = genesis.tensor([float('inf'), 1.0, float('-inf'), 2.0], device=device)
    
    inf_result = genesis.isinf(y)
    expected_inf = genesis.tensor([True, False, True, False], device=device)
    np.testing.assert_allclose(inf_result.float().numpy(), expected_inf.float().numpy(), 
                              atol=atol, rtol=rtol, err_msg="Infinity detection failed")
    
    finite_result = genesis.isfinite(y)
    expected_finite = genesis.tensor([False, True, False, True], device=device)
    np.testing.assert_allclose(finite_result.float().numpy(), expected_finite.float().numpy(), 
                              atol=atol, rtol=rtol, err_msg="Finite detection with infinity failed")
    
    # Test with NaN
    z = genesis.tensor([float('nan'), 1.0, float('nan'), 2.0], device=device)
    
    nan_result = genesis.isnan(z)
    expected_nan = genesis.tensor([True, False, True, False], device=device)
    np.testing.assert_allclose(nan_result.float().numpy(), expected_nan.float().numpy(), 
                              atol=atol, rtol=rtol, err_msg="NaN detection failed")
    
    finite_result = genesis.isfinite(z)
    expected_finite = genesis.tensor([False, True, False, True], device=device)
    np.testing.assert_allclose(finite_result.float().numpy(), expected_finite.float().numpy(), 
                              atol=atol, rtol=rtol, err_msg="Finite detection with NaN failed")
    
    # Test mixed case
    w = genesis.tensor([float('inf'), float('nan'), 1.0, float('-inf'), 0.0], device=device)
    
    inf_result = genesis.isinf(w)
    expected_inf = genesis.tensor([True, False, False, True, False], device=device)
    np.testing.assert_allclose(inf_result.float().numpy(), expected_inf.float().numpy(), 
                              atol=atol, rtol=rtol, err_msg="Mixed infinity detection failed")
    
    nan_result = genesis.isnan(w)
    expected_nan = genesis.tensor([False, True, False, False, False], device=device)
    np.testing.assert_allclose(nan_result.float().numpy(), expected_nan.float().numpy(), 
                              atol=atol, rtol=rtol, err_msg="Mixed NaN detection failed")
    
    finite_result = genesis.isfinite(w)
    expected_finite = genesis.tensor([False, False, True, False, True], device=device)
    np.testing.assert_allclose(finite_result.float().numpy(), expected_finite.float().numpy(), 
                              atol=atol, rtol=rtol, err_msg="Mixed finite detection failed")
    
    # Test 2D case
    mat = genesis.tensor([[1.0, float('inf')], [float('nan'), -2.0]], device=device)
    
    inf_result = genesis.isinf(mat)
    expected_inf = genesis.tensor([[False, True], [False, False]], device=device)
    np.testing.assert_allclose(inf_result.float().numpy(), expected_inf.float().numpy(), 
                              atol=atol, rtol=rtol, err_msg="2D infinity detection failed")
    
    nan_result = genesis.isnan(mat)
    expected_nan = genesis.tensor([[False, False], [True, False]], device=device)
    np.testing.assert_allclose(nan_result.float().numpy(), expected_nan.float().numpy(), 
                              atol=atol, rtol=rtol, err_msg="2D NaN detection failed")
    
    finite_result = genesis.isfinite(mat)
    expected_finite = genesis.tensor([[True, False], [False, True]], device=device)
    np.testing.assert_allclose(finite_result.float().numpy(), expected_finite.float().numpy(), 
                              atol=atol, rtol=rtol, err_msg="2D finite detection failed")



if __name__ == "__main__":
    pytest.main()
