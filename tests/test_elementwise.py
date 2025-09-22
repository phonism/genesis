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
    backward_grad = out.op.gradient_as_tuple(genesis.tensor(c, device=args[0].device), out)
    error = sum(
            np.linalg.norm(backward_grad[i].numpy() - numerical_grad[i]) for i in range(len(args)))
    assert error < 4.2e-1
    return [g.numpy() for g in backward_grad]

_DEVICES = [
        genesis.device('cpu'),
        pytest.param(
            genesis.device("cuda"), 
            marks=pytest.mark.skipif(not genesis.cuda.is_available(), reason="No GPU"))]

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
    A = genesis.tensor(_A, device=device, dtype=dtype[0], requires_grad=True)
    B = genesis.tensor(_B, device=device, dtype=dtype[0], requires_grad=True)
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


@pytest.mark.parametrize("shape", GENERAL_SHAPES)
@pytest.mark.parametrize("device", _DEVICES, ids=["cpu", "cuda"])
@pytest.mark.parametrize("dtype", _DTYPE, ids=["float32", "float16"])
def test_add_inplace(shape, device, dtype):
    """Test in-place addition operation.

    Args:
        shape: Input tensor shape
        device: Device to run test on (CPU or CUDA)
        dtype: Data type for tensors (float32 or float16)

    Tests:
        - In-place operations work correctly on non-leaf tensors
        - Proper error handling for leaf tensors with requires_grad
        - Forward pass matches PyTorch behavior
    """
    if dtype[0] == genesis.float16:
        atol, rtol = 1e-3, 1e-3
        _A = np.random.randn(*shape).astype(np.float16)
        _B = np.random.randn(*shape).astype(np.float16)
    else:
        atol, rtol = 1e-5, 1e-5
        _A = np.random.randn(*shape).astype(np.float32)
        _B = np.random.randn(*shape).astype(np.float32)

    # Test 1: Verify that in-place on leaf with requires_grad raises error
    A_leaf = genesis.tensor(_A, device=device, dtype=dtype[0], requires_grad=True)
    B = genesis.tensor(_B, device=device, dtype=dtype[0])

    TA_leaf = torch.Tensor(_A).to(dtype[1])
    TA_leaf.requires_grad = True
    TB = torch.Tensor(_B).to(dtype[1])

    # Both should raise the same error
    with pytest.raises(RuntimeError, match="leaf Variable"):
        A_leaf += B

    with pytest.raises(RuntimeError, match="leaf Variable"):
        TA_leaf += TB

    # Test 2: In-place on non-leaf tensors should work
    A = genesis.tensor(_A, device=device, dtype=dtype[0], requires_grad=True)
    A_nonleaf = A * 1.0  # Create non-leaf tensor
    B = genesis.tensor(_B, device=device, dtype=dtype[0])

    TA = torch.Tensor(_A).to(dtype[1])
    TA.requires_grad = True
    TA_nonleaf = TA * 1.0  # Create non-leaf tensor
    TB = torch.Tensor(_B).to(dtype[1])

    # Perform in-place addition
    A_nonleaf += B
    TA_nonleaf += TB

    # Results should match
    np.testing.assert_allclose(
        TA_nonleaf.detach().numpy(),
        A_nonleaf.detach().numpy(),
        atol=atol, rtol=rtol
    )

    # Test 3: In-place on tensors without requires_grad should work
    A_no_grad = genesis.tensor(_A, device=device, dtype=dtype[0])
    B_no_grad = genesis.tensor(_B, device=device, dtype=dtype[0])

    TA_no_grad = torch.Tensor(_A).to(dtype[1])
    TB_no_grad = torch.Tensor(_B).to(dtype[1])

    A_no_grad += B_no_grad
    TA_no_grad += TB_no_grad

    np.testing.assert_allclose(
        TA_no_grad.numpy(),
        A_no_grad.detach().numpy(),
        atol=atol, rtol=rtol
    )


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
    A = genesis.tensor(_A, device=device, dtype=dtype[0], requires_grad=True)
    np.testing.assert_allclose(fn(TA, _B).detach().numpy(), fn(A, _B).detach().numpy(), atol=atol, rtol=rtol)

    fn(TA, _B).sum().backward()
    fn(A, _B).sum().backward()
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
    A = genesis.tensor(_A, device=device, dtype=dtype[0], requires_grad=True)
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
    A = genesis.tensor(_A, device=device, dtype=dtype[0], requires_grad=True)
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
    A = genesis.tensor(_A, device=device, dtype=dtype[0], requires_grad=True)
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
    A = genesis.tensor(_A, device=device, requires_grad=True)
    TA = torch.Tensor(_A)
    TA.requires_grad = True
    np.testing.assert_allclose(
            torch.sqrt(TA).detach().numpy(), 
            F.sqrt(A).detach().numpy(), atol=atol, rtol=rtol)

    torch.sqrt(TA).sum().backward()
    F.sqrt(A).sum().backward()
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
    A = genesis.tensor(_A, device=device, requires_grad=True)
    B = genesis.tensor(_B, device=device)
    C = genesis.tensor(_A, device=device)
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
    A = genesis.tensor(_A, device=device, requires_grad=True)
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
    A = genesis.tensor(_A, device=device, requires_grad=True)
    TA = torch.Tensor(_A)
    TA.requires_grad = True
    B = A.cos()
    TB = TA.cos()
    np.testing.assert_allclose(TB.detach().numpy(), B.detach().numpy(), atol=atol, rtol=rtol)

    B.sum().backward()
    TB.sum().backward()
    np.testing.assert_allclose(TA.grad.detach().numpy(), A.grad.detach().numpy(), atol=atol, rtol=rtol)


if __name__ == "__main__":
    pytest.main()
