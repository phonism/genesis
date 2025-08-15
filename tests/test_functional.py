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
        genesis.cpu(),
        pytest.param(
            genesis.cuda(), 
            marks=pytest.mark.skipif(not genesis.cuda().enabled(), reason="No GPU"))]

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
    ((8, 3, 2048), 1)
]
@pytest.mark.parametrize("shape, axes", SUMMATION_PARAMETERS)
@pytest.mark.parametrize("device", _DEVICES, ids=["cpu", "cuda"])
@pytest.mark.parametrize("dtype", _DTYPE, ids=["float32", "float16"])
def test_summation(shape, axes, device, dtype):
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

@pytest.mark.parametrize("shape, axes", SUMMATION_PARAMETERS)
@pytest.mark.parametrize("device", _DEVICES, ids=["cpu", "cuda"])
@pytest.mark.parametrize("dtype", _DTYPE, ids=["float32", "float16"])
def test_max(shape, axes, device, dtype):
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

@pytest.mark.parametrize("shape, axes", SUMMATION_PARAMETERS)
@pytest.mark.parametrize("device", _DEVICES, ids=["cpu", "cuda"])
def test_logsumexp(shape, axes, device):
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
    """Test basic indexing: int, slice, ellipsis, None."""
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
    """Test advanced indexing: boolean mask, integer array, tensor indexing."""
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
    """Test mixed indexing: combining basic and advanced indexing."""
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
    """Test basic setitem: int, slice assignments."""
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
    """Test advanced setitem: boolean mask, integer array assignments."""
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
    """Test backward with duplicate indices to verify scatter-add behavior."""
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


if __name__ == "__main__":
    pytest.main()
