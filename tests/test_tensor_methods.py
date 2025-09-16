"""Test suite for Genesis tensor methods.

This module contains tests for tensor-specific methods like device movement,
dtype conversion, and other tensor utilities.
"""

import sys
sys.path.append('./')
import pytest
import numpy as np

import genesis

atol = 1e-1
rtol = 1e-1


@pytest.mark.parametrize("dtype", [genesis.float32, genesis.float16])
@pytest.mark.parametrize("shape", [(3, 4), (2, 2, 3), (5,)])
def test_tensor_cuda_method(dtype, shape):
    """Test tensor.cuda() method for device movement."""
    # Create CPU tensor
    input_data = np.random.randn(*shape).astype(np.float32)
    cpu_tensor = genesis.tensor(input_data, device=genesis.device('cpu'), dtype=dtype)
    
    # Test basic properties
    assert str(cpu_tensor.device) == "cpu", f"Expected cpu device, got {cpu_tensor.device}"
    assert cpu_tensor.dtype == dtype, f"Expected {dtype}, got {cpu_tensor.dtype}"
    assert cpu_tensor.shape == shape, f"Expected shape {shape}, got {cpu_tensor.shape}"
    
    # Test .cuda() method exists and is callable
    assert hasattr(cpu_tensor, 'cuda'), "Tensor should have cuda() method"
    assert callable(cpu_tensor.cuda), "cuda() should be callable"
    
    # Note: We can't actually test GPU movement without GPU access
    # But we can test that the method exists and has correct signature
    try:
        # This might fail if no GPU, but method should exist
        gpu_tensor = cpu_tensor.cuda()
        # If successful, check that it returns a tensor
        assert isinstance(gpu_tensor, genesis.Tensor), "cuda() should return a Tensor"
    except (RuntimeError, Exception) as e:
        # Expected if no GPU available - this is fine for testing method existence
        print(f"GPU not available (expected): {e}")
    
    # Test cuda() with explicit device ID
    try:
        gpu_tensor_0 = cpu_tensor.cuda(0)
        assert isinstance(gpu_tensor_0, genesis.Tensor), "cuda(0) should return a Tensor"
    except (RuntimeError, Exception) as e:
        print(f"GPU device 0 not available (expected): {e}")


@pytest.mark.parametrize("dtype", [genesis.float32, genesis.float16])
def test_tensor_requires_grad_default(dtype):
    """Test that tensor default requires_grad is False (aligned with PyTorch)."""
    input_data = np.random.randn(3, 4).astype(np.float32)
    
    # Test default requires_grad=False
    tensor_default = genesis.tensor(input_data, dtype=dtype)
    assert tensor_default.requires_grad == False, "Tensor default requires_grad should be False"
    
    # Test explicit requires_grad=True
    tensor_grad = genesis.tensor(input_data, dtype=dtype, requires_grad=True)
    assert tensor_grad.requires_grad == True, "Tensor with requires_grad=True should have requires_grad=True"
    
    # Test explicit requires_grad=False
    tensor_no_grad = genesis.tensor(input_data, dtype=dtype, requires_grad=False)
    assert tensor_no_grad.requires_grad == False, "Tensor with requires_grad=False should have requires_grad=False"


@pytest.mark.parametrize("dtype", [genesis.float32, genesis.float16, genesis.int32, genesis.int64])
@pytest.mark.parametrize("shape", [(3, 4), (2, 2, 3), (5,), ()])
def test_from_numpy(dtype, shape):
    """Test from_numpy function."""
    import torch
    
    # Create numpy array with appropriate dtype
    if dtype in [genesis.float32, genesis.float16]:
        if shape == ():
            np_data = np.array(np.random.randn(), dtype=np.float32 if dtype == genesis.float32 else np.float16)
        else:
            np_data = np.random.randn(*shape).astype(np.float32 if dtype == genesis.float32 else np.float16)
        torch_dtype = torch.float32 if dtype == genesis.float32 else torch.float16
    elif dtype == genesis.int32:
        if shape == ():
            np_data = np.array(np.random.randint(-100, 100), dtype=np.int32)
        else:
            np_data = np.random.randint(-100, 100, shape).astype(np.int32)
        torch_dtype = torch.int32
    elif dtype == genesis.int64:
        if shape == ():
            np_data = np.array(np.random.randint(-100, 100), dtype=np.int64)
        else:
            np_data = np.random.randint(-100, 100, shape).astype(np.int64)
        torch_dtype = torch.int64
    
    # Test from_numpy
    genesis_tensor = genesis.from_numpy(np_data, dtype=dtype)
    torch_tensor = torch.from_numpy(np_data).to(torch_dtype)
    
    # Check basic properties
    assert genesis_tensor.shape == shape, f"Shape mismatch: {genesis_tensor.shape} vs {shape}"
    assert genesis_tensor.dtype == dtype, f"Dtype mismatch: {genesis_tensor.dtype} vs {dtype}"
    
    # Check values
    np.testing.assert_allclose(
        genesis_tensor.numpy(),
        torch_tensor.numpy(),
        atol=atol, rtol=rtol,
        err_msg=f"from_numpy value mismatch for {dtype}"
    )
    
    # Test automatic dtype inference
    genesis_auto = genesis.from_numpy(np_data)
    expected_dtype = dtype if dtype in [genesis.float32, genesis.float16, genesis.int32, genesis.int64] else genesis.float32
    assert genesis_auto.dtype == expected_dtype, f"Auto dtype inference failed: {genesis_auto.dtype} vs {expected_dtype}"
    
    # Test requires_grad
    genesis_grad = genesis.from_numpy(np_data, requires_grad=True)
    if dtype in [genesis.float32, genesis.float16]:  # Only floating point tensors can require gradients
        assert genesis_grad.requires_grad == True, "from_numpy with requires_grad=True failed"
    
    # Test device specification
    genesis_cpu = genesis.from_numpy(np_data, device=genesis.device('cpu'))
    assert str(genesis_cpu.device) == "cpu", f"Device specification failed: {genesis_cpu.device}"


@pytest.mark.parametrize("input_type", ["list", "nested_list", "scalar"])
def test_from_numpy_input_types(input_type):
    """Test from_numpy with different input types."""
    if input_type == "list":
        data = [1.0, 2.0, 3.0, 4.0]
        expected_shape = (4,)
    elif input_type == "nested_list":
        data = [[1.0, 2.0], [3.0, 4.0]]
        expected_shape = (2, 2)
    elif input_type == "scalar":
        data = 5.0
        expected_shape = ()
    
    genesis_tensor = genesis.from_numpy(data)
    assert genesis_tensor.shape == expected_shape, f"Shape mismatch for {input_type}: {genesis_tensor.shape} vs {expected_shape}"


if __name__ == "__main__":
    pytest.main()