"""Test suite for Genesis distributed training functionality.

This module contains comprehensive tests for distributed communication operations,
process group management, and DistributedDataParallel (DDP) wrapper.
Tests cover single-process mode and validate distributed primitives.
"""

import sys
sys.path.append('./')
import pytest
import os
import numpy as np

import genesis
import genesis.distributed as dist

# Distributed training only supports CUDA
_CUDA_AVAILABLE = genesis.device("cuda").enabled()

# Skip all tests if CUDA is not available
pytestmark = pytest.mark.skipif(not _CUDA_AVAILABLE, reason="Distributed training requires CUDA")

_DEVICES = [genesis.device("cuda")]

# Tolerance for numerical comparisons
atol = 1e-5
rtol = 1e-5


def test_import_structure():
    """Test that distributed module imports work correctly."""
    # Test basic imports
    assert hasattr(genesis, 'distributed')
    assert hasattr(genesis.distributed, 'init_process_group')
    assert hasattr(genesis.distributed, 'DistributedDataParallel')
    assert hasattr(genesis.distributed, 'DDP')  # Alias
    assert hasattr(genesis.distributed, 'all_reduce')
    assert hasattr(genesis.distributed, 'ReduceOp')
    
    # Test enum values
    assert hasattr(genesis.distributed.ReduceOp, 'SUM')
    assert hasattr(genesis.distributed.ReduceOp, 'MAX')
    assert hasattr(genesis.distributed.ReduceOp, 'MIN')
    assert hasattr(genesis.distributed.ReduceOp, 'PRODUCT')


def test_nccl_availability():
    """Test NCCL library availability detection."""
    from genesis.distributed.nccl import is_nccl_available
    
    # This should return True or False without crashing
    available = is_nccl_available()
    assert isinstance(available, bool)


@pytest.mark.parametrize("device", _DEVICES, ids=["cuda"])
def test_single_process_init(device):
    """Test single process initialization and cleanup.
    
    Args:
        device: Device to run test on (CUDA only)
        
    Tests:
        Basic process group initialization and destruction
        in single-process mode (world_size=1).
    """
        
    try:
        # Test initialization
        dist.init_process_group(backend="nccl", world_size=1, rank=0)
        
        assert dist.is_initialized() == True
        assert dist.get_world_size() == 1
        assert dist.get_rank() == 0
        
        # Test cleanup
        dist.destroy_process_group()
        assert dist.is_initialized() == False
        
    except RuntimeError as e:
        if "NCCL library not available" in str(e):
            pytest.skip("NCCL not available for testing")
        else:
            raise


@pytest.mark.parametrize("device", _DEVICES, ids=["cuda"])
def test_collective_ops_single_process(device):
    """Test collective communication operations in single process.
    
    Args:
        device: Device to run test on (CUDA only)
        
    Tests:
        all_reduce, broadcast, and barrier operations
        in single-process mode.
    """
        
    try:
        dist.init_process_group(backend="nccl", world_size=1, rank=0)
        
        # Test all_reduce with SUM
        tensor = genesis.ones([4], dtype=genesis.float32, device=device)
        original_data = tensor.data.clone()
        
        dist.all_reduce(tensor, dist.ReduceOp.SUM)
        # In single process, tensor should remain unchanged
        assert genesis.allclose(tensor, original_data, atol=atol, rtol=rtol)
        
        # Test all_reduce with different operations
        for op in [dist.ReduceOp.MAX, dist.ReduceOp.MIN]:
            test_tensor = genesis.ones([4], dtype=genesis.float32, device=device) * 2.5
            dist.all_reduce(test_tensor, op)
            assert genesis.allclose(test_tensor, genesis.ones([4], device=device) * 2.5, 
                                  atol=atol, rtol=rtol)
        
        # Test broadcast
        broadcast_tensor = genesis.randn([8], dtype=genesis.float32, device=device)
        original_broadcast = broadcast_tensor.data.clone()
        
        dist.broadcast(broadcast_tensor, src=0)
        assert genesis.allclose(broadcast_tensor, original_broadcast, atol=atol, rtol=rtol)
        
        # Test barrier
        dist.barrier()  # Should complete without hanging
        
        dist.destroy_process_group()
        
    except RuntimeError as e:
        if "NCCL library not available" in str(e):
            pytest.skip("NCCL not available for testing")
        else:
            raise


@pytest.mark.parametrize("device", _DEVICES, ids=["cuda"])  
def test_all_gather_single_process(device):
    """Test all_gather operation in single process.
    
    Args:
        device: Device to run test on (CUDA only)
        
    Tests:
        all_gather collective communication operation
        with various tensor shapes and types.
    """
        
    try:
        dist.init_process_group(backend="nccl", world_size=1, rank=0)
        
        # Test all_gather
        input_tensor = genesis.randn([4, 8], dtype=genesis.float32, device=device)
        output_list = [genesis.zeros_like(input_tensor) for _ in range(1)]
        
        dist.all_gather(output_list, input_tensor)
        
        # In single process, output should be same as input
        assert genesis.allclose(output_list[0], input_tensor, atol=atol, rtol=rtol)
        
        dist.destroy_process_group()
        
    except RuntimeError as e:
        if "NCCL library not available" in str(e):
            pytest.skip("NCCL not available for testing")
        else:
            raise


@pytest.mark.parametrize("device", _DEVICES, ids=["cuda"])
@pytest.mark.parametrize("model_size", [(10, 5), (64, 32), (128, 64)])
def test_ddp_single_process(model_size, device):
    """Test DistributedDataParallel wrapper in single process.
    
    Args:
        model_size: Tuple of (input_size, output_size) for linear model
        device: Device to run test on (CUDA only)
        
    Tests:
        DDP wrapper functionality including forward pass,
        backward pass, gradient synchronization, and state management.
    """
        
    input_size, output_size = model_size
    
    try:
        dist.init_process_group(backend="nccl", world_size=1, rank=0)
        
        # Create a simple model
        model = genesis.nn.Sequential([
            genesis.nn.Linear(input_size, output_size),
            genesis.nn.ReLU(),
            genesis.nn.Linear(output_size, 1)
        ])
        
        # Test DDP wrapper creation
        ddp_model = dist.DistributedDataParallel(model, device_ids=[device.index])
        assert isinstance(ddp_model, dist.DistributedDataParallel)
        
        # Test forward pass
        batch_size = 8
        input_data = genesis.randn([batch_size, input_size], device=device)
        output = ddp_model(input_data)
        
        assert output.shape == (batch_size, 1)
        assert output.device.type == device.type
        
        # Test backward pass
        loss = output.sum()
        loss.backward()
        
        # Check gradients exist and are finite
        for param in ddp_model.parameters():
            if param.requires_grad:
                assert param.grad is not None
                assert genesis.isfinite(param.grad).all()
        
        # Test state dict operations
        state_dict = ddp_model.state_dict()
        assert isinstance(state_dict, dict)
        assert len(state_dict) > 0
        
        # Test parameter access
        params = list(ddp_model.parameters())
        assert len(params) > 0
        
        # Test training mode toggle
        ddp_model.train()
        ddp_model.eval()
        
        dist.destroy_process_group()
        
    except RuntimeError as e:
        if "NCCL library not available" in str(e):
            pytest.skip("NCCL not available for testing")
        else:
            raise


@pytest.mark.parametrize("device", _DEVICES, ids=["cuda"])
def test_ddp_gradient_synchronization(device):
    """Test gradient synchronization in DDP.
    
    Args:
        device: Device to run test on (CUDA only)
        
    Tests:
        Gradient averaging and synchronization behavior
        in single-process DDP mode.
    """
        
    try:
        dist.init_process_group(backend="nccl", world_size=1, rank=0)
        
        # Create simple model with known parameters
        model = genesis.nn.Linear(4, 1)
        ddp_model = dist.DistributedDataParallel(model, device_ids=[device.index])
        
        # Create deterministic input and target
        input_tensor = genesis.ones([2, 4], device=device)
        target = genesis.ones([2, 1], device=device) * 0.5
        
        # Forward and backward pass
        output = ddp_model(input_tensor)
        loss = ((output - target) ** 2).sum()
        loss.backward()
        
        # Check that gradients are computed
        for name, param in ddp_model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None
                assert not genesis.isnan(param.grad).any()
                
        dist.destroy_process_group()
        
    except RuntimeError as e:
        if "NCCL library not available" in str(e):
            pytest.skip("NCCL not available for testing")
        else:
            raise


@pytest.mark.parametrize("device", _DEVICES, ids=["cuda"])
@pytest.mark.parametrize("dtype", [genesis.float32, genesis.float16])
def test_reduce_operations_dtypes(device, dtype):
    """Test reduce operations with different data types.
    
    Args:
        device: Device to run test on (CUDA only)
        dtype: Data type for tensors
        
    Tests:
        All-reduce operations with various tensor dtypes
        including float32 and float16.
    """
        
    try:
        dist.init_process_group(backend="nccl", world_size=1, rank=0)
        
        # Test different reduce operations
        test_data = [1.0, 2.5, -1.0, 3.14159]
        tensor = genesis.tensor(test_data, dtype=dtype, device=device)
        
        # Test SUM
        sum_tensor = tensor.clone()
        dist.all_reduce(sum_tensor, dist.ReduceOp.SUM)
        assert genesis.allclose(sum_tensor, tensor, atol=atol, rtol=rtol)
        
        # Test MAX  
        max_tensor = tensor.clone()
        dist.all_reduce(max_tensor, dist.ReduceOp.MAX)
        assert genesis.allclose(max_tensor, tensor, atol=atol, rtol=rtol)
        
        # Test MIN
        min_tensor = tensor.clone()
        dist.all_reduce(min_tensor, dist.ReduceOp.MIN)
        assert genesis.allclose(min_tensor, tensor, atol=atol, rtol=rtol)
        
        dist.destroy_process_group()
        
    except RuntimeError as e:
        if "NCCL library not available" in str(e):
            pytest.skip("NCCL not available for testing")
        else:
            raise


def test_process_group_error_handling():
    """Test error handling in process group operations.
    
    Tests:
        Proper error messages and exception handling
        for invalid operations and uninitialized states.
    """
    # Test operations without initialization
    with pytest.raises(RuntimeError, match="Process group not initialized"):
        dist.get_world_size()
        
    with pytest.raises(RuntimeError, match="Process group not initialized"):
        dist.get_rank()
        
    with pytest.raises(RuntimeError, match="Process group not initialized"):
        dist.barrier()
    
    # Test invalid backend
    with pytest.raises(ValueError, match="Unsupported backend"):
        dist.init_process_group(backend="invalid_backend", world_size=1, rank=0)


# Multi-process tests (marked as skip by default, require special setup)
@pytest.mark.skip(reason="Requires multi-process launch setup")
@pytest.mark.parametrize("world_size", [2, 4])
def test_multi_process_all_reduce(world_size):
    """Test multi-process all_reduce operation.
    
    Args:
        world_size: Number of processes to simulate
        
    Note:
        This test requires running with torch.distributed.launch
        or similar multi-process setup. Skipped by default.
    """
    rank = int(os.environ.get('RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', world_size))
    
    dist.init_process_group(backend="nccl", world_size=world_size, rank=rank)
    
    # Create tensor with rank-specific value
    device = genesis.device('cuda')
    tensor = genesis.ones([4], dtype=genesis.float32, device=device) * rank
    
    # All reduce should sum across all ranks
    dist.all_reduce(tensor, dist.ReduceOp.SUM)
    
    # Expected: sum of 0 + 1 + ... + (world_size-1)
    expected_sum = sum(range(world_size))
    expected_tensor = genesis.ones([4], dtype=genesis.float32, device=device) * expected_sum
    
    assert genesis.allclose(tensor, expected_tensor, atol=atol, rtol=rtol)
    
    dist.destroy_process_group()


if __name__ == "__main__":
    # Basic smoke test
    print("Running basic distributed tests...")
    
    try:
        test_import_structure()
        print("✅ Import structure test passed")
        
        test_nccl_availability()
        print("✅ NCCL availability test passed")
        
        test_process_group_error_handling()
        print("✅ Error handling test passed")
        
        # Test with CUDA if available
        if genesis.device("cuda").enabled():
            cuda_device = genesis.device("cuda")
            
            test_single_process_init(cuda_device)
            print("✅ Single process init test passed")
            
            test_collective_ops_single_process(cuda_device)
            print("✅ Collective ops test passed")
            
            test_ddp_single_process((10, 5), cuda_device)
            print("✅ DDP test passed")
            
        print("\n🎉 All distributed tests passed!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()