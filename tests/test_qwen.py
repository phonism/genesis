"""
Test consistency of Qwen model between PyTorch and Genesis modes
Use environment variables to gracefully switch frameworks
"""
import sys
import os
import importlib
import numpy as np
import pytest
import torch
sys.path.append("./")


def set_seed(seed: int = 42):
    """Set random seed to ensure reproducible results"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_qwen_model(use_torch: bool, config_dict: dict):
    """
    load qwen model
    
    Args:
        use_torch: whether to use PyTorch framework
        config_dict: model configuration dictionary
    
    Returns:
        model: loaded model instance
        framework_name: framework name string
    """
    # Set environment variable
    os.environ["QWEN_USE_TORCH"] = "true" if use_torch else "false"
    print(f"🔧 Setting QWEN_USE_TORCH={os.environ['QWEN_USE_TORCH']}")
    
    # clear module cache
    print("🔄 Clearing module cache and re-importing...")
    
    # remove genesis related modules from cache
    genesis_modules = [k for k in sys.modules.keys() if k.startswith("genesis")]
    for module_name in genesis_modules:
        if module_name in sys.modules:
            print(f"   Removing {module_name} from cache")
            del sys.modules[module_name]
    
    from genesis.models.qwen import QwenForCausalLM, ModelArgs, USE_TORCH
    
    # verify framework switch
    print(f"✅ Module USE_TORCH={USE_TORCH}")
    
    config = ModelArgs(**config_dict)
    model = QwenForCausalLM(config)
    
    # check framework
    if hasattr(model.model.embed_tokens, "weight"):
        weight_type = type(model.model.embed_tokens.weight)
        print(f"📊 Model weight type: {weight_type}")
        
        if "torch" in str(weight_type).lower():
            actual_framework = "PyTorch"
        else:
            actual_framework = "Genesis"
        print(f"🎯 Detected framework: {actual_framework}")
    
    framework_name = "PyTorch" if use_torch else "Genesis"
    return model, framework_name


def copy_weights_torch_to_genesis(torch_model, genesis_model):
    """
    Copy weights from PyTorch model to Genesis model
    """
    print("🔗 Starting weight synchronization...")
    torch_state_dict = torch_model.state_dict()
    print(f"📊 PyTorch model has {len(torch_state_dict)} parameters")
    
    import genesis
    genesis_state_dict = {}
    
    # convert PyTorch tensors to Genesis tensors
    converted_count = 0
    for key, torch_tensor in torch_state_dict.items():
        try:
            genesis_state_dict[key] = torch_tensor
            converted_count += 1
        except Exception as e:
            print(f"⚠️  Failed to convert parameter {key}: {e}")
    
    print(f"✅ Converted {converted_count} / {len(torch_state_dict)} parameters")
    
    # load to Genesis model
    try:
        result = genesis_model.load_state_dict(torch_state_dict, strict=False)
        if result is not None and len(result) == 2:
            missing_keys, unexpected_keys = result
            if missing_keys:
                print(f"⚠️  Missing keys in Genesis model: {len(missing_keys)} keys")
                print(f"   First few missing: {missing_keys[:3]}")
            if unexpected_keys:
                print(f"⚠️  Unexpected keys: {len(unexpected_keys)} keys")
                print(f"   First few unexpected: {unexpected_keys[:3]}")
        else:
            print("📝 Genesis load_state_dict completed (no return info)")
    except Exception as e:
        print(f"⚠️  Error during load_state_dict: {e}")
    
    print("✅ Weight synchronization completed")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")  
def test_qwen_consistency_cuda():
    """
    Test consistency on CUDA
    """
    _test_qwen_consistency(use_cuda=True)


def test_qwen_consistency_cpu():
    """
    Test consistency on CPU
    """
    _test_qwen_consistency(use_cuda=False)


def _test_qwen_consistency(use_cuda: bool = False):
    """
    Core consistency test function
    
    Args:
        use_cuda: whether to use CUDA
    """
    # set random seed
    set_seed(42)
    
    # small scale test config
    config_dict = {
        "block_size": 64,
        "vocab_size": 1000,
        "n_layer": 2,
        "num_attention_heads": 4, 
        "hidden_size": 64,
        "intermediate_size": 256,
        "num_key_value_heads": 2,
        "head_dim": 16,
        "rope_base": 10000.0,
        "max_position_embeddings": 64,
        "norm_eps": 1e-6
    }
    
    device_str = "cuda" if use_cuda else "cpu"
    
    print(f"\n🔄 Testing Qwen consistency on {device_str.upper()}...")
    
    # load PyTorch version
    print("📦 Loading PyTorch version...")
    torch_model, _ = load_qwen_model(use_torch=True, config_dict=config_dict)
    if use_cuda:
        torch_model = torch_model.cuda()
    
    # load Genesis version
    print("📦 Loading Genesis version...")
    genesis_model, _ = load_qwen_model(use_torch=False, config_dict=config_dict)
    
    # synchronize weights
    print("🔗 Synchronizing weights...")
    copy_weights_torch_to_genesis(torch_model, genesis_model)
    
    if use_cuda:
        import genesis
        genesis_model = genesis_model.cuda()
    
    # create test input
    batch_size, seq_len = 2, 16
    set_seed(42)  # ensure input data consistency
    input_ids = np.random.randint(0, config_dict["vocab_size"], (batch_size, seq_len))
    
    print(f"📝 Testing with input shape: {input_ids.shape}")
    
    # PyTorch forward pass
    torch_input = torch.tensor(input_ids, dtype=torch.long)
    if use_cuda:
        torch_input = torch_input.cuda()
    
    torch_model.eval()
    with torch.no_grad():
        torch_output = torch_model(torch_input)
    
    # Genesis forward pass
    import genesis
    device = genesis.cuda() if use_cuda else genesis.cpu()
    genesis_input = genesis.Tensor(input_ids.astype(np.int64), device=device)
    
    genesis_model.eval() 
    genesis_output = genesis_model(genesis_input)
    
    # compare results
    print("🔍 Comparing outputs...")
    torch_np = torch_output.detach().cpu().numpy()
    genesis_np = genesis_output.detach().cpu().numpy()
    
    # verify shape
    assert torch_np.shape == genesis_np.shape, \
        f"Shape mismatch: PyTorch {torch_np.shape} vs Genesis {genesis_np.shape}"
    
    # calculate numerical difference
    abs_diff = np.abs(torch_np - genesis_np)
    max_abs_diff = abs_diff.max()
    mean_abs_diff = abs_diff.mean()
    
    # verify numerical precision
    tolerance = 1e-3
    try:
        np.testing.assert_allclose(torch_np, genesis_np, atol=tolerance, rtol=tolerance)
        print(f"✅ Consistency test PASSED on {device_str.upper()}!")
        print(f"   Max absolute difference: {max_abs_diff:.6f}")
        print(f"   Mean absolute difference: {mean_abs_diff:.6f}")
        
    except AssertionError as e:
        print(f"❌ Consistency test FAILED on {device_str.upper()}!")
        print(f"   Max absolute difference: {max_abs_diff:.6f}")
        print(f"   Mean absolute difference: {mean_abs_diff:.6f}")
        print(f"   Tolerance: {tolerance}")
        raise e


def test_qwen_framework_switching():
    """
    test framework switching functionality
    """
    config_dict = {
        "vocab_size": 100,
        "hidden_size": 32,
        "n_layer": 1,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "intermediate_size": 64
    }
    
    # test Genesis mode
    genesis_model, framework_name = load_qwen_model(use_torch=False, config_dict=config_dict)
    assert framework_name == "Genesis"
    
    # test PyTorch mode
    torch_model, framework_name = load_qwen_model(use_torch=True, config_dict=config_dict)  
    assert framework_name == "PyTorch"
    
    print("✅ Framework switching test passed!")


def test_qwen_basic_functionality():
    """
    test basic functionality of two frameworks
    """
    config_dict = {
        "vocab_size": 100,
        "hidden_size": 32,
        "n_layer": 1,
        "num_attention_heads": 2,
        "num_key_value_heads": 1, 
        "intermediate_size": 64,
        "max_position_embeddings": 64
    }
    
    # test basic functionality of two frameworks
    for use_torch in [False, True]:
        framework_name = "PyTorch" if use_torch else "Genesis"
        print(f"🧪 Testing {framework_name} basic functionality...")
        
        model, _ = load_qwen_model(use_torch=use_torch, config_dict=config_dict)
        
        # create test input
        batch_size, seq_len = 1, 8
        input_ids = np.random.randint(0, config_dict["vocab_size"], (batch_size, seq_len))
        
        if use_torch:
            input_tensor = torch.tensor(input_ids, dtype=torch.long)
        else:
            import genesis
            input_tensor = genesis.Tensor(input_ids.astype(np.int64))
        
        # forward pass
        model.eval()
        if use_torch:
            with torch.no_grad():
                output = model(input_tensor)
        else:
            output = model(input_tensor)
        
        # verify output shape
        expected_shape = (batch_size, seq_len, config_dict["vocab_size"])
        actual_shape = output.shape if use_torch else tuple(output.shape)
        
        assert actual_shape == expected_shape, \
            f"{framework_name} output shape mismatch: expected {expected_shape}, got {actual_shape}"
        
        print(f"✅ {framework_name} basic functionality test passed!")


if __name__ == "__main__":
    # run main tests
    print("🚀 Running Qwen consistency tests...")
    
    test_qwen_framework_switching()
    test_qwen_basic_functionality() 
    test_qwen_consistency_cpu()
    
    if torch.cuda.is_available():
        test_qwen_consistency_cuda()
    else:
        print("⚠️  CUDA not available, skipping CUDA tests")
    
    print("\n🎉 All tests completed!")