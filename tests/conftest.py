"""
Pytest configuration and fixtures
Optimize CUDA initialization performance
"""

import pytest
import genesis
import time

@pytest.fixture(scope="session", autouse=True)
def cuda_warmup(pytestconfig):
    """Session-level CUDA warmup to avoid reinitialization for each test"""
    # Use pytest output system
    if pytestconfig.option.verbose:
        print("\n🔥 Warming up CUDA (session-level)...")
    
    start_time = time.perf_counter()
    
    # Warmup CUDA - create a simple tensor to trigger initialization
    try:
        if genesis.cuda().enabled():
            warmup_tensor = genesis.tensor([1.0], device=genesis.cuda())
            # Trigger some basic operations to ensure full initialization
            _ = warmup_tensor + warmup_tensor
            
            elapsed = time.perf_counter() - start_time
            if pytestconfig.option.verbose:
                print(f"✅ CUDA warmed up in {elapsed:.4f}s")
        else:
            if pytestconfig.option.verbose:
                print("⚠️  CUDA not available, skipping warmup")
    except Exception as e:
        elapsed = time.perf_counter() - start_time
        if pytestconfig.option.verbose:
            print(f"❌ CUDA warmup failed after {elapsed:.4f}s: {e}")
    
    yield  # Run all tests here
    
    if pytestconfig.option.verbose:
        print("\n🏁 Test session completed")

@pytest.fixture(scope="function")
def device_info():
    """Fixture providing device information"""
    return {
        'cpu_available': True,
        'cuda_available': genesis.cuda().enabled()
    }