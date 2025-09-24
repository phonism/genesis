"""
Pytest configuration and fixtures
Optimize CUDA initialization performance and add hang detection
"""

import pytest
import genesis
import time
import threading
import traceback
import signal
import sys
import os
# Test logger functionality removed - not needed

@pytest.fixture(scope="session", autouse=True)
def cuda_warmup(pytestconfig):
    """Session-level CUDA warmup to avoid reinitialization for each test"""
    try:
        if genesis.cuda.is_available():
            # Warm up common Triton kernels that are used in tests
            warmup_tensor = genesis.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], device=genesis.device("cuda"), requires_grad=True)
            
            # Basic arithmetic operations
            _ = warmup_tensor + warmup_tensor
            _ = warmup_tensor * 2.0
            
            # Critical reduce operations (these are slow on first compile)
            _ = warmup_tensor.sum()
            _ = warmup_tensor.max()
            _ = warmup_tensor.mean()
            
            # Basic tensor operations
            _ = warmup_tensor.transpose()
            _ = warmup_tensor.reshape(-1)
            
            # Autograd operations (triggers backward kernels)
            loss = warmup_tensor.sum()
            loss.backward()
            
    except Exception:
        pass  # Silent failure - don't break tests if warmup fails
    
    yield  # Run all tests here

@pytest.fixture(scope="function")
def device_info():
    """Fixture providing device information"""
    return {
        'cpu_available': True,
        'cuda_available': genesis.cuda.is_available()
    }

# Memory manager reset removed - let it persist across tests for better performance


# Global variables for hang detection
_test_watchdog_timer = None
_current_test_name = None

def _timeout_handler(signum, frame):
    """Signal handler for test timeouts."""
    if _current_test_name:
        stack = traceback.format_stack(frame)
        print(f"Test {_current_test_name} timed out after 300s")
        print(''.join(stack))
    print("🚨 Test execution timed out - sending SIGTERM")
    os._exit(1)  # Force exit

# Simplified watchdog - only enable if explicitly requested
@pytest.fixture(scope="function", autouse=True)
def test_timeout_watchdog(request):
    """Minimal watchdog timer."""
    if os.environ.get('GENESIS_HANG_DEBUG') == '1':
        global _test_watchdog_timer, _current_test_name
        test_name = request.node.nodeid
        _current_test_name = test_name
        timeout_seconds = int(os.environ.get('GENESIS_TEST_TIMEOUT', '300'))
        old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
        signal.alarm(timeout_seconds)
    
    yield
    
    if os.environ.get('GENESIS_HANG_DEBUG') == '1':
        signal.alarm(0)
        if 'old_handler' in locals():
            signal.signal(signal.SIGALRM, old_handler)
        _current_test_name = None


# Minimal logging - only for failures and session
@pytest.hookimpl(tryfirst=True, hookwrapper=True)
def pytest_runtest_makereport(item, call):
    """Store test outcome - minimal logging."""
    outcome = yield
    rep = outcome.get_result()
    setattr(item, f"rep_{rep.when}", rep)
    
    # Only print failures
    if rep.when == "call" and rep.failed:
        print(f"❌ {item.nodeid} FAILED")