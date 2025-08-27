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
from tests.test_logger import test_logger, log_test_start, log_test_end, log_hang_detection, log_operation

@pytest.fixture(scope="session", autouse=True)
def cuda_warmup(pytestconfig):
    """Session-level CUDA warmup to avoid reinitialization for each test"""
    with log_operation("Session-CUDA-Warmup", timeout=10.0):
        # Warmup CUDA - create a simple tensor to trigger initialization
        try:
            if genesis.device("cuda").enabled():
                warmup_tensor = genesis.tensor([1.0], device=genesis.device("cuda"))
                # Trigger some basic operations to ensure full initialization
                _ = warmup_tensor + warmup_tensor
                test_logger.info("✅ CUDA warmed up successfully")
            else:
                test_logger.info("⚠️  CUDA not available, skipping warmup")
        except Exception as e:
            test_logger.error(f"❌ CUDA warmup failed: {e}")
    
    yield  # Run all tests here
    
    test_logger.info("🏁 Test session completed")

@pytest.fixture(scope="function")
def device_info():
    """Fixture providing device information"""
    return {
        'cpu_available': True,
        'cuda_available': genesis.device("cuda").enabled()
    }

@pytest.fixture(scope="function", autouse=True)
def reset_memory_manager():
    """Reset memory manager state between tests to avoid state pollution"""
    # This fixture runs before each test automatically
    yield  # Run the test here
    
    # After test cleanup with logging and timeout protection
    with log_operation("Memory-Manager-Reset", timeout=5.0):
        try:
            # Import here to avoid circular imports
            from genesis.ndarray import cuda_memory_manager
            
            # Reset global memory manager to force fresh state for next test
            if hasattr(cuda_memory_manager, '_memory_manager'):
                # Clean up existing manager
                if cuda_memory_manager._memory_manager is not None:
                    try:
                        cuda_memory_manager._memory_manager.empty_cache()
                    except Exception as e:
                        test_logger.warning(f"Memory manager cleanup error: {e}")
                    
                # Reset to None so next test gets fresh manager
                cuda_memory_manager._memory_manager = None
                
        except ImportError:
            pass  # Memory manager not available, skip cleanup


# Global variables for hang detection
_test_watchdog_timer = None
_current_test_name = None

def _timeout_handler(signum, frame):
    """Signal handler for test timeouts."""
    if _current_test_name:
        stack = traceback.format_stack(frame)
        log_hang_detection(_current_test_name, 300.0, ''.join(stack))
    test_logger.error("🚨 Test execution timed out - sending SIGTERM")
    os._exit(1)  # Force exit

@pytest.fixture(scope="function", autouse=True)
def test_timeout_watchdog(request):
    """Watchdog timer to detect hanging tests."""
    global _test_watchdog_timer, _current_test_name
    
    test_name = request.node.nodeid
    _current_test_name = test_name
    
    # Set up timeout handler (if not using pytest-timeout)
    timeout_seconds = int(os.environ.get('GENESIS_TEST_TIMEOUT', '300'))  # 5 minutes default
    
    if os.environ.get('GENESIS_HANG_DEBUG') == '1':
        old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
        signal.alarm(timeout_seconds)
        test_logger.debug(f"🕐 Watchdog set for {test_name} ({timeout_seconds}s)")
    
    log_test_start(test_name)
    
    yield
    
    # Clean up
    if os.environ.get('GENESIS_HANG_DEBUG') == '1':
        signal.alarm(0)  # Cancel alarm
        if 'old_handler' in locals():
            signal.signal(signal.SIGALRM, old_handler)
    
    # Log test completion
    outcome = "PASSED"
    if hasattr(request.node, 'rep_call') and request.node.rep_call.failed:
        outcome = "FAILED"
    elif hasattr(request.node, 'rep_setup') and request.node.rep_setup.failed:
        outcome = "ERROR"
    
    log_test_end(test_name, outcome)
    _current_test_name = None


# Pytest hooks for detailed logging
def pytest_runtest_setup(item):
    """Called before each test is run."""
    test_logger.info(f"🔧 SETUP: {item.nodeid}")

def pytest_runtest_call(item):
    """Called when the test is executed."""
    test_logger.info(f"🏃 CALL: {item.nodeid}")

def pytest_runtest_teardown(item):
    """Called after each test is run.""" 
    test_logger.info(f"🧹 TEARDOWN: {item.nodeid}")

@pytest.hookimpl(tryfirst=True, hookwrapper=True)
def pytest_runtest_makereport(item, call):
    """Store test outcome for logging."""
    outcome = yield
    rep = outcome.get_result()
    
    # Store the report in the item for access in fixtures
    setattr(item, f"rep_{rep.when}", rep)
    
    if rep.when == "call":
        if rep.failed:
            test_logger.error(f"❌ {item.nodeid} FAILED: {rep.longrepr}")
        elif rep.passed:
            test_logger.info(f"✅ {item.nodeid} PASSED")

def pytest_sessionstart(session):
    """Called after the Session object has been created."""
    test_logger.info("🚀 Test session starting")

def pytest_sessionfinish(session, exitstatus):
    """Called after whole test run finished."""
    test_logger.info(f"🏁 Test session finished with exit status: {exitstatus}")