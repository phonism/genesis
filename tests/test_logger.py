"""
Comprehensive logging system for pytest hanging diagnosis.
"""

import logging
import time
import threading
import functools
import sys
import os
import traceback
from typing import Any, Callable, Optional
from contextlib import contextmanager

# Configure test logger
test_logger = logging.getLogger("genesis_test")
test_logger.setLevel(logging.DEBUG)

# Create formatter
formatter = logging.Formatter(
    '%(asctime)s [%(levelname)8s] %(name)s: %(message)s (%(filename)s:%(lineno)d)',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Add console handler if not already present
if not test_logger.handlers:
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(formatter)
    test_logger.addHandler(console_handler)

# Thread-local storage for nested timing
_local = threading.local()

def get_thread_info() -> str:
    """Get current thread information."""
    thread = threading.current_thread()
    return f"Thread-{thread.ident}"

@contextmanager
def log_operation(operation_name: str, timeout: Optional[float] = None):
    """
    Context manager to log operation start/end with timing.
    
    Args:
        operation_name: Name of the operation being logged
        timeout: Optional timeout in seconds to warn about slow operations
    """
    thread_info = get_thread_info()
    start_time = time.perf_counter()
    
    test_logger.info(f"🔄 START {operation_name} [{thread_info}]")
    
    try:
        yield
        elapsed = time.perf_counter() - start_time
        
        if timeout and elapsed > timeout:
            test_logger.warning(f"⚠️  SLOW {operation_name} completed in {elapsed:.4f}s (>{timeout}s) [{thread_info}]")
        else:
            test_logger.info(f"✅ END {operation_name} completed in {elapsed:.4f}s [{thread_info}]")
            
    except Exception as e:
        elapsed = time.perf_counter() - start_time
        test_logger.error(f"❌ FAILED {operation_name} after {elapsed:.4f}s: {e} [{thread_info}]")
        test_logger.debug(f"Stack trace:\n{traceback.format_exc()}")
        raise

def log_function(timeout: Optional[float] = None, log_args: bool = False):
    """
    Decorator to log function calls with timing.
    
    Args:
        timeout: Warn if function takes longer than this many seconds
        log_args: Whether to log function arguments
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            func_name = f"{func.__module__}.{func.__qualname__}"
            
            # Log arguments if requested
            args_str = ""
            if log_args:
                args_repr = [repr(arg) for arg in args[:3]]  # Limit to first 3 args
                kwargs_repr = [f"{k}={repr(v)}" for k, v in list(kwargs.items())[:3]]
                if len(args) > 3:
                    args_repr.append("...")
                if len(kwargs) > 3:
                    kwargs_repr.append("...")
                args_str = f"({', '.join(args_repr + kwargs_repr)})"
            
            with log_operation(f"{func_name}{args_str}", timeout=timeout):
                return func(*args, **kwargs)
                
        return wrapper
    return decorator

@contextmanager
def log_cuda_operation(operation: str, device_id: Optional[int] = None):
    """Log CUDA operations with device information."""
    device_str = f"device:{device_id}" if device_id is not None else "default_device"
    with log_operation(f"CUDA-{operation}[{device_str}]", timeout=5.0):
        yield

@contextmanager 
def log_memory_operation(operation: str, size_mb: Optional[float] = None):
    """Log memory operations with size information."""
    size_str = f"{size_mb:.2f}MB" if size_mb is not None else "unknown_size"
    with log_operation(f"MEM-{operation}[{size_str}]", timeout=2.0):
        yield

def log_test_start(test_name: str):
    """Log the start of a test."""
    test_logger.info(f"🧪 TEST START: {test_name}")
    if not hasattr(_local, 'test_start_time'):
        _local.test_start_time = {}
    _local.test_start_time[test_name] = time.perf_counter()

def log_test_end(test_name: str, status: str = "PASSED"):
    """Log the end of a test with timing."""
    if hasattr(_local, 'test_start_time') and test_name in _local.test_start_time:
        elapsed = time.perf_counter() - _local.test_start_time[test_name]
        del _local.test_start_time[test_name]
    else:
        elapsed = 0.0
    
    status_emoji = "✅" if status == "PASSED" else "❌" if status == "FAILED" else "⚠️"
    test_logger.info(f"{status_emoji} TEST {status}: {test_name} ({elapsed:.4f}s)")

def log_hang_detection(test_name: str, elapsed_time: float, stack_trace: str):
    """Log when a test appears to be hanging."""
    test_logger.error(f"🚨 HANG DETECTED: {test_name} has been running for {elapsed_time:.1f}s")
    test_logger.error(f"Stack trace at hang detection:\n{stack_trace}")

def log_system_info():
    """Log system information for debugging."""
    import psutil
    import platform
    
    test_logger.info("📊 System Information:")
    test_logger.info(f"  Platform: {platform.platform()}")
    test_logger.info(f"  Python: {sys.version}")
    test_logger.info(f"  CPU count: {os.cpu_count()}")
    test_logger.info(f"  Memory: {psutil.virtual_memory().total / (1024**3):.1f} GB")
    
    # CUDA info if available
    try:
        import genesis
        if genesis.device("cuda").enabled():
            test_logger.info(f"  CUDA: Available")
        else:
            test_logger.info(f"  CUDA: Not available")
    except Exception as e:
        test_logger.info(f"  CUDA: Error checking - {e}")

# Environment variable controlled debugging
HANG_DEBUG = os.environ.get('GENESIS_HANG_DEBUG', '0') == '1'
DEBUG_MODE = os.environ.get('GENESIS_DEBUG', '0') == '1'

if HANG_DEBUG:
    test_logger.setLevel(logging.DEBUG)
    test_logger.info("🔍 Hang debugging enabled")

if DEBUG_MODE:
    test_logger.info("🐛 Debug mode enabled")
    log_system_info()