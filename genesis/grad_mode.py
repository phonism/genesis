"""
Context managers for controlling gradient computation.
"""

import threading
from typing import Any, ContextManager

# Thread-local storage for gradient enabled state
_grad_enabled = threading.local()

def _get_grad_enabled():
    """Get current gradient enabled state."""
    if not hasattr(_grad_enabled, 'enabled'):
        _grad_enabled.enabled = True
    return _grad_enabled.enabled

def _set_grad_enabled(enabled: bool):
    """Set gradient enabled state."""
    _grad_enabled.enabled = enabled

def is_grad_enabled() -> bool:
    """Return True if gradient computation is enabled."""
    return _get_grad_enabled()

class no_grad:
    """Context manager that disables gradient calculation.
    
    Disabling gradient calculation is useful for inference, when you are sure 
    that you will not call Tensor.backward(). It will reduce memory consumption 
    for computations that would otherwise have requires_grad=True.
    
    Example:
        >>> x = genesis.tensor([1.], requires_grad=True)
        >>> with genesis.no_grad():
        ...     y = x * 2
        >>> y.requires_grad
        False
    """
    
    def __init__(self):
        self.prev = None
    
    def __enter__(self):
        self.prev = _get_grad_enabled()
        _set_grad_enabled(False)
        return self
    
    def __exit__(self, *args):
        _set_grad_enabled(self.prev)
        return False

class enable_grad:
    """Context manager that enables gradient calculation.
    
    Example:
        >>> x = genesis.tensor([1.], requires_grad=True)
        >>> with genesis.no_grad():
        ...     with genesis.enable_grad():
        ...         y = x * 2
        >>> y.requires_grad
        True
    """
    
    def __init__(self):
        self.prev = None
    
    def __enter__(self):
        self.prev = _get_grad_enabled()
        _set_grad_enabled(True)
        return self
    
    def __exit__(self, *args):
        _set_grad_enabled(self.prev)
        return False

class set_grad_enabled:
    """Context manager that sets gradient calculation on or off.
    
    Args:
        mode (bool): Flag whether to enable grad (True), or disable (False).
                     This can be used to conditionally enable gradients.
                     
    Example:
        >>> x = genesis.tensor([1.], requires_grad=True)
        >>> is_train = False
        >>> with genesis.set_grad_enabled(is_train):
        ...     y = x * 2
        >>> y.requires_grad
        False
    """
    
    def __init__(self, mode: bool):
        self.prev = None
        self.mode = mode
    
    def __enter__(self):
        self.prev = _get_grad_enabled()
        _set_grad_enabled(self.mode)
        return self
    
    def __exit__(self, *args):
        _set_grad_enabled(self.prev)
        return False