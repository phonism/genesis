"""
Context managers and decorators for controlling gradient computation.
"""

import threading
import functools
from typing import Any, Callable, ContextManager, TypeVar

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
    """Context manager and decorator that disables gradient calculation.

    Disabling gradient calculation is useful for inference, when you are sure
    that you will not call Tensor.backward(). It will reduce memory consumption
    for computations that would otherwise have requires_grad=True.

    Can be used as a context manager or as a function decorator.

    Example (context manager):
        >>> x = genesis.tensor([1.], requires_grad=True)
        >>> with genesis.no_grad():
        ...     y = x * 2
        >>> y.requires_grad
        False

    Example (decorator):
        >>> @genesis.no_grad()
        ... def evaluation_step(model, x):
        ...     return model(x)
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

    def __call__(self, func: Callable) -> Callable:
        """Enable using no_grad as a decorator.

        Args:
            func: Function to wrap with gradient disabled context

        Returns:
            Wrapped function that runs with gradients disabled
        """
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with self.__class__():
                return func(*args, **kwargs)
        return wrapper

class enable_grad:
    """Context manager and decorator that enables gradient calculation.

    Can be used as a context manager or as a function decorator.

    Example (context manager):
        >>> x = genesis.tensor([1.], requires_grad=True)
        >>> with genesis.no_grad():
        ...     with genesis.enable_grad():
        ...         y = x * 2
        >>> y.requires_grad
        True

    Example (decorator):
        >>> @genesis.enable_grad()
        ... def training_step(model, x):
        ...     return model(x)
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

    def __call__(self, func: Callable) -> Callable:
        """Enable using enable_grad as a decorator.

        Args:
            func: Function to wrap with gradient enabled context

        Returns:
            Wrapped function that runs with gradients enabled
        """
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with self.__class__():
                return func(*args, **kwargs)
        return wrapper

class set_grad_enabled:
    """Context manager and decorator that sets gradient calculation on or off.

    Args:
        mode (bool): Flag whether to enable grad (True), or disable (False).
                     This can be used to conditionally enable gradients.

    Can be used as a context manager or as a function decorator.

    Example (context manager):
        >>> x = genesis.tensor([1.], requires_grad=True)
        >>> is_train = False
        >>> with genesis.set_grad_enabled(is_train):
        ...     y = x * 2
        >>> y.requires_grad
        False

    Example (decorator):
        >>> @genesis.set_grad_enabled(False)
        ... def inference_step(model, x):
        ...     return model(x)
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

    def __call__(self, func: Callable) -> Callable:
        """Enable using set_grad_enabled as a decorator.

        Args:
            func: Function to wrap with gradient mode set

        Returns:
            Wrapped function that runs with specified gradient mode
        """
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with self.__class__(self.mode):
                return func(*args, **kwargs)
        return wrapper