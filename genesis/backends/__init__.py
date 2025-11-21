"""Backend storage implementations for Genesis framework.

This module provides the storage layer abstraction for different computing devices,
including CPU (PyTorch-based) and CUDA (custom kernels) backends.
"""

from genesis.backends.base import Storage
from genesis.backends.cpu import CPUStorage
from genesis.backends.cuda import CUDAStorage

__all__ = ["Storage", "CPUStorage", "CUDAStorage"]