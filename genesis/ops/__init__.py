"""
Operations layer for Genesis framework.

This module provides high-level operations with automatic device dispatch.
Importing this module automatically registers all CPU and CUDA operations.
"""

# Import submodules to trigger operation registration
from . import cpu
from . import cuda
from .dispatcher import OperationDispatcher, DeviceType

__all__ = ["OperationDispatcher", "DeviceType"]