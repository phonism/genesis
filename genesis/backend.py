"""Backend device abstraction and array API for Genesis.

This module provides device management and array operations that abstract
over CPU (NumPy) and GPU (CUDA) backends for unified tensor computation.
"""

from . import ndarray as array_api
from .ndarray import (
        all_devices,      # List of all available devices
        device,           # Generic device constructor
        default_device,   # Get default device
        Device,           # Device class
)
NDArray = array_api.NDArray  # Core N-dimensional array class

