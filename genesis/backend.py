"""Logic for backend selection"""
from . import ndarray as array_api
from .ndarray import (
        all_devices,
        cpu,
        cuda,
        device,
        default_device,
        Device,
)
NDArray = array_api.NDArray

