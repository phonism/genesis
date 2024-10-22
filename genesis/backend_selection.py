"""Logic for backend selection"""
import os

from . import backend_ndarray as array_api
from .backend_ndarray import (
        all_devices,
        cpu,
        cuda,
        default_device,
        BackendDevice as Device,
)
NDArray = array_api.NDArray
