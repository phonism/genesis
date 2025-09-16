"""
Device management for Genesis framework.

This module provides device abstraction for CPU and CUDA devices.
"""

from enum import Enum
from typing import Optional


class DeviceType(Enum):
    """Supported device types."""
    CPU = "cpu"
    CUDA = "cuda"


class Device:
    """Device abstraction for CPU and GPU computation."""
    
    def __init__(self, device_str: str):
        """Initialize device from string descriptor.
        
        Args:
            device_str: Device string like 'cpu', 'cuda', 'cuda:0'
        """
        if device_str == "cpu":
            self.type = DeviceType.CPU
            self.index = None
        elif device_str.startswith("cuda"):
            self.type = DeviceType.CUDA
            if ":" in device_str:
                self.index = int(device_str.split(":")[1])
            else:
                self.index = 0
        else:
            raise ValueError(f"Unknown device: {device_str}")
    
    def __str__(self):
        if self.type == DeviceType.CPU:
            return "cpu"
        elif self.index is not None and self.index > 0:
            return f"cuda:{self.index}"
        else:
            return "cuda"
    
    def __repr__(self):
        return f"Device('{str(self)}')"
    
    def __eq__(self, other):
        if isinstance(other, Device):
            return self.type == other.type and self.index == other.index
        return False
    
    def is_cuda(self) -> bool:
        """Check if device is CUDA."""
        return self.type == DeviceType.CUDA
    
    def is_cpu(self) -> bool:
        """Check if device is CPU."""
        return self.type == DeviceType.CPU


# Global default device
_default_device = Device("cpu")


def device(device_input) -> Device:
    """Create a device from string or return existing Device.

    Args:
        device_input: Device string like 'cpu', 'cuda', 'cuda:0' or existing Device object

    Returns:
        Device instance
    """
    if isinstance(device_input, Device):
        return device_input
    return Device(device_input)


def default_device() -> Device:
    """Get the default device.
    
    Returns:
        Default device (CPU by default)
    """
    return _default_device


def set_default_device(device_str: str):
    """Set the default device.
    
    Args:
        device_str: Device string like 'cpu', 'cuda', 'cuda:0'
    """
    global _default_device
    _default_device = Device(device_str)


def cuda(device_id: int = 0) -> Device:
    """Create a CUDA device.
    
    Args:
        device_id: CUDA device ID (default 0)
        
    Returns:
        CUDA device
    """
    if device_id == 0:
        return Device("cuda")
    return Device(f"cuda:{device_id}")


def cpu() -> Device:
    """Create a CPU device.
    
    Returns:
        CPU device
    """
    return Device("cpu")