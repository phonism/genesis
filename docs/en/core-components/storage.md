# Storage Layer

Genesis's storage layer provides an abstraction interface for managing tensor data storage across different devices.

## 📋 Overview

The storage layer is a key component of Genesis v2.0 architecture, providing:
- Abstraction of device-specific storage implementations
- Memory lifecycle management
- Efficient data transfer between devices
- Support for various data types and memory layouts

## 🏗️ Architecture

```mermaid
graph TB
    subgraph "Storage Abstraction"
        A[Storage Interface] --> B[create_storage()]
        A --> C[storage.to()]
        A --> D[storage.copy_()]
    end

    subgraph "Device Implementations"
        E[CPUStorage] --> A
        F[CUDAStorage] --> A
        G[FutureStorage] --> A
    end

    subgraph "Memory Management"
        H[Memory Allocation] --> I[Memory Pool]
        H --> J[Reference Counting]
        H --> K[Garbage Collection]
    end

    style A fill:#e1f5fe
    style E fill:#e8f5e9
    style F fill:#ffeb3b
```

## 🎯 Core Concepts

### Storage Interface
The base interface for all storage implementations:

```python
class Storage:
    """Abstract interface for tensor data storage."""

    def __init__(self, shape, dtype, device):
        self.shape = shape
        self.dtype = dtype
        self.device = device
        self._data = None

    def to(self, device):
        """Transfer storage to another device."""
        raise NotImplementedError

    def copy_(self, other):
        """In-place copy from another storage."""
        raise NotImplementedError

    def clone(self):
        """Create a deep copy of storage."""
        raise NotImplementedError

    @property
    def data_ptr(self):
        """Get underlying data pointer."""
        return self._data
```

### Storage Creation
Creating storage appropriate for the device:

```python
def create_storage(data, device, dtype=None):
    """Create storage for given device."""
    if device.type == 'cpu':
        return CPUStorage(data, dtype)
    elif device.type == 'cuda':
        return CUDAStorage(data, dtype)
    else:
        raise ValueError(f"Unsupported device type: {device.type}")

# Usage
storage = create_storage([1, 2, 3], genesis.device("cuda"))
```

## 🔗 See Also

- [Device Abstraction](device.md) - Device management system
- [Memory Management](../backends/memory.md) - Advanced memory management
- [Backend System](../backends/index.md) - Backend implementations
- [Tensor System](tensor.md) - Tensor class and storage integration