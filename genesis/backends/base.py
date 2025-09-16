"""
Base storage interface for Genesis backends.

This module defines the abstract base class for storage implementations,
ensuring consistent API across CPU and CUDA backends.
"""

from abc import ABC, abstractmethod
from typing import Any, Optional, Tuple, Union
import numpy as np


class Storage(ABC):
    """Abstract base class for storage backends."""
    
    @abstractmethod
    def to_numpy(self) -> np.ndarray:
        """Convert storage to numpy array."""
        pass
    
    @abstractmethod
    def clone(self):
        """Create a deep copy of the storage."""
        pass
    
    @abstractmethod
    def contiguous(self):
        """Return contiguous version of storage."""
        pass
    
    @abstractmethod
    def is_contiguous(self) -> bool:
        """Check if storage is contiguous in memory."""
        pass
    
    @property
    @abstractmethod
    def shape(self) -> Tuple[int, ...]:
        """Get shape of the storage."""
        pass
    
    @property
    @abstractmethod
    def dtype(self):
        """Get data type of the storage."""
        pass
    
    @abstractmethod
    def __getitem__(self, key):
        """Index into the storage."""
        pass
    
    @abstractmethod
    def __setitem__(self, key, value):
        """Set values in the storage."""
        pass