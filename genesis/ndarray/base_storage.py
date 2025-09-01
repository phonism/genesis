"""
Base storage interface for unified tensor storage abstraction.

Provides a common interface for CPU and GPU storage backends to enable
consistent access patterns across different device types.
"""

from abc import ABC, abstractmethod
from typing import Tuple
import numpy as np


class BaseStorage(ABC):
    """
    Abstract base class for tensor storage backends.
    
    Defines the minimum interface that all storage implementations must provide
    to ensure consistency across CPU and GPU backends.
    """
    
    @property
    @abstractmethod
    def shape(self) -> Tuple[int, ...]:
        """
        Return the shape of the tensor as a tuple of integers.
        
        Returns:
            Tuple[int, ...]: The shape of the tensor
        """
        pass
    
    @property
    @abstractmethod
    def dtype(self) -> str:
        """
        Return the data type of the tensor as a string.
        
        Returns:
            str: The data type (e.g., "float32", "float16", "int64")
        """
        pass
    
    @abstractmethod
    def to_numpy(self) -> np.ndarray:
        """
        Convert the storage to a NumPy array.
        
        Returns:
            np.ndarray: NumPy representation of the tensor data
        """
        pass