"""
Random number generation state management for Genesis.

This module provides comprehensive RNG state management including
manual seed setting, state saving/loading, and thread-safe random number generation.
"""

import time
import os
import threading
import random
from typing import Optional
import numpy as np


class Generator:
    """
    Random number generator for reproducible computations.

    Manages random state for reproducible random number generation
    across different Genesis operations.
    """
    
    def __init__(self, device="cpu"):
        """
        Initialize Generator.
        
        Args:
            device: Device for this generator (currently only supports "cpu")
        """
        self.device = device
        self._seed = None
        self._initial_seed = None
        self._counter = 0
        self._lock = threading.Lock()
        
        # Initialize with time-based seed
        self._set_default_seed()
    
    def _set_default_seed(self):
        """Set default seed based on current time and process ID."""
        default_seed = int((time.time() * 1000000) % (2**31)) + os.getpid()
        self.manual_seed(default_seed)
    
    def manual_seed(self, seed: int):
        """
        Set manual seed for this generator.
        
        Args:
            seed: Integer seed value
            
        Returns:
            Generator: Self for chaining
        """
        with self._lock:
            self._seed = int(seed) % (2**31)
            self._initial_seed = self._seed
            self._counter = 0
        return self
    
    def seed(self) -> int:
        """Generate and set a random seed."""
        new_seed = random.randint(0, 2**31 - 1)
        self.manual_seed(new_seed)
        return new_seed
    
    def initial_seed(self) -> int:
        """Get the initial seed that was set."""
        return self._initial_seed
    
    def get_state(self) -> dict:
        """
        Get current generator state.
        
        Returns:
            dict: Generator state containing seed and counter
        """
        with self._lock:
            return {
                'seed': self._seed,
                'initial_seed': self._initial_seed,
                'counter': self._counter
            }
    
    def set_state(self, state: dict):
        """
        Set generator state.
        
        Args:
            state: Generator state dict from get_state()
        """
        with self._lock:
            self._seed = state['seed']
            self._initial_seed = state['initial_seed'] 
            self._counter = state['counter']
    
    def next_seed(self) -> int:
        """
        Generate next seed in sequence.
        
        Returns:
            int: Next seed value for operations
        """
        with self._lock:
            # Use Linear Congruential Generator for next seed
            # Same parameters as used in random_ops.py for consistency
            a = 1664525
            c = 1013904223
            m = 2**32
            
            current_seed = (self._seed + self._counter) % (2**31)
            self._counter += 1
            
            return current_seed


# Global default generator
_default_generator = Generator()

def seed() -> int:
    """
    Generate a random seed and set it as the manual seed.
    
    Returns:
        int: The generated seed
    """
    return _default_generator.seed()

def manual_seed(seed: int) -> Generator:
    """
    Set manual seed for default generator.
    
    Args:
        seed: Integer seed value
        
    Returns:
        Generator: The default generator
    """
    return _default_generator.manual_seed(seed)

def initial_seed() -> int:
    """
    Get the initial seed of the default generator.
    
    Returns:
        int: Initial seed value
    """
    return _default_generator.initial_seed()

def get_rng_state() -> dict:
    """
    Get RNG state of default generator.
    
    Returns:
        dict: Current RNG state
    """
    return _default_generator.get_state()

def set_rng_state(state: dict):
    """
    Set RNG state of default generator.
    
    Args:
        state: RNG state dict from get_rng_state()
    """
    _default_generator.set_state(state)

def default_generator() -> Generator:
    """
    Get the default generator.
    
    Returns:
        Generator: Default generator instance
    """
    return _default_generator

def get_seed_for_operation() -> int:
    """
    Get next seed for random operations.
    
    This is used internally by operations like dropout, randn, etc.
    to get consistent, reproducible random seeds.
    
    Returns:
        int: Seed for next operation
    """
    return _default_generator.next_seed()

def fork_rng(devices=None, enabled=True):
    """
    Context manager to fork the RNG state.
    
    Args:
        devices: Not used currently (for PyTorch compatibility)
        enabled: Whether forking is enabled
        
    Returns:
        Context manager that preserves RNG state
    """
    class ForkRNG:
        def __init__(self, enabled=True):
            self.enabled = enabled
            self.saved_state = None
            
        def __enter__(self):
            if self.enabled:
                self.saved_state = get_rng_state()
            return self
            
        def __exit__(self, *args):
            if self.enabled and self.saved_state is not None:
                set_rng_state(self.saved_state)
    
    return ForkRNG(enabled)
