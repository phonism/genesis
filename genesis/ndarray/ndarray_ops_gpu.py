"""
GPU operations backend for Genesis framework.

This module provides GPU operations by importing from the organized submodules.
The operations are split into different categories for better maintainability:
- basic_ops: Arithmetic operations (add, mul, div, etc.)
- comparison_ops: Comparison operations (eq, lt, gt, etc.) 
- reduction_ops: Reduction operations (sum, max)
- shape_ops: Shape manipulation (reshape, view, permute, etc.)
- matrix_ops: Matrix operations (matmul)
- indexing_ops: Indexing and manipulation (getitem, setitem, cat)
- utils: Utility functions (broadcast_shapes, from_numpy, etc.)
"""

# Import all GPU operations from the modularized structure
from .gpu_ops import *