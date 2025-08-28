"""
GPU operations module for Genesis framework.
"""
# Import all functions from submodules to maintain API compatibility
from .basic_ops import (
    add, sub, iadd, mul, truediv, rtruediv, pow, log, exp, sin, cos, sqrt, 
    abs, sign, clamp, greater_equal, less_equal, ones, zeros, maximum, 
    mul_scalar_kernel_wrapper, one_hot, arange, where, zeros_like,
    argmax, argmin, isinf, isnan, isfinite
)
from .comparison_ops import eq, ge, gt, le, lt
from .reduction_ops import reduce_sum, reduce_max
from .shape_ops import reshape, view, expand, permute, broadcast_to, repeat_interleave
from .indexing_ops import getitem, setitem, fill, fill_tensor, cat, gather, scatter, scatter_add
from .matrix_ops import matmul
from .utils import prod, broadcast_shapes, from_numpy, from_tensor, array, clone
from .random_ops import randn, rand
from .tensor_ops import triu, split, to_dtype, squeeze, unsqueeze, topk, argsort, bincount