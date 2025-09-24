"""
CPU reduction operations implementation.
"""

import torch
from genesis.ops.dispatcher import register_cpu


@register_cpu("sum")
def sum(x, axis=None, keepdims=False):
    """Sum reduction operation."""
    if axis is None:
        # Sum all elements
        result = x.sum()
        if keepdims:
            # Keep original number of dimensions
            shape = [1] * len(x.shape)
            result = result.view(shape)
        return result
    else:
        # Sum along specific axis
        if isinstance(axis, int):
            axis = (axis,)
        elif isinstance(axis, (list, tuple)):
            axis = tuple(axis)
        
        result = x
        for ax in sorted(axis, reverse=True):  # Process in reverse order to maintain indices
            result = torch.sum(result, dim=ax, keepdim=keepdims)
        
        return result


@register_cpu("max")
def max(x, axis=None, keepdims=False):
    """Max reduction operation."""
    if axis is None:
        result = x.max()
        if keepdims:
            shape = [1] * len(x.shape)
            result = result.view(shape)
        return result
    else:
        if isinstance(axis, int):
            result = torch.max(x, dim=axis, keepdim=keepdims)[0]  # torch.max returns (values, indices)
        elif isinstance(axis, (list, tuple)):
            # Handle multi-axis reduction
            result = x
            for ax in sorted(axis, reverse=True):  # Process in reverse order to maintain indices
                result = torch.max(result, dim=ax, keepdim=keepdims)[0]
        else:
            raise NotImplementedError(f"Unsupported axis type: {type(axis)}")
        return result


@register_cpu("mean")
def mean(x, axis=None, keepdims=False):
    """Mean reduction operation."""
    if axis is None:
        result = x.mean()
        if keepdims:
            shape = [1] * len(x.shape)
            result = result.view(shape)
        return result
    else:
        if isinstance(axis, int):
            result = torch.mean(x, dim=axis, keepdim=keepdims)
        else:
            raise NotImplementedError("Multi-axis mean not implemented yet")
        return result


@register_cpu("sum_to_shape")
def sum_to_shape(x, target_shape):
    """Sum tensor to match target shape."""
    result = x

    # First, reduce leading dimensions if needed
    while len(result.shape) > len(target_shape):
        result = result.sum(dim=0)

    # Then, reduce dimensions where target is 1 but current is not
    for i in range(len(target_shape)):
        if i < len(result.shape) and target_shape[i] == 1 and result.shape[i] != 1:
            result = result.sum(dim=i, keepdim=True)

    return result