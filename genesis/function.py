"""Function classes for automatic differentiation in Genesis.

This module contains the base Function class and related utilities for
implementing automatic differentiation operations.
"""

import genesis
from genesis.tensor import Tensor
from typing import List, Optional, NamedTuple, Tuple, Union, Any
import weakref


class AccumulateGrad:
    """Special function node for leaf tensors.

    Leaf tensors (requires_grad=True, creator=None) need a grad_fn
    to participate in the backward graph. AccumulateGrad holds a
    weak reference to the leaf tensor and accumulates gradients to it.
    """

    def __init__(self, tensor):
        """Initialize AccumulateGrad for a leaf tensor.

        Args:
            tensor: The leaf tensor to accumulate gradients for
        """
        self.variable = weakref.ref(tensor)
        self.next_functions = []  # Leaf nodes have no inputs

    def apply_grad(self, grad):
        """Accumulate gradient to the leaf tensor.

        Args:
            grad: Gradient to accumulate
        """
        tensor = self.variable()
        if tensor is not None:  # If tensor still exists
            if tensor.grad is None:
                tensor.grad = grad
            else:
                tensor.grad = tensor.grad + grad


class Context:
    """Stores intermediate values during forward pass for backward computation."""
    
    def __init__(self):
        """Initialize context for storing intermediate values."""
        self._saved_tensors: List[Tensor] = []

    def save_for_backward(self, *tensors: Tensor) -> None:
        """Save tensors needed for backward pass.

        Args:
            *tensors: Tensors to save for backward computation
        """
        self._saved_tensors.extend(tensors)

    @property
    def saved_tensors(self) -> List[Tensor]:
        """Get saved tensors for backward computation."""
        return self._saved_tensors


def _cast(value: Any, dtype: 'genesis.DType') -> Any:
    """Cast tensors to target dtype for mixed precision training.

    Args:
        value: Value to cast (tensor, dict, list, tuple, or other)
        dtype: Target data type for casting

    Returns:
        Casted value with same structure but converted tensors
    """
    if hasattr(value, 'is_floating_point') and value.is_floating_point():
        if dtype == genesis.float16:
            return value.half()
        else:
            return value.float()
    elif isinstance(value, dict):
        return {_cast(k, dtype): _cast(v, dtype) for k, v in value.items()}
    elif isinstance(value, list) or isinstance(value, tuple):
        return type(value)(_cast(v, dtype) for v in value)
    else:
        return value


def check_dtype(value: Any, dtype: 'genesis.DType') -> bool:
    """Check if value contains tensors of specified dtype.

    Args:
        value: Value to check (tensor, dict, list, tuple, or other)
        dtype: Data type to check for

    Returns:
        bool: True if value contains tensors of the specified dtype
    """
    if hasattr(value, 'dtype') and value.dtype == dtype:
        return True
    elif isinstance(value, dict):
        return any(check_dtype(k, dtype) or check_dtype(v, dtype) for k, v in value.items())
    elif isinstance(value, list) or isinstance(value, tuple):
        return any(check_dtype(v, dtype) for v in value)
    else:
        return False


class Function:
    """Base class for differentiable operations.

    Implements the dual-number automatic differentiation paradigm where
    operations define both forward computation and backward gradient propagation.
    """

    def __init__(self):
        """Initialize Function with empty next_functions and context."""
        self.next_functions = []  # List of (creator, output_idx) tuples
        self.ctx = Context()

    @staticmethod
    def forward(ctx, *args, **kwargs):
        """Calculate forward pass of operator.

        Args:
            ctx: Context object to save values for backward pass
            *args: Input arrays to the function
            **kwargs: Additional keyword arguments

        Returns:
            Array: Array output of the operation
        """
        raise NotImplementedError()

    @staticmethod
    def backward(ctx, *args) -> Union["Tensor", Tuple["Tensor"]]:
        """Compute partial adjoint for each input value for a given output adjoint.

        Args:
            ctx: Context object containing saved values from forward pass
            *args: The adjoint with respect to the output value

        Returns:
            Tensor or Tuple[Tensor]: Partial gradient adjoints to be propagated to each input node
        """
        raise NotImplementedError()

    @classmethod
    def apply(cls, *args, **kwargs):
        """Apply operation with automatic mixed precision and gradient tracking."""
        instance = cls()

        # Handle mixed precision casting
        if hasattr(genesis, 'enable_autocast') and genesis.enable_autocast and hasattr(genesis, 'upgrade') and genesis.upgrade is False:
            result = cls.forward(
                    instance.ctx, *_cast(args, genesis.float16), **_cast(kwargs, genesis.float16))
        else:
            has_float32 = check_dtype(args, genesis.float32) or check_dtype(kwargs, genesis.float32)
            has_float16 = check_dtype(args, genesis.float16) or check_dtype(kwargs, genesis.float16)
            if has_float32 and has_float16:
                result = cls.forward(instance.ctx, *_cast(args, genesis.float32), **_cast(kwargs, genesis.float32))
            else:
                result = cls.forward(instance.ctx, *args, **kwargs)
        
        instance.is_tuple_result = isinstance(result, tuple)

        # Only set creator for gradient tracking if gradients are enabled
        if genesis.is_grad_enabled():
            # Set creator for gradient tracking - check both tensor types
            
            if instance.is_tuple_result:
                for idx, res in enumerate(result):
                    if hasattr(res, 'requires_grad') and res.requires_grad:
                        if hasattr(res, 'set_creator'):
                            res.set_creator(instance, idx)
            elif hasattr(result, 'requires_grad') and result.requires_grad:
                if hasattr(result, 'set_creator'):
                    result.set_creator(instance)

        # Build next_functions graph (creator graph, not tensor graph)
        # This allows intermediate tensors to be garbage collected
        instance.next_functions = []
        for t in args:
            if hasattr(t, 'requires_grad') and t.requires_grad:
                if hasattr(t, 'creator') and t.creator is not None:
                    # Intermediate tensor: save its creator
                    idx = t.idx if hasattr(t, 'idx') else 0
                    instance.next_functions.append((t.creator, idx))
                else:
                    # Leaf tensor: get or create AccumulateGrad node
                    if not hasattr(t, '_grad_fn'):
                        t._grad_fn = AccumulateGrad(t)
                    instance.next_functions.append((t._grad_fn, 0))
            elif isinstance(t, list):
                # Handle list of tensors
                for tt in t:
                    if hasattr(tt, 'requires_grad') and tt.requires_grad:
                        if hasattr(tt, 'creator') and tt.creator is not None:
                            idx = tt.idx if hasattr(tt, 'idx') else 0
                            instance.next_functions.append((tt.creator, idx))
                        else:
                            if not hasattr(tt, '_grad_fn'):
                                tt._grad_fn = AccumulateGrad(tt)
                            instance.next_functions.append((tt._grad_fn, 0))

        return result
