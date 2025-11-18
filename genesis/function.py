"""Function classes for automatic differentiation in Genesis.

This module contains the base Function class and related utilities for
implementing automatic differentiation operations.
"""

import genesis
from genesis.tensor import Tensor
from genesis.amp import AMPPolicy, get_amp_dtype
from genesis.amp.amp_cache import get_amp_cache
from genesis.grad_mode import no_grad
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

    CRITICAL: Uses Function-based casting with caching optimization!
    - For leaf tensors (parameters): cache FP16 versions to avoid repeated conversion
    - For intermediate tensors: use Function-based casting
    - All paths preserve gradient flow through the computation graph

    Args:
        value: Value to cast (tensor, dict, list, tuple, or other)
        dtype: Target data type for casting

    Returns:
        Casted value with same structure but converted tensors
    """
    if isinstance(value, Tensor) and value.is_floating_point():
        # Use Function-based casting for FP16 and FP32
        if dtype == genesis.float16:
            # Optimization: cache leaf tensor conversions (parameters)
            # Intermediate tensors still use Function-based casting
            if value.is_leaf and value.requires_grad:
                # Use cache for parameters (with Function-based conversion)
                cache = get_amp_cache()
                tensor_id = id(value)

                if tensor_id in cache._cache:
                    return cache._cache[tensor_id]

                # Convert using Function (preserves gradients)
                converted = cast_to_fp16(value)
                cache._cache[tensor_id] = converted
                return converted
            else:
                # Intermediate tensors: always use Function-based casting
                return cast_to_fp16(value)
        elif dtype == genesis.float32:
            # CRITICAL: Use Function-based casting to preserve creator chain!
            return cast_to_fp32(value)
        else:
            return value.to(dtype)
    elif isinstance(value, dict):
        return {_cast(k, dtype): _cast(v, dtype) for k, v in value.items()}
    elif isinstance(value, (list, tuple)):
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
    if isinstance(value, Tensor) and value.dtype == dtype:
        return True
    elif isinstance(value, dict):
        return any(check_dtype(k, dtype) or check_dtype(v, dtype) for k, v in value.items())
    elif isinstance(value, (list, tuple)):
        return any(check_dtype(v, dtype) for v in value)
    else:
        return False


class Function:
    """Base class for differentiable operations.

    Implements the dual-number automatic differentiation paradigm where
    operations define both forward computation and backward gradient propagation.

    AMP Policy System:
        Subclasses can declare their AMP behavior by setting the `amp_policy` class attribute:
        - AMPPolicy.FP16: Cast to FP16 (for Tensor Core ops like matmul)
        - AMPPolicy.FP32: Cast to FP32 (for numerical stability like softmax)
        - AMPPolicy.PROMOTE: Promote mixed dtypes to FP32
        - AMPPolicy.PRESERVE: Preserve input dtype
        - AMPPolicy.DEFAULT: No special handling (default)

    Example:
        from genesis.amp import AMPPolicy

        class MatMul(Function):
            amp_policy = AMPPolicy.FP16  # Benefit from Tensor Core

            @staticmethod
            def forward(ctx, a, b):
                return a @ b
    """

    # Default AMP policy: preserve dtype (no casting)
    amp_policy = None  # Will use AMPPolicy.DEFAULT

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
        """Apply operation with automatic mixed precision and gradient tracking.

        Implements metadata-driven AMP: each Function subclass declares its
        amp_policy, and this method applies the appropriate dtype casting.
        """
        instance = cls()

        # === AMP Casting Logic ===
        if genesis.enable_autocast:
            # Get target dtype based on operation's AMP policy
            # Default is PROMOTE (safe for mixed dtypes)
            policy = cls.amp_policy if hasattr(cls, 'amp_policy') and cls.amp_policy is not None else None
            target_dtype = get_amp_dtype(policy, *args, **kwargs)
            if target_dtype is not None:
                args = _cast(args, target_dtype)
                kwargs = _cast(kwargs, target_dtype)

        # Handle mixed dtypes when autocast is disabled (promote to FP32)
        elif not genesis.enable_autocast:
            has_fp32 = check_dtype(args, genesis.float32) or check_dtype(kwargs, genesis.float32)
            has_fp16 = check_dtype(args, genesis.float16) or check_dtype(kwargs, genesis.float16)
            if has_fp32 and has_fp16:
                args = _cast(args, genesis.float32)
                kwargs = _cast(kwargs, genesis.float32)

        # Execute forward pass in no_grad context to prevent intermediate node creation
        # This ensures only the final output is tracked in the autograd graph
        with no_grad():
            result = cls.forward(instance.ctx, *args, **kwargs)

        instance.is_tuple_result = isinstance(result, tuple)

        # Set creator for gradient tracking
        if genesis.is_grad_enabled():
            if instance.is_tuple_result:
                # For tuple results, count how many outputs need backward
                # Only clear saved_tensors after all outputs have done backward
                instance.ctx._backward_count = len(result)
                for idx, res in enumerate(result):
                    if isinstance(res, Tensor) and res.requires_grad:
                        res.set_creator(instance, idx)
            elif isinstance(result, Tensor) and result.requires_grad:
                result.set_creator(instance)

        # Build next_functions graph (creator graph, not tensor graph)
        # This allows intermediate tensors to be garbage collected
        instance.next_functions = []
        for t in args:
            if isinstance(t, Tensor) and t.requires_grad:
                if t.creator is not None:
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
                    if isinstance(tt, Tensor) and tt.requires_grad:
                        if tt.creator is not None:
                            idx = tt.idx if hasattr(tt, 'idx') else 0
                            instance.next_functions.append((tt.creator, idx))
                        else:
                            if not hasattr(tt, '_grad_fn'):
                                tt._grad_fn = AccumulateGrad(tt)
                            instance.next_functions.append((tt._grad_fn, 0))

        return result


class CastToFP16(Function):
    """Cast tensor to FP16 with gradient support.

    This Function ensures type conversion maintains the computational graph
    for proper gradient flow in mixed precision training.
    """

    # CRITICAL: PRESERVE policy prevents recursive AMP casting
    amp_policy = AMPPolicy.PRESERVE

    @staticmethod
    def forward(ctx, a):
        """Forward: convert to FP16."""
        # Store original dtype for backward
        ctx.original_dtype = a.dtype

        if a.dtype == genesis.float16:
            return a

        # Convert to FP16 using storage conversion
        result = a.to_dtype(genesis.float16)
        result.requires_grad = a.requires_grad
        return result

    @staticmethod
    def backward(ctx, out_grad):
        """Backward: convert gradient back to original dtype."""
        if ctx.original_dtype == genesis.float16:
            return (out_grad,)

        # Convert gradient back to original dtype (typically FP32)
        grad = out_grad.to_dtype(ctx.original_dtype)
        grad.requires_grad = False
        return (grad,)


class CastToFP32(Function):
    """Cast tensor to FP32 with gradient support.

    This Function ensures type conversion maintains the computational graph
    for proper gradient flow in mixed precision training.
    """

    # CRITICAL: PRESERVE policy prevents recursive AMP casting
    amp_policy = AMPPolicy.PRESERVE

    @staticmethod
    def forward(ctx, a):
        """Forward: convert to FP32."""
        # Store original dtype for backward (even if already FP32)
        ctx.original_dtype = a.dtype

        if a.dtype == genesis.float32:
            return a

        # Convert to FP32 using storage conversion
        result = a.to_dtype(genesis.float32)
        result.requires_grad = a.requires_grad
        return result

    @staticmethod
    def backward(ctx, out_grad):
        """Backward: convert gradient back to original dtype."""
        if ctx.original_dtype == genesis.float32:
            return (out_grad,)

        # Convert gradient back to original dtype
        grad = out_grad.to_dtype(ctx.original_dtype)
        grad.requires_grad = False
        return (grad,)


def cast_to_fp16(a):
    """Cast tensor to FP16 with gradient support.

    CRITICAL: If already FP16, return immediately to preserve existing creator!
    """
    if a.dtype == genesis.float16:
        return a
    return CastToFP16.apply(a)


def cast_to_fp32(a):
    """Cast tensor to FP32 with gradient support.

    CRITICAL: If already FP32, return immediately to preserve existing creator!
    """
    if a.dtype == genesis.float32:
        return a
    return CastToFP32.apply(a)
