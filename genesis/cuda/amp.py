"""Automatic Mixed Precision (AMP) training utilities.

This module provides standard interfaces for mixed precision training,
including autocast context manager and gradient scaling for numerical stability.
"""

import genesis
from enum import Enum
from .amp_cache import get_amp_cache


# ============================================================================
# AMP Policy System (Metadata-Driven Approach)
# ============================================================================
# Following standard autocast strategy with metadata-driven design


class AMPPolicy(Enum):
    """AMP casting policy for operations.

    This enum defines how operations should handle dtype casting in autocast mode.
    Each Function class declares its policy via the `amp_policy` attribute.

    Default Behavior:
        Operations without explicit amp_policy will use PROMOTE (safe default).
        Only operations that need special handling should declare a policy.
    """

    # Cast inputs to FP16 (for Tensor Core acceleration)
    FP16 = "fp16"

    # Cast inputs to FP32 (for numerical stability)
    FP32 = "fp32"

    # Promote mixed dtypes to FP32 (for consistent arithmetic)
    # This is also the DEFAULT behavior for operations without explicit policy
    PROMOTE = "promote"

    # Preserve input dtype (for view/shape operations)
    PRESERVE = "preserve"


def get_amp_dtype(policy: AMPPolicy, *args, **kwargs):
    """Determine target dtype based on AMP policy and inputs.

    Args:
        policy: AMPPolicy enum value (or None for default PROMOTE behavior)
        *args: Input arguments (may contain tensors)
        **kwargs: Input keyword arguments

    Returns:
        Target dtype for casting, or None if no casting needed
    """
    if policy == AMPPolicy.FP16:
        return genesis.float16

    elif policy == AMPPolicy.FP32:
        return genesis.float32

    elif policy == AMPPolicy.PRESERVE:
        return None  # No casting

    elif policy == AMPPolicy.PROMOTE or policy is None:
        # Default behavior: PROMOTE mixed dtypes to FP32
        # Check if inputs have mixed dtypes
        has_fp32 = _check_dtype(args, genesis.float32) or _check_dtype(kwargs, genesis.float32)
        has_fp16 = _check_dtype(args, genesis.float16) or _check_dtype(kwargs, genesis.float16)
        if has_fp32 and has_fp16:
            return genesis.float32  # Promote to FP32
        return None  # Keep original dtypes

    else:
        # Unknown policy - use PROMOTE as safe default
        return get_amp_dtype(AMPPolicy.PROMOTE, *args, **kwargs)


def _check_dtype(value, dtype):
    """Check if value contains tensors of specified dtype."""
    if hasattr(value, 'dtype') and value.dtype == dtype:
        return True
    elif isinstance(value, dict):
        return any(_check_dtype(k, dtype) or _check_dtype(v, dtype) for k, v in value.items())
    elif isinstance(value, (list, tuple)):
        return any(_check_dtype(v, dtype) for v in value)
    return False


class autocast:
    """Context manager for automatic mixed precision training.

    When enabled, operations inside the context will use lower precision (fp16/bf16)
    for faster computation while maintaining numerical stability.

    Example:
        with amp.autocast():
            output = model(input)
            loss = criterion(output, target)
    """

    def __enter__(self):
        """Enter autocast context and clear old cached conversions."""
        # Clear cache from previous iteration
        # This prevents memory accumulation while allowing backward pass to use cached tensors
        cache = get_amp_cache()
        cache.clear()

        genesis.enable_autocast = True
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Exit autocast context.

        Note: We don't clear the cache here to allow backward pass to use cached FP16 tensors.
        Cache will be cleared at the start of next forward pass.
        """
        genesis.enable_autocast = False
        return False


class GradScaler:
    """Gradient scaler for mixed precision training (standard API).

    Scales loss to prevent gradient underflow in fp16 training, then unscales
    gradients before optimizer step. Dynamically adjusts scale factor based on
    whether inf/nan gradients are detected.

    Args:
        init_scale: Initial scale factor (default: 2^16)
        growth_factor: Factor to multiply scale by when no inf/nan found (default: 2.0)
        backoff_factor: Factor to multiply scale by when inf/nan found (default: 0.5)
        growth_interval: Number of iterations before growing scale (default: 2000)

    Example:
        scaler = amp.GradScaler()

        for data, target in dataloader:
            optimizer.zero_grad()

            with amp.autocast():
                output = model(data)
                loss = criterion(output, target)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
    """

    def __init__(self, init_scale=2.**16, growth_factor=2.0, backoff_factor=0.5, growth_interval=2000, enabled=True):
        """Initialize GradScaler.

        Args:
            init_scale: Initial scale factor (default: 65536)
            growth_factor: Factor to multiply scale by when no inf/nan detected
            backoff_factor: Factor to multiply scale by when inf/nan detected
            growth_interval: Number of iterations before growing scale
            enabled: If False, GradScaler becomes a no-op (for debugging)
        """
        self._enabled = enabled
        self._scale = init_scale if enabled else 1.0
        self._growth_factor = growth_factor
        self._backoff_factor = backoff_factor
        self._growth_interval = growth_interval
        self._growth_tracker = 0
        self._found_inf_per_device = {}
        self._unscaled = set()  # Track which optimizers have been unscaled

    def scale(self, loss):
        """Multiply loss by the current scale factor.

        Args:
            loss: Loss tensor to scale

        Returns:
            Scaled loss tensor
        """
        return loss * self._scale

    def unscale_(self, optimizer):
        """Unscale gradients by dividing by current scale factor (in-place).

        Args:
            optimizer: Optimizer whose parameters' gradients should be unscaled
        """
        # Avoid unscaling twice
        if id(optimizer) in self._unscaled:
            return

        inv_scale = 1.0 / self._scale

        for param in optimizer.params:
            if param.grad is not None:
                # Replace gradient with unscaled version (standard pattern)
                param.grad = param.grad * inv_scale

        self._unscaled.add(id(optimizer))

    def _unscale_and_check(self, optimizer):
        """Unscale gradients and check for inf/nan in a single pass (optimized).

        This combines unscaling and inf/nan checking in one loop over gradients,
        reducing overhead compared to two separate passes.

        Args:
            optimizer: Optimizer whose parameters' gradients should be unscaled and checked

        Returns:
            True if any gradient contains inf or nan, False otherwise
        """
        # Avoid unscaling twice
        if id(optimizer) in self._unscaled:
            # Already unscaled, just check
            return self._check_inf_nan(optimizer)

        inv_scale = 1.0 / self._scale

        # Collect all gradients
        grads = [param.grad for param in optimizer.params if param.grad is not None]

        if len(grads) == 0:
            self._unscaled.add(id(optimizer))
            return False

        # Compute total squared norm (if any grad has inf/nan, norm will be inf/nan)
        total_norm_sq = genesis.tensor(0.0, device=grads[0].device, dtype=genesis.float32)
        for grad in grads:
            # Convert to FP32 for accurate accumulation
            grad_fp32 = grad if grad.dtype == genesis.float32 else grad.to(genesis.float32)
            total_norm_sq = total_norm_sq + (grad_fp32 * grad_fp32).sum()

        # Check if norm is finite
        found_inf = not genesis.isfinite(total_norm_sq)

        # Unscale gradients (in-place)
        for param in optimizer.params:
            if param.grad is not None:
                param.grad = param.grad * inv_scale

        self._unscaled.add(id(optimizer))
        return found_inf

    def _check_inf_nan(self, optimizer):
        """Check if any gradients contain inf or nan.

        Optimized implementation that computes total gradient norm in a single pass.
        This is much faster than per-gradient checks.

        Args:
            optimizer: Optimizer whose parameters' gradients should be checked

        Returns:
            True if any gradient contains inf or nan, False otherwise
        """
        # Collect all gradients
        grads = [param.grad for param in optimizer.params if param.grad is not None]

        if len(grads) == 0:
            return False

        # Compute total squared norm of all gradients
        # If any gradient has inf/nan, the total norm will be inf/nan
        total_norm_sq = genesis.tensor(0.0, device=grads[0].device, dtype=genesis.float32)
        for grad in grads:
            # Convert to FP32 for accurate accumulation
            grad_fp32 = grad if grad.dtype == genesis.float32 else grad.to(genesis.float32)
            total_norm_sq = total_norm_sq + (grad_fp32 * grad_fp32).sum()

        # Check if total norm is finite
        return not genesis.isfinite(total_norm_sq)

    def step(self, optimizer):
        """Step the optimizer if gradients are finite, otherwise skip.

        This follows the standard pattern:
        1. Unscale gradients and check for inf/nan (optimized single pass)
        2. Only call optimizer.step() if gradients are finite

        Args:
            optimizer: Optimizer to step

        Note:
            Unlike older implementations, this does NOT call optimizer.zero_grad().
            User must call zero_grad() separately (standard behavior).
        """
        # Unscale and check in a single pass (optimization)
        found_inf = self._unscale_and_check(optimizer)

        # Only step if no inf/nan found
        if not found_inf:
            optimizer.step()

        # Track whether inf/nan was found for update()
        self._found_inf_per_device[0] = found_inf

    def update(self):
        """Update the scale factor based on whether inf/nan was found.

        If inf/nan detected: reduce scale by backoff_factor and reset growth tracker
        If no inf/nan: increment growth tracker, grow scale after growth_interval iterations
        """
        # Check if any device found inf/nan
        found_inf = any(self._found_inf_per_device.values()) if self._found_inf_per_device else False

        if found_inf:
            # Reduce scale when inf/nan found
            self._scale *= self._backoff_factor
            self._growth_tracker = 0
        else:
            # Increment growth tracker when no inf/nan
            self._growth_tracker += 1
            if self._growth_tracker >= self._growth_interval:
                # Increase scale after growth_interval successful iterations
                self._scale *= self._growth_factor
                self._growth_tracker = 0

        # Clear tracking for next iteration
        self._found_inf_per_device.clear()
        self._unscaled.clear()
