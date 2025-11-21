"""
Global cache for Automatic Mixed Precision (AMP) dtype conversions.

This module implements efficient caching for FP32→FP16 conversions to avoid
repeated dtype conversions during the forward pass. The cache is cleared after
each forward pass (when exiting autocast context).
"""

from typing import Dict, Any
import weakref
from genesis.dtypes import float16, float32


class AMPCache:
    """Global cache for automatic mixed precision dtype conversions.

    This cache stores FP16 versions of FP32 parameters to avoid repeated
    conversions during forward passes. The cache is automatically cleared
    when exiting the autocast context.

    Caching Strategy:
    - Only cache FP32 → FP16 conversions
    - Only cache leaf tensors with requires_grad=True (i.e., parameters)
    - Cache key is id(tensor) - unique identifier for tensor lifetime
    - Cache is cleared at end of each forward pass

    Example:
        >>> cache = AMPCache()
        >>> # First use: converts and caches
        >>> weight_fp16 = cache.cached_cast(weight_fp32, genesis.float16)
        >>> # Second use in same forward: cache hit, no conversion
        >>> weight_fp16_2 = cache.cached_cast(weight_fp32, genesis.float16)
        >>> assert weight_fp16 is weight_fp16_2  # Same object!
        >>> # After forward pass
        >>> cache.clear()
    """

    def __init__(self):
        """Initialize empty cache."""
        self._cache: Dict[int, Any] = {}
        self._enabled = True
        self._cache_hits = 0
        self._cache_misses = 0

    def cached_cast(self, tensor, target_dtype):
        """
        Cast tensor to target dtype with intelligent caching.

        Only caches FP32→FP16 conversions of leaf tensors (parameters).
        All other conversions are performed without caching.

        Args:
            tensor: Input tensor to convert
            target_dtype: Target dtype (e.g., float16)

        Returns:
            Converted tensor (may be cached)
        """
        # Determine if this conversion should be cached
        should_cache = (
            self._enabled and
            hasattr(tensor, 'dtype') and
            hasattr(tensor, 'requires_grad') and
            hasattr(tensor, 'is_leaf') and
            target_dtype == float16 and
            tensor.dtype == float32 and
            tensor.requires_grad and
            tensor.is_leaf
        )

        if should_cache:
            # Check cache
            tensor_id = id(tensor)
            if tensor_id in self._cache:
                # Cache hit!
                self._cache_hits += 1
                return self._cache[tensor_id]

            # Cache miss - convert and store
            self._cache_misses += 1
            converted = tensor.to_dtype(target_dtype)
            self._cache[tensor_id] = converted
            return converted
        else:
            # Non-cacheable case - just convert
            if hasattr(tensor, 'to_dtype'):
                return tensor.to_dtype(target_dtype)
            else:
                # Not a tensor, return as-is
                return tensor

    def clear(self):
        """Clear the cache.

        This should be called at the end of each forward pass (when exiting
        autocast context) to ensure fresh conversions in the next iteration.
        """
        self._cache.clear()

    def set_enabled(self, enabled: bool):
        """Enable or disable caching.

        Args:
            enabled: If True, caching is enabled. If False, all conversions
                     are performed without caching.
        """
        self._enabled = enabled
        if not enabled:
            self.clear()

    def get_stats(self) -> Dict[str, int]:
        """Get cache statistics.

        Returns:
            Dictionary with cache hits, misses, and hit rate
        """
        total = self._cache_hits + self._cache_misses
        hit_rate = self._cache_hits / total if total > 0 else 0.0

        return {
            'cache_hits': self._cache_hits,
            'cache_misses': self._cache_misses,
            'total_requests': total,
            'hit_rate': hit_rate,
            'cache_size': len(self._cache),
        }

    def reset_stats(self):
        """Reset cache statistics."""
        self._cache_hits = 0
        self._cache_misses = 0


# Global singleton instance
_global_amp_cache = AMPCache()


def get_amp_cache() -> AMPCache:
    """
    Get the global AMP cache instance.

    Returns:
        Global AMPCache singleton
    """
    return _global_amp_cache
