"""
Lightweight CUDA memory allocator optimized for stable training workloads.

Design Goals:
- Ultra-fast hot path: single dict/list lookup + minimal branching
- Stable training: no cudaMalloc after warmup for common sizes
- Size-based pooling: small (<1MB) and large (1-16MB) bins prevent fragmentation
- Stream-safe but cheap: only cross-stream deallocations need events
- Optional observability: stats/fragmentation/GC are off hot path
"""

try:
    from cuda import cuda
except ImportError:
    from cuda.bindings import driver as cuda

import threading
import os
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
from genesis.backends.cuda_error import check_cuda_error

# Memory statistics control - disabled by default for performance
ENABLE_MEM_STATS = bool(int(os.getenv("GENESIS_MEM_STATS", "0")))

# Size class definitions
# Small pool: <1MB - most common tensor sizes in training
SMALL_SIZES = [
    64, 128, 256, 512,  # Tiny tensors
    1024, 2048, 4096, 8192,  # Small tensors (1KB - 8KB)
    16384, 32768, 65536,  # Medium-small (16KB - 64KB)
    131072, 262144, 524288,  # Large-small (128KB - 512KB)
    1048576  # 1MB
]

# Large pool: 1MB - 1GB - attention/MLP intermediate tensors and gradients
# Optimized with finer granularity to reduce allocation wastage
LARGE_SIZES = [
    2 * 1024 * 1024,    # 2MB
    4 * 1024 * 1024,    # 4MB
    6 * 1024 * 1024,    # 6MB - for small weights
    8 * 1024 * 1024,    # 8MB
    10 * 1024 * 1024,   # 10MB - common activation size (2048 * 1216 * 4)
    12 * 1024 * 1024,   # 12MB
    14 * 1024 * 1024,   # 14MB
    16 * 1024 * 1024,   # 16MB
    18 * 1024 * 1024,   # 18MB - QKV weights
    20 * 1024 * 1024,   # 20MB
    24 * 1024 * 1024,   # 24MB - FFN weights
    28 * 1024 * 1024,   # 28MB
    32 * 1024 * 1024,   # 32MB
    36 * 1024 * 1024,   # 36MB
    40 * 1024 * 1024,   # 40MB - FFN activations (2048 * 4 * 1216 * 4)
    44 * 1024 * 1024,   # 44MB
    48 * 1024 * 1024,   # 48MB
    52 * 1024 * 1024,   # 52MB
    56 * 1024 * 1024,   # 56MB
    64 * 1024 * 1024,   # 64MB
    80 * 1024 * 1024,   # 80MB
    96 * 1024 * 1024,   # 96MB
    128 * 1024 * 1024,  # 128MB
    160 * 1024 * 1024,  # 160MB
    192 * 1024 * 1024,  # 192MB
    224 * 1024 * 1024,  # 224MB
    256 * 1024 * 1024,  # 256MB - attention scores/weights
    320 * 1024 * 1024,  # 320MB
    384 * 1024 * 1024,  # 384MB
    512 * 1024 * 1024,  # 512MB
    640 * 1024 * 1024,  # 640MB
    768 * 1024 * 1024,  # 768MB
    896 * 1024 * 1024,  # 896MB
    1024 * 1024 * 1024  # 1GB
]

# Direct allocation threshold
DIRECT_ALLOC_THRESHOLD = 1024 * 1024 * 1024  # >1GB

# Memory pressure thresholds (dynamic pool management)
# Keep caching until memory pressure is high
MEMORY_PRESSURE_THRESHOLD = float(os.getenv("GENESIS_MEMORY_PRESSURE_THRESHOLD", "0.9"))  # Start releasing at 90% usage
MEMORY_CRITICAL_THRESHOLD = float(os.getenv("GENESIS_MEMORY_CRITICAL_THRESHOLD", "0.95"))  # Aggressive release at 95%

# Pending cleanup frequency
PENDING_CHECK_INTERVAL = int(os.getenv("GENESIS_PENDING_CHECK_INTERVAL", "10"))


import bisect

def round_size(nbytes: int) -> Tuple[int, bool]:
    """
    Round requested size to nearest bucket size using binary search.

    Args:
        nbytes: Requested allocation size in bytes

    Returns:
        Tuple of (rounded_size, is_small_pool)
    """
    if nbytes <= SMALL_SIZES[-1]:
        # Small pool - use bisect for O(log N) lookup
        idx = bisect.bisect_left(SMALL_SIZES, nbytes)
        return SMALL_SIZES[idx], True
    elif nbytes <= DIRECT_ALLOC_THRESHOLD:
        # Large pool - use bisect for O(log N) lookup
        idx = bisect.bisect_left(LARGE_SIZES, nbytes)
        return LARGE_SIZES[idx], False
    else:
        # Direct allocation - no rounding
        return nbytes, False


class CudaCachingAllocator:
    """
    Lightweight caching allocator for CUDA memory.

    Optimized for stable training workloads where tensor sizes are consistent
    after warmup. Uses fixed-size buckets to minimize fragmentation.
    """

    def __init__(self, device: int = 0, default_stream: int = 0):
        """
        Initialize allocator for specific device (standard API).

        Args:
            device: CUDA device ID this allocator manages
            default_stream: Primary CUDA stream for training (usually 0)
        """
        # Device binding (immutable after creation)
        self._target_device = device

        # Memory pools
        self.small_bins: Dict[int, List[int]] = defaultdict(list)  # <1MB
        self.large_bins: Dict[int, List[int]] = defaultdict(list)  # 1-16MB
        self.pending: List[Tuple[int, int, int]] = []  # (ptr, size, event)

        # Configuration
        self.default_stream = default_stream
        self.memory_pressure_threshold = MEMORY_PRESSURE_THRESHOLD
        self.memory_critical_threshold = MEMORY_CRITICAL_THRESHOLD

        # Thread safety
        self.lock = threading.RLock()

        # Memory pressure tracking
        self._last_memory_check_time = 0
        self._cached_memory_pressure = 0.0
        self._memory_check_interval = 0.1  # Check at most every 0.1s

        # Statistics (optional)
        self.alloc_count = 0
        self.free_count = 0
        self.pending_checks = 0
        self.cuda_alloc_count = 0
        self.cache_hits = 0

        # CUDA initialization flag
        self._cuda_initialized = False
        self._cuda_context = None
        self._device_idx = None  # Actual device index after initialization

        # Reference counting for memory sharing (compatibility)
        self.ref_counts: Dict[int, int] = {}

    def _ensure_cuda_initialized(self):
        """Lazy CUDA initialization on first allocation (standard API)"""
        if not self._cuda_initialized:
            # Initialize CUDA driver
            result = cuda.cuInit(0)
            err = result[0] if isinstance(result, tuple) else result
            if err != cuda.CUresult.CUDA_SUCCESS:
                raise RuntimeError(f"Failed to initialize CUDA: {err}")

            # Use the target device assigned to this allocator
            device_idx = self._target_device
            self._device_idx = device_idx

            # Get device handle
            result = cuda.cuDeviceGet(device_idx)
            if result[0] != cuda.CUresult.CUDA_SUCCESS:
                raise RuntimeError(f"Failed to get CUDA device {device_idx}: {result[0]}")
            device = result[1]

            # Check if there's already a context for this device
            ctx_result = cuda.cuCtxGetCurrent()
            if ctx_result[0] == cuda.CUresult.CUDA_SUCCESS and ctx_result[1] is not None:
                # Context exists - verify it's for the right device
                dev_result = cuda.cuCtxGetDevice()
                if dev_result[0] == cuda.CUresult.CUDA_SUCCESS and dev_result[1] == device_idx:
                    # Perfect - reuse existing context
                    self._cuda_context = ctx_result[1]
                    self._cuda_initialized = True
                    return

            # No suitable context - create primary context for our device
            result = cuda.cuDevicePrimaryCtxRetain(device)
            if result[0] != cuda.CUresult.CUDA_SUCCESS:
                raise RuntimeError(f"Failed to retain primary context for device {device_idx}: {result[0]}")
            self._cuda_context = result[1]

            # Set as current context
            result = cuda.cuCtxSetCurrent(self._cuda_context)
            if isinstance(result, tuple):
                if result[0] != cuda.CUresult.CUDA_SUCCESS:
                    raise RuntimeError(f"Failed to set current context for device {device_idx}: {result[0]}")
            elif result != cuda.CUresult.CUDA_SUCCESS:
                raise RuntimeError(f"Failed to set current context for device {device_idx}: {result}")

            self._cuda_initialized = True

    def _cuda_alloc(self, size: int) -> int:
        """
        Allocate memory directly from CUDA using async allocation when available.

        Note: Caller must hold self.lock. OOM recovery is handled in allocate_memory().
        """
        self._ensure_cuda_initialized()

        # Ensure our context is current before allocation (PyTorch does cudaSetDevice before cudaMalloc)
        # Context may have been switched by operations on other devices
        ctx_set = cuda.cuCtxSetCurrent(self._cuda_context)
        if isinstance(ctx_set, tuple):
            if ctx_set[0] != cuda.CUresult.CUDA_SUCCESS:
                raise RuntimeError(f"Failed to set context for device {self._device_idx}: {ctx_set[0]}")
        elif ctx_set != cuda.CUresult.CUDA_SUCCESS:
            raise RuntimeError(f"Failed to set context for device {self._device_idx}: {ctx_set}")

        result = cuda.cuMemAlloc(size)

        # cuMemAlloc returns (CUresult, pointer)
        if result[0] == cuda.CUresult.CUDA_SUCCESS:
            if ENABLE_MEM_STATS:
                self.cuda_alloc_count += 1
            return int(result[1])

        # Return None on allocation failure (OOM recovery handled in allocate_memory)
        return None

    def _cuda_free(self, ptr: int):
        """Free memory directly to CUDA"""
        try:
            result = cuda.cuMemFree(ptr)
            # Ignore errors during cleanup (cuMemFree returns just CUresult)
            pass
        except Exception:
            # Ignore errors during cleanup
            pass

    def _create_event(self) -> int:
        """Create CUDA event for stream synchronization"""
        result = cuda.cuEventCreate(cuda.CUevent_flags.CU_EVENT_DISABLE_TIMING)
        if result[0] != cuda.CUresult.CUDA_SUCCESS:
            raise RuntimeError(f"Failed to create CUDA event: {result[0]}")
        return int(result[1])

    def _record_event(self, event: int, stream: int):
        """Record event on stream"""
        result = cuda.cuEventRecord(event, stream)
        if result != cuda.CUresult.CUDA_SUCCESS:
            raise RuntimeError(f"Failed to record CUDA event: {result}")

    def _event_query(self, event: int) -> bool:
        """Check if event has completed"""
        result = cuda.cuEventQuery(event)
        return result == cuda.CUresult.CUDA_SUCCESS

    def _destroy_event(self, event: int):
        """Destroy CUDA event"""
        try:
            result = cuda.cuEventDestroy(event)
            # Ignore result during cleanup
        except Exception:
            pass

    def _get_memory_pressure(self) -> float:
        """
        Get current GPU memory pressure (0.0 to 1.0).

        Uses cached value to avoid expensive cuMemGetInfo calls.

        Returns:
            Memory usage ratio (0.0 = empty, 1.0 = full)
        """
        import time
        current_time = time.time()

        # Use cached value if recent enough
        if current_time - self._last_memory_check_time < self._memory_check_interval:
            return self._cached_memory_pressure

        try:
            result = cuda.cuMemGetInfo()
            if result[0] == cuda.CUresult.CUDA_SUCCESS:
                free_bytes = result[1]
                total_bytes = result[2]
                used_bytes = total_bytes - free_bytes
                pressure = used_bytes / max(1, total_bytes)

                # Cache result
                self._cached_memory_pressure = pressure
                self._last_memory_check_time = current_time

                return pressure
        except Exception:
            pass

        # Default to moderate pressure if check fails
        return 0.5

    def _clear_cache_unsafe(self):
        """
        Clear all cached memory in pools WITHOUT acquiring lock.

        UNSAFE: Caller must hold self.lock before calling this.
        This is used internally when we're already inside a lock context.
        """
        # Free all blocks in small pool
        for bucket_size, bins in list(self.small_bins.items()):
            for ptr in bins:
                self._cuda_free(ptr)
            bins.clear()

        # Free all blocks in large pool
        for bucket_size, bins in list(self.large_bins.items()):
            for ptr in bins:
                self._cuda_free(ptr)
            bins.clear()

        # Note: Don't drain pending here - those are still in use

    def _clear_cache(self):
        """
        Clear all cached memory in pools (emergency cleanup on OOM).

        This is called when we hit OOM to free up memory for allocation retry.
        Similar to torch.cuda.empty_cache().
        """
        with self.lock:
            self._clear_cache_unsafe()

    def drain_pending(self):
        """
        Process pending deallocations.

        Checks which events have completed and returns their blocks to the pool.
        This is called periodically during allocations to avoid unbounded growth.
        """
        if not self.pending:
            return

        if ENABLE_MEM_STATS:
            self.pending_checks += 1

        # Scan pending list for completed events
        still_pending = []

        for ptr, size, event in self.pending:
            if self._event_query(event):
                # Event completed, can safely return to pool
                self._destroy_event(event)

                # Return to appropriate pool
                bucket, is_small = round_size(size)
                bins = self.small_bins if is_small else self.large_bins
                max_cached = self.max_small_cached if is_small else self.max_large_cached

                if len(bins[bucket]) < max_cached:
                    bins[bucket].append(ptr)
                else:
                    # Pool is full, free directly
                    self._cuda_free(ptr)
            else:
                # Still in use, keep in pending
                still_pending.append((ptr, size, event))

        self.pending = still_pending

    def allocate_memory(self, nbytes: int, stream: Optional[int] = None) -> int:
        """
        Allocate GPU memory with OOM recovery.

        Hot path optimized:
        1. Check pool (fast dict/list lookup)
        2. If miss, allocate directly from CUDA
        3. On OOM: trigger GC, clear cache, retry once

        Args:
            nbytes: Number of bytes to allocate
            stream: CUDA stream (optional, defaults to default_stream)

        Returns:
            GPU pointer (int)

        Raises:
            RuntimeError: If allocation fails even after OOM recovery
        """
        if stream is None:
            stream = self.default_stream

        with self.lock:
            if ENABLE_MEM_STATS:
                self.alloc_count += 1

            # Periodically drain pending (lightweight check)
            if self.alloc_count % PENDING_CHECK_INTERVAL == 0:
                self.drain_pending()

            # Calculate bucket
            bucket, is_small = round_size(nbytes)

            # Direct allocation for very large sizes
            if bucket > DIRECT_ALLOC_THRESHOLD:
                ptr = self._cuda_alloc(bucket)
                if ptr is not None:
                    return ptr
                # OOM - will handle below
            else:
                # Try to hit exact bucket (fast path)
                bins = self.small_bins if is_small else self.large_bins

                if bins[bucket]:
                    # Cache hit - fast path!
                    ptr = bins[bucket].pop()

                    if ENABLE_MEM_STATS:
                        self.cache_hits += 1

                    return ptr

                # Cache miss - allocate from CUDA
                ptr = self._cuda_alloc(bucket)
                if ptr is not None:
                    return ptr
                # OOM - will handle below

        # OOM recovery (outside lock to avoid deadlock)
        import gc
        gc.collect()
        self._clear_cache()  # This acquires its own lock

        # Retry allocation
        with self.lock:
            ptr = self._cuda_alloc(bucket)
            if ptr is not None:
                return ptr

        # Still failed after recovery
        raise RuntimeError(f"CUDA out of memory: failed to allocate {bucket} bytes even after cache cleanup")

    def free_memory(self, ptr: int, size: int, stream: Optional[int] = None):
        """
        Free GPU memory (return to pool or defer).

        Dynamic pool management based on memory pressure:
        - Low pressure (<90%): Always cache
        - High pressure (90-95%): Cache small blocks, free large blocks
        - Critical pressure (>95%): Free immediately

        Args:
            ptr: GPU pointer to free
            size: Original allocation size
            stream: CUDA stream (optional)
        """
        if stream is None:
            stream = self.default_stream

        with self.lock:
            if ENABLE_MEM_STATS:
                self.free_count += 1

            # Calculate bucket
            bucket, is_small = round_size(size)

            # Check memory pressure
            memory_pressure = self._get_memory_pressure()

            # Direct allocation (>1GB) - don't cache, always free
            # Large blocks are rarely reused and waste GPU memory if cached
            if bucket > DIRECT_ALLOC_THRESHOLD:
                self._cuda_free(ptr)
                return

            bins = self.small_bins if is_small else self.large_bins

            # Decide whether to cache based on memory pressure
            should_cache = True
            if memory_pressure >= self.memory_critical_threshold:
                # Critical pressure - free immediately
                should_cache = False
            elif memory_pressure >= self.memory_pressure_threshold:
                # High pressure - only cache small blocks
                should_cache = is_small

            # Check stream
            if stream == self.default_stream:
                # Same stream - can directly return to pool (fast path!)
                if should_cache:
                    bins[bucket].append(ptr)
                else:
                    # Memory pressure too high, free directly
                    self._cuda_free(ptr)
            else:
                # Different stream - need to wait for completion
                event = self._create_event()
                self._record_event(event, stream)
                self.pending.append((ptr, size, event))

    def increase_ref_count(self, ptr: int):
        """Increase reference count for shared memory (compatibility)"""
        with self.lock:
            self.ref_counts[ptr] = self.ref_counts.get(ptr, 1) + 1

    def decrease_ref_count(self, ptr: int, stream: Optional[int] = None) -> bool:
        """
        Decrease reference count for shared memory.

        Args:
            ptr: GPU pointer
            stream: CUDA stream

        Returns:
            True if this was a ref-counted block, False otherwise
        """
        with self.lock:
            if ptr in self.ref_counts:
                self.ref_counts[ptr] -= 1
                if self.ref_counts[ptr] <= 0:
                    del self.ref_counts[ptr]
                    # NOTE: We don't know the size here, so we can't free properly
                    # This is a compatibility shim - proper usage should call free_memory directly
                return True
            return False

    def warmup_pool(self, size_list: List[int]):
        """
        Pre-allocate memory for common sizes to avoid allocation during training.

        Args:
            size_list: List of sizes (in bytes) to pre-allocate
        """
        with self.lock:
            for size in size_list:
                bucket, is_small = round_size(size)
                bins = self.small_bins if is_small else self.large_bins

                # Allocate one block per size if not already in pool
                if not bins[bucket]:
                    ptr = self._cuda_alloc(bucket)
                    bins[bucket].append(ptr)

    def memory_stats(self) -> Dict:
        """
        Get memory statistics (optional, not on hot path).

        Returns:
            Dict with allocation statistics
        """
        with self.lock:
            # Count blocks in pools
            small_blocks = sum(len(blocks) for blocks in self.small_bins.values())
            large_blocks = sum(len(blocks) for blocks in self.large_bins.values())

            # Calculate cached memory
            small_cached_bytes = sum(
                size * len(blocks) for size, blocks in self.small_bins.items()
            )
            large_cached_bytes = sum(
                size * len(blocks) for size, blocks in self.large_bins.items()
            )

            # Get memory pressure
            memory_pressure = self._get_memory_pressure()

            stats = {
                'alloc_count': self.alloc_count,
                'free_count': self.free_count,
                'cache_hits': self.cache_hits,
                'cache_hit_rate': self.cache_hits / max(1, self.alloc_count),
                'cuda_alloc_count': self.cuda_alloc_count,
                'small_pool_blocks': small_blocks,
                'large_pool_blocks': large_blocks,
                'small_pool_bytes': small_cached_bytes,
                'large_pool_bytes': large_cached_bytes,
                'total_cached_bytes': small_cached_bytes + large_cached_bytes,
                'pending_deallocations': len(self.pending),
                'pending_checks': self.pending_checks,
                'memory_pressure': memory_pressure,
                'memory_pressure_threshold': self.memory_pressure_threshold,
                'memory_critical_threshold': self.memory_critical_threshold,
            }

            return stats

    def trigger_gc(self):
        """
        Manually trigger garbage collection (clear all cached memory).

        This is an emergency operation for memory pressure situations.
        Not recommended during training.
        """
        with self.lock:
            # Drain pending first
            self.drain_pending()

            # Free all small blocks
            for bucket, blocks in self.small_bins.items():
                for ptr in blocks:
                    self._cuda_free(ptr)
            self.small_bins.clear()

            # Free all large blocks
            for bucket, blocks in self.large_bins.items():
                for ptr in blocks:
                    self._cuda_free(ptr)
            self.large_bins.clear()

            # Clear ref counts
            self.ref_counts.clear()


# Per-device allocator instances (standard design pattern)
_device_allocators: Dict[int, CudaCachingAllocator] = {}
_allocator_lock = threading.Lock()


def get_memory_manager(device: Optional[int] = None) -> CudaCachingAllocator:
    """
    Get or create memory manager for specific device (standard API).

    Each CUDA device has its own allocator instance with its own context.
    This implements efficient caching allocation for CUDA memory.

    Args:
        device: CUDA device ID (None = use current device)

    Returns:
        CudaCachingAllocator instance for the specified device
    """
    global _device_allocators

    # Get current device if not specified
    if device is None:
        try:
            ctx_result = cuda.cuCtxGetCurrent()
            if ctx_result[0] == cuda.CUresult.CUDA_SUCCESS and ctx_result[1] is not None:
                dev_result = cuda.cuCtxGetDevice()
                if dev_result[0] == cuda.CUresult.CUDA_SUCCESS:
                    # CRITICAL: convert CUdevice to int for use as dict key
                    device = int(dev_result[1])
                else:
                    device = 0  # Fallback to device 0
            else:
                device = 0  # No context, use device 0
        except Exception:
            device = 0

    # Get or create allocator for this device
    if device not in _device_allocators:
        with _allocator_lock:
            if device not in _device_allocators:
                _device_allocators[device] = CudaCachingAllocator(device=device)

    return _device_allocators[device]


# Public API - matches old interface for compatibility
def allocate_memory(nbytes: int, stream: Optional[int] = None) -> int:
    """
    Allocate GPU memory.

    Args:
        nbytes: Number of bytes to allocate
        stream: CUDA stream (optional)

    Returns:
        GPU pointer (int)
    """
    manager = get_memory_manager()
    return manager.allocate_memory(nbytes, stream)


def free_memory(ptr: int, nbytes: int, stream: Optional[int] = None):
    """
    Free GPU memory.

    Args:
        ptr: GPU pointer
        nbytes: Original allocation size
        stream: CUDA stream (optional)
    """
    manager = get_memory_manager()
    manager.free_memory(ptr, nbytes, stream)


def increase_ref_count(ptr: int):
    """Increase reference count for shared memory"""
    manager = get_memory_manager()
    manager.increase_ref_count(ptr)


def decrease_ref_count(ptr: int, stream: Optional[int] = None) -> bool:
    """
    Decrease reference count for shared memory.

    Returns:
        True if this was a ref-counted block
    """
    manager = get_memory_manager()
    return manager.decrease_ref_count(ptr, stream)


def memory_stats() -> Dict:
    """Get memory statistics"""
    manager = get_memory_manager()
    return manager.memory_stats()


def trigger_gc():
    """Manually trigger garbage collection"""
    manager = get_memory_manager()
    manager.trigger_gc()


def get_memory_info() -> Dict:
    """
    Get GPU memory information.

    Returns:
        Dict with GPU memory statistics (total, used, free, usage_ratio)
    """
    try:
        result = cuda.cuMemGetInfo()
        # cuMemGetInfo returns (CUresult, free_bytes, total_bytes)
        if result[0] != cuda.CUresult.CUDA_SUCCESS:
            raise RuntimeError(f"Failed to get memory info: {result[0]}")

        free_bytes = result[1]
        total_bytes = result[2]
        used_bytes = total_bytes - free_bytes
        usage_ratio = used_bytes / max(1, total_bytes)

        return {
            'gpu_memory': {
                'total_mb': total_bytes / (1024 * 1024),
                'used_mb': used_bytes / (1024 * 1024),
                'free_mb': free_bytes / (1024 * 1024),
                'usage_ratio': f'{usage_ratio:.3f}'
            },
            'system_memory': {
                'total_mb': 'N/A',
                'used_mb': 'N/A',
                'available_mb': 'N/A'
            }
        }
    except Exception as e:
        return {
            'gpu_memory': {'error': str(e)},
            'system_memory': {'error': 'Not available'}
        }


def defragment_memory() -> Optional[Dict]:
    """
    Defragment memory pools (compatibility function).

    In the lightweight allocator, defragmentation is implicit:
    - Fixed-size buckets prevent fragmentation
    - GC clears all pools if needed

    Returns:
        Dict with defragmentation stats (mostly no-op for this allocator)
    """
    manager = get_memory_manager()

    # Drain pending deallocations
    manager.drain_pending()

    return {
        'defragmented': False,
        'reason': 'Lightweight allocator uses fixed-size buckets (no fragmentation)',
        'small_pool_blocks': sum(len(blocks) for blocks in manager.small_bins.values()),
        'large_pool_blocks': sum(len(blocks) for blocks in manager.large_bins.values()),
    }


def empty_cache():
    """
    Release all cached memory back to GPU.

    This function clears all memory blocks that are cached in the allocator pools
    but not currently in use. Similar to torch.cuda.empty_cache().

    This is useful when:
    - You want to free memory for other applications
    - You're between training iterations and want to minimize memory footprint
    - You're debugging memory usage

    Note: This does NOT free memory that is currently allocated to tensors.
    Only cached/free blocks are released.
    """
    manager = get_memory_manager()
    manager._clear_cache()


def get_memory_stats() -> Dict:
    """
    Get detailed memory allocator statistics.

    Returns:
        Dict with memory statistics
    """
    manager = get_memory_manager()

    small_cached = sum(len(blocks) for blocks in manager.small_bins.values())
    large_cached = sum(len(blocks) for blocks in manager.large_bins.values())

    stats = {
        'alloc_count': manager.alloc_count,
        'free_count': manager.free_count,
        'cuda_alloc_count': manager.cuda_alloc_count,
        'cache_hits': manager.cache_hits,
        'small_pool_cached_blocks': small_cached,
        'large_pool_cached_blocks': large_cached,
        'pending_blocks': len(manager.pending),
    }

    # Calculate efficiency metrics
    if manager.alloc_count > 0:
        stats['cache_hit_rate'] = manager.cache_hits / manager.alloc_count
    else:
        stats['cache_hit_rate'] = 0.0

    return stats


def analyze_memory_fragmentation() -> Dict:
    """
    Analyze memory fragmentation (compatibility function).

    Returns:
        Dict with fragmentation analysis
    """
    manager = get_memory_manager()

    small_blocks = sum(len(blocks) for blocks in manager.small_bins.values())
    large_blocks = sum(len(blocks) for blocks in manager.large_bins.values())

    return {
        'fragmentation_ratio': 0.0,  # Fixed-size buckets prevent fragmentation
        'small_pool_blocks': small_blocks,
        'large_pool_blocks': large_blocks,
        'pending_blocks': len(manager.pending),
        'recommendation': 'No action needed - using fixed-size bucket allocator'
    }


def get_fragmentation_stats() -> Dict:
    """
    Get fragmentation statistics (compatibility function).

    Returns:
        Dict with fragmentation stats
    """
    return analyze_memory_fragmentation()


def set_memory_config(config: Dict):
    """
    Set memory allocator configuration (compatibility function).

    Args:
        config: Configuration dict (currently ignored for lightweight allocator)
    """
    # Lightweight allocator uses environment variables for configuration
    # This is a no-op for compatibility
    pass
