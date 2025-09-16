"""
CUDA utilities and memory management for Genesis.

This module provides PyTorch-compatible interfaces for CUDA device management
and memory operations, organized similarly to torch.cuda.
"""

import os
from typing import Dict, Optional, List, Any

import threading

try:
    from cuda import cuda
    from cuda.bindings import driver
except ImportError:
    from cuda.bindings import driver as cuda
    from cuda.bindings import driver

# Import device class for type hints
from ..device import Device

# Import memory management functions
from ..backends.cuda_memory import (
    get_memory_manager, trigger_gc, get_memory_info,
    defragment_memory, analyze_memory_fragmentation,
    get_fragmentation_stats, set_memory_config
)

# CUDA initialization state
_cuda_initialized = False
_cuda_available = False
_init_lock = threading.Lock()


def _ensure_cuda_initialized():
    """Ensure CUDA is initialized. Thread-safe and lazy."""
    global _cuda_initialized, _cuda_available

    if _cuda_initialized:
        return _cuda_available

    with _init_lock:
        # Double-check after acquiring lock
        if _cuda_initialized:
            return _cuda_available

        try:
            result = cuda.cuInit(0)
            _cuda_available = (result[0] == cuda.CUresult.CUDA_SUCCESS)
        except Exception:
            _cuda_available = False
        finally:
            _cuda_initialized = True

    return _cuda_available


# Device management functions
def device_count() -> int:
    """Get the number of available CUDA devices."""
    if not _ensure_cuda_initialized():
        return 0
    device_count = cuda.cuDeviceGetCount()
    return device_count[1] if device_count[0] == cuda.CUresult.CUDA_SUCCESS else 0


def current_device() -> int:
    """Get the current CUDA device index."""
    device = cuda.cuCtxGetDevice()
    return device[1] if device[0] == cuda.CUresult.CUDA_SUCCESS else 0


def set_device(device: int) -> None:
    """Set the current CUDA device."""
    # Initialize CUDA if not already done
    cuda.cuInit(0)
    
    # Get device handle
    dev_result = cuda.cuDeviceGet(device)
    if dev_result[0] != cuda.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"Failed to get CUDA device {device}")
    dev = dev_result[1]
    
    # Get primary context and set it as current
    ctx_result = cuda.cuDevicePrimaryCtxRetain(dev)
    if ctx_result[0] != cuda.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"Failed to retain primary context for device {device}")
    ctx = ctx_result[1]
    
    set_result = cuda.cuCtxSetCurrent(ctx)
    if set_result != cuda.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"Failed to set CUDA device {device}")


def get_device_name(device: Optional[int] = None) -> str:
    """Get the name of a CUDA device."""
    if device is None:
        device = current_device()
    
    name = cuda.cuDeviceGetName(128, device)
    return name[1].decode('utf-8') if name[0] == cuda.CUresult.CUDA_SUCCESS else f"CUDA Device {device}"


def is_available() -> bool:
    """Check if CUDA is available."""
    return device_count() > 0


def synchronize(device: Optional[int] = None) -> None:
    """
    Synchronize CUDA operations.
    
    This ensures all CUDA operations are complete before returning.
    """
    if device is not None:
        # Synchronize specific device
        current = current_device()
        if current != device:
            # Would need to switch context, for now just sync current
            pass
    # Use cuCtxSynchronize instead of cuDeviceSynchronize
    result = cuda.cuCtxSynchronize()
    if result[0] != cuda.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"CUDA synchronization failed: {result[0]}")


# Memory management functions
def empty_cache() -> Dict[str, Any]:
    """
    Clear the CUDA memory cache.
    
    Similar to torch.cuda.empty_cache(), this function frees all cached memory
    that is not currently being used by any tensors.
    
    Returns:
        Dict containing information about the cleanup operation:
        - freed_memory_mb: Amount of memory freed in MB
        - cleared_pools: Number of memory pools cleared
        - defragmentation_performed: Whether defragmentation was performed
        - stats: Detailed statistics about the cleanup
    """
    _ensure_config_applied()
    manager = get_memory_manager()
    
    # Get stats before cleanup
    stats_before = manager.get_stats()
    pool_blocks_before = stats_before['ref_pool']['pool_blocks']
    pool_size_before = stats_before['ref_pool']['pool_size_mb']
    
    # Perform aggressive cleanup
    cleanup_stats = trigger_gc()
    
    # Check if defragmentation would be beneficial
    frag_analysis = analyze_memory_fragmentation()
    defrag_performed = False
    defrag_stats = None
    
    if frag_analysis['needs_defrag']:
        defrag_stats = defragment_memory()
        defrag_performed = defrag_stats is not None
    
    # Get stats after cleanup
    stats_after = manager.get_stats()
    pool_blocks_after = stats_after['ref_pool']['pool_blocks']
    pool_size_after = stats_after['ref_pool']['pool_size_mb']
    
    # Calculate freed memory
    freed_memory_mb = pool_size_before - pool_size_after
    cleared_pools = pool_blocks_before - pool_blocks_after
    
    result = {
        'freed_memory_mb': freed_memory_mb,
        'cleared_pools': cleared_pools,
        'defragmentation_performed': defrag_performed,
        'stats': {
            'cleanup': cleanup_stats,
            'defragmentation': defrag_stats,
            'before': stats_before,
            'after': stats_after
        }
    }
    
    return result


def memory_stats() -> Dict[str, Any]:
    """
    Get detailed CUDA memory statistics.
    
    Similar to torch.cuda.memory_stats(), returns comprehensive information
    about CUDA memory usage, cache performance, and allocation patterns.
    
    Returns:
        Dict containing detailed memory statistics
    """
    manager = get_memory_manager()
    stats = manager.get_stats()
    
    # Get additional memory info
    memory_info = get_memory_info()
    frag_stats = get_fragmentation_stats()
    
    # Enhance stats with additional information
    enhanced_stats = {
        'allocation': {
            'total_allocations': stats['total_alloc_count'],
            'active_blocks': stats['active_blocks'],
            'current_active_memory_mb': stats.get('current_memory_mb', 0)
        },
        'cache': {
            'pool_hits': stats['ref_pool']['pool_hits'],
            'pool_misses': stats['ref_pool']['pool_misses'],
            'hit_rate': stats['ref_pool']['hit_rate'],
            'pool_blocks': stats['ref_pool']['pool_blocks'],
            'pool_size_mb': stats['ref_pool']['pool_size_mb'],
            'ref_count_saves': stats['ref_pool']['ref_count_saves']
        },
        'pressure': {
            'memory_pressure': stats['ref_pool']['memory_pressure'],
            'critical_pressure': stats['ref_pool']['critical_pressure'],
            'pressure_cleanups': stats['ref_pool']['pressure_cleanups'],
            'critical_cleanups': stats['ref_pool']['critical_cleanups'],
            'pressure_threshold': stats['ref_pool']['pressure_monitor']['pressure_threshold'],
            'critical_threshold': stats['ref_pool']['pressure_monitor']['critical_threshold']
        },
        'fragmentation': {
            'overall_fragmentation': float(frag_stats.get('current_fragmentation', {}).get('overall_fragmentation', 0.0)),
            'pool_fragmentation': float(frag_stats.get('current_fragmentation', {}).get('pool_fragmentation', {}).get('fragmentation_ratio', 0.0)),
            'defrag_operations': int(frag_stats.get('defrag_history', {}).get('defrag_operations', 0)),
            'fragmentation_threshold': float(frag_stats.get('defrag_history', {}).get('fragmentation_threshold', 0.3)),
            'trend': str(frag_stats.get('defrag_history', {}).get('trend', 'stable')),
            'recommendation': str(frag_stats.get('defrag_history', {}).get('recommendation', 'monitor'))
        },
        'memory_info': memory_info
    }
    
    return enhanced_stats


def memory_summary() -> str:
    """
    Get a human-readable summary of CUDA memory usage.
    
    Similar to torch.cuda.memory_summary(), returns a formatted string
    with key memory statistics and recommendations.
    
    Returns:
        Formatted string with memory usage summary
    """
    stats = memory_stats()
    
    # Extract key metrics
    alloc = stats['allocation']
    cache = stats['cache']
    pressure = stats['pressure']
    frag = stats['fragmentation']
    mem_info = stats['memory_info']
    
    summary_lines = [
        "=" * 60,
        "Genesis CUDA Memory Summary",
        "=" * 60,
        "",
        "ALLOCATION:",
        f"  Total allocations: {alloc['total_allocations']:,}",
        f"  Active blocks: {alloc['active_blocks']:,}",
        f"  Current memory: {alloc['current_active_memory_mb']:.2f} MB",
        "",
        "CACHE PERFORMANCE:",
        f"  Pool hits: {cache['pool_hits']:,}",
        f"  Pool misses: {cache['pool_misses']:,}",
        f"  Hit rate: {cache['hit_rate']}",
        f"  Pool blocks: {cache['pool_blocks']:,}",
        f"  Pool size: {cache['pool_size_mb']:.2f} MB",
        f"  Reference count saves: {cache['ref_count_saves']:,}",
        "",
        "MEMORY PRESSURE:",
        f"  Current pressure: {'Yes' if pressure['memory_pressure'] else 'No'}",
        f"  Critical pressure: {'Yes' if pressure['critical_pressure'] else 'No'}",
        f"  Pressure cleanups: {pressure['pressure_cleanups']:,}",
        f"  Critical cleanups: {pressure['critical_cleanups']:,}",
        f"  Pressure threshold: {pressure['pressure_threshold']}",
        "",
        "FRAGMENTATION:",
        f"  Overall fragmentation: {frag['overall_fragmentation']:.3f}",
        f"  Pool fragmentation: {frag['pool_fragmentation']:.3f}",
        f"  Defrag operations: {frag['defrag_operations']:,}",
        f"  Fragmentation trend: {frag['trend']}",
        f"  Recommendation: {frag['recommendation']}",
    ]
    
    # Add GPU memory info if available
    if 'gpu_memory' in mem_info and 'error' not in mem_info['gpu_memory']:
        gpu_mem = mem_info['gpu_memory']
        summary_lines.extend([
            "",
            "GPU MEMORY:",
            f"  Total: {gpu_mem['total_mb']:.1f} MB",
            f"  Used: {gpu_mem['used_mb']:.1f} MB",
            f"  Free: {gpu_mem['free_mb']:.1f} MB",
            f"  Usage: {gpu_mem['usage_ratio']:.1%}",
        ])
    
    # Add recommendations
    summary_lines.extend([
        "",
        "RECOMMENDATIONS:",
    ])
    
    if pressure['memory_pressure']:
        summary_lines.append("  ⚠️  Memory pressure detected - consider calling empty_cache()")
    
    if frag['overall_fragmentation'] > 0.3:
        summary_lines.append("  🔧 High fragmentation - defragmentation recommended")
    
    if cache['hit_rate'].rstrip('%') and float(cache['hit_rate'].rstrip('%')) < 50:
        summary_lines.append("  📈 Low cache hit rate - memory access patterns may need optimization")
    
    if not any(line.startswith("  ") for line in summary_lines[-10:]):
        summary_lines.append("  ✅ Memory usage is optimal")
    
    summary_lines.extend([
        "",
        "=" * 60
    ])
    
    return "\n".join(summary_lines)


def set_memory_fraction(fraction: float) -> None:
    """
    Set the memory fraction for CUDA memory allocation.
    
    Args:
        fraction: Fraction of GPU memory to use (0.0 to 1.0)
    """
    if not 0.0 <= fraction <= 1.0:
        raise ValueError("Memory fraction must be between 0.0 and 1.0")
    
    # Convert to pool size configuration
    mem_info = get_memory_info()
    if 'gpu_memory' in mem_info and 'error' not in mem_info['gpu_memory']:
        total_mb = mem_info['gpu_memory']['total_mb']
        max_pool_mb = int(total_mb * fraction)
        set_memory_config(max_pool_size_mb=max_pool_mb)
    else:
        # Fallback: set a reasonable default based on fraction
        default_pool_mb = int(2048 * fraction)  # Assume 2GB default
        set_memory_config(max_pool_size_mb=default_pool_mb)


def reset_max_memory_allocated() -> None:
    """Reset the maximum memory allocation tracker."""
    manager = get_memory_manager()
    if hasattr(manager, 'max_allocated_bytes'):
        manager.max_allocated_bytes = 0


def reset_max_memory_cached() -> None:
    """Reset the maximum memory cached tracker."""
    manager = get_memory_manager()
    if hasattr(manager.ref_pool, 'max_pool_size_reached'):
        manager.ref_pool.max_pool_size_reached = 0


# Configuration parsing for GENESIS_CUDA_ALLOC_CONF
def _parse_memory_config() -> Dict[str, Any]:
    """
    Parse GENESIS_CUDA_ALLOC_CONF environment variable.
    
    Format: "key1=value1,key2=value2,..."
    
    Supported keys:
    - gc_threshold: Garbage collection threshold (0.0-1.0)
    - max_pool_size_mb: Maximum pool size in MB
    - max_split_size_mb: Maximum split size in MB  
    - fragmentation_threshold: Fragmentation threshold (0.0-1.0)
    """
    config = {}
    
    env_config = os.environ.get('GENESIS_CUDA_ALLOC_CONF', '')
    if not env_config:
        return config
    
    try:
        for pair in env_config.split(','):
            if '=' in pair:
                key, value = pair.split('=', 1)
                key = key.strip()
                value = value.strip()
                
                # Parse different value types
                if key in ['gc_threshold', 'fragmentation_threshold']:
                    config[key] = float(value)
                elif key in ['max_pool_size_mb', 'max_split_size_mb']:
                    config[key] = int(value)
                else:
                    config[key] = value
                    
    except (ValueError, AttributeError) as e:
        print(f"Warning: Failed to parse GENESIS_CUDA_ALLOC_CONF: {e}")
    
    return config


def apply_memory_config() -> None:
    """Apply memory configuration from environment variables."""
    config = _parse_memory_config()
    if config:
        set_memory_config(**config)
        print(f"Applied GENESIS_CUDA_ALLOC_CONF: {config}")


# Global flag to track if config has been applied
_config_applied = False

def _ensure_config_applied():
    """Ensure memory configuration is applied exactly once."""
    global _config_applied
    if not _config_applied:
        apply_memory_config()
        _config_applied = True


# Note: Device creation is now done via genesis.device("cuda") or genesis.device("cuda:0")
# This follows PyTorch's pattern: torch.device('cuda') instead of torch.cuda()


# Note: Configuration will be applied lazily on first use to avoid 
# import-time CUDA initialization issues during batch testing