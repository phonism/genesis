"""
High-performance CUDA memory manager
"""

try:
    from cuda import cuda
    from cuda.bindings import driver
except ImportError:
    from cuda.bindings import driver as cuda
    from cuda.bindings import driver

import threading
import os
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass
from collections import defaultdict
import time
from .memory_stats_collector import get_stats_collector


def _ok(ret):
    """
    Helper for unified CUDA Driver API return handling
    """
    code = ret[0]
    if code != cuda.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"CUDA driver error: {code}")
    return ret[1] if len(ret) > 1 else None

@dataclass
class Block:
    """
    block
    """
    ptr: int          # GPU pointer
    size: int         # block size  
    is_free: bool     # is free
    segment_id: int   # segment id


@dataclass
class RefCountedBlock:
    """Reference-counted memory block for efficient pooling"""
    ptr: int
    size: int
    is_free: bool
    segment_id: int
    ref_count: int = 1
    last_used_time: float = 0.0
    stream_id: Optional[int] = None
    
class Segment:
    """
    segment
    """
    
    def __init__(self, segment_id: int, size: int):
        self.segment_id = segment_id
        self.total_size = size
        
        # allocate entire segment from CUDA
        try:
            self.base_ptr = _ok(cuda.cuMemAlloc(size))
        except RuntimeError as e:
            # Fast fail on OOM with clear error message
            raise RuntimeError(f"CUDA OOM: Failed to allocate large segment of {size} bytes "
                             f"({size // (1024*1024)} MB). Consider reducing batch size or model size.") from e
        
        # NOTE: Skip memory initialization for performance
        # CUDA memory is generally clean, and clearing 1GB takes ~3 seconds
        # If precision issues occur, we can implement on-demand clearing
        # _ok(cuda.cuMemsetD8(self.base_ptr, 0, size))
        
        # initialize as a single large free block
        self.blocks: List[Block] = [
            Block(ptr=self.base_ptr, size=size, is_free=True, segment_id=segment_id)
        ]
        
        # quick lookup of free blocks (by size)
        self.free_blocks_by_size: Dict[int, List[Block]] = {size: [self.blocks[0]]}
        self.used_bytes = 0
        
    def allocate(self, size: int) -> Optional[int]:
        """Allocate a block of the specified size from the segment"""
        # Best-fit: find the smallest free block that is large enough
        best_block = None
        best_size = float('inf')
        
        for block_size, blocks in self.free_blocks_by_size.items():
            if block_size >= size and block_size < best_size:
                if blocks:  # ensure there are available blocks
                    best_size = block_size
                    best_block = blocks[0]
        
        if not best_block:
            return None  # no available block
        
        # remove from free list
        self.free_blocks_by_size[best_block.size].remove(best_block)
        if not self.free_blocks_by_size[best_block.size]:
            del self.free_blocks_by_size[best_block.size]
        
        # if the block is larger than needed, split it
        if best_block.size > size:
            # create remaining free block
            remaining_size = best_block.size - size
            # handle pointer arithmetic (CUDA pointers may be objects rather than integers)
            ptr_val = int(best_block.ptr) if hasattr(best_block.ptr, '__int__') else best_block.ptr
            remaining_block = Block(
                ptr=ptr_val + size,
                size=remaining_size,
                is_free=True,
                segment_id=self.segment_id
            )
            
            # insert into blocks list (keep address order)
            idx = self.blocks.index(best_block)
            self.blocks.insert(idx + 1, remaining_block)
            
            # add to free block index
            if remaining_size not in self.free_blocks_by_size:
                self.free_blocks_by_size[remaining_size] = []
            self.free_blocks_by_size[remaining_size].append(remaining_block)
            
            # adjust original block size
            best_block.size = size
        
        # mark as used
        best_block.is_free = False
        self.used_bytes += best_block.size
        
        return best_block.ptr
    
    def free(self, ptr: int) -> bool:
        """Free the block at the specified pointer"""
        # find the corresponding block
        block = None
        for b in self.blocks:
            if b.ptr == ptr and not b.is_free:
                block = b
                break
        
        if not block:
            return False  # not found
        
        # mark as free
        block.is_free = True
        self.used_bytes -= block.size
        
        # try to merge with adjacent free blocks
        idx = self.blocks.index(block)
        
        # merge with previous block
        if idx > 0 and self.blocks[idx - 1].is_free:
            prev_block = self.blocks[idx - 1]
            # remove from free list
            self.free_blocks_by_size[prev_block.size].remove(prev_block)
            if not self.free_blocks_by_size[prev_block.size]:
                del self.free_blocks_by_size[prev_block.size]
            
            # merge
            prev_block.size += block.size
            self.blocks.remove(block)
            block = prev_block
            idx -= 1
        
        # merge with next block
        if idx < len(self.blocks) - 1 and self.blocks[idx + 1].is_free:
            next_block = self.blocks[idx + 1]
            # remove from free list
            self.free_blocks_by_size[next_block.size].remove(next_block)
            if not self.free_blocks_by_size[next_block.size]:
                del self.free_blocks_by_size[next_block.size]
            
            # merge
            block.size += next_block.size
            self.blocks.remove(next_block)
        
        # add to free block index
        if block.size not in self.free_blocks_by_size:
            self.free_blocks_by_size[block.size] = []
        self.free_blocks_by_size[block.size].append(block)
        
        return True
    
    def __del__(self):
        """Release the entire segment"""
        if hasattr(self, 'base_ptr'):
            try:
                cuda.cuMemFree(self.base_ptr)
            except:
                pass


class FragmentationDetector:
    """Detect and analyze memory fragmentation patterns"""
    
    def __init__(self):
        self.fragmentation_threshold = 0.3  # 30% fragmentation triggers defrag
        self.min_defrag_size = 1024 * 1024  # 1MB minimum for defrag consideration
        self.fragmentation_history = []
        self.defrag_operations = 0
        
    def analyze_fragmentation(self, memory_pool: dict, segments: list) -> Dict:
        """Analyze current memory fragmentation levels"""
        stats = {
            'pool_fragmentation': self._analyze_pool_fragmentation(memory_pool),
            'segment_fragmentation': self._analyze_segment_fragmentation(segments),
            'overall_fragmentation': 0.0,
            'needs_defrag': False,
            'recommended_action': 'none'
        }
        
        # Calculate overall fragmentation
        pool_frag = stats['pool_fragmentation']['fragmentation_ratio']
        segment_frag = stats['segment_fragmentation']['average_fragmentation']
        stats['overall_fragmentation'] = (pool_frag + segment_frag) / 2
        
        # Determine if defragmentation is needed
        if stats['overall_fragmentation'] > self.fragmentation_threshold:
            stats['needs_defrag'] = True
            stats['recommended_action'] = 'defragment'
        elif pool_frag > 0.5:  # High pool fragmentation
            stats['recommended_action'] = 'compact_pool'
        elif segment_frag > 0.4:  # High segment fragmentation
            stats['recommended_action'] = 'merge_segments'
        
        # Record in history
        self.fragmentation_history.append({
            'timestamp': time.time(),
            'overall_fragmentation': stats['overall_fragmentation'],
            'pool_fragmentation': pool_frag,
            'segment_fragmentation': segment_frag
        })
        
        # Keep only recent history
        if len(self.fragmentation_history) > 100:
            self.fragmentation_history = self.fragmentation_history[-50:]
        
        return stats
    
    def _analyze_pool_fragmentation(self, memory_pool: dict) -> Dict:
        """Analyze fragmentation in memory pool buckets"""
        if not memory_pool:
            return {
                'fragmentation_ratio': 0.0,
                'wasted_space': 0,
                'bucket_distribution': {},
                'largest_contiguous': 0
            }
        
        bucket_sizes = list(memory_pool.keys())
        bucket_counts = {size: len(blocks) for size, blocks in memory_pool.items()}
        
        # Calculate wasted space due to bucket size granularity
        total_wasted = 0
        total_allocated = 0
        largest_bucket = 0
        
        for bucket_size, blocks in memory_pool.items():
            block_count = len(blocks)
            total_allocated += bucket_size * block_count
            largest_bucket = max(largest_bucket, bucket_size)
            
            # Estimate waste: assume average allocation is 75% of bucket size
            avg_usage = bucket_size * 0.75
            total_wasted += (bucket_size - avg_usage) * block_count
        
        fragmentation_ratio = total_wasted / max(1, total_allocated)
        
        return {
            'fragmentation_ratio': fragmentation_ratio,
            'wasted_space': total_wasted,
            'bucket_distribution': bucket_counts,
            'largest_contiguous': largest_bucket,
            'total_buckets': len(bucket_sizes)
        }
    
    def _analyze_segment_fragmentation(self, segments: list) -> Dict:
        """Analyze fragmentation in memory segments"""
        if not segments:
            return {
                'average_fragmentation': 0.0,
                'worst_fragmentation': 0.0,
                'fragmented_segments': 0,
                'total_segments': 0
            }
        
        segment_frags = []
        fragmented_count = 0
        
        for segment in segments:
            if hasattr(segment, 'blocks') and segment.blocks:
                frag_ratio = self._calculate_segment_fragmentation(segment)
                segment_frags.append(frag_ratio)
                if frag_ratio > 0.3:  # Consider >30% as fragmented
                    fragmented_count += 1
        
        if not segment_frags:
            return {
                'average_fragmentation': 0.0,
                'worst_fragmentation': 0.0,
                'fragmented_segments': 0,
                'total_segments': len(segments)
            }
        
        return {
            'average_fragmentation': sum(segment_frags) / len(segment_frags),
            'worst_fragmentation': max(segment_frags),
            'fragmented_segments': fragmented_count,
            'total_segments': len(segments)
        }
    
    def _calculate_segment_fragmentation(self, segment) -> float:
        """Calculate fragmentation ratio for a single segment"""
        if not hasattr(segment, 'blocks') or not segment.blocks:
            return 0.0
        
        # Count free blocks and gaps
        free_blocks = [b for b in segment.blocks if b.is_free]
        if len(free_blocks) <= 1:
            return 0.0  # No fragmentation with 0 or 1 free block
        
        # Calculate fragmentation based on number of free blocks vs total free space
        total_free_space = sum(b.size for b in free_blocks)
        if total_free_space == 0:
            return 0.0
        
        # More free blocks = more fragmentation
        # Ideal case: 1 large free block
        # Worst case: many small free blocks
        fragmentation = (len(free_blocks) - 1) / len(free_blocks)
        
        return min(fragmentation, 1.0)
    
    def get_defrag_stats(self) -> Dict:
        """Get defragmentation statistics"""
        recent_history = self.fragmentation_history[-10:] if self.fragmentation_history else []
        
        return {
            'defrag_operations': self.defrag_operations,
            'fragmentation_threshold': self.fragmentation_threshold,
            'recent_fragmentation': [h['overall_fragmentation'] for h in recent_history],
            'trend': self._calculate_fragmentation_trend(),
            'recommendation': self._get_defrag_recommendation()
        }
    
    def _calculate_fragmentation_trend(self) -> str:
        """Calculate fragmentation trend over recent history"""
        if len(self.fragmentation_history) < 3:
            return 'insufficient_data'
        
        recent = self.fragmentation_history[-5:]
        if len(recent) < 2:
            return 'stable'
        
        # Simple trend calculation
        first_half = sum(h['overall_fragmentation'] for h in recent[:len(recent)//2])
        second_half = sum(h['overall_fragmentation'] for h in recent[len(recent)//2:])
        
        first_avg = first_half / max(1, len(recent)//2)
        second_avg = second_half / max(1, len(recent) - len(recent)//2)
        
        if second_avg > first_avg * 1.1:
            return 'increasing'
        elif second_avg < first_avg * 0.9:
            return 'decreasing'
        else:
            return 'stable'
    
    def _get_defrag_recommendation(self) -> str:
        """Get recommendation for defragmentation"""
        if not self.fragmentation_history:
            return 'monitor'
        
        current = self.fragmentation_history[-1]['overall_fragmentation']
        trend = self._calculate_fragmentation_trend()
        
        if current > 0.5:
            return 'urgent_defrag'
        elif current > self.fragmentation_threshold:
            if trend == 'increasing':
                return 'schedule_defrag'
            else:
                return 'monitor_closely'
        else:
            return 'monitor'
    
    def get_defrag_history(self) -> Dict:
        """Get complete defragmentation history and statistics"""
        recent_history = self.fragmentation_history[-20:] if self.fragmentation_history else []
        
        return {
            'defrag_operations_count': self.defrag_operations,
            'fragmentation_threshold': self.fragmentation_threshold,
            'min_defrag_size_mb': self.min_defrag_size / (1024 * 1024),
            'recent_fragmentation': [
                {
                    'timestamp': h['timestamp'],
                    'overall_fragmentation': h['overall_fragmentation'],
                    'pool_fragmentation': h['pool_fragmentation'],
                    'segment_fragmentation': h['segment_fragmentation']
                }
                for h in recent_history
            ],
            'fragmentation_trend': self._calculate_fragmentation_trend(),
            'recommendation': self._get_defrag_recommendation(),
            'history_length': len(self.fragmentation_history)
        }
    
    def defragment(self, memory_pool: dict, segments: list) -> Optional[Dict]:
        """Perform memory defragmentation on pool and segments"""
        if not memory_pool:
            return None
            
        # Analyze current fragmentation
        frag_stats = self.analyze_fragmentation(memory_pool, segments)
        
        # Only defragment if needed
        if not frag_stats['needs_defrag'] and frag_stats['overall_fragmentation'] < 0.1:
            return None
            
        buckets_before = len([bucket for bucket, blocks in memory_pool.items() if blocks])
        blocks_before = sum(len(blocks) for blocks in memory_pool.values())
        
        # Consolidate small buckets into larger ones
        blocks_consolidated = 0
        memory_freed = 0
        
        # Find small buckets that can be consolidated
        small_buckets = [(size, blocks) for size, blocks in memory_pool.items() 
                        if blocks and size < self.min_defrag_size and len(blocks) > 1]
        
        for bucket_size, blocks in small_buckets:
            if len(blocks) > 1:
                # Keep one block, free others to be reallocated as larger blocks
                blocks_to_consolidate = blocks[1:]  # Keep first block
                for block in blocks_to_consolidate:
                    try:
                        from cuda import cuda
                        _ok(cuda.cuMemFree(block.ptr))
                        memory_freed += block.size
                        blocks_consolidated += 1
                    except:
                        pass  # Continue on error
                
                # Update the bucket
                memory_pool[bucket_size] = blocks[:1]  # Keep only first block
        
        buckets_after = len([bucket for bucket, blocks in memory_pool.items() if blocks])
        
        # Record defragmentation operation
        self.defrag_operations += 1
        
        # Calculate improvement
        frag_after = self.analyze_fragmentation(memory_pool, segments)
        improvement = frag_stats['overall_fragmentation'] - frag_after['overall_fragmentation']
        
        return {
            'buckets_before': buckets_before,
            'buckets_after': buckets_after,
            'blocks_consolidated': blocks_consolidated,
            'memory_freed': memory_freed,
            'fragmentation_improvement': improvement,
            'blocks_before': blocks_before,
            'blocks_after': sum(len(blocks) for blocks in memory_pool.values())
        }


class MemoryPressureMonitor:
    """Monitor GPU memory pressure and trigger cleanup"""
    
    def __init__(self):
        self.pressure_threshold = 0.8  # 80% memory usage triggers cleanup
        self.critical_threshold = 0.95  # 95% triggers aggressive cleanup
        self.last_check_time = 0.0
        self.check_interval = 1.0  # Check every 1 second
        self.gc_triggered_count = 0
        self.critical_gc_count = 0
        
    def check_memory_pressure(self) -> tuple[float, bool, bool]:
        """Check current memory pressure, returns (usage_ratio, needs_gc, critical)"""
        current_time = time.time()
        
        # Rate limit checks to avoid overhead
        if current_time - self.last_check_time < self.check_interval:
            return 0.0, False, False
            
        self.last_check_time = current_time
        
        try:
            # Get GPU memory info
            free_bytes, total_bytes = _ok(cuda.cuMemGetInfo())
            used_bytes = total_bytes - free_bytes
            usage_ratio = used_bytes / total_bytes
            
            needs_gc = usage_ratio > self.pressure_threshold
            critical = usage_ratio > self.critical_threshold
            
            if critical:
                self.critical_gc_count += 1
            elif needs_gc:
                self.gc_triggered_count += 1
                
            return usage_ratio, needs_gc, critical
            
        except Exception:
            # If memory info fails, assume no pressure
            return 0.0, False, False
    
    def get_stats(self) -> Dict:
        """Get memory pressure monitoring statistics"""
        return {
            'pressure_threshold': f'{self.pressure_threshold * 100:.1f}%',
            'critical_threshold': f'{self.critical_threshold * 100:.1f}%',
            'gc_triggered_count': self.gc_triggered_count,
            'critical_gc_count': self.critical_gc_count,
            'last_check_time': self.last_check_time
        }


class RefCountedMemoryPool:
    """Reference-counted memory pool with lazy cleanup and pressure-based eviction"""
    
    def __init__(self):
        # Memory pool: bucket_size -> [RefCountedBlock]
        self.memory_pool = defaultdict(list)
        
        # Pre-compute bucket lookup table for faster allocation
        self._bucket_lookup = self._build_bucket_lookup()
        
        # Active blocks: ptr -> RefCountedBlock (for reference counting)
        self.active_blocks = {}
        
        # Thread-local storage for lock-free access
        self._thread_local = threading.local()
        
        # Stream-level cache - Level 1 (very fast, no events)
        self.stream_cache = defaultdict(lambda: defaultdict(list))  # stream -> bucket -> [RefCountedBlock]
        
        # Global cache - Level 2 (with event synchronization)
        self.global_cache = defaultdict(list)  # bucket -> [(RefCountedBlock, event)]
        
        # Fine-grained locks
        self.stream_locks = defaultdict(threading.Lock)
        self.global_lock = threading.Lock()
        # Use re-entrant lock to avoid self-deadlock when object destructors
        # (which may free memory) run during an allocation while the pool lock
        # is already held in the same thread.
        self.pool_lock = threading.RLock()
        
        # Configuration with PyTorch-like parameters
        self.max_pool_size = 2 * 1024 * 1024 * 1024  # 2GB pool limit
        self.current_pool_size = 0
        self.gc_threshold = 0.8  # trigger cleanup at 80% usage
        self.max_split_size_mb = 128  # prevent over-fragmentation
        self.expandable_segments = True  # allow segment expansion
        
        # Memory pressure monitoring
        self.pressure_monitor = MemoryPressureMonitor()
        
        # Fragmentation detection and management
        self.fragmentation_detector = FragmentationDetector()
        
        # Statistics
        self.pool_hits = 0
        self.pool_misses = 0
        self.ref_count_saves = 0  # blocks saved from immediate deallocation
        self.pressure_cleanups = 0  # cleanups triggered by memory pressure
        self.critical_cleanups = 0  # critical memory situation cleanups
        self.defrag_operations = 0  # defragmentation operations performed
        
        # Warmup pool for common sizes
        self._warmup_pool()
        
    def allocate_block(self, size: int, stream: Optional[int] = None) -> 'RefCountedBlock':
        """Allocate a reference-counted block from pool or create new one"""
        bucket = self._get_bucket(size)
        current_time = time.time()
        
        # First try thread-local pool (lock-free, fastest)
        thread_pool = self._get_thread_pool()
        if thread_pool[bucket]:
            block = thread_pool[bucket].pop()
            block.ref_count = 1
            block.is_free = False
            block.last_used_time = current_time
            block.stream_id = stream
            # We need to update active_blocks with lock for consistency
            with self.pool_lock:
                self.active_blocks[block.ptr] = block
            self.pool_hits += 1
            self._thread_local.hits += 1
            return block
        
        # Try to get from global memory pool
        with self.pool_lock:
            if self.memory_pool[bucket]:
                block = self.memory_pool[bucket].pop(0)
                block.ref_count = 1
                block.is_free = False
                block.last_used_time = current_time
                block.stream_id = stream
                self.active_blocks[block.ptr] = block
                self.current_pool_size -= bucket
                self.pool_hits += 1
                return block
        
        # Pool miss - try stream cache
        if stream is not None:
            with self.stream_locks[stream]:
                if self.stream_cache[stream][bucket]:
                    block = self.stream_cache[stream][bucket].pop()
                    block.ref_count = 1
                    block.is_free = False
                    block.last_used_time = current_time
                    with self.pool_lock:
                        self.active_blocks[block.ptr] = block
                    self.pool_hits += 1
                    return block
        
        # Cache miss - allocate new block
        try:
            ptr = _ok(cuda.cuMemAlloc(bucket))
            block = RefCountedBlock(
                ptr=ptr,
                size=bucket,
                is_free=False,
                segment_id=-1,  # pool blocks don't belong to segments
                ref_count=1,
                last_used_time=current_time,
                stream_id=stream
            )
            self.active_blocks[ptr] = block
            self.pool_misses += 1
            return block
        except RuntimeError as e:
            # Fast fail on OOM - provide clear error message
            raise RuntimeError(f"CUDA OOM: Failed to allocate {bucket} bytes ({nbytes} requested). "
                             f"Pool stats: {len(self.active_blocks)} active blocks, "
                             f"{sum(len(blocks) for blocks in self.pool.values())} cached blocks.") from e
    
    def decrease_ref(self, ptr: int, stream: Optional[int] = None) -> bool:
        """Decrease reference count and potentially return to pool"""
        with self.pool_lock:
            if ptr not in self.active_blocks:
                return False
            
            block = self.active_blocks[ptr]
            block.ref_count -= 1
            
            if block.ref_count <= 0:
                # Reference count reached zero - move to pool instead of immediate free
                del self.active_blocks[ptr]
                block.is_free = True
                block.last_used_time = time.time()
                
                # First try to return to thread-local pool (fastest)
                thread_pool = self._get_thread_pool()
                bucket = block.size
                thread_pool_limit = 32  # Limit thread-local pool size to prevent memory bloat
                
                if len(thread_pool[bucket]) < thread_pool_limit:
                    thread_pool[bucket].append(block)
                    self.ref_count_saves += 1
                    return True
                
                # Thread-local pool full, try global pool
                if self.current_pool_size + block.size <= self.max_pool_size:
                    self.memory_pool[block.size].append(block)
                    self.current_pool_size += block.size
                    self.ref_count_saves += 1
                    return True
                else:
                    # Pool is full - actually free to CUDA
                    try:
                        _ok(cuda.cuMemFree(ptr))
                    except:
                        pass
                    return True
            
            return False  # Still has references
    
    def increase_ref(self, ptr: int) -> bool:
        """Increase reference count for shared ownership"""
        with self.pool_lock:
            if ptr in self.active_blocks:
                self.active_blocks[ptr].ref_count += 1
                return True
            return False
    
    def _trigger_gc(self, max_age: float = 60.0):
        """Standard garbage collection of old blocks"""
        current_time = time.time()
        
        stats = {
            'freed_bytes': 0,
            'freed_blocks': 0,
            'pool_blocks_freed': 0,
            'stream_blocks_freed': 0
        }
        
        with self.pool_lock:
            # Clean up old blocks from pool
            total_freed = 0
            blocks_freed = 0
            for bucket, blocks in list(self.memory_pool.items()):
                remaining_blocks = []
                for block in blocks:
                    if current_time - block.last_used_time > max_age:
                        # Free old block
                        try:
                            _ok(cuda.cuMemFree(block.ptr))
                            total_freed += block.size
                            blocks_freed += 1
                        except:
                            pass
                    else:
                        remaining_blocks.append(block)
                
                if remaining_blocks:
                    self.memory_pool[bucket] = remaining_blocks
                else:
                    del self.memory_pool[bucket]
            
            self.current_pool_size -= total_freed
            stats['freed_bytes'] += total_freed
            stats['freed_blocks'] += blocks_freed
            stats['pool_blocks_freed'] = blocks_freed
            
        # Also clean up stream caches
        stream_blocks_freed = 0
        for stream_id in list(self.stream_cache.keys()):
            with self.stream_locks[stream_id]:
                for bucket in list(self.stream_cache[stream_id].keys()):
                    remaining = []
                    for block in self.stream_cache[stream_id][bucket]:
                        if current_time - block.last_used_time <= max_age:
                            remaining.append(block)
                        else:
                            try:
                                _ok(cuda.cuMemFree(block.ptr))
                                stats['freed_bytes'] += block.size
                                stats['freed_blocks'] += 1
                                stream_blocks_freed += 1
                            except:
                                pass
                    
                    if remaining:
                        self.stream_cache[stream_id][bucket] = remaining
                    else:
                        del self.stream_cache[stream_id][bucket]
        
        stats['stream_blocks_freed'] = stream_blocks_freed
        return stats
    
    def _emergency_cleanup(self):
        """Emergency cleanup when system is under critical memory pressure"""
        # Clear all caches immediately
        total_freed = self.clear_all_caches()
        
        # Force aggressive GC with very short max age
        gc_stats = self._trigger_gc(max_age=5.0)  # Keep only blocks used in last 5 seconds
        
        self.critical_cleanups += 1
        
        return {
            'cache_freed': total_freed,
            'gc_stats': gc_stats,
            'total_freed': total_freed + gc_stats['freed_bytes']
        }
    
    def put_to_pool(self, block: 'RefCountedBlock', stream: Optional[int] = None):
        """Put block into appropriate cache level"""
        bucket = block.size
        block.last_used_time = time.time()
        
        # 1. Priority: stream-local cache (Level 1, no events, stream-order guarantee)
        if stream is not None:
            with self.stream_locks[stream]:
                if len(self.stream_cache[stream][bucket]) < 10:
                    self.stream_cache[stream][bucket].append(block)
                    return
        
        # 2. Second priority: global cache (Level 2, must synchronize events)
        if stream is not None:
            event = _ok(cuda.cuEventCreate(0))
            _ok(cuda.cuEventRecord(event, stream))
            with self.global_lock:
                if len(self.global_cache[bucket]) < 100:
                    self.global_cache[bucket].append((block, event))
                    return
                else:
                    # Global cache is full, destroy event and fall through to pool
                    _ok(cuda.cuEventDestroy(event))
        
        # 3. Final fallback: memory pool
        with self.pool_lock:
            if self.current_pool_size + bucket <= self.max_pool_size:
                self.memory_pool[bucket].append(block)
                self.current_pool_size += bucket
            else:
                # Pool is full - actually free to CUDA
                try:
                    _ok(cuda.cuMemFree(block.ptr))
                except:
                    pass
    
    def _build_bucket_lookup(self) -> dict:
        """Pre-compute bucket lookup table for sizes up to 1MB"""
        lookup = {}
        
        # Generate all bucket sizes (power of 2, starting from 64)
        bucket_sizes = []
        bucket = 64
        while bucket <= 1024 * 1024:  # 1MB limit for ref_pool
            bucket_sizes.append(bucket)
            bucket *= 2
        
        # Build lookup table for all sizes up to 1MB
        for size in range(1, 1024 * 1024 + 1):
            # Find smallest bucket >= size
            for bucket in bucket_sizes:
                if bucket >= size:
                    lookup[size] = bucket
                    break
        
        return lookup
    
    def _get_thread_pool(self):
        """Get or create thread-local memory pool for lock-free access"""
        if not hasattr(self._thread_local, 'pool'):
            # Initialize thread-local pool
            self._thread_local.pool = defaultdict(list)  # bucket_size -> [RefCountedBlock]
            self._thread_local.hits = 0
            self._thread_local.misses = 0
        return self._thread_local.pool
    
    def _get_bucket(self, size: int) -> int:
        """Get bucket size - optimized with pre-computed lookup"""
        if size <= 1048576 and size in self._bucket_lookup:  # 1MB
            return self._bucket_lookup[size]
        
        # Fallback for very large sizes (>1MB, shouldn't happen in ref_pool)
        if size <= 64:
            return 64
        
        # Use bit manipulation for power of 2 (faster than loop)
        return 1 << (size - 1).bit_length()
    
    def _warmup_pool(self):
        """Warm up pool with common allocation sizes for better cache hit rate."""
        # Common sizes from benchmark and deep learning workloads (< 1MB for ref_pool)
        warmup_sizes = [
            # Benchmark common sizes
            256, 512, 1024, 2048, 4096, 8192, 16384, 32768,
            # Additional common sizes
            64, 128, 384, 768, 1536, 3072, 6144, 12288, 24576, 49152, 98304,
            # Common tensor sizes for small models
            196608, 393216, 786432  # up to ~768KB
        ]
        
        warmup_count_per_size = 8  # Pre-allocate 8 blocks per size
        
        try:
            for size in warmup_sizes:
                if size >= 1024 * 1024:  # Skip sizes >= 1MB (ref_pool limit)
                    continue
                    
                bucket = self._get_bucket(size)
                
                # Skip if already warmed up
                if len(self.memory_pool[bucket]) >= warmup_count_per_size:
                    continue
                
                # Pre-allocate blocks
                for _ in range(warmup_count_per_size):
                    try:
                        ptr = _ok(cuda.cuMemAlloc(bucket))
                        
                        block = RefCountedBlock(
                            ptr=ptr,
                            size=bucket,
                            is_free=True,
                            segment_id=0,
                            ref_count=0,
                            last_used_time=time.time()
                        )
                        
                        self.memory_pool[bucket].append(block)
                        self.current_pool_size += bucket
                        
                        # Don't exceed pool size limit
                        if self.current_pool_size >= self.max_pool_size * 0.1:  # Use 10% for warmup
                            return
                            
                    except Exception:
                        # If allocation fails, skip remaining blocks for this size
                        break
                        
        except Exception:
            # If warmup fails completely, continue without it
            pass
    
    def clear_all_caches(self):
        """Clear all caches and free all pooled memory"""
        total_freed = 0
        
        with self.global_lock:
            # Clear global cache, destroy events and free memory
            for bucket_list in self.global_cache.values():
                for block, event in bucket_list:
                    try:
                        _ok(cuda.cuEventDestroy(event))
                    except:
                        pass  # Ignore destroy failure
                    try:
                        _ok(cuda.cuMemFree(block.ptr))
                        total_freed += block.size
                    except:
                        pass
            self.global_cache.clear()
            
            # Clear stream-local cache
            for stream in list(self.stream_cache.keys()):
                with self.stream_locks[stream]:
                    for bucket, blocks in self.stream_cache[stream].items():
                        for block in blocks:
                            try:
                                _ok(cuda.cuMemFree(block.ptr))
                                total_freed += block.size
                            except:
                                pass
                    self.stream_cache[stream].clear()
        
        # Clear memory pool
        with self.pool_lock:
            for bucket, blocks in self.memory_pool.items():
                for block in blocks:
                    try:
                        _ok(cuda.cuMemFree(block.ptr))
                        total_freed += block.size
                    except:
                        pass
            self.memory_pool.clear()
            self.current_pool_size = 0
        
        return total_freed
    
    def get_pool_stats(self) -> Dict:
        """Get comprehensive memory pool statistics"""
        with self.pool_lock:
            pool_blocks = sum(len(blocks) for blocks in self.memory_pool.values())
            active_blocks = len(self.active_blocks)
            total_refs = sum(block.ref_count for block in self.active_blocks.values())
        
        # Get current memory pressure
        usage_ratio, needs_gc, critical = self.pressure_monitor.check_memory_pressure()
        
        # Count cached blocks
        stream_cached = 0
        global_cached = 0
        for stream_cache in self.stream_cache.values():
            for bucket_blocks in stream_cache.values():
                stream_cached += len(bucket_blocks)
        
        for bucket_blocks in self.global_cache.values():
            global_cached += len(bucket_blocks)
        
        stats = {
            'pool_hits': self.pool_hits,
            'pool_misses': self.pool_misses,
            'ref_count_saves': self.ref_count_saves,
            'pool_size_mb': self.current_pool_size / (1024 * 1024),
            'pool_blocks': pool_blocks,
            'active_blocks': active_blocks,
            'total_references': total_refs,
            'hit_rate': f'{self.pool_hits / max(1, self.pool_hits + self.pool_misses) * 100:.1f}%',
            
            # Memory pressure stats
            'memory_usage_ratio': f'{usage_ratio * 100:.1f}%',
            'memory_pressure': needs_gc,
            'critical_pressure': critical,
            'pressure_cleanups': self.pressure_cleanups,
            'critical_cleanups': self.critical_cleanups,
            
            # Cache distribution
            'stream_cached_blocks': stream_cached,
            'global_cached_blocks': global_cached,
            
            # Pressure monitor stats
            'pressure_monitor': self.pressure_monitor.get_stats()
        }
        
        return stats

# ============= Main Memory Manager =============

class CUDAMemoryManager:
    """
    High-performance CUDA memory manager
    Integrating Priority A-C optimizations
    """
    
    def __init__(self):
        # Initialize CUDA and context
        self._init_cuda()
        
        # Phase 2: Bucket caching for small allocations
        self.free_blocks = defaultdict(list)  # size -> [ptr_list] 
        self.active_blocks = {}  # ptr -> size
        self.lock = threading.RLock()
        
        # Enhanced statistics collector
        self.stats_collector = get_stats_collector()
        
        # Phase 3: Block allocator for large allocations
        self.segments: List[Segment] = []
        self.next_segment_id = 0
        self.segment_size = 1024 * 1024 * 1024  # 1GB per segment
        self.block_allocator_threshold = 1024 * 1024  # 1MB threshold
        
        # Configuration
        self.alignment = 512  # 512B alignment
        self.max_cache_size = 1024 * 1024 * 1024  # 1GB cache limit
        self.current_cache_size = 0
        
        # Large memory cache (>=1MB)
        self.large_memory_cache = defaultdict(list)  # size -> [ptr_list]
        self.max_large_cache_size = 512 * 1024 * 1024  # 512MB cache for large blocks
        self.current_large_cache_size = 0
        self.max_cached_blocks_per_size = 4  # Max 4 cached blocks per size
        
        # Warmup configuration - preallocation for common sizes
        self.warmup_enabled = True
        self.warmup_sizes = [
            # Phase 2 Optimization: Enhanced small tensor sizes based on deep learning patterns
            # Very small tensors (scalars, small vectors)
            64, 128, 192, 256, 320, 384, 448, 512,
            # Medium tensors (embeddings, attention weights)
            768, 1024, 1536, 2048, 3072, 4096,
            # Large tensors (model-specific sizes)
            896 * 4,      # hidden_size * sizeof(float32)
            896 * 32 * 4, # (batch, seq_len, hidden_size) for small batches
            896 * 128 * 4, # (batch, seq_len, hidden_size) * sizeof(float32)
            4864 * 4,     # intermediate_size * sizeof(float32)
            14 * 64 * 4,  # num_heads * head_dim * sizeof(float32)
            # Additional common sizes found in deep learning
            7168,  # 896 * 8 (common intermediate calculations)
            14336, # 896 * 16
        ]
        self.warmup_count_per_size = 12  # Phase 2: Increased to 12 blocks per size for better cache hit rate
        
        # Statistics  
        self.alloc_count = 0
        self.cache_hits = 0
        self.cache_misses = 0
        self.block_alloc_count = 0
        self.block_hits = 0
        self.block_misses = 0
        
        # Initialize warmup if enabled
        if self.warmup_enabled:
            self._warmup_cache()
        
        # Initialize default stream (align with Triton / runtime default stream)
        # Use CUDA default (legacy) stream 0 to avoid interop issues when PyTorch
        # initializes CUDA first and Triton launches kernels on the default stream.
        # A dedicated stream here can cause ordering/visibility surprises across APIs.
        self.default_stream = 0  # CUstream legacy default
        
        # Initialize reference counting memory pool
        self.ref_pool = RefCountedMemoryPool()
        
    def _init_cuda(self):
        """Initialize CUDA environment using primary context"""
        _ok(cuda.cuInit(0))
        dev = _ok(cuda.cuDeviceGet(0))
        ctx = _ok(cuda.cuDevicePrimaryCtxRetain(dev))
        _ok(cuda.cuCtxSetCurrent(ctx))
        self.context = ctx
    
    def _create_stream(self, flags: int = 0):
        """Create a CUDA stream"""
        return _ok(cuda.cuStreamCreate(flags))
    
    def _create_segment(self, size: int) -> Segment:
        """Create new segment"""
        segment = Segment(self.next_segment_id, size)
        self.next_segment_id += 1
        self.segments.append(segment)
        self.block_alloc_count += 1
        return segment
    
    def _round_up(self, size: int, alignment: int) -> int:
        """Round up size to alignment boundary"""
        return ((size + alignment - 1) // alignment) * alignment
    
    def _get_bucket_size(self, size: int) -> int:
        """
        Optimized bucket size for deep learning workloads
        Uses smaller increments for common sizes to reduce waste
        """
        # For very small sizes, use smaller increments
        if size <= 256:
            return ((size + 63) // 64) * 64  # 64B increments
        elif size <= 1024:
            return ((size + 127) // 128) * 128  # 128B increments  
        elif size <= 4096:
            return ((size + 255) // 256) * 256  # 256B increments
        elif size <= self.alignment:
            return self.alignment
        
        # For larger sizes, use power of 2 with alignment
        bucket = self.alignment
        while bucket < size:
            bucket *= 2
        
        # Cap at 16MB for very large allocations to avoid excessive waste
        max_bucket = 16 * 1024 * 1024
        if bucket > max_bucket:
            # For very large allocations, use exact size alignment
            return self._round_up(size, self.alignment)
        
        return bucket
    
    def _warmup_cache(self):
        """Pre-allocate common sizes to improve cache hit rate"""
        try:
            for size in self.warmup_sizes:
                bucket_size = self._get_bucket_size(size)
                # Skip if already cached
                if len(self.free_blocks[bucket_size]) >= self.warmup_count_per_size:
                    continue
                    
                # Pre-allocate blocks
                for _ in range(self.warmup_count_per_size):
                    try:
                        ptr = _ok(cuda.cuMemAlloc(bucket_size))
                        self.free_blocks[bucket_size].append(ptr)
                        self.current_cache_size += bucket_size
                        
                        # Don't exceed cache limit
                        if self.current_cache_size >= self.max_cache_size * 0.8:
                            return
                    except Exception:
                        # If allocation fails, skip this size
                        break
        except Exception:
            # If warmup fails, continue without it
            pass
    
    def allocate(self, nbytes: int, stream: Optional[int] = None) -> int:
        """
        Phase 3: Hybrid allocator
        - Small allocations (<1MB): bucket caching
        - Large allocations (>=1MB): block allocator
        """
        if nbytes == 0:
            return 0
            
        # Decide allocation strategy based on size
        if nbytes < self.block_allocator_threshold:
            return self._allocate_small(nbytes)
        else:
            return self._allocate_large(nbytes)
    
    def _allocate_small(self, nbytes: int) -> int:
        """Allocate small memory using optimized bucket caching"""
        start_time = time.perf_counter_ns()
        bucket_size = self._get_bucket_size(nbytes)
        
        with self.lock:
            # Try to get from cache (bucket match)
            if self.free_blocks[bucket_size]:
                ptr = self.free_blocks[bucket_size].pop()
                self.active_blocks[ptr] = bucket_size
                self.current_cache_size -= bucket_size
                self.cache_hits += 1
                
                # Record cache hit statistics
                alloc_time = time.perf_counter_ns() - start_time
                self.stats_collector.record_allocation(
                    size=nbytes,
                    bucket_size=bucket_size,
                    cache_hit=True,
                    allocator_type='small',
                    allocation_time_ns=alloc_time
                )
                return ptr
        
        # Cache miss - check if we should do batch allocation
        batch_size = self._get_batch_allocation_size(bucket_size)
        
        # Allocate one block for immediate return
        try:
            ptr = _ok(cuda.cuMemAlloc(bucket_size))
        except RuntimeError as e:
            # Fast fail on OOM with clear error message
            raise RuntimeError(f"CUDA OOM: Failed to allocate {bucket_size} bytes "
                             f"(bucket for {nbytes} bytes request). Try reducing model size or batch size.") from e
        
        # If cache is not full, allocate additional blocks for future use
        if batch_size > 1 and self.current_cache_size + (batch_size - 1) * bucket_size <= self.max_cache_size:
            try:
                for _ in range(batch_size - 1):
                    extra_ptr = _ok(cuda.cuMemAlloc(bucket_size))
                    self.free_blocks[bucket_size].append(extra_ptr)
                    self.current_cache_size += bucket_size
            except Exception:
                # If batch allocation fails, continue with single allocation
                pass
        
        with self.lock:
            self.active_blocks[ptr] = bucket_size
            self.alloc_count += 1
            self.cache_misses += 1
        
        # Record cache miss statistics
        alloc_time = time.perf_counter_ns() - start_time
        self.stats_collector.record_allocation(
            size=nbytes,
            bucket_size=bucket_size,
            cache_hit=False,
            allocator_type='small',
            allocation_time_ns=alloc_time
        )
        return ptr
    
    def _get_batch_allocation_size(self, bucket_size: int) -> int:
        """Phase 2 Optimization: Adaptive batch allocation based on bucket size and memory pressure"""
        # Dynamic batch sizes based on available memory and usage patterns
        if bucket_size <= 512:
            return 8  # Very small tensors: batch allocate 8 at once
        elif bucket_size <= 2048:
            return 6  # Small tensors: batch allocate 6 at once
        elif bucket_size <= 8192:
            return 3  # Medium tensors: batch allocate 3 at once
        elif bucket_size <= 32768:
            return 2  # Large tensors: batch allocate 2 at once
        else:
            return 1  # Very large tensors: single allocation
    
    def _allocate_large(self, nbytes: int) -> int:
        """Allocate large memory using block allocator with caching"""
        start_time = time.perf_counter_ns()
        aligned_size = self._round_up(nbytes, self.alignment)
        
        with self.lock:
            # First try large memory cache
            if self.large_memory_cache[aligned_size]:
                ptr = self.large_memory_cache[aligned_size].pop()
                self.active_blocks[ptr] = aligned_size
                self.current_large_cache_size -= aligned_size
                self.block_hits += 1
                
                # Record cache hit statistics
                alloc_time = time.perf_counter_ns() - start_time
                self.stats_collector.record_allocation(
                    size=nbytes,
                    bucket_size=aligned_size,
                    cache_hit=True,
                    allocator_type='large_cache',
                    allocation_time_ns=alloc_time
                )
                return ptr
            
            # Try to allocate from existing segments
            for segment in self.segments:
                ptr = segment.allocate(aligned_size)
                if ptr is not None:
                    self.active_blocks[ptr] = aligned_size
                    self.block_hits += 1
                    
                    # Record block hit statistics
                    alloc_time = time.perf_counter_ns() - start_time
                    self.stats_collector.record_allocation(
                        size=nbytes,
                        bucket_size=aligned_size,
                        cache_hit=True,
                        allocator_type='large',
                        allocation_time_ns=alloc_time
                    )
                    return ptr
            
            # No suitable block found - create new segment
            # Use adaptive segment size: smaller for small requests to avoid waste
            if aligned_size < 16 * 1024 * 1024:  # < 16MB
                segment_size = max(aligned_size * 8, 64 * 1024 * 1024)  # 8x request or 64MB min
            else:
                segment_size = max(aligned_size * 2, self.segment_size)  # Original logic for large
            segment = self._create_segment(segment_size)
            ptr = segment.allocate(aligned_size)
            if ptr is not None:
                self.active_blocks[ptr] = aligned_size
                self.block_misses += 1
                
                # Record new segment allocation statistics
                alloc_time = time.perf_counter_ns() - start_time
                self.stats_collector.record_allocation(
                    size=nbytes,
                    bucket_size=aligned_size,
                    cache_hit=False,
                    allocator_type='large',
                    allocation_time_ns=alloc_time
                )
                return ptr
            
            # Fallback: direct allocation
            ptr = _ok(cuda.cuMemAlloc(aligned_size))
            self.active_blocks[ptr] = aligned_size
            self.block_misses += 1
            
            # Record direct allocation statistics
            alloc_time = time.perf_counter_ns() - start_time
            self.stats_collector.record_allocation(
                size=nbytes,
                bucket_size=aligned_size,
                cache_hit=False,
                allocator_type='large_direct',
                allocation_time_ns=alloc_time
            )
            return ptr
    
    def free(self, ptr: int, stream: Optional[int] = None):
        """
        Phase 3: Hybrid free
        - Small allocations: bucket cache
        - Large allocations: block allocator
        """
        if not ptr or int(ptr) == 0:
            return
            
        with self.lock:
            if ptr not in self.active_blocks:
                return  # Already freed or not from our allocator
                
            size = self.active_blocks.pop(ptr)
            
            # Record deallocation statistics (note: we don't have allocation timestamp here)
            self.stats_collector.record_deallocation(size=size)
            
            # Decide free strategy based on size
            if size < self.block_allocator_threshold:
                self._free_small(ptr, size)
            else:
                self._free_large(ptr, size)
    
    def _free_small(self, ptr: int, size: int):
        """Free small memory to bucket cache"""
        # Try to return to cache
        if self.current_cache_size + size <= self.max_cache_size:
            self.free_blocks[size].append(ptr)
            self.current_cache_size += size
            return
        
        # Cache full - actually free to CUDA
        try:
            _ok(cuda.cuMemFree(ptr))
        except:
            pass
    
    def _free_large(self, ptr: int, size: int):
        """Free large memory to cache or block allocator"""
        # Try to cache the block if there's space and not too many cached
        if (self.current_large_cache_size + size <= self.max_large_cache_size and 
            len(self.large_memory_cache[size]) < self.max_cached_blocks_per_size):
            
            self.large_memory_cache[size].append(ptr)
            self.current_large_cache_size += size
            return
            
        # Cache full or too many blocks of this size - try to free to segment
        for segment in self.segments:
            if segment.free(ptr):
                return
        
        # Not found in any segment - direct free
        try:
            _ok(cuda.cuMemFree(ptr))
        except:
            pass
    
    def get_stats(self) -> Dict:
        """Get comprehensive allocator statistics"""
        with self.lock:
            # Small allocation stats
            total_small_requests = self.cache_hits + self.cache_misses
            small_hit_rate = (self.cache_hits / total_small_requests * 100) if total_small_requests > 0 else 0
            
            # Large allocation stats
            total_large_requests = self.block_hits + self.block_misses
            large_hit_rate = (self.block_hits / total_large_requests * 100) if total_large_requests > 0 else 0
            
            # Count cached blocks and total cached memory
            cached_blocks = sum(len(block_list) for block_list in self.free_blocks.values())
            
            # Bucket statistics
            bucket_info = {}
            for bucket_size, block_list in self.free_blocks.items():
                if block_list:
                    bucket_info[f'{bucket_size//1024}KB'] = len(block_list)
            
            # Segment statistics
            segment_stats = []
            total_segment_memory = 0
            total_used_memory = 0
            for segment in self.segments:
                total_segment_memory += segment.total_size
                total_used_memory += segment.used_bytes
                segment_stats.append({
                    'id': segment.segment_id,
                    'total_mb': segment.total_size // (1024 * 1024),
                    'used_mb': segment.used_bytes // (1024 * 1024),
                    'utilization': f'{segment.used_bytes / segment.total_size * 100:.1f}%'
                })
            
            # Get ref pool stats
            ref_pool_stats = self.ref_pool.get_pool_stats()
            
            # Get memory info
            memory_info = get_memory_info()
            
            stats = {
                # Overall stats
                'total_alloc_count': self.alloc_count + self.block_alloc_count,
                'active_blocks': len(self.active_blocks),
                
                # Small allocation (bucket cache) stats
                'small_cache_hits': self.cache_hits,
                'small_cache_misses': self.cache_misses,
                'small_hit_rate': f'{small_hit_rate:.1f}%',
                'cached_memory_mb': self.current_cache_size / (1024 * 1024),
                'cached_blocks': cached_blocks,
                'bucket_distribution': bucket_info,
                
                # Large allocation (block allocator) stats
                'large_block_hits': self.block_hits,
                'large_block_misses': self.block_misses,
                'large_hit_rate': f'{large_hit_rate:.1f}%',
                'segment_count': len(self.segments),
                'segment_memory_mb': total_segment_memory // (1024 * 1024),
                'segment_used_mb': total_used_memory // (1024 * 1024),
                'segment_utilization': f'{total_used_memory / max(1, total_segment_memory) * 100:.1f}%',
                'segments': segment_stats,
                
                # Reference counting pool stats
                'ref_pool': ref_pool_stats,
                
                # Memory info
                'memory_info': memory_info
            }
            
            return stats
    
    def get_enhanced_stats(self) -> Dict:
        """Get enhanced memory statistics with detailed insights."""
        # Get basic stats
        basic_stats = self.get_stats()
        
        # Get enhanced stats from collector
        enhanced_stats = self.stats_collector.get_enhanced_stats()
        
        # Combine and return
        return {
            'basic_stats': basic_stats,
            'enhanced_stats': enhanced_stats,
            'collection_active': True
        }
    
    def empty_cache(self):
        """Empty all cached memory (small and large)"""
        with self.lock:
            # Free all small cached blocks to CUDA
            for size, ptr_list in self.free_blocks.items():
                for ptr in ptr_list:
                    try:
                        _ok(cuda.cuMemFree(ptr))
                    except:
                        pass
            
            # Free all large cached blocks to CUDA
            for size, ptr_list in self.large_memory_cache.items():
                for ptr in ptr_list:
                    try:
                        _ok(cuda.cuMemFree(ptr))
                    except:
                        pass
            
            # Clear caches
            self.free_blocks.clear()
            self.large_memory_cache.clear()
            self.current_cache_size = 0
            self.current_large_cache_size = 0
            print(f"Cache cleared: freed all small and large cached blocks")
    
    def __del__(self):
        """Cleanup CUDA resources in proper order"""
        try:
            # 1. Clear cache and return segments
            self.empty_cache()
            
            # 2. Bottom-up release segments
            for segment in self.segments:
                try:
                    cuda.cuMemFree(segment.base_ptr)
                except:
                    pass
            
            # 3. Destroy default stream
            if hasattr(self, 'default_stream') and self.default_stream not in (None, 0):
                try:
                    _ok(cuda.cuStreamDestroy(self.default_stream))
                except:
                    pass
            
            # 4. Release primary context
            if hasattr(self, 'context'):
                try:
                    device = _ok(cuda.cuCtxGetDevice())
                    _ok(cuda.cuDevicePrimaryCtxRelease(device))
                except:
                    pass
        except:
            pass

# Global memory manager instance
_memory_manager = None

def get_memory_manager() -> CUDAMemoryManager:
    """Get global memory manager instance"""
    global _memory_manager
    if _memory_manager is None:
        _memory_manager = CUDAMemoryManager()
    return _memory_manager

# ============= External Interfaces =============

def allocate_memory(nbytes: int, stream: Optional[int] = None) -> int:
    """Allocate GPU memory with reference counting for small allocations"""
    manager = get_memory_manager()
    
    # Use ref pool for small allocations (< 1MB)
    if nbytes < 1024 * 1024:
        block = manager.ref_pool.allocate_block(nbytes, stream)
        return block.ptr
    else:
        # Use traditional allocator for large allocations
        return manager.allocate(nbytes, stream)

def free_memory(ptr: int, nbytes: int = 0, stream: Optional[int] = None):
    """Free GPU memory with size info for caching"""
    manager = get_memory_manager()
    
    # Route to appropriate allocator based on size (same logic as allocate_memory)
    if nbytes > 0 and nbytes < 1024 * 1024:
        # Small allocations: use ref pool
        manager.ref_pool.decrease_ref(ptr, stream)
    else:
        # Large allocations or unknown size: use traditional allocator
        manager.free(ptr, stream)

def increase_ref_count(ptr: int) -> bool:
    """Increase reference count for shared tensor ownership"""
    return get_memory_manager().ref_pool.increase_ref(ptr)

def decrease_ref_count(ptr: int, stream: Optional[int] = None) -> bool:
    """Decrease reference count and potentially return to pool"""
    return get_memory_manager().ref_pool.decrease_ref(ptr, stream)

def trigger_gc():
    """Manually trigger garbage collection"""
    return get_memory_manager().ref_pool._trigger_gc()

def memory_stats() -> Dict:
    """Get memory statistics"""
    return get_memory_manager().get_stats()

def enhanced_memory_stats() -> Dict:
    """Get enhanced memory statistics with detailed insights"""
    return get_memory_manager().get_enhanced_stats()

def empty_cache():
    """Empty memory cache"""
    get_memory_manager().empty_cache()

def get_memory_info() -> Dict:
    """Get system memory information"""
    try:
        from cuda import cuda
        free_bytes, total_bytes = _ok(cuda.cuMemGetInfo())
        used_bytes = total_bytes - free_bytes
        usage_ratio = used_bytes / total_bytes
        
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

def check_memory_pressure() -> bool:
    """Check if system is under memory pressure"""
    manager = get_memory_manager()
    usage_ratio, needs_gc, critical = manager.ref_pool.pressure_monitor.check_memory_pressure()
    return needs_gc or critical

def set_memory_config(**kwargs):
    """Set memory management configuration"""
    manager = get_memory_manager()
    
    if 'gc_threshold' in kwargs:
        manager.ref_pool.gc_threshold = kwargs['gc_threshold']
        manager.ref_pool.pressure_monitor.pressure_threshold = kwargs['gc_threshold']
    
    if 'max_pool_size_mb' in kwargs:
        manager.ref_pool.max_pool_size = kwargs['max_pool_size_mb'] * 1024 * 1024
    
    if 'max_split_size_mb' in kwargs:
        manager.ref_pool.max_split_size_mb = kwargs['max_split_size_mb']
    
    if 'fragmentation_threshold' in kwargs:
        manager.ref_pool.fragmentation_detector.fragmentation_threshold = kwargs['fragmentation_threshold']

def defragment_memory() -> Optional[Dict]:
    """Manually trigger memory defragmentation"""
    manager = get_memory_manager()
    return manager.ref_pool.fragmentation_detector.defragment(
        manager.ref_pool.memory_pool, 
        []  # No segments for pool-based defrag
    )

def get_fragmentation_stats() -> Dict:
    """Get memory fragmentation statistics"""
    manager = get_memory_manager()
    defrag_history = manager.ref_pool.fragmentation_detector.get_defrag_history()
    
    return {
        'defrag_history': defrag_history
    }

def analyze_memory_fragmentation() -> Dict:
    """Analyze current memory fragmentation"""
    manager = get_memory_manager()
    return manager.ref_pool.fragmentation_detector.analyze_fragmentation(
        manager.ref_pool.memory_pool,
        []  # No segments for pool-based analysis
    )

# ============= Test Functions =============

def test_performance():
    """Test memory manager performance"""
    import numpy as np
    
    print("🚀 High-performance CUDA memory manager test\n")
    
    manager = get_memory_manager()
    
    # Test configuration
    shape = (4096, 4096)
    nbytes = int(np.prod(shape) * 4)  # float32
    iterations = 20
    
    # ✅ Use the same stream to ensure cache hit
    test_stream = manager.default_stream
    print(f"Test configuration: {shape} float32 tensor, {iterations} iterations")
    
    # Test allocation performance
    start = time.perf_counter()
    ptrs = [manager.allocate(nbytes, stream=test_stream) for _ in range(iterations)]
    alloc_time = time.perf_counter() - start
    
    print(f"\n✅ Allocation performance: Total time {alloc_time*1000:.2f}ms | Average {alloc_time*1000/iterations:.3f}ms/iter")
    
    # Test deallocation performance - ✅ Use the same stream
    start = time.perf_counter()
    for ptr in ptrs:
        manager.free(ptr, stream=test_stream)
    free_time = time.perf_counter() - start
    
    print(f"\n✅ Deallocation performance: Total time {free_time*1000:.2f}ms | Average {free_time*1000/iterations:.3f}ms/iter")
    
    # Statistics
    stats = manager.get_stats()
    print("\n📊 Statistics:")
    for key, value in stats.items():
        print(f"   {key}: {value}")
    
    # Compare with PyTorch
    pytorch_baseline = 0.217  # ms from previous tests
    our_time = alloc_time * 1000 / iterations
    speedup = pytorch_baseline / our_time
    
    print(f"\n🏁 Performance comparison:")
    print(f"   PyTorch: {pytorch_baseline:.3f}ms/tensor")
    print(f"   Optimized: {our_time:.3f}ms/tensor")
    print(f"   Speedup: {speedup:.2f}x")
    
    if speedup > 0.5:
        print("✅ Excellent! Close to PyTorch performance")
    elif speedup > 0.1:
        print("⚡ Good! But still has optimization space")
    else:
        print("⚠️ Need further optimization")
    
    return our_time

if __name__ == "__main__":
    test_performance()
