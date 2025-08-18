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
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass
from collections import defaultdict
import time


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
    
class Segment:
    """
    segment
    """
    
    def __init__(self, segment_id: int, size: int):
        self.segment_id = segment_id
        self.total_size = size
        
        # allocate entire segment from CUDA
        self.base_ptr = _ok(cuda.cuMemAlloc(size))
        
        # initialize memory to zero to avoid dirty data causing precision issues
        _ok(cuda.cuMemsetD8(self.base_ptr, 0, size))
        
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


class TwoLevelCache:
    """Two-level cache architecture - Stream-local cache + Global cache (with event synchronization)"""
    
    def __init__(self):
        # Stream-level cache - Level 1 (very fast, no events)
        self.stream_cache = defaultdict(lambda: defaultdict(list))  # stream -> bucket -> [ptr_list]
        
        # Global cache - Level 2 (with event synchronization)
        self.global_cache = defaultdict(list)  # bucket -> [(ptr, event)]
        
        # Fine-grained locks
        self.stream_locks = defaultdict(threading.Lock)
        self.global_lock = threading.Lock()
        
        # Statistics
        self.cache_hits = 0
        self.cache_misses = 0
    
    def get_from_cache(self, size: int, stream: int) -> Optional[int]:
        """Get memory block from cache"""
        bucket = self._get_bucket(size)
        
        # 1. Try stream-local cache (Level 1, very fast, no events)
        with self.stream_locks[stream]:
            if self.stream_cache[stream][bucket]:
                self.cache_hits += 1
                ptr = self.stream_cache[stream][bucket].pop()
                # Comment out memory cleanup to avoid affecting gradient calculations
                # _ok(cuda.cuMemsetD8(ptr, 0, bucket))
                return ptr
        
        # 2. Try global cache (Level 2, cross-stream reuse requires event synchronization)
        with self.global_lock:
            if self.global_cache[bucket]:
                ptr, event = self.global_cache[bucket].pop()
                # Wait for the event recorded by the release stream
                _ok(cuda.cuEventDestroy(event))
                # Comment out memory cleanup to avoid affecting gradient calculations
                # _ok(cuda.cuMemsetD8(ptr, 0, bucket))
                self.cache_hits += 1
                return ptr
        
        self.cache_misses += 1
        return None
    
    def put_to_cache(self, ptr: int, size: int, stream: int):
        """Put memory block into cache"""
        bucket = self._get_bucket(size)
        
        # 1. Priority: stream-local cache (Level 1, no events, stream-order guarantee)
        with self.stream_locks[stream]:
            if len(self.stream_cache[stream][bucket]) < 10:
                self.stream_cache[stream][bucket].append(ptr)
                return
        
        # 2. Second priority: global cache (Level 2, must synchronize events)
        event = _ok(cuda.cuEventCreate(0))
        _ok(cuda.cuEventRecord(event, stream))
        with self.global_lock:
            if len(self.global_cache[bucket]) < 100:
                self.global_cache[bucket].append((ptr, event))
            else:
                # Global cache is full, fallback strategy: immediately destroy event and discard (defer to upper layer to free to Segment)
                _ok(cuda.cuEventDestroy(event))
    
    def _get_bucket(self, size: int) -> int:
        """Get bucket size - align up to 64B, reduce memory waste"""
        if size <= 64:
            return 64
        
        # Power of 2 buckets starting from 64
        bucket = 64
        while bucket < size:
            bucket *= 2
        return bucket
    
    def clear_cache_with(self, ptr_size_map, free_cb):
        """Clear cache and return pointers to segment"""
        with self.global_lock:
            # Clear global cache, destroy events and return pointers
            for bucket_list in self.global_cache.values():
                for ptr, event in bucket_list:
                    try:
                        _ok(cuda.cuEventDestroy(event))
                    except:
                        pass  # Ignore destroy failure
                    free_cb(ptr)  # Return pointer to segment
            self.global_cache.clear()
            
            # Clear stream-local cache
            for stream in list(self.stream_cache.keys()):
                with self.stream_locks[stream]:
                    for bucket, lst in self.stream_cache[stream].items():
                        for ptr in lst:
                            free_cb(ptr)  # Return pointer to segment
                    self.stream_cache[stream].clear()
    
    def clear_cache(self):
        """Clear all cache - compatible interface, but will drop pointers!"""
        # This interface has memory leak risk, recommend using clear_cache_with
        with self.global_lock:
            # Clear global cache, destroy events
            for bucket_list in self.global_cache.values():
                for ptr, event in bucket_list:
                    try:
                        _ok(cuda.cuEventDestroy(event))
                    except:
                        pass  # Ignore destroy failure
            self.global_cache.clear()
            
            # Clear stream-local cache
            for stream in list(self.stream_cache.keys()):
                with self.stream_locks[stream]:
                    self.stream_cache[stream].clear()

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
        
        # Phase 3: Block allocator for large allocations
        self.segments: List[Segment] = []
        self.next_segment_id = 0
        self.segment_size = 1024 * 1024 * 1024  # 1GB per segment
        self.block_allocator_threshold = 1024 * 1024  # 1MB threshold
        
        # Configuration
        self.alignment = 512  # 512B alignment
        self.max_cache_size = 1024 * 1024 * 1024  # 1GB cache limit
        self.current_cache_size = 0
        
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
        
        # Initialize default stream (already have current context)
        self.default_stream = self._create_stream()
        
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
        bucket_size = self._get_bucket_size(nbytes)
        
        with self.lock:
            # Try to get from cache (bucket match)
            if self.free_blocks[bucket_size]:
                ptr = self.free_blocks[bucket_size].pop()
                self.active_blocks[ptr] = bucket_size
                self.current_cache_size -= bucket_size
                self.cache_hits += 1
                return ptr
        
        # Cache miss - check if we should do batch allocation
        batch_size = self._get_batch_allocation_size(bucket_size)
        
        # Allocate one block for immediate return
        ptr = _ok(cuda.cuMemAlloc(bucket_size))
        
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
        """Allocate large memory using block allocator"""
        aligned_size = self._round_up(nbytes, self.alignment)
        
        with self.lock:
            # Try to allocate from existing segments
            for segment in self.segments:
                ptr = segment.allocate(aligned_size)
                if ptr is not None:
                    self.active_blocks[ptr] = aligned_size
                    self.block_hits += 1
                    return ptr
            
            # No suitable block found - create new segment
            segment = self._create_segment(max(aligned_size * 2, self.segment_size))
            ptr = segment.allocate(aligned_size)
            if ptr is not None:
                self.active_blocks[ptr] = aligned_size
                self.block_misses += 1
                return ptr
            
            # Fallback: direct allocation
            ptr = _ok(cuda.cuMemAlloc(aligned_size))
            self.active_blocks[ptr] = aligned_size
            self.block_misses += 1
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
        """Free large memory to block allocator"""
        # Try to free to segment
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
                'segments': segment_stats
            }
            
            return stats
    
    def empty_cache(self):
        """Phase 2: Empty all bucket cached memory"""
        with self.lock:
            # Free all cached blocks to CUDA
            for size, ptr_list in self.free_blocks.items():
                for ptr in ptr_list:
                    try:
                        _ok(cuda.cuMemFree(ptr))
                    except:
                        pass
            
            # Clear cache
            self.free_blocks.clear()
            self.current_cache_size = 0
            print(f"Cache cleared: freed all cached blocks")
    
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
            if hasattr(self, 'default_stream'):
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
    """Allocate GPU memory"""
    return get_memory_manager().allocate(nbytes, stream)

def free_memory(ptr: int, nbytes: int = 0, stream: Optional[int] = None):
    """Free GPU memory with size info for caching"""
    get_memory_manager().free(ptr, stream)

def memory_stats() -> Dict:
    """Get memory statistics"""
    return get_memory_manager().get_stats()

def empty_cache():
    """Empty memory cache"""
    get_memory_manager().empty_cache()

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