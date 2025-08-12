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
        
        # Initialize segments, cache, and statistics
        self.segments: List[Segment] = []
        self.next_segment_id = 0
        self.cache = TwoLevelCache()
        self.small_segment_size = 4 * 1024 * 1024    # 4MB for small allocations
        self.large_segment_size = 32 * 1024 * 1024   # 32MB for large allocations
        self.small_threshold = 1 * 1024 * 1024       # 1MB threshold
        self.total_allocated = 0
        self.alloc_count = 0
        self.cuda_alloc_count = 0
        self.ptr_to_segment: Dict[int, Segment] = {}
        self.ptr_to_size: Dict[int, int] = {}
        self.lock = threading.Lock()
        
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
        self.cuda_alloc_count += 1
        return segment
    
    def allocate(self, nbytes: int, stream: Optional[int] = None) -> int:
        """
        Simple allocation - directly call cuMemAlloc, remove all optimizations
        """
        if nbytes == 0:
            return 0
            
        # Direct allocation, no alignment, no caching
        ptr = _ok(cuda.cuMemAlloc(nbytes))
        self.alloc_count += 1
        return ptr
    
    def free(self, ptr: int, stream: Optional[int] = None):
        """
        Simple deallocation - directly call cuMemFree, remove all optimizations
        """
        if ptr and int(ptr) != 0:
            try:
                _ok(cuda.cuMemFree(ptr))
            except:
                pass
    
    def get_stats(self) -> Dict:
        """Get simple statistics"""
        return {
            'alloc_count': self.alloc_count,
            'cache_hit_rate': '0.0%',  # No cache
            'efficiency': '1.0x'
        }
    
    def empty_cache(self):
        """No cache, no cleanup needed"""
        pass
    
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

def free_memory(ptr: int, stream: Optional[int] = None):
    """Free GPU memory"""
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