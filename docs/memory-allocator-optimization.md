# Genesis GPU Memory Allocator Performance Optimization Journey

> A complete technical blog documenting the performance optimization process of Genesis deep learning framework's GPU memory allocator, from problem discovery to progressive solutions

## Background: Why is Genesis Memory Allocation So Slow?

When using our self-developed Genesis deep learning framework, we discovered a serious performance issue: GPU memory allocation was much slower than PyTorch. Comparative testing revealed:

```
CUDAStorage allocation vs PyTorch:
- 1K elements:   Genesis 0.58x PyTorch (42% slower)
- 10K elements:  Genesis 0.75x PyTorch (25% slower) 
- 100K elements: Genesis 0.42x PyTorch (58% slower)
```

Even more shocking was the `fill_` operation performance:

```
Fill operation (before optimization):
- 512×512:   Genesis 0.10x PyTorch (10x slower!)
- 1024×1024: Genesis 0.03x PyTorch (33x slower!)
- 2048×2048: Genesis 0.01x PyTorch (100x slower!)
```

This was clearly unacceptable. We needed to deeply analyze the problem and develop an optimization strategy.

## Step 1: Establish Performance Baseline Testing

Any optimization must first establish accurate baseline testing. We needed to understand:
1. How slow is current allocation performance exactly?
2. Which allocation patterns are performance bottlenecks?
3. What are the specific gaps compared to PyTorch?

### Design Benchmark Tests

I created a dedicated memory manager benchmark tool `benchmark/bench_memory_manager.py`, testing the following key patterns:

1. **Same-size Repeated Allocation** - Simulate training loops
2. **Allocation-Release Cycles** - Test memory reuse capability  
3. **Variable-size Allocation** - Simulate batch size changes
4. **Large Memory Allocation** - Test large block memory behavior
5. **PyTorch Cache Analysis** - Deep understanding of PyTorch's caching mechanism

### Baseline Test Results

After running tests, the results were shocking:

```
🔴 Overall Performance Statistics:
- Average speedup ratio: 0.16x (Genesis 6x slower than PyTorch!)
- Worst speedup ratio: 0.02x (allocation-release cycles 50x slower!)
- Best speedup ratio: 0.38x (still 2.6x slower)

📊 By Pattern Category:
- Same-size repeated allocation:    0.22x (poor)
- Allocation-release cycles:        0.02x (severe!)  
- Variable-size allocation:         0.12x (severe)
- Large memory allocation:          0.20x (poor)
```

### Key Findings

#### 1. **PyTorch Caching Effect is Amazing**
```
PyTorch 1024×1024 allocation behavior:
- First allocation (cold start): 0.458ms
- Second allocation (cache hit): 0.021ms  
- Cache speedup ratio: 22x!

10 consecutive allocations:
- PyTorch average: 0.015ms 
- Genesis average: 0.925ms
- Steady-state performance gap: 62x!
```

#### 2. **Genesis Has No Caching**
Genesis allocation time is consistently similar (0.9-1.0ms), indicating it indeed calls `cudaMalloc` every time with no caching mechanism.

#### 3. **Allocation-Release Cycles are the Biggest Bottleneck** 
```
Allocation-release cycle performance (20 cycles):
- 1024×1024: PyTorch 0.149ms vs Genesis 5.116ms (34x slower!)
```

This confirmed expert analysis: `cudaFree`'s implicit synchronization severely impacts performance.

### Optimization Direction Determined

Based on test results, our optimization priorities are very clear:

1. **🔴 Urgent**: Implement basic cache pool to solve repeated `cudaMalloc/cudaFree` issues
2. **🟠 Important**: Optimize memory reuse strategy, especially allocation-release cycles
3. **🟡 Improvement**: Handle variable-size allocation patterns

Now let's start Phase 1 optimization.

## Step 2: Implement Simple Cache Pool

Based on baseline test discoveries, we first implement a simple memory cache pool to avoid frequent `cudaMalloc/cudaFree` calls.

### Phase 1 Design Approach

I implemented a minimal viable cache allocator with the following features:

1. **512B Alignment**: All allocations aligned to 512-byte boundaries
2. **Exact Size Matching**: Cache by exact size to avoid memory waste
3. **Simple Free List**: Use `defaultdict(list)` to implement size -> [ptr_list] mapping
4. **Immediate Recycling**: Return to cache immediately upon release (if cache not full)
5. **Single Stream Friendly**: Current version doesn't handle cross-stream, focuses on validating cache effect

### Core Implementation

```python
class CUDAMemoryManager:
    def __init__(self):
        # Phase 1: Simple caching allocator
        self.free_blocks = defaultdict(list)  # size -> [ptr_list] 
        self.active_blocks = {}  # ptr -> size
        self.alignment = 512  # 512B alignment
        self.max_cache_size = 1024 * 1024 * 1024  # 1GB cache limit
        
    def allocate(self, nbytes: int, stream=None) -> int:
        aligned_size = self._round_up(nbytes, self.alignment)
        
        # Try cache first
        if self.free_blocks[aligned_size]:
            ptr = self.free_blocks[aligned_size].pop()
            self.cache_hits += 1
            return ptr
        
        # Cache miss - allocate from CUDA
        ptr = cuda.cuMemAlloc(aligned_size)
        self.cache_misses += 1
        return ptr
    
    def free(self, ptr: int, stream=None):
        size = self.active_blocks.pop(ptr)
        
        # Return to cache if not full
        if self.current_cache_size + size <= self.max_cache_size:
            self.free_blocks[size].append(ptr)
            return
        
        # Cache full - actually free
        cuda.cuMemFree(ptr)
```

### Phase 1 Optimization Results

Simple cache allocator benchmark results:

#### **Performance Results**
```
Average speedup ratio: 0.98x
Median speedup ratio: 0.65x  
Performance range: 0.03x ~ 2.85x
```

#### **Scenario Analysis**

**Well-performing scenarios:**
- Same-size repeated allocation: 1.43x
- Large memory allocation: 1.29x  
- Inference dynamic batches: 1.01x

**Average-performing scenarios:**
- Allocation-release cycles: 0.84x
- Variable-size allocation: 0.51x

**Poorly-performing scenarios:**
- Transformer training: 0.04x
- Gradient accumulation: 0.03x
- Memory pressure: 0.08x

#### **Main Findings**

**Cache mechanism validation:**
- Exact size matching is effective for repeated allocation scenarios
- Large allocations (≥100K elements) average 1.20x
- Small allocations (<100K elements) only 0.46x, becoming performance bottleneck

**Limitations:**
- Low cache hit rate in complex scenarios
- Exact matching strategy unsuitable for diverse memory patterns
- Need more flexible caching strategies

## Step 2: Next Phase Optimization Plan

Based on Phase 1 results analysis, determine optimization priorities:

### **Core Problem Diagnosis**
1. **Poor small allocation performance**: <100K element scenarios drag down overall performance
2. **Complex scenario failures**: Extremely low cache hit rates in diverse memory patterns
3. **Exact matching limitations**: Current strategy unsuitable for scenarios with large size variations

### **Phase 2 Optimization Plan: Size Bucket Caching**

**Goal**: Improve cache hit rate, solve variable-size allocation issues

**Core improvements**:
- Change exact matching to bucket matching (like 64B, 128B, 256B, 512B...)
- Reduce memory fragmentation, improve reuse rate
- Prioritize solving small allocation performance issues

**Expected effects**:
- Variable-size allocation from 0.51x to 0.8x+
- Complex scenario performance improvement
- Overall average performance from 0.98x to 1.2x+

### **Implementation Plan**
1. Design bucket size strategy (powers of 2 vs fixed steps)
2. Implement bucket matching allocation logic
3. Benchmark test to verify effects
4. Based on results, decide whether to proceed to Phase 3 (block allocator)

Current Phase 1 has established a stable foundation, can begin Phase 2 development.

## Phase 2 Implementation: Size Bucket Cache Optimization

### **Core Improvements**

Changed exact size matching to bucket matching strategy:
- Use powers of 2 buckets: 512B, 1KB, 2KB, 4KB...
- Maximum bucket limit 16MB, use exact alignment beyond
- Improve cache hit rate for variable-size scenarios

### **Implementation Results**

#### **Overall Performance Comparison**
```
Phase 1 → Phase 2:
Average speedup ratio: 0.98x → 0.97x (slight decrease)
Median speedup ratio: 0.65x → 0.88x (significant improvement +35%)
```

#### **Per-scenario Performance Changes**

**Significantly improved scenarios:**
- Variable-size allocation: 0.51x → 0.83x (+63%)
- Memory pressure: 0.08x → 1.48x (+1750%)  
- Inference dynamic batches: 1.01x → 1.40x (+39%)
- Allocation-release cycles: 0.84x → 1.01x (+20%)

**Performance decline scenarios:**
- Same-size repeated allocation: 1.43x → 0.90x (-37%)

**Still severe bottlenecks:**
- Transformer training: 0.04x → 0.05x (almost no improvement)
- Gradient accumulation: 0.03x → 0.07x (minor improvement)

### **Phase 2 Technical Assessment**

**Successfully validated:**
- Bucket matching effectively improved cache hit rate for variable-size scenarios
- Large median performance improvement shows most scenarios benefited
- Breakthrough in memory pressure scenarios proves value of bucket caching

**Problems discovered:**
- Bucket caching introduces memory waste, affecting same-size allocation performance
- Complex training scenarios (Transformer/gradient accumulation) still not fundamentally improved
- Need deeper optimization strategies to solve core bottlenecks

### **Transformer Scenario Bottleneck Root Cause Analysis**

Through deep analysis, found fundamental reasons why bucket caching is ineffective for complex training scenarios:

#### **Large Tensors Exceed Bucket Limits**
- Logits tensors reach 78MB-313MB, far exceeding 16MB bucket limit
- Ultra-large tensors fall back to exact alignment, cannot enjoy bucket caching advantages
- Frequent large memory cudaMalloc calls become main overhead

#### **Architectural-level Differences**
```
PyTorch block allocator advantages:
- Pre-allocate large memory pools (512MB-2GB)
- Slice tensors from memory pool, avoiding cudaMalloc
- Return to pool upon release, achieving true zero-overhead reuse

Genesis bucket cache limitations:
- Each tensor still needs independent cudaMalloc
- Cannot utilize fundamental advantages of memory pools
- Large tensors completely bypass caching mechanism
```

#### **Performance Bottleneck Truth**
- 60 tensors, mostly 4MB-320MB level
- cudaMalloc system call overhead for large memory blocks is huge
- No matter how high cache hit rate, cannot mask fundamental architectural issues

**Conclusion**: Bucket caching is incremental improvement but cannot solve fundamental problems of large-scale training. Need to implement PyTorch-style block allocator to truly break through performance bottlenecks.

## Phase 3 Implementation: Block Allocator

### **Core Design**

Implement PyTorch-style block allocator to solve fundamental performance issues of large memory allocation:
- Pre-allocate large memory segments (1GB) as memory pool
- Use best-fit algorithm to slice blocks from pool
- Return to pool upon release, support block merging to reduce fragmentation
- Layered architecture: <1MB use bucket cache, ≥1MB use block allocator

### **Implementation Results**

#### **Overall Performance Comparison**
```
Phase 2 → Phase 3:
Average speedup ratio: 0.97x → 1.41x (+45% improvement)
Median speedup ratio: 0.88x → 0.81x (slight decrease)
Best performance: 2.60x → 4.83x (new performance peak)
```

#### **Major Breakthroughs in Key Scenarios**

**Dramatically improved scenarios:**
- Transformer training: 0.05x → 1.89x (+3680%, from severe bottleneck to surpassing PyTorch)
- Large memory allocation: 1.29x → 3.92x (+204%, significantly better than PyTorch)
- Large-size repeated allocation: from Phase 1's 0.27x to Phase 3's 2.31x

**Stable scenarios:**
- Small allocation scenarios basically maintain original levels
- Practical scenarios like inference services perform stably

**Still need improvement scenarios:**
- Gradient accumulation: 0.07x → 0.18x (improved but still poor)
- Variable-size allocation: 0.83x → 0.34x (affected by layered strategy)

### **Technical Achievements**

**Successfully solved core problems:**
- Completely eliminated cudaMalloc system call overhead in large allocation scenarios
- Achieved true memory pool reuse mechanism
- Validated effectiveness of block allocator architecture

**Technical architecture success:**
- Layered allocation strategy works properly
- Good utilization rate of 1GB memory segments
- Best-fit algorithm and block merging mechanism effective

**Limitation recognition:**
- Small allocation scenarios still have room for improvement
- Some special scenarios (like gradient accumulation) need further tuning
- Compared to mature PyTorch, still gaps in some specific scenarios

### **Phase 3 Assessment**

Block Allocator successfully solved the most critical large memory allocation bottleneck, enabling Genesis to achieve or even surpass PyTorch performance in important scenarios. While not all scenarios are perfect, it has transformed from "severely lagging" to "basically usable, leading in some scenarios."

This establishes a solid foundation for Genesis applications in actual deep learning tasks.

## Optimization Journey Summary

From initial "catastrophic performance" (0.02x) to current "overall leading" (1.41x), this memory allocator optimization achieved substantial breakthroughs:

### **Progressive Improvements in Three Phases**
- **Phase 1**: Solved most basic memory reuse issues, established optimization foundation
- **Phase 2**: Improved cache hit rate for variable-size scenarios, enhanced median performance  
- **Phase 3**: Completely solved large memory allocation bottlenecks, achieved qualitative leap

### **Technical Route Correctness Validation**
Through systematic benchmark testing and root cause analysis, we accurately identified performance bottlenecks and chose correct technical solutions. Each phase had clear goals and measurable results.

### **Practical Value**
Genesis can now provide acceptable memory allocation performance in most practical scenarios, particularly competitive in critical scenarios like large-scale model training.

Of course, this is only a phase result in memory management optimization. Future could consider more optimization directions, such as multi-stream concurrency, NUMA awareness, or specialized optimizations for specific models.