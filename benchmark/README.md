# Genesis Performance Benchmarks

This directory contains comprehensive performance benchmarks comparing Genesis with PyTorch across various operations and scenarios.

## 🎯 Overview

The benchmark suite is designed to:
- **Measure Performance**: Compare Genesis vs PyTorch execution times
- **Identify Bottlenecks**: Find operations that need optimization
- **Track Progress**: Monitor performance improvements over time
- **Validate Optimization**: Ensure optimizations don't break functionality

## 📊 Benchmark Categories

### Core Operations
- **[bench_ops.py](bench_ops.py)** - Element-wise operations (add, mul, div, activations)
- **[bench_matmul.py](bench_matmul.py)** - Matrix multiplication in various configurations

### Neural Network Layers
- **[bench_attention.py](bench_attention.py)** - Multi-head attention mechanisms
- **[bench_softmax.py](bench_softmax.py)** - Softmax operations
- **[bench_layernorm.py](bench_layernorm.py)** - Layer normalization
- **[bench_dropout.py](bench_dropout.py)** - Dropout operations

## 🚀 Quick Start

### Run All Benchmarks
```bash
# Run the complete benchmark suite
./run.sh
```

### Run Individual Benchmarks
```bash
# Set GPU device (optional, defaults to GPU 0)
export CUDA_VISIBLE_DEVICES=0

# Element-wise operations - Full comprehensive benchmark
python bench_ops.py

# Quick test mode (faster for development)
python bench_ops.py --fast

# Test specific operation
python bench_ops.py --op add
python bench_ops.py --op relu

# Test specific tensor size category
python bench_ops.py --size large
python bench_ops.py --size small

# Combined options for targeted testing
python bench_ops.py --op add --size small --fast

# List available options
python bench_ops.py --list-ops
python bench_ops.py --list-sizes

# Other benchmarks (traditional interface)
python bench_matmul.py
python bench_attention.py
python bench_softmax.py
python bench_layernorm.py
python bench_dropout.py
```

## 📈 Understanding Results

### Performance Metrics
Each benchmark reports:
- **Mean Time**: Average execution time over multiple runs
- **Standard Deviation**: Variability in execution times
- **Min/Max Times**: Best and worst case performance
- **Speedup**: Genesis performance relative to PyTorch (>1.0 means faster)

### Result Interpretation
```
Operation            PyTorch         Genesis         Speedup   
------------------------------------------------------------
Addition             0.019±0.001ms   0.111±0.004ms   0.17x
```

- **PyTorch**: 0.019ms ± 0.001ms (very fast, highly optimized)
- **Genesis**: 0.111ms ± 0.004ms (slower due to overhead)
- **Speedup**: 0.17x (Genesis is ~6x slower than PyTorch)

### Performance Categories
- **🟢 Excellent (≥0.8x)**: Close to PyTorch performance
- **🟡 Good (0.5-0.8x)**: Acceptable performance gap
- **🟠 Fair (0.2-0.5x)**: Notable performance gap
- **🔴 Needs Work (<0.2x)**: Significant optimization needed

## 🔍 Benchmark Details

### System Requirements
- **CUDA**: GPU with CUDA support
- **PyTorch**: 2.0+ with CUDA enabled
- **Genesis**: Latest version with GPU backend

### Test Methodology
- **Warmup**: Multiple warmup iterations to stabilize GPU
- **Timing**: Multiple test iterations for statistical accuracy
- **Memory**: CUDA memory pool enabled for optimal allocation
- **Precision**: Tests both float32 and float16 where applicable

### Tensor Sizes Tested
- **Small**: 1024×1024 (development/debugging)
- **Medium**: 2048×2048 (typical training)
- **Large**: 4096×4096 (large model training)
- **Batch**: Various batch dimensions for practical scenarios

## 📊 Current Performance Status

### Matrix Multiplication (4096×4096)
- **PyTorch**: 7.3ms
- **Genesis**: 12.1ms  
- **Efficiency**: 60% ⭐⭐⭐

### Element-wise Operations (4096×4096)
- **Addition**: 5% efficiency 🔴
- **Multiplication**: 8% efficiency 🔴
- **Division**: 6% efficiency 🔴

### Neural Network Layers
- **Attention**: Performance varies by configuration
- **LayerNorm**: Competitive with PyTorch
- **Softmax**: Good performance on large tensors

## 🔧 Optimization Opportunities

### High Priority
1. **Element-wise Operations**: Major optimization needed
2. **Small Tensor Performance**: Poor efficiency on small sizes
3. **Memory Management**: Reduce allocation overhead

### Medium Priority
1. **Batch Operations**: Improve batched computation efficiency
2. **Mixed Precision**: Optimize float16 performance
3. **Kernel Fusion**: Fuse multiple operations

### Low Priority
1. **Edge Cases**: Optimize uncommon tensor shapes
2. **Memory Bandwidth**: Improve memory access patterns

## 🛠️ Development Workflow

### Quick Development Testing
For rapid development and optimization cycles:

```bash
# Quick test of specific operation during development
python bench_ops.py --op add --fast

# Test on small tensors only (faster iteration)
python bench_ops.py --size small --fast

# Full test of optimized operation
python bench_ops.py --op add --size large
```

### Optimization Workflow
1. **Identify Bottleneck**: Run full benchmark to find slow operations
2. **Quick Iteration**: Use `--fast --op <operation>` for rapid testing
3. **Size Analysis**: Test specific sizes with `--size <category>`
4. **Validate**: Run full benchmark to confirm improvements

### Adding New Benchmarks
1. **Create Test File**: Follow naming convention `bench_<operation>.py`
2. **Use Template**: Copy structure from existing benchmarks
3. **Include Shapes**: Test multiple tensor sizes and configurations
4. **Add to Suite**: Update `run.sh` to include new benchmark

### Benchmark Template
```python
#!/usr/bin/env python3
"""
<Operation> Benchmark: Genesis vs PyTorch
"""

import time
import torch
import genesis
from typing import Dict, List

class BenchmarkTimer:
    def __init__(self, warmup_iters=10, test_iters=100):
        self.warmup_iters = warmup_iters
        self.test_iters = test_iters
    
    def __enter__(self):
        # Warmup
        for _ in range(self.warmup_iters):
            torch.cuda.synchronize()
        
        torch.cuda.synchronize()
        self.start_time = time.perf_counter()
        return self
    
    def __exit__(self, *args):
        torch.cuda.synchronize()
        self.elapsed_time = time.perf_counter() - self.start_time

def benchmark_operation(shape):
    """Benchmark specific operation."""
    # Your benchmark code here
    pass

if __name__ == "__main__":
    shapes = [(1024, 1024), (2048, 2048), (4096, 4096)]
    for shape in shapes:
        benchmark_operation(shape)
```

### Performance Testing Best Practices
1. **Consistent Environment**: Same GPU, CUDA version, and system state
2. **Multiple Runs**: Average over many iterations for statistical validity
3. **Memory Warmup**: Ensure CUDA memory pool is properly initialized
4. **Synchronization**: Always synchronize CUDA before timing
5. **Realistic Workloads**: Test shapes and patterns used in real models

## 📝 Interpreting Benchmark Output

### Sample Output Analysis
```
================================================================================
Matrix Multiplication Benchmark (torch.float32)
================================================================================
Size            Implementation       Performance                    Speedup   
------------------------------------------------------------------------
4096x4096       PyTorch              Mean: 7.256ms ± 0.011ms      
                Genesis              Mean: 12.089ms ± 4.141ms     0.60x
```

**Analysis:**
- Genesis takes ~1.67x longer than PyTorch
- High standard deviation (4.141ms) suggests inconsistent performance
- 60% efficiency indicates significant room for improvement

### Red Flags to Watch For
- **High Variance**: Large standard deviations suggest unstable performance
- **Memory Issues**: OOM errors or excessive memory usage
- **Kernel Failures**: CUDA errors or incorrect results
- **Regression**: Performance getting worse over time

## 🎯 Performance Goals

### Short Term (Q1 2025)
- Element-wise operations: >50% efficiency
- Matrix multiplication: >80% efficiency
- Neural network layers: >70% efficiency

### Medium Term (Q2 2025)  
- Overall performance: >75% of PyTorch
- Memory efficiency: <1.5x PyTorch memory usage
- Kernel optimization: Fused operations

### Long Term (Q3 2025)
- Performance parity: >90% of PyTorch
- Specialized optimizations: Better than PyTorch in some cases
- Production ready: Stable performance across all operations

## 🤝 Contributing

When contributing performance improvements:

1. **Benchmark First**: Always run benchmarks before and after changes
2. **Document Changes**: Record what was optimized and expected impact
3. **Verify Correctness**: Ensure optimizations don't break functionality
4. **Test Multiple Sizes**: Verify improvements across tensor sizes
5. **Update Baselines**: Update expected performance numbers

## 📚 Related Documentation

- [Performance Guide](../docs/performance/) - Optimization techniques
- [Architecture Overview](../docs/architecture/) - System design
- [CUDA Backend](../docs/api/ndarray/cuda_tensor.md) - Low-level implementation
- [Triton Kernels](../genesis/ndarray/ndarray_ops_gpu.py) - GPU kernel implementations

---

**Last Updated**: January 2025  
**Genesis Version**: 0.3.0+  
**Benchmark Environment**: NVIDIA A800 GPU, CUDA 12.4, PyTorch 2.6.0