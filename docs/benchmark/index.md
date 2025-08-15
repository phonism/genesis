# Genesis Benchmark Reports

Welcome to the Genesis framework performance benchmark reports. This directory contains comprehensive performance analysis comparing Genesis with PyTorch.

## About Genesis Benchmarks

Genesis benchmarks provide detailed performance analysis to help users understand:
- **Operation-level performance**: How individual operations compare to PyTorch
- **End-to-end model performance**: Complete model benchmarking including memory usage
- **Optimization opportunities**: Detailed recommendations for performance improvements

## Benchmark Categories

### 📊 Operations Benchmarks
Detailed performance analysis of individual operations and operation categories:
- **Element-wise operations**: add, subtract, multiply, divide, etc.
- **Activation functions**: ReLU, sigmoid, tanh, etc.
- **Reduction operations**: sum, mean, max, etc.
- **Matrix operations**: matrix multiplication
- **Memory operations**: transpose, reshape, etc.

### 🤖 Model Benchmarks
End-to-end performance analysis of complete models:
- **Qwen Language Models**: Forward/backward pass timing and memory analysis
- **Scalability testing**: Performance across different batch sizes and sequence lengths

## Performance Indicators

We use a standardized performance grading system:

- 🟢 **Excellent (≥90%)**: Genesis performs at 90% or better vs PyTorch
- 🟡 **Good (70-90%)**: Acceptable performance gap
- 🟠 **Fair (50-70%)**: Notable performance gap, optimization recommended
- 🔴 **Poor (20-50%)**: Significant optimization needed
- ❌ **Critical (<20%)**: Major performance issues requiring attention

## How to Run Benchmarks

### Quick Start
```bash
# Navigate to benchmark directory
cd benchmark

# Run all benchmarks (generates reports in docs/benchmark/)
./run.sh
```

### Specific Benchmarks

#### Operations Benchmarks
```bash
# Test all operations (comprehensive)
python bench_ops.py

# Test specific operation category
python bench_ops.py --category element

# Test specific operation
python bench_ops.py --op add

# Fast mode (reduced iterations)
python bench_ops.py --fast
```

#### Model Benchmarks
```bash
# Test Qwen model (if available)
python bench_qwen.py --size 0.5B --batch-size 1,2,4 --seq-len 128,256,512

# Quick test
python bench_qwen.py --size 0.5B --batch-size 1,2 --seq-len 128,256 --fast
```

## Understanding the Reports

### Operations Reports
Each operations benchmark report includes:
- **System Information**: GPU details and theoretical performance limits
- **Performance Summary**: Overall statistics and success rates
- **Category Breakdown**: Performance by operation type
- **Detailed Results**: Individual operation results with speedup and bandwidth
- **Top Performers**: Best and worst performing operations
- **Optimization Recommendations**: Specific improvement suggestions

### Model Reports
Model benchmark reports provide:
- **Model Configuration**: Model size, batch sizes, sequence lengths tested
- **Performance Summary**: Overall speedup and memory efficiency
- **Operation Analysis**: Performance breakdown by model operation
- **Scalability Analysis**: Performance across different input sizes
- **Memory Analysis**: Memory usage patterns and efficiency
- **Optimization Priorities**: Focused recommendations for improvement

## Benchmark Infrastructure

### Reliability Features
- **Statistical outlier detection**: Robust timing measurements
- **Adaptive iterations**: Automatic iteration count adjustment
- **Memory bandwidth analysis**: Theoretical performance comparison
- **Error handling**: Graceful handling of failed operations

### Timing Modes
- **Real timing**: Includes all overheads (realistic user experience)
- **Pure timing**: Minimized overhead (peak computational performance)

## Getting Started

1. **Install Requirements**: Ensure Genesis, PyTorch, and CUDA are properly installed
2. **Run Quick Test**: `cd benchmark && python bench_ops.py --fast`
3. **Generate Reports**: `./run.sh` to run all benchmarks and generate documentation
4. **Review Results**: Check this directory for generated reports

## Continuous Improvement

These benchmarks are designed to:
- Track Genesis performance improvements over time
- Identify optimization opportunities
- Validate new features and optimizations
- Provide transparency in performance characteristics

---

**Note**: To generate fresh benchmark reports, run `./run.sh` from the benchmark directory. This will create timestamped reports with the latest performance data.

For technical details about the benchmark implementation, see the source code in the `benchmark/` directory.