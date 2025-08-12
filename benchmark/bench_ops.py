#!/usr/bin/env python3
"""
Genesis vs PyTorch Element-wise Operations Benchmark

Comprehensive performance testing with two timing modes:

TIMING MODES:
- Real: Per-operation timing (includes CUDA sync overhead) - real user experience
- Pure: Batch timing (minimal overhead) - pure computational performance

FEATURES:
- Multiple tensor sizes and operation types
- Professional metrics: bandwidth, efficiency, performance categorization
- Adaptive iterations and statistical analysis
- Detailed optimization recommendations

Usage:
    python bench_ops.py                    # Full benchmark
    python bench_ops.py --fast             # Quick test 
    python bench_ops.py --op add           # Test specific operation
    python bench_ops.py --size large       # Test specific size category
    python bench_ops.py --dtype float16    # Test with different precision

Options:
    --fast          Quick mode with reduced iterations
    --op OPERATION  Test specific operation: add, sub, mul, div, pow, relu, sigmoid, tanh
    --size SIZE     Test specific size category: small, medium, large, very_large, batch
    --dtype TYPE    Data type: float32, float16, bfloat16
    --list-ops      List available operations
    --list-sizes    List available size categories
"""

import sys
import os
import argparse
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import numpy as np
import torch
import torch.nn.functional as F
import genesis
import genesis.nn.functional as gF
from typing import Dict, List, Callable, Tuple
import gc

# Ensure we're using GPU
assert torch.cuda.is_available(), "CUDA is not available"

class BenchmarkTimer:
    """Professional benchmark timer with comprehensive metrics"""
    
    def __init__(self, warmup_iters=20, test_iters=100):
        self.warmup_iters = warmup_iters
        self.test_iters = test_iters
        self.gpu_properties = torch.cuda.get_device_properties(0)
        self.theoretical_bandwidth_gb_s = self._get_theoretical_bandwidth()
        # Create CUDA events for precise timing
        self.start_event = torch.cuda.Event(enable_timing=True)
        self.end_event = torch.cuda.Event(enable_timing=True)
    
    def _get_theoretical_bandwidth(self) -> float:
        """Get theoretical memory bandwidth for current GPU"""
        gpu_bandwidths = {
            'A100': 1555, 'A800': 1555, 'V100': 900, 'RTX 4090': 1008,
            'RTX 3090': 936, 'Tesla T4': 320, 'RTX 3080': 760
        }
        gpu_name = self.gpu_properties.name
        for gpu_type, bandwidth in gpu_bandwidths.items():
            if gpu_type in gpu_name:
                return bandwidth
        return 500  # Conservative default
    
    def calculate_flops(self, operation: str, tensor_size: int) -> float:
        """Calculate FLOPs for operations"""
        flop_counts = {
            'add': tensor_size, 'sub': tensor_size, 'mul': tensor_size,
            'div': tensor_size, 'pow': tensor_size * 2,
            'relu': tensor_size, 'sigmoid': tensor_size * 4,
            'tanh': tensor_size * 6
        }
        return flop_counts.get(operation.lower(), tensor_size)
    
    def calculate_memory_bandwidth(self, operation: str, tensor_sizes: List[int], 
                                 time_ms: float, dtype_bytes: int = 4) -> float:
        """Calculate memory bandwidth in GB/s with proper read/write modeling"""
        if not tensor_sizes or time_ms <= 0:
            return 0
        
        # Define read/write patterns for each operation
        op_patterns = {
            'add': (2, 1),     # 2 reads (A, B), 1 write (C)
            'sub': (2, 1),     # 2 reads (A, B), 1 write (C)
            'mul': (2, 1),     # 2 reads (A, B), 1 write (C)
            'div': (2, 1),     # 2 reads (A, B), 1 write (C)
            'pow': (1, 1),     # 1 read (A), 1 write (C)
            'relu': (1, 1),    # 1 read (A), 1 write (C)
            'sigmoid': (1, 1), # 1 read (A), 1 write (C)
            'tanh': (1, 1),    # 1 read (A), 1 write (C)
            'silu': (1, 1),    # 1 read (A), 1 write (C)
        }
        
        reads, writes = op_patterns.get(operation.lower(), (1, 1))
        
        # Calculate total bytes considering broadcasting
        if len(tensor_sizes) > 1 and tensor_sizes[0] != tensor_sizes[1]:
            # Broadcasting case - use actual tensor sizes
            total_bytes = sum(size * dtype_bytes for size in tensor_sizes) + tensor_sizes[-1] * dtype_bytes
        else:
            # Normal case
            tensor_size = tensor_sizes[0] if tensor_sizes else 0
            total_bytes = (reads + writes) * tensor_size * dtype_bytes
        
        return (total_bytes / 1e9) / (time_ms / 1000)
    
    def _get_iterations(self, tensor_size: int) -> tuple:
        """Get iteration counts - simplified logic"""
        return self.warmup_iters, self.test_iters
    
    def benchmark(self, fn, operation: str = "unknown", 
                 tensor_sizes: List[int] = None, *args, **kwargs):
        """Run benchmark with CUDA events for precise timing"""
        tensor_sizes = tensor_sizes or [0]
        
        # Warmup
        for _ in range(self.warmup_iters):
            try:
                _ = fn(*args, **kwargs)
            except Exception:
                # Silent failure during warmup
                break
        
        # Single sync after warmup
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        # Actual timing using CUDA events
        times = []
        for i in range(self.test_iters):
            try:
                self.start_event.record()
                result = fn(*args, **kwargs)
                self.end_event.record()
                
                # Wait for completion and get elapsed time
                torch.cuda.synchronize()
                elapsed_ms = self.start_event.elapsed_time(self.end_event)
                times.append(elapsed_ms)
            except Exception as e:
                # Print error details for the first few failures
                if i < 3:
                    print(f"Warning: Benchmark iteration {i} failed: {str(e)}")
                break
        
        if not times:
            return {'mean': float('inf'), 'std': 0, 'min': float('inf'),
                   'max': float('inf'), 'median': float('inf'),
                   'gflops': 0, 'bandwidth_gb_s': 0}
        
        mean_time = np.mean(times)
        tensor_size = tensor_sizes[0] if tensor_sizes else 0
        
        # Calculate professional metrics
        flops = self.calculate_flops(operation, tensor_size)
        gflops = (flops / 1e9) / (mean_time / 1000) if mean_time > 0 else 0
        bandwidth = self.calculate_memory_bandwidth(operation, tensor_sizes, mean_time, 4)  # Default to float32
        
        return {
            'mean': mean_time,
            'std': np.std(times),
            'min': np.min(times),
            'max': np.max(times),
            'median': np.median(times),
            'gflops': gflops,
            'bandwidth_gb_s': bandwidth
        }

    def benchmark_pure_compute(self, fn, operation: str = "unknown", 
                             tensor_sizes: List[int] = None, *args, **kwargs):
        """Benchmark pure computational performance (batch timing with CUDA events)"""
        tensor_sizes = tensor_sizes or [0]
        
        tensor_size = tensor_sizes[0] if tensor_sizes else 0
        
        # Get iteration counts
        warmup_iters, test_iters = self._get_iterations(tensor_size)
        
        # Warmup to eliminate compilation overhead
        for _ in range(warmup_iters):
            try:
                _ = fn(*args, **kwargs)
            except Exception:
                # Silent failure during warmup
                break
        
        # Single sync after warmup
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        # Batch timing with CUDA events
        self.start_event.record()
        for _ in range(test_iters):
            try:
                result = fn(*args, **kwargs)
            except Exception:
                break
        self.end_event.record()
        
        # Wait and calculate average time per operation
        torch.cuda.synchronize()
        total_time_ms = self.start_event.elapsed_time(self.end_event)
        mean_time = total_time_ms / test_iters
        
        # Calculate metrics
        tensor_size = tensor_sizes[0] if tensor_sizes else 0
        flops = self.calculate_flops(operation, tensor_size)
        gflops = (flops / 1e9) / (mean_time / 1000) if mean_time > 0 else 0
        bandwidth = self.calculate_memory_bandwidth(operation, tensor_sizes, mean_time, 4)  # Default to float32
        
        return {
            'mean': mean_time,
            'std': 0,  # No std for batch timing
            'min': mean_time,
            'max': mean_time,
            'median': mean_time,
            'gflops': gflops,
            'bandwidth_gb_s': bandwidth
        }


def get_comprehensive_shapes() -> Dict[str, List[Tuple[int, ...]]]:
    """Get comprehensive test shapes categorized by size"""
    return {
        'small': [
            (256, 256),          # 64K elements
            (512, 512),          # 256K elements  
            (1024, 1024),        # 1M elements
            (32, 32, 32),        # 32K elements (3D)
            (64, 64, 64),        # 256K elements (3D)
        ],
        'medium': [
            (2048, 2048),        # 4M elements
            (4096, 1024),        # 4M elements (non-square)
            (1024, 4096),        # 4M elements (non-square)
            (128, 128, 128),     # 2M elements (3D)
            (32, 256, 256),      # 2M elements (3D)
        ],
        'large': [
            (4096, 4096),        # 16M elements
            (8192, 2048),        # 16M elements (non-square)
            (2048, 8192),        # 16M elements (non-square)
            (256, 256, 256),     # 16M elements (3D)
            (64, 512, 512),      # 16M elements (3D)
        ],
        'very_large': [
            (8192, 8192),        # 64M elements
            (16384, 4096),       # 64M elements (non-square)
            (512, 512, 512),     # 128M elements (3D)
        ],
        'batch': [
            (32, 128, 768),      # Transformer-like (BERT base)
            (64, 256, 256),      # CNN-like
            (128, 512, 512),     # Large CNN-like
            (16, 1024, 1024),    # Large batch
        ]
    }

def format_size(size: int) -> str:
    """Format tensor size in human readable form"""
    if size >= 1e9:
        return f"{size/1e9:.1f}B"
    elif size >= 1e6:
        return f"{size/1e6:.1f}M"
    elif size >= 1e3:
        return f"{size/1e3:.1f}K"
    else:
        return str(size)

def categorize_performance(efficiency: float) -> Tuple[str, str]:
    """Categorize performance level"""
    if efficiency >= 0.9:
        return "🟢 EXCELLENT", "Competitive with PyTorch"
    elif efficiency >= 0.7:
        return "🟡 GOOD", "Acceptable performance gap"  
    elif efficiency >= 0.5:
        return "🟠 FAIR", "Notable performance gap"
    elif efficiency >= 0.2:
        return "🔴 POOR", "Significant optimization needed"
    else:
        return "❌ CRITICAL", "Major performance issues"

def format_results(results: Dict[str, float]) -> str:
    """Format benchmark results"""
    return f"{results['mean']:.3f}±{results['std']:.3f}ms"

def print_professional_header():
    """Print professional result table header"""
    print(f"{'Shape':<15} {'Size':<8} {'PyTorch':<10} {'Genesis(Real)':<12} {'Genesis(Pure)':<12} "
          f"{'Real Speedup':<12} {'Pure Speedup':<12} {'Efficiency':<12} {'Status':<15}")
    print("-" * 140)

def print_professional_row(shape: Tuple[int, ...], pytorch_result: Dict, 
                          genesis_real_result: Dict, genesis_pure_result: Dict, operation: str):
    """Print formatted result row with dual performance metrics"""
    real_speedup = pytorch_result['mean'] / genesis_real_result['mean'] if genesis_real_result['mean'] > 0 else 0
    pure_speedup = pytorch_result['mean'] / genesis_pure_result['mean'] if genesis_pure_result['mean'] > 0 else 0
    
    # Calculate relative efficiency vs PyTorch 
    relative_efficiency = real_speedup * 100  # Convert speedup to percentage
    
    status, _ = categorize_performance(real_speedup)  # Use real performance for status
    
    shape_str = "×".join(map(str, shape))
    if len(shape_str) > 14:
        shape_str = shape_str[:11] + "..."
    
    tensor_size = np.prod(shape)
    
    print(f"{shape_str:<15} {format_size(tensor_size):<8} "
          f"{pytorch_result['mean']:.3f}ms{'':<2} {genesis_real_result['mean']:.3f}ms{'':<4} {genesis_pure_result['mean']:.3f}ms{'':<4} "
          f"{real_speedup:.2f}x{'':<7} {pure_speedup:.2f}x{'':<7} "
          f"{relative_efficiency:.1f}%{'':<8} {status:<15}")


def benchmark_activation_functions(shapes: List[Tuple[int, ...]], dtype=torch.float32):
    """Benchmark activation functions"""
    print(f"\n{'='*80}")
    print(f"Activation Functions Benchmark ({dtype})")
    print(f"{'='*80}")
    
    timer = BenchmarkTimer(warmup_iters=10, test_iters=100)
    
    activations = [
        ("ReLU", F.relu, gF.relu),
        ("Sigmoid", torch.sigmoid, lambda x: x.sigmoid()),
        ("Tanh", torch.tanh, lambda x: x.tanh()),
        ("SiLU", F.silu, gF.silu),
    ]
    
    for shape in shapes:
        print(f"\nShape: {shape}")
        print(f"{'Activation':<20} {'PyTorch':<15} {'Genesis':<15} {'Speedup':<10}")
        print(f"{'-'*60}")
        
        # Create test data
        np_x = np.random.randn(*shape).astype(np.float32 if dtype == torch.float32 else np.float16)
        
        torch_x = torch.from_numpy(np_x).cuda()
        genesis_x = genesis.tensor(np_x, device=genesis.cuda())
        
        for act_name, torch_act, genesis_act in activations:
            torch_results = timer.benchmark(torch_act, torch_x)
            genesis_real_results = timer.benchmark(genesis_act, genesis_x)
            
            speedup = torch_results['mean'] / genesis_real_results['mean']
            
            print(f"{act_name:<20} {format_results(torch_results):<15} "
                  f"{format_results(genesis_real_results):<15} {speedup:.2f}x")
        
        # Clean up
        del torch_x, genesis_x
        gc.collect()
        torch.cuda.empty_cache()

def benchmark_reduction_ops(shapes: List[Tuple[int, ...]], dtype=torch.float32):
    """Benchmark reduction operations"""
    print(f"\n{'='*80}")
    print(f"Reduction Operations Benchmark ({dtype})")
    print(f"{'='*80}")
    
    timer = BenchmarkTimer(warmup_iters=10, test_iters=100)
    
    for shape in shapes:
        print(f"\nShape: {shape}")
        print(f"{'Operation':<25} {'PyTorch':<15} {'Genesis':<15} {'Speedup':<10}")
        print(f"{'-'*65}")
        
        # Create test data
        np_x = np.random.randn(*shape).astype(np.float32 if dtype == torch.float32 else np.float16)
        
        torch_x = torch.from_numpy(np_x).cuda()
        genesis_x = genesis.tensor(np_x, device=genesis.cuda())
        
        # Test different reduction operations
        reductions = [
            ("Sum (all)", lambda x: x.sum(), lambda x: x.sum()),
            ("Mean (all)", lambda x: x.mean(), lambda x: x.mean()),
            ("Max (all)", lambda x: x.max(), lambda x: x.max()),
        ]
        
        # Add axis-specific reductions for multi-dimensional tensors
        if len(shape) > 1:
            reductions.extend([
                ("Sum (axis=0)", lambda x: x.sum(dim=0), lambda x: x.sum(axis=0)),
                ("Sum (axis=-1)", lambda x: x.sum(dim=-1), lambda x: x.sum(axis=-1)),
                ("Mean (axis=0)", lambda x: x.mean(dim=0), lambda x: x.mean(axis=0)),
                ("Max (axis=-1)", lambda x: x.max(dim=-1)[0], lambda x: x.max(axis=-1)),
            ])
        
        for op_name, torch_op, genesis_op in reductions:
            torch_results = timer.benchmark(torch_op, torch_x)
            genesis_real_results = timer.benchmark(genesis_op, genesis_x)
            
            speedup = torch_results['mean'] / genesis_real_results['mean']
            
            print(f"{op_name:<25} {format_results(torch_results):<15} "
                  f"{format_results(genesis_real_results):<15} {speedup:.2f}x")
        
        # Clean up
        del torch_x, genesis_x
        gc.collect()
        torch.cuda.empty_cache()

def benchmark_memory_ops(shapes: List[Tuple[int, ...]], dtype=torch.float32):
    """Benchmark memory operations (transpose, reshape, etc.)"""
    print(f"\n{'='*80}")
    print(f"Memory Operations Benchmark ({dtype})")
    print(f"{'='*80}")
    
    timer = BenchmarkTimer(warmup_iters=10, test_iters=100)
    
    for shape in shapes:
        if len(shape) < 2:
            continue  # Skip 1D tensors for transpose
            
        print(f"\nShape: {shape}")
        print(f"{'Operation':<25} {'PyTorch':<15} {'Genesis':<15} {'Speedup':<10}")
        print(f"{'-'*65}")
        
        # Create test data
        np_x = np.random.randn(*shape).astype(np.float32 if dtype == torch.float32 else np.float16)
        
        torch_x = torch.from_numpy(np_x).cuda()
        genesis_x = genesis.tensor(np_x, device=genesis.cuda())
        
        # Test memory operations
        operations = [
            ("Transpose", 
             lambda x: x.transpose(-2, -1), 
             lambda x: x.transpose()),
            ("Reshape (flatten)", 
             lambda x: x.reshape(-1), 
             lambda x: x.reshape(-1)),
            ("Contiguous", 
             lambda x: x.contiguous(), 
             lambda x: x.contiguous()),
        ]
        
        # Add permute for 3D+ tensors
        if len(shape) >= 3:
            perm = list(range(len(shape)))
            perm[-2], perm[-1] = perm[-1], perm[-2]
            operations.append(
                ("Permute", 
                 lambda x: x.permute(*perm), 
                 lambda x: x.permute(*perm))
            )
        
        for op_name, torch_op, genesis_op in operations:
            torch_results = timer.benchmark(torch_op, torch_x)
            genesis_real_results = timer.benchmark(genesis_op, genesis_x)
            
            speedup = torch_results['mean'] / genesis_real_results['mean']
            
            print(f"{op_name:<25} {format_results(torch_results):<15} "
                  f"{format_results(genesis_real_results):<15} {speedup:.2f}x")
        
        # Clean up
        del torch_x, genesis_x
        gc.collect()
        torch.cuda.empty_cache()

def benchmark_broadcast_ops(dtype=torch.float32):
    """Benchmark broadcasting operations"""
    print(f"\n{'='*80}")
    print(f"Broadcasting Operations Benchmark ({dtype})")
    print(f"{'='*80}")
    
    timer = BenchmarkTimer(warmup_iters=10, test_iters=100)
    
    broadcast_cases = [
        ((1000, 1), (1000, 100), "Vector-Matrix broadcast"),
        ((100, 100, 1), (100, 100, 100), "3D broadcast"),
        ((32, 1, 768), (32, 128, 768), "Batch broadcast"),
        ((1, 512, 512), (64, 512, 512), "Batch expansion"),
    ]
    
    print(f"{'Case':<30} {'Shape A':<15} {'Shape B':<15} {'PyTorch':<12} {'Genesis':<12} {'Speedup':<10}")
    print(f"{'-'*94}")
    
    for shape_a, shape_b, description in broadcast_cases:
        # Create test data
        np_a = np.random.randn(*shape_a).astype(np.float32 if dtype == torch.float32 else np.float16)
        np_b = np.random.randn(*shape_b).astype(np.float32 if dtype == torch.float32 else np.float16)
        
        torch_a = torch.from_numpy(np_a).cuda()
        torch_b = torch.from_numpy(np_b).cuda()
        genesis_a = genesis.tensor(np_a, device=genesis.cuda())
        genesis_b = genesis.tensor(np_b, device=genesis.cuda())
        
        # Benchmark addition with broadcasting
        torch_results = timer.benchmark(lambda: torch_a + torch_b)
        genesis_real_results = timer.benchmark(lambda: genesis_a + genesis_b)
        
        speedup = torch_results['mean'] / genesis_real_results['mean']
        
        print(f"{description:<30} {str(shape_a):<15} {str(shape_b):<15} "
              f"{format_results(torch_results):<12} {format_results(genesis_real_results):<12} {speedup:.2f}x")
        
        # Clean up
        del torch_a, torch_b, genesis_a, genesis_b
        gc.collect()
        torch.cuda.empty_cache()

def print_summary(results: Dict[str, float]):
    """Print benchmark summary"""
    print(f"\n{'='*80}")
    print("Performance Summary")
    print(f"{'='*80}")
    
    # Calculate overall statistics
    speedups = list(results.values())
    avg_speedup = np.mean(speedups)
    median_speedup = np.median(speedups)
    
    print(f"Average Speedup: {avg_speedup:.2f}x")
    print(f"Median Speedup: {median_speedup:.2f}x")
    print(f"Best Speedup: {max(speedups):.2f}x")
    print(f"Worst Speedup: {min(speedups):.2f}x")

def generate_professional_summary(arithmetic_results: Dict, activation_results: Dict,
                                timer: BenchmarkTimer):
    """Generate comprehensive professional summary"""
    print(f"\n{'='*120}")
    print("COMPREHENSIVE PERFORMANCE SUMMARY")
    print(f"{'='*120}")
    
    # System information
    gpu_props = timer.gpu_properties
    print(f"\n📊 SYSTEM INFORMATION")
    print(f"GPU: {gpu_props.name}")
    print(f"Memory: {gpu_props.total_memory / 1024**3:.1f} GB")
    print(f"Theoretical Bandwidth: {timer.theoretical_bandwidth_gb_s:.0f} GB/s")
    print(f"Multi-processors: {gpu_props.multi_processor_count}")
    
    # Collect all efficiency data
    all_efficiencies = []
    category_stats = {}
    
    for result_dict in [arithmetic_results, activation_results]:
        for category, results in result_dict.items():
            if category not in category_stats:
                category_stats[category] = []
            
            for result in results:
                eff = result['efficiency']
                if 0 < eff < float('inf'):
                    all_efficiencies.append(eff)
                    category_stats[category].append(eff)
    
    # Overall statistics
    if all_efficiencies:
        print(f"\n📈 OVERALL PERFORMANCE STATISTICS")
        print(f"Average Efficiency: {np.mean(all_efficiencies)*100:.1f}%")
        print(f"Median Efficiency: {np.median(all_efficiencies)*100:.1f}%")
        print(f"Best Performance: {np.max(all_efficiencies)*100:.1f}%")
        print(f"Worst Performance: {np.min(all_efficiencies)*100:.1f}%")
        print(f"Standard Deviation: {np.std(all_efficiencies)*100:.1f}%")
        print(f"Tests Completed: {len(all_efficiencies)}")
    
    # Category breakdown
    print(f"\n📊 PERFORMANCE BY TENSOR SIZE")
    print(f"{'Category':<15} {'Avg Efficiency':<15} {'Tests':<8} {'Status':<15}")
    print("-" * 65)
    
    for category, efficiencies in category_stats.items():
        if efficiencies:
            avg_eff = np.mean(efficiencies)
            status, _ = categorize_performance(avg_eff)
            print(f"{category.capitalize():<15} {avg_eff*100:.1f}%{'':<10} {len(efficiencies):<8} {status:<15}")
    
    # Performance distribution
    if all_efficiencies:
        print(f"\n🎯 PERFORMANCE DISTRIBUTION")
        excellent = sum(1 for e in all_efficiencies if e >= 0.9)
        good = sum(1 for e in all_efficiencies if 0.7 <= e < 0.9)
        fair = sum(1 for e in all_efficiencies if 0.5 <= e < 0.7)
        poor = sum(1 for e in all_efficiencies if 0.2 <= e < 0.5)
        critical = sum(1 for e in all_efficiencies if e < 0.2)
        total = len(all_efficiencies)
        
        print(f"🟢 Excellent (≥90%): {excellent:3d} ({excellent/total*100:4.1f}%)")
        print(f"🟡 Good (70-90%):     {good:3d} ({good/total*100:4.1f}%)")
        print(f"🟠 Fair (50-70%):     {fair:3d} ({fair/total*100:4.1f}%)")
        print(f"🔴 Poor (20-50%):     {poor:3d} ({poor/total*100:4.1f}%)")
        print(f"❌ Critical (<20%):   {critical:3d} ({critical/total*100:4.1f}%)")
    
    # Optimization recommendations
    print(f"\n🔧 OPTIMIZATION RECOMMENDATIONS")
    print("HIGH PRIORITY:")
    
    if category_stats.get('small', []):
        small_avg = np.mean(category_stats['small'])
        if small_avg < 0.5:
            print("• Small tensor performance needs major optimization")
    
    if category_stats.get('large', []):
        large_avg = np.mean(category_stats['large'])
        if large_avg < 0.7:
            print("• Large tensor kernels need optimization")
    
    print("• Element-wise operations show significant overhead")
    print("• Consider kernel fusion for multiple operations")
    print("• Memory allocation overhead reduction needed")
    
    print("\nMEDIUM PRIORITY:")
    print("• Improve small batch performance")
    print("• Optimize non-contiguous tensor operations")
    print("• Better GPU memory management")
    
    # Framework readiness assessment
    avg_performance = np.mean(all_efficiencies) if all_efficiencies else 0
    readiness = "Production Ready" if avg_performance > 0.8 else \
               "Beta Quality" if avg_performance > 0.6 else \
               "Development Stage"
    
    print(f"\n✅ BENCHMARK COMPLETED")
    print(f"Total operations tested: {len(all_efficiencies)}")
    print(f"Framework readiness: {readiness}")
    print(f"Overall efficiency: {avg_performance*100:.1f}%")

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Professional Element-wise Operations Benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python bench_ops.py                    # Full comprehensive benchmark  
    python bench_ops.py --fast             # Quick test mode
    python bench_ops.py --op add           # Test only addition operation
    python bench_ops.py --size large       # Test only large tensors
    python bench_ops.py --op add --size small --fast  # Quick add test on small tensors
        """
    )
    
    parser.add_argument("--fast", action="store_true", 
                       help="Quick mode with reduced iterations and fewer shapes")
    parser.add_argument("--op", type=str, choices=[
        "add", "sub", "mul", "div", "pow", "relu", "sigmoid", "tanh"
    ], help="Test specific operation only")
    parser.add_argument("--size", type=str, choices=[
        "small", "medium", "large", "very_large", "batch"
    ], help="Test specific size category only")
    parser.add_argument("--list-ops", action="store_true",
                       help="List available operations and exit")
    parser.add_argument("--list-sizes", action="store_true", 
                       help="List available size categories and exit")
    
    return parser.parse_args()

def list_operations():
    """List all available operations"""
    operations = {
        "Arithmetic Operations": ["add", "sub", "mul", "div", "pow"],
        "Activation Functions": ["relu", "sigmoid", "tanh"]
    }
    
    print("Available Operations:")
    print("=" * 50)
    for category, ops in operations.items():
        print(f"\n{category}:")
        for op in ops:
            print(f"  - {op}")

def list_size_categories():
    """List all available size categories"""
    shapes_dict = get_comprehensive_shapes()
    
    print("Available Size Categories:")
    print("=" * 50)
    for category, shapes in shapes_dict.items():
        print(f"\n{category.upper()}:")
        for shape in shapes:
            size = np.prod(shape)
            print(f"  - {shape} ({format_size(size)} elements)")

def filter_operations(all_operations, target_op):
    """Filter operations based on target operation"""
    if not target_op:
        return all_operations
    
    filtered = []
    for op_name, op_key, torch_op, genesis_op in all_operations:
        if op_key.lower() == target_op.lower():
            filtered.append((op_name, op_key, torch_op, genesis_op))
    
    return filtered

def get_benchmark_config(args):
    """Get benchmark configuration based on arguments"""
    if args.fast:
        warmup_iters = 5
        test_iters = 20
        max_shapes_per_category = 2
    else:
        warmup_iters = 20
        test_iters = 100
        max_shapes_per_category = None
    
    # Convert dtype string to torch dtype
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16
    }
    
    return {
        "warmup_iters": warmup_iters,
        "test_iters": test_iters,
        "max_shapes_per_category": max_shapes_per_category,
        "dtype": dtype_map[args.dtype],
        "dtype_bytes": 4 if args.dtype == "float32" else 2
    }

def filter_shapes(shapes_dict, target_size, max_shapes=None):
    """Filter shapes based on target size category"""
    if target_size:
        if target_size in shapes_dict:
            filtered = {target_size: shapes_dict[target_size]}
        else:
            filtered = {}
    else:
        filtered = shapes_dict.copy()
    
    # Limit shapes per category if specified
    if max_shapes:
        for category in filtered:
            filtered[category] = filtered[category][:max_shapes]
    
    return filtered

def main():
    """Run professional comprehensive benchmarks"""
    args = parse_args()
    
    # Handle list operations
    if args.list_ops:
        list_operations()
        return
    
    if args.list_sizes:
        list_size_categories()
        return
    
    # Print configuration
    mode = "FAST" if args.fast else "COMPREHENSIVE"
    print(f"🚀 Genesis Professional Element-wise Operations Benchmark ({mode})")
    print("=" * 80)
    
    if args.op:
        print(f"🎯 Testing operation: {args.op.upper()}")
    if args.size:
        print(f"📏 Testing size category: {args.size.upper()}")
    if args.fast:
        print("⚡ Fast mode: Reduced iterations and shapes")
    
    # Add default dtype if not present
    if not hasattr(args, 'dtype'):
        args.dtype = 'float32'
    
    # Get benchmark configuration
    config = get_benchmark_config(args)
    timer = BenchmarkTimer(
        warmup_iters=config["warmup_iters"], 
        test_iters=config["test_iters"]
    )
    
    # Get and filter test shapes
    shapes_dict = get_comprehensive_shapes()
    shapes_dict = filter_shapes(shapes_dict, args.size, config["max_shapes_per_category"])
    
    if not shapes_dict:
        print(f"❌ No shapes found for size category: {args.size}")
        return
    
    # Run benchmarks
    try:
        arithmetic_results = benchmark_arithmetic_ops_professional(
            shapes_dict, dtype=config["dtype"], target_op=args.op, timer=timer
        )
    except Exception as e:
        print(f"⚠️  Arithmetic benchmark failed: {e}")
        arithmetic_results = {}
    
    # Only run activation benchmarks if no specific operation is requested
    # or if the requested operation is an activation function
    activation_functions = ["relu", "sigmoid", "tanh"]
    if not args.op or args.op in activation_functions:
        try:
            # For activation functions, just use one representative shape per category
            test_shapes = {}
            for category, shapes in shapes_dict.items():
                test_shapes[category] = shapes[:1] if shapes else []
            
            activation_results = benchmark_activation_functions_professional(
                test_shapes, dtype=config["dtype"], target_op=args.op, timer=timer
            )
        except Exception as e:
            print(f"⚠️  Activation benchmark failed: {e}")
            activation_results = {}
    else:
        activation_results = {}
    
    # Generate summary
    if arithmetic_results or activation_results:
        generate_professional_summary(arithmetic_results, activation_results, timer)
    else:
        print("\n⚠️  No results available")
        print("✅ Benchmark completed!")

# Add the new professional arithmetic benchmark function
def benchmark_arithmetic_ops_professional(shapes_dict: Dict[str, List[Tuple[int, ...]]],
                                         dtype=torch.float32, target_op=None, timer=None) -> Dict[str, List[Dict]]:
    """Professional benchmark for arithmetic operations"""
    print(f"\n{'='*120}")
    print(f"ARITHMETIC OPERATIONS BENCHMARK ({dtype})")
    print(f"{'='*120}")
    
    if timer is None:
        timer = BenchmarkTimer(warmup_iters=20, test_iters=100)
    operations = [
        ("Addition", "add", lambda x, y: x + y, lambda x, y: x + y),
        ("Subtraction", "sub", lambda x, y: x - y, lambda x, y: x - y),
        ("Multiplication", "mul", lambda x, y: x * y, lambda x, y: x * y),
        ("Division", "div", lambda x, y: x / y, lambda x, y: x / y),
        ("Power", "pow", lambda x: x ** 2, lambda x: x ** 2),
    ]
    
    # Filter operations if target_op is specified
    if target_op:
        operations = filter_operations(operations, target_op)
        if not operations:
            print(f"❌ Operation '{target_op}' not found in arithmetic operations")
            return {}
    
    all_results = {}
    
    for category, shapes in shapes_dict.items():
        print(f"\n{category.upper()} TENSORS")
        print("-" * 50)
        
        category_results = []
        
        for op_name, op_key, torch_op, genesis_op in operations:
            print(f"\n{op_name} Operation:")
            print_professional_header()
            
            for shape in shapes:
                try:
                    # Create test data
                    np_a = np.random.randn(*shape).astype(np.float32)
                    np_b = np.random.randn(*shape).astype(np.float32)
                    
                    torch_a = torch.from_numpy(np_a).cuda()
                    torch_b = torch.from_numpy(np_b).cuda() if 'pow' not in op_key else None
                    genesis_a = genesis.tensor(np_a, device=genesis.cuda())
                    genesis_b = genesis.tensor(np_b, device=genesis.cuda()) if 'pow' not in op_key else None
                    
                    tensor_size = np.prod(shape)
                    tensor_sizes = [tensor_size] if 'pow' in op_key else [tensor_size, tensor_size]
                    
                    # Calculate dtype bytes for bandwidth calculation
                    dtype_bytes = 4 if dtype == torch.float32 else 2
                    
                    # Benchmark
                    if 'pow' in op_key:
                        pytorch_result = timer.benchmark(torch_op, op_key, tensor_sizes, torch_a)
                        genesis_real_result = timer.benchmark(genesis_op, op_key, tensor_sizes, genesis_a)
                        genesis_pure_result = timer.benchmark_pure_compute(genesis_op, op_key, tensor_sizes, genesis_a)
                    else:
                        pytorch_result = timer.benchmark(torch_op, op_key, tensor_sizes, torch_a, torch_b)
                        genesis_real_result = timer.benchmark(genesis_op, op_key, tensor_sizes, genesis_a, genesis_b)
                        genesis_pure_result = timer.benchmark_pure_compute(genesis_op, op_key, tensor_sizes, genesis_a, genesis_b)
                    
                    # Print result
                    print_professional_row(shape, pytorch_result, genesis_real_result, genesis_pure_result, op_key)
                    
                    # Store for summary
                    speedup = pytorch_result['mean'] / genesis_real_result['mean'] if genesis_real_result['mean'] > 0 else 0
                    category_results.append({
                        'operation': op_name,
                        'shape': shape,
                        'efficiency': speedup,  # Keep as 'efficiency' for backward compatibility but it's actually speedup
                        'pytorch_time': pytorch_result['mean'],
                        'genesis_time': genesis_real_result['mean'],
                        'gflops': genesis_real_result['gflops'],
                        'bandwidth': genesis_real_result['bandwidth_gb_s']
                    })
                    
                    # Clean up
                    del torch_a, genesis_a
                    if torch_b is not None:
                        del torch_b, genesis_b
                        
                except Exception as e:
                    print(f"{'×'.join(map(str, shape)):<15} ERROR: {str(e)[:50]}")
                    continue
            
            gc.collect()
            torch.cuda.empty_cache()
        
        all_results[category] = category_results
    
    return all_results

def benchmark_activation_functions_professional(shapes_dict: Dict[str, List[Tuple[int, ...]]],
                                               dtype=torch.float32, target_op=None, timer=None) -> Dict[str, List[Dict]]:
    """Professional benchmark for activation functions"""
    print(f"\n{'='*120}")
    print(f"ACTIVATION FUNCTIONS BENCHMARK ({dtype})")
    print(f"{'='*120}")
    
    if timer is None:
        timer = BenchmarkTimer(warmup_iters=20, test_iters=100)
    
    operations = [
        ("ReLU", "relu", F.relu, gF.relu),
        ("Sigmoid", "sigmoid", torch.sigmoid, lambda x: x.sigmoid()),
        ("Tanh", "tanh", torch.tanh, lambda x: x.tanh()),
    ]
    
    # Add SiLU if available in genesis
    try:
        operations.append(("SiLU", "silu", F.silu, gF.silu))
    except AttributeError:
        pass  # Skip SiLU if not available
    
    # Filter operations if target_op is specified
    if target_op:
        operations = filter_operations(operations, target_op)
        if not operations:
            print(f"❌ Operation '{target_op}' not found in activation functions")
            return {}
    
    all_results = {}
    
    for category, shapes in shapes_dict.items():
        print(f"\n{category.upper()} TENSORS")
        print("-" * 50)
        
        category_results = []
        
        for op_name, op_key, torch_op, genesis_op in operations:
            print(f"\n{op_name} Activation:")
            print_professional_header()
            
            for shape in shapes:
                try:
                    # Create test data with appropriate dtype
                    if dtype == torch.float32:
                        np_dtype = np.float32
                    elif dtype == torch.float16:
                        np_dtype = np.float16
                    else:  # bfloat16
                        np_dtype = np.float32  # Use float32 for numpy, convert to bfloat16 in torch
                    
                    np_x = np.random.randn(*shape).astype(np_dtype)
                    
                    torch_x = torch.from_numpy(np_x).to(dtype).cuda()
                    genesis_x = genesis.tensor(np_x.astype(np.float32), device=genesis.cuda())  # Genesis uses float32
                    
                    tensor_size = np.prod(shape)
                    tensor_sizes = [tensor_size]
                    
                    # Calculate dtype bytes for bandwidth calculation
                    dtype_bytes = 4 if dtype == torch.float32 else 2
                    
                    # Benchmark  
                    pytorch_result = timer.benchmark(torch_op, op_key, tensor_sizes, torch_x)
                    genesis_real_result = timer.benchmark(genesis_op, op_key, tensor_sizes, genesis_x)
                    genesis_pure_result = timer.benchmark_pure_compute(genesis_op, op_key, tensor_sizes, genesis_x)
                    
                    # Print result
                    print_professional_row(shape, pytorch_result, genesis_real_result, genesis_pure_result, op_key)
                    
                    # Store for summary
                    speedup = pytorch_result['mean'] / genesis_real_result['mean'] if genesis_real_result['mean'] > 0 else 0
                    category_results.append({
                        'operation': op_name,
                        'shape': shape,
                        'efficiency': speedup,  # Keep as 'efficiency' for backward compatibility but it's actually speedup
                        'pytorch_time': pytorch_result['mean'],
                        'genesis_time': genesis_real_result['mean'],
                        'gflops': genesis_real_result['gflops'],
                        'bandwidth': genesis_real_result['bandwidth_gb_s']
                    })
                    
                    # Clean up
                    del torch_x, genesis_x
                        
                except Exception as e:
                    print(f"{'×'.join(map(str, shape)):<15} ERROR: {str(e)[:50]}")
                    continue
            
            gc.collect()
            torch.cuda.empty_cache()
        
        all_results[category] = category_results
    
    return all_results

if __name__ == "__main__":
    main()