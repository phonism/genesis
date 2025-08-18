#!/usr/bin/env python3
"""
CUDA Storage vs PyTorch Tensor Benchmark

Comprehensive comparison between Genesis CUDAStorage and PyTorch tensors:
- Memory allocation/deallocation
- Data transfer (CPU <-> GPU)
- Basic tensor operations
- Shape manipulation operations
- Indexing operations

Usage:
    python bench_cuda_storage.py                    # Full benchmark
    python bench_cuda_storage.py --fast             # Quick test
    python bench_cuda_storage.py --category memory  # Test specific category
    python bench_cuda_storage.py --size large       # Test specific sizes
"""

import sys
import os
import argparse
import time
import gc
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import genesis
from genesis.ndarray.cuda_storage import CUDAStorage

# Ensure CUDA is available
assert torch.cuda.is_available(), "CUDA is not available"

class BenchCategory(Enum):
    """Benchmark categories"""
    MEMORY = "memory"          # Allocation/deallocation
    DATA_TRANSFER = "transfer" # CPU <-> GPU transfers
    BASIC_OPS = "basic"        # Element-wise operations
    SHAPE_OPS = "shape"        # Shape manipulation
    INDEXING = "indexing"      # Indexing operations
    ADVANCED = "advanced"      # Advanced operations
    BASELINE = "baseline"      # Baseline allocation patterns

@dataclass
class BenchResult:
    """Benchmark result"""
    operation: str
    category: str
    shape: Tuple[int, ...]
    pytorch_time_ms: float
    genesis_time_ms: float
    speedup: float
    status: str
    error: Optional[str] = None

class CUDAStorageBenchmarkTimer:
    """Professional benchmark timer for CUDA Storage operations"""
    
    def __init__(self, warmup_iters: int = 10, test_iters: int = 50):
        self.warmup_iters = warmup_iters
        self.test_iters = test_iters
        self.start_event = torch.cuda.Event(enable_timing=True)
        self.end_event = torch.cuda.Event(enable_timing=True)
        self.gpu_props = torch.cuda.get_device_properties(0)
    
    def benchmark(self, fn, *args, **kwargs) -> float:
        """Run benchmark with precise CUDA timing"""
        # Warmup
        for _ in range(self.warmup_iters):
            try:
                _ = fn(*args, **kwargs)
            except Exception:
                break
        
        torch.cuda.synchronize()
        
        # Actual timing
        times = []
        for i in range(self.test_iters):
            try:
                self.start_event.record()
                result = fn(*args, **kwargs)
                self.end_event.record()
                
                torch.cuda.synchronize()
                elapsed_ms = self.start_event.elapsed_time(self.end_event)
                times.append(elapsed_ms)
            except Exception as e:
                if i < 3:
                    print(f"Warning: Benchmark iteration {i} failed: {str(e)}")
                break
        
        if not times:
            return float('inf')
        
        return np.mean(times)

class CUDAStorageBenchmarkSuite:
    """Main benchmark suite for CUDA Storage vs PyTorch"""
    
    def __init__(self, warmup_iters: int = 10, test_iters: int = 50):
        self.timer = CUDAStorageBenchmarkTimer(warmup_iters, test_iters)
        self.results: List[BenchResult] = []
    
    def _get_test_shapes(self, category: BenchCategory) -> List[Tuple[int, ...]]:
        """Get test shapes for different categories"""
        shape_sets = {
            BenchCategory.MEMORY: [
                (1000,), (10000,), (100000,), (1000000,),
                (100, 100), (1000, 1000), (2000, 2000),
                (64, 64, 64), (128, 128, 128)
            ],
            BenchCategory.DATA_TRANSFER: [
                (1000,), (10000,), (100000,), (1000000,),
                (512, 512), (1024, 1024), (2048, 2048),
                (32, 32, 32), (64, 64, 64)
            ],
            BenchCategory.BASIC_OPS: [
                (512, 512), (1024, 1024), (2048, 2048),
                (64, 64, 64), (128, 128, 128),
                (32, 128, 256), (16, 32, 32, 32)
            ],
            BenchCategory.SHAPE_OPS: [
                (512, 512), (1024, 1024), (2048, 2048),
                (64, 64, 64), (32, 128, 256)
            ],
            BenchCategory.INDEXING: [
                (1000,), (10000,), (512, 512), (1024, 1024),
                (64, 64, 64), (32, 128, 256)
            ],
            BenchCategory.ADVANCED: [
                (512, 512), (1024, 1024), (64, 64, 64)
            ]
        }
        return shape_sets.get(category, [(512, 512)])
    
    def _format_size(self, size: int) -> str:
        """Format tensor size in human readable form"""
        if size >= 1e9:
            return f"{size/1e9:.1f}B"
        elif size >= 1e6:
            return f"{size/1e6:.1f}M"
        elif size >= 1e3:
            return f"{size/1e3:.1f}K"
        else:
            return str(size)
    
    def _benchmark_operation(self, operation: str, category: BenchCategory, 
                           shape: Tuple[int, ...], pytorch_fn, genesis_fn) -> BenchResult:
        """Benchmark a single operation"""
        try:
            # Run benchmarks
            pytorch_time = self.timer.benchmark(pytorch_fn)
            genesis_time = self.timer.benchmark(genesis_fn)
            
            # Calculate metrics
            speedup = pytorch_time / genesis_time if genesis_time > 0 else 0
            
            # Determine status
            if speedup >= 0.9:
                status = "🟢 EXCELLENT"
            elif speedup >= 0.7:
                status = "🟡 GOOD"
            elif speedup >= 0.5:
                status = "🟠 FAIR"
            elif speedup >= 0.2:
                status = "🔴 POOR"
            else:
                status = "❌ CRITICAL"
            
            return BenchResult(
                operation=operation,
                category=category.value,
                shape=shape,
                pytorch_time_ms=pytorch_time,
                genesis_time_ms=genesis_time,
                speedup=speedup,
                status=status
            )
            
        except Exception as e:
            return BenchResult(
                operation=operation,
                category=category.value,
                shape=shape,
                pytorch_time_ms=float('inf'),
                genesis_time_ms=float('inf'),
                speedup=0,
                status="❌ ERROR",
                error=str(e)
            )
        finally:
            # Clean up
            gc.collect()
            torch.cuda.empty_cache()
    
    def benchmark_memory_operations(self, shapes: List[Tuple[int, ...]]) -> List[BenchResult]:
        """Benchmark memory allocation and deallocation"""
        print(f"\n{'='*80}")
        print("MEMORY OPERATIONS BENCHMARK")
        print(f"{'='*80}")
        
        results = []
        
        # Test allocation
        for shape in shapes:
            print(f"\nShape: {shape} ({self._format_size(np.prod(shape))} elements)")
            
            # PyTorch allocation
            def pytorch_alloc():
                return torch.empty(shape, device='cuda', dtype=torch.float32)
            
            # Genesis CUDAStorage allocation  
            def genesis_alloc():
                return CUDAStorage(shape, dtype="float32")
            
            result = self._benchmark_operation(
                "allocation", BenchCategory.MEMORY, shape,
                pytorch_alloc, genesis_alloc
            )
            results.append(result)
            
            shape_str = "×".join(map(str, shape))
            size_str = self._format_size(np.prod(shape))
            
            if result.error:
                print(f"  Allocation    {shape_str:<20} {size_str:<8} ERROR: {result.error[:50]}")
            else:
                print(f"  Allocation    {shape_str:<20} {size_str:<8} "
                      f"PyTorch: {result.pytorch_time_ms:.3f}ms | "
                      f"Genesis: {result.genesis_time_ms:.3f}ms | "
                      f"Speedup: {result.speedup:.2f}x | {result.status}")
        
        self.results.extend(results)
        return results
    
    def benchmark_data_transfer(self, shapes: List[Tuple[int, ...]]) -> List[BenchResult]:
        """Benchmark CPU <-> GPU data transfers"""
        print(f"\n{'='*80}")
        print("DATA TRANSFER BENCHMARK")
        print(f"{'='*80}")
        
        results = []
        
        for shape in shapes:
            print(f"\nShape: {shape} ({self._format_size(np.prod(shape))} elements)")
            
            # Create CPU data
            cpu_data = np.random.randn(*shape).astype(np.float32)
            
            # CPU -> GPU transfer
            def pytorch_to_gpu():
                return torch.from_numpy(cpu_data).cuda()
            
            def genesis_to_gpu():
                gpu_tensor = CUDAStorage(shape, dtype="float32")
                gpu_tensor.from_numpy(cpu_data)
                return gpu_tensor
            
            # Test if genesis has from_numpy method
            try:
                test_storage = CUDAStorage((10,), dtype="float32")
                if hasattr(test_storage, 'from_numpy'):
                    result = self._benchmark_operation(
                        "cpu_to_gpu", BenchCategory.DATA_TRANSFER, shape,
                        pytorch_to_gpu, genesis_to_gpu
                    )
                else:
                    # Skip if method doesn't exist
                    result = BenchResult(
                        operation="cpu_to_gpu",
                        category=BenchCategory.DATA_TRANSFER.value,
                        shape=shape,
                        pytorch_time_ms=float('inf'),
                        genesis_time_ms=float('inf'),
                        speedup=0,
                        status="❌ ERROR",
                        error="from_numpy method not available"
                    )
                del test_storage
            except Exception as e:
                result = BenchResult(
                    operation="cpu_to_gpu",
                    category=BenchCategory.DATA_TRANSFER.value,
                    shape=shape,
                    pytorch_time_ms=float('inf'),
                    genesis_time_ms=float('inf'),
                    speedup=0,
                    status="❌ ERROR",
                    error=str(e)
                )
            
            results.append(result)
            
            shape_str = "×".join(map(str, shape))
            size_str = self._format_size(np.prod(shape))
            
            if result.error:
                print(f"  CPU->GPU      {shape_str:<20} {size_str:<8} ERROR: {result.error[:50]}")
            else:
                print(f"  CPU->GPU      {shape_str:<20} {size_str:<8} "
                      f"PyTorch: {result.pytorch_time_ms:.3f}ms | "
                      f"Genesis: {result.genesis_time_ms:.3f}ms | "
                      f"Speedup: {result.speedup:.2f}x | {result.status}")
            
            # Also test GPU -> CPU transfer
            def pytorch_to_cpu():
                gpu_tensor = torch.randn(shape, device='cuda', dtype=torch.float32)
                return gpu_tensor.cpu().numpy()
            
            def genesis_to_cpu():
                gpu_tensor = CUDAStorage(shape, dtype="float32")
                return gpu_tensor.to_numpy()
            
            # Test GPU -> CPU transfer
            try:
                result2 = self._benchmark_operation(
                    "gpu_to_cpu", BenchCategory.DATA_TRANSFER, shape,
                    pytorch_to_cpu, genesis_to_cpu
                )
                results.append(result2)
                
                if result2.error:
                    print(f"  GPU->CPU      {shape_str:<20} {size_str:<8} ERROR: {result2.error[:50]}")
                else:
                    print(f"  GPU->CPU      {shape_str:<20} {size_str:<8} "
                          f"PyTorch: {result2.pytorch_time_ms:.3f}ms | "
                          f"Genesis: {result2.genesis_time_ms:.3f}ms | "
                          f"Speedup: {result2.speedup:.2f}x | {result2.status}")
            except Exception as e:
                result2 = BenchResult(
                    operation="gpu_to_cpu",
                    category=BenchCategory.DATA_TRANSFER.value,
                    shape=shape,
                    pytorch_time_ms=float('inf'),
                    genesis_time_ms=float('inf'),
                    speedup=0,
                    status="❌ ERROR",
                    error=str(e)
                )
                results.append(result2)
                print(f"  GPU->CPU      {shape_str:<20} {size_str:<8} ERROR: {str(e)[:50]}")
        
        self.results.extend(results)
        return results
    
    def benchmark_basic_operations(self, shapes: List[Tuple[int, ...]]) -> List[BenchResult]:
        """Benchmark basic tensor operations"""
        print(f"\n{'='*80}")
        print("BASIC OPERATIONS BENCHMARK")
        print(f"{'='*80}")
        
        results = []
        
        for shape in shapes:
            print(f"\nShape: {shape}")
            
            # Create test tensors
            pytorch_tensor = torch.randn(shape, device='cuda', dtype=torch.float32)
            genesis_tensor = CUDAStorage(shape, dtype="float32")
            
            # Fill operation - simple memset
            def pytorch_fill():
                pytorch_tensor.fill_(1.0)
                return pytorch_tensor
            
            def genesis_fill():
                genesis_tensor.fill_(1.0)
                return genesis_tensor
            
            result = self._benchmark_operation(
                "fill", BenchCategory.BASIC_OPS, shape,
                pytorch_fill, genesis_fill
            )
            results.append(result)
            
            shape_str = "×".join(map(str, shape))
            size_str = self._format_size(np.prod(shape))
            
            if result.error:
                print(f"  fill         {shape_str:<20} {size_str:<8} ERROR: {result.error[:50]}")
            else:
                print(f"  fill         {shape_str:<20} {size_str:<8} "
                      f"PyTorch: {result.pytorch_time_ms:.3f}ms | "
                      f"Genesis: {result.genesis_time_ms:.3f}ms | "
                      f"Speedup: {result.speedup:.2f}x | {result.status}")
            
            # Clean up
            del pytorch_tensor, genesis_tensor
            gc.collect()
            torch.cuda.empty_cache()
        
        self.results.extend(results)
        return results
    
    def benchmark_shape_operations(self, shapes: List[Tuple[int, ...]]) -> List[BenchResult]:
        """Benchmark shape manipulation operations"""
        print(f"\n{'='*80}")
        print("SHAPE OPERATIONS BENCHMARK")
        print(f"{'='*80}")
        
        results = []
        
        operations = [
            ("reshape", 
             lambda x: x.reshape(-1), 
             lambda x: x.reshape((-1,))),
            ("transpose", 
             lambda x: x.transpose(-1, -2) if len(x.shape) >= 2 else x, 
             lambda x: x.transpose(-1, -2) if len(x.shape) >= 2 else x),
            ("contiguous",
             lambda x: x.contiguous(),
             lambda x: x.contiguous()),
        ]
        
        for shape in shapes:
            print(f"\nShape: {shape}")
            
            for op_name, pytorch_op, genesis_op in operations:
                # Create test tensors
                pytorch_tensor = torch.randn(shape, device='cuda', dtype=torch.float32)
                genesis_tensor = CUDAStorage(shape, dtype="float32")
                
                def pytorch_fn():
                    return pytorch_op(pytorch_tensor)
                
                def genesis_fn():
                    return genesis_op(genesis_tensor)
                
                result = self._benchmark_operation(
                    op_name, BenchCategory.SHAPE_OPS, shape,
                    pytorch_fn, genesis_fn
                )
                results.append(result)
                
                shape_str = "×".join(map(str, shape))
                size_str = self._format_size(np.prod(shape))
                
                if result.error:
                    print(f"  {op_name:<12} {shape_str:<20} {size_str:<8} ERROR: {result.error[:50]}")
                else:
                    print(f"  {op_name:<12} {shape_str:<20} {size_str:<8} "
                          f"PyTorch: {result.pytorch_time_ms:.3f}ms | "
                          f"Genesis: {result.genesis_time_ms:.3f}ms | "
                          f"Speedup: {result.speedup:.2f}x | {result.status}")
                
                # Clean up
                del pytorch_tensor, genesis_tensor
                gc.collect()
                torch.cuda.empty_cache()
        
        self.results.extend(results)
        return results
    
    def benchmark_indexing_operations(self, shapes: List[Tuple[int, ...]]) -> List[BenchResult]:
        """Benchmark indexing operations"""
        print(f"\n{'='*80}")
        print("INDEXING OPERATIONS BENCHMARK")
        print(f"{'='*80}")
        
        results = []
        
        for shape in shapes:
            print(f"\nShape: {shape}")
            
            # Create test tensors
            pytorch_tensor = torch.randn(shape, device='cuda', dtype=torch.float32)
            genesis_tensor = CUDAStorage(shape, dtype="float32")
            
            # Simple slice operations
            operations = []
            
            if len(shape) >= 1:
                operations.append(("slice_0", lambda x: x[:shape[0]//2]))
            
            if len(shape) >= 2:
                operations.append(("slice_2d", lambda x: x[:shape[0]//2, :shape[1]//2]))
            
            for op_name, op_func in operations:
                def pytorch_fn():
                    return op_func(pytorch_tensor)
                
                def genesis_fn():
                    return op_func(genesis_tensor)
                
                result = self._benchmark_operation(
                    op_name, BenchCategory.INDEXING, shape,
                    pytorch_fn, genesis_fn
                )
                results.append(result)
                
                shape_str = "×".join(map(str, shape))
                size_str = self._format_size(np.prod(shape))
                
                if result.error:
                    print(f"  {op_name:<12} {shape_str:<20} {size_str:<8} ERROR: {result.error[:50]}")
                else:
                    print(f"  {op_name:<12} {shape_str:<20} {size_str:<8} "
                          f"PyTorch: {result.pytorch_time_ms:.3f}ms | "
                          f"Genesis: {result.genesis_time_ms:.3f}ms | "
                          f"Speedup: {result.speedup:.2f}x | {result.status}")
            
            # Clean up
            del pytorch_tensor, genesis_tensor
            gc.collect()
            torch.cuda.empty_cache()
        
        self.results.extend(results)
        return results
    
    def benchmark_strided_copy_operations(self, shapes: List[Tuple[int, ...]]) -> List[BenchResult]:
        """Benchmark strided copy operations - our Phase 1 optimization target"""
        print(f"\n{'='*80}")
        print("STRIDED COPY OPERATIONS BENCHMARK (Phase 1 Optimization)")
        print(f"{'='*80}")
        
        results = []
        
        for shape in shapes:
            if len(shape) < 2:
                continue  # Skip 1D tensors for strided copy tests
                
            print(f"\nShape: {shape} ({self._format_size(np.prod(shape))} elements)")
            
            # Create test data
            cpu_data = np.random.randn(*shape).astype(np.float32)
            
            # Test 1: Copy to non-contiguous tensor (transpose)
            def pytorch_strided_copy():
                # Create contiguous tensor and transpose it
                torch_tensor = torch.from_numpy(cpu_data.copy()).cuda()
                if len(shape) == 2:
                    torch_strided = torch_tensor.T  # Non-contiguous
                else:
                    # For higher dims, permute first two dimensions
                    dims = list(range(len(shape)))
                    dims[0], dims[1] = dims[1], dims[0]
                    torch_strided = torch_tensor.permute(dims)
                
                # Copy new data to non-contiguous tensor
                new_data = np.random.randn(*torch_strided.shape).astype(np.float32)
                torch_strided.copy_(torch.from_numpy(new_data).cuda())
                return torch_strided
            
            def genesis_strided_copy():
                # Create contiguous tensor and transpose it  
                genesis_tensor = CUDAStorage(shape, dtype="float32")
                genesis_tensor.from_numpy(cpu_data.copy())
                
                if len(shape) == 2:
                    genesis_strided = genesis_tensor.T  # Non-contiguous
                else:
                    # For higher dims, permute first two dimensions
                    dims = list(range(len(shape)))
                    dims[0], dims[1] = dims[1], dims[0]
                    genesis_strided = genesis_tensor.permute(dims)
                
                # Copy new data to non-contiguous tensor - this uses our optimized kernel!
                new_data = np.random.randn(*genesis_strided.shape).astype(np.float32)
                genesis_strided.from_numpy(new_data)
                return genesis_strided
            
            result = self._benchmark_operation(
                "strided_copy", BenchCategory.ADVANCED, shape,
                pytorch_strided_copy, genesis_strided_copy
            )
            results.append(result)
            
            shape_str = "×".join(map(str, shape))
            size_str = self._format_size(np.prod(shape))
            
            if result.error:
                print(f"  Strided Copy  {shape_str:<20} {size_str:<8} ERROR: {result.error[:50]}")
            else:
                print(f"  Strided Copy  {shape_str:<20} {size_str:<8} "
                      f"PyTorch: {result.pytorch_time_ms:.3f}ms | "
                      f"Genesis: {result.genesis_time_ms:.3f}ms | "
                      f"Speedup: {result.speedup:.2f}x | {result.status}")
                
                # Calculate throughput
                elements = np.prod(shape)
                torch_throughput = elements / result.pytorch_time_ms if result.pytorch_time_ms > 0 else 0
                genesis_throughput = elements / result.genesis_time_ms if result.genesis_time_ms > 0 else 0
                
                print(f"               Throughput: PyTorch {torch_throughput:.0f} elem/ms | "
                      f"Genesis {genesis_throughput:.0f} elem/ms")
            
            # Clean up
            gc.collect()
            torch.cuda.empty_cache()
        
        self.results.extend(results)
        return results
    
    def benchmark_category(self, category: BenchCategory, max_shapes: Optional[int] = None) -> List[BenchResult]:
        """Benchmark a specific category"""
        shapes = self._get_test_shapes(category)
        if max_shapes:
            shapes = shapes[:max_shapes]
        
        if category == BenchCategory.MEMORY:
            return self.benchmark_memory_operations(shapes)
        elif category == BenchCategory.DATA_TRANSFER:
            return self.benchmark_data_transfer(shapes)
        elif category == BenchCategory.BASIC_OPS:
            return self.benchmark_basic_operations(shapes)
        elif category == BenchCategory.SHAPE_OPS:
            return self.benchmark_shape_operations(shapes)
        elif category == BenchCategory.INDEXING:
            return self.benchmark_indexing_operations(shapes)
        elif category == BenchCategory.ADVANCED:
            return self.benchmark_strided_copy_operations(shapes)
        else:
            print(f"Category {category.value} not implemented yet")
            return []
    
    def benchmark_all(self, max_shapes: Optional[int] = None) -> List[BenchResult]:
        """Benchmark all categories"""
        print(f"🚀 COMPREHENSIVE CUDA STORAGE BENCHMARK")
        print(f"GPU: {self.timer.gpu_props.name}")
        print(f"Memory: {self.timer.gpu_props.total_memory / 1024**3:.1f} GB")
        
        all_results = []
        
        for category in BenchCategory:
            try:
                category_results = self.benchmark_category(category, max_shapes)
                all_results.extend(category_results)
            except Exception as e:
                print(f"⚠️  Category {category.value} failed: {e}")
        
        return all_results
    
    def generate_summary(self, results: Optional[List[BenchResult]] = None):
        """Generate comprehensive benchmark summary"""
        if results is None:
            results = self.results
        
        if not results:
            print("❌ No results to summarize")
            return
        
        print(f"\n{'='*100}")
        print("CUDA STORAGE BENCHMARK SUMMARY")
        print(f"{'='*100}")
        
        # System information
        print(f"\n📊 SYSTEM INFORMATION")
        print(f"GPU: {self.timer.gpu_props.name}")
        print(f"Memory: {self.timer.gpu_props.total_memory / 1024**3:.1f} GB")
        
        # Overall statistics
        valid_results = [r for r in results if r.error is None and r.speedup > 0]
        failed_results = [r for r in results if r.error is not None]
        
        if valid_results:
            speedups = [r.speedup for r in valid_results]
            
            print(f"\n📈 OVERALL PERFORMANCE STATISTICS")
            print(f"Total Operations Tested: {len(results)}")
            print(f"Successful Tests: {len(valid_results)}")
            print(f"Failed Tests: {len(failed_results)}")
            print(f"Success Rate: {len(valid_results)/len(results)*100:.1f}%")
            print(f"Average Speedup: {np.mean(speedups):.2f}x")
            print(f"Median Speedup: {np.median(speedups):.2f}x")
            print(f"Best Speedup: {np.max(speedups):.2f}x")
            print(f"Worst Speedup: {np.min(speedups):.2f}x")
        
        # Category breakdown
        print(f"\n📊 PERFORMANCE BY CATEGORY")
        print(f"{'Category':<15} {'Tests':<8} {'Success':<8} {'Avg Speedup':<12} {'Status':<15}")
        print("-" * 70)
        
        categories = set(r.category for r in results)
        for category in categories:
            cat_results = [r for r in results if r.category == category]
            cat_valid = [r for r in cat_results if r.error is None and r.speedup > 0]
            
            if cat_valid:
                speedups = [r.speedup for r in cat_valid]
                avg_speedup = np.mean(speedups)
                success_rate = len(cat_valid) / len(cat_results) * 100
                
                if avg_speedup >= 0.8:
                    status = "🟢 EXCELLENT"
                elif avg_speedup >= 0.6:
                    status = "🟡 GOOD"
                elif avg_speedup >= 0.4:
                    status = "🟠 FAIR"
                else:
                    status = "🔴 POOR"
                
                print(f"{category:<15} {len(cat_results):<8} {success_rate:5.1f}%{'':<2} "
                      f"{avg_speedup:.2f}x{'':<7} {status}")
            else:
                print(f"{category:<15} {len(cat_results):<8} {'0.0%':<7} {'N/A':<12} {'❌ FAILED'}")
        
        # Top and bottom performers
        if valid_results:
            print(f"\n🏆 TOP 5 PERFORMERS")
            top_performers = sorted(valid_results, key=lambda x: x.speedup, reverse=True)[:5]
            for i, result in enumerate(top_performers, 1):
                shape_str = "×".join(map(str, result.shape))
                print(f"{i}. {result.operation} ({shape_str}): {result.speedup:.2f}x")
            
            print(f"\n⚠️  BOTTOM 5 PERFORMERS")
            bottom_performers = sorted(valid_results, key=lambda x: x.speedup)[:5]
            for i, result in enumerate(bottom_performers, 1):
                shape_str = "×".join(map(str, result.shape))
                print(f"{i}. {result.operation} ({shape_str}): {result.speedup:.2f}x - {result.status}")
        
        # Failed operations
        if failed_results:
            print(f"\n❌ FAILED OPERATIONS ({len(failed_results)} total)")
            for result in failed_results[:10]:  # Show first 10 failures
                print(f"• {result.operation}: {result.error[:80]}")
        
        print(f"\n✅ BENCHMARK COMPLETED")
        success_rate = len(valid_results) / len(results) * 100 if results else 0
        avg_performance = np.mean([r.speedup for r in valid_results]) if valid_results else 0
        print(f"Success Rate: {success_rate:.1f}%")
        print(f"Average Performance: {avg_performance:.2f}x")

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="CUDA Storage vs PyTorch Benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument("--fast", action="store_true",
                       help="Quick mode with reduced iterations and shapes")
    parser.add_argument("--category", type=str, 
                       choices=[cat.value for cat in BenchCategory],
                       help="Test specific category only")
    parser.add_argument("--size", type=str, choices=["small", "medium", "large"],
                       help="Test specific size category")
    
    return parser.parse_args()

def main():
    """Main benchmark execution"""
    args = parse_args()
    
    # Configure benchmark
    if args.fast:
        warmup_iters, test_iters = 3, 10
        max_shapes = 3
        print("⚡ Fast mode: Reduced iterations and shapes")
    else:
        warmup_iters, test_iters = 10, 50
        max_shapes = None
    
    # Print configuration
    mode = "FAST" if args.fast else "COMPREHENSIVE"
    print(f"🚀 CUDA Storage {mode} Benchmark")
    print("=" * 80)
    
    if args.category:
        print(f"📏 Testing category: {args.category.upper()}")
    
    # Create benchmark suite
    suite = CUDAStorageBenchmarkSuite(warmup_iters, test_iters)
    
    # Run benchmarks
    if args.category:
        try:
            category = BenchCategory(args.category)
            results = suite.benchmark_category(category, max_shapes)
        except ValueError:
            print(f"❌ Invalid category: {args.category}")
            return
    else:
        results = suite.benchmark_all(max_shapes)
    
    # Generate summary
    suite.generate_summary(results)

if __name__ == "__main__":
    main()