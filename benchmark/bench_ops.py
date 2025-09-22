"""
Genesis Comprehensive Operations Benchmark

Unified benchmark testing ALL operations in Genesis against PyTorch with:

TIMING MODES:
- Real: Per-operation timing (includes CUDA sync overhead) - real user experience
- Pure: Batch timing (minimal overhead) - pure computational performance

RELIABILITY IMPROVEMENTS (v2.0):
- Statistical outlier detection using IQR or Z-score methods
- Median-based timing (more robust than mean)
- Reliability scoring with coefficient of variation
- Automatic warning for low-reliability measurements
- Configurable outlier filtering (--no-outlier-filter to disable)
- Minimum iteration requirements for statistical validity

FEATURES:
- Automatic operation discovery from genesis.nn.functional
- Multiple tensor sizes and operation categories
- Professional metrics: bandwidth, GFLOPS, efficiency
- Adaptive iterations and statistical analysis
- Detailed optimization recommendations
- Support for all data types and operation categories

CATEGORIES TESTED:
1. Element-wise operations (add, sub, mul, div, pow, etc.)
2. Activation functions (relu, sigmoid, tanh, etc.)
3. Reduction operations (sum, max, logsumexp, etc.)
4. Shape operations (transpose, reshape, expand, etc.)
5. Matrix operations (matmul)
6. Tensor manipulation (stack, cat, squeeze, etc.)
7. Broadcasting operations
8. Triton fused operations (when available)

Usage:
    python bench_ops.py                    # Full comprehensive benchmark
    python bench_ops.py --fast             # Quick test mode
    python bench_ops.py --op add           # Test specific operation
    python bench_ops.py --category element # Test specific category
    python bench_ops.py --size large       # Test specific size category
    python bench_ops.py --dtype float16    # Test with different precision
    python bench_ops.py --timing pure      # Use pure compute timing
    python bench_ops.py --timing both      # Test with both timing modes
    
    # Reliability options (NEW):
    python bench_ops.py --no-outlier-filter         # Disable outlier filtering
    python bench_ops.py --outlier-method zscore     # Use Z-score instead of IQR
    python bench_ops.py --min-iterations 200        # Force minimum iterations
    python bench_ops.py --show-reliability          # Show detailed reliability metrics

Options:
    --fast                  Quick mode with reduced iterations
    --op OPERATION          Test specific operation (supports partial matching)
    --category CAT          Test specific category: element, activation, reduction, shape, matrix, tensor, broadcast, triton
    --size SIZE             Test specific size category: small, medium, large, very_large, batch
    --dtype TYPE            Data type: float32, float16, bfloat16
    --timing MODE           Timing mode: real, pure, both
    --no-outlier-filter     Disable statistical outlier filtering
    --outlier-method METHOD Outlier detection method: iqr, zscore (default: iqr)
    --min-iterations N      Minimum iterations for statistical reliability
    --show-reliability      Show detailed reliability metrics in output
    --list-ops              List available operations
    --list-categories       List available categories
    --list-sizes            List available size categories
"""

import sys
import os
import argparse
import inspect
from typing import Dict, List, Callable, Tuple, Any, Optional, Union
from dataclasses import dataclass
from enum import Enum
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import numpy as np
import torch
import torch.nn.functional as F
import genesis
import genesis.nn.functional as gF
import gc
import statistics

# Ensure we're using GPU
assert torch.cuda.is_available(), "CUDA is not available"

class OpCategory(Enum):
    """Operation categories for systematic testing"""
    ELEMENT_WISE = "element"
    ACTIVATION = "activation"
    REDUCTION = "reduction"
    SHAPE = "shape"
    MATRIX_OPS = "matrix"
    TENSOR_MANIPULATION = "tensor"
    BROADCAST = "broadcast"
    TRITON_FUSED = "triton"
    MEMORY_OPS = "memory"

@dataclass
class OpSpec:
    """Operation specification for benchmarking"""
    name: str
    category: OpCategory
    genesis_func: Callable
    torch_func: Callable
    test_shapes: List[Tuple[int, ...]]
    requires_second_tensor: bool = False
    requires_scalar: bool = False
    special_args: Optional[Dict[str, Any]] = None
    description: str = ""
    skip_reason: Optional[str] = None

class ComprehensiveOpRegistry:
    """Registry for all operations with automatic discovery and manual definitions"""
    
    def __init__(self):
        self.operations: Dict[str, OpSpec] = {}
        self._discover_operations()
    
    def _get_test_shapes(self, category: OpCategory) -> List[Tuple[int, ...]]:
        """Get appropriate test shapes for different operation categories"""
        base_shapes = {
            "small": [
                (256, 256),          # 64K elements
                (512, 512),          # 256K elements  
                (1024, 1024),        # 1M elements
                (32, 32, 32),        # 32K elements (3D)
                (64, 64, 64),        # 256K elements (3D)
            ],
            "medium": [
                (2048, 2048),        # 4M elements
                (4096, 1024),        # 4M elements (non-square)
                (1024, 4096),        # 4M elements (non-square)
                (128, 128, 128),     # 2M elements (3D)
                (32, 256, 256),      # 2M elements (3D)
            ],
            "large": [
                (4096, 4096),        # 16M elements
                (8192, 2048),        # 16M elements (non-square)
                (2048, 8192),        # 16M elements (non-square)
                (256, 256, 256),     # 16M elements (3D)
                (64, 512, 512),      # 16M elements (3D)
            ],
            "very_large": [
                (8192, 8192),        # 64M elements
                (16384, 4096),       # 64M elements (non-square)
                (512, 512, 512),     # 128M elements (3D)
            ],
            "batch": [
                (32, 128, 768),      # Transformer-like (BERT base)
                (64, 256, 256),      # CNN-like
                (128, 512, 512),     # Large CNN-like
                (16, 1024, 1024),    # Large batch
            ]
        }
        
        # Category-specific shape preferences
        if category in [OpCategory.ELEMENT_WISE, OpCategory.ACTIVATION]:
            return base_shapes["small"] + base_shapes["medium"][:2]
        elif category == OpCategory.REDUCTION:
            return base_shapes["small"] + base_shapes["medium"][:2]
        elif category == OpCategory.SHAPE:
            return base_shapes["medium"][:3] + base_shapes["batch"][:2]
        elif category == OpCategory.MATRIX_OPS:
            # Transformer-specific MatMul shapes (batch=4, seq_len=512, various hidden sizes)
            return [
                # Small to medium transformers
                (512, 768),           # BERT-base hidden size
                (768, 2304),          # QKV projection (768 -> 3*768)
                (768, 3072),          # MLP up projection (768 -> 4*768)
                (3072, 768),          # MLP down projection (4*768 -> 768)
                # Qwen-0.5B sizes
                (896, 896),           # Self-attention projection
                (896, 2688),          # QKV projection (896 -> 3*896)
                (896, 4864),          # MLP up projection
                (4864, 896),          # MLP down projection
                # Large model (LM Head - the main bottleneck!)
                (896, 151936),        # LM Head (hidden -> vocab)
                (2048, 896),          # Larger batch size
                (2048, 151936),       # Large batch LM Head
            ]
        elif category == OpCategory.TENSOR_MANIPULATION:
            return base_shapes["small"][ :3] + base_shapes["medium"][ :2]
        elif category == OpCategory.BROADCAST:
            return [(1000, 1), (100, 100, 1), (32, 1, 768), (1, 512, 512)]
        else:
            return base_shapes["small"][ :3] + base_shapes["medium"][ :2]
    
    def _discover_operations(self):
        """Automatically discover and categorize all operations"""
        
        # Element-wise binary operations
        element_wise_binary = [
            ("add", gF.add, lambda x, y: x + y, True),
            ("sub", gF.sub, lambda x, y: x - y, True),
            ("multiply", gF.multiply, lambda x, y: x * y, True),
            ("divide", gF.divide, lambda x, y: x / y, True),
        ]
        
        # Element-wise unary operations
        element_wise_unary = [
            ("negate", gF.negate, torch.neg, False),
            ("sin", gF.sin, torch.sin, False),
            ("cos", gF.cos, torch.cos, False),
            ("log", gF.log, torch.log, False),
            ("exp", gF.exp, torch.exp, False),
            ("sqrt", gF.sqrt, torch.sqrt, False),
        ]
        
        # Add binary operations
        for name, gfunc, tfunc, needs_second in element_wise_binary:
            if hasattr(gF, name):
                self.operations[name] = OpSpec(
                    name=name,
                    category=OpCategory.ELEMENT_WISE,
                    genesis_func=gfunc,
                    torch_func=tfunc,
                    test_shapes=self._get_test_shapes(OpCategory.ELEMENT_WISE),
                    requires_second_tensor=needs_second,
                    description=f"Element-wise {name} operation"
                )
        
        # Add unary operations
        for name, gfunc, tfunc, needs_second in element_wise_unary:
            if hasattr(gF, name):
                self.operations[name] = OpSpec(
                    name=name,
                    category=OpCategory.ELEMENT_WISE,
                    genesis_func=gfunc,
                    torch_func=tfunc,
                    test_shapes=self._get_test_shapes(OpCategory.ELEMENT_WISE),
                    requires_second_tensor=needs_second,
                    description=f"Element-wise {name} operation"
                )
        
        # Scalar operations
        scalar_ops = [
            ("add_scalar", gF.add_scalar, lambda x: x + 2.5, {"scalar": 2.5}),
            ("mul_scalar", gF.mul_scalar, lambda x: x * 2.5, {"scalar": 2.5}),
            ("divide_scalar", gF.divide_scalar, lambda x: x / 2.5, {"scalar": 2.5}),
            ("pow_scalar", gF.pow_scalar, lambda x: x ** 2, {"scalar": 2}),
        ]
        
        for name, gfunc, tfunc, args in scalar_ops:
            if hasattr(gF, name):
                self.operations[name] = OpSpec(
                    name=name,
                    category=OpCategory.ELEMENT_WISE,
                    genesis_func=gfunc,
                    torch_func=tfunc,
                    test_shapes=self._get_test_shapes(OpCategory.ELEMENT_WISE),
                    requires_scalar=True,
                    special_args=args,
                    description=f"Scalar {name.replace('_scalar', '')} operation"
                )
        
        # Activation functions
        activation_ops = [
            ("relu", gF.relu, F.relu),
        ]
        
        # Try to get triton fused activations if available
        try:
            from genesis.nn.triton_ops import softmax, dropout, safe_softmax
            activation_ops.extend([
                ("softmax_triton", lambda x: softmax(x), lambda x: F.softmax(x, dim=-1)),
                ("safe_softmax_triton", lambda x: safe_softmax(x), lambda x: F.softmax(x, dim=-1)),
                ("dropout_triton", lambda x: dropout(x, 0.1), lambda x: F.dropout(x, 0.1)),
            ])
        except ImportError:
            pass
        
        for name, gfunc, tfunc in activation_ops:
            category = OpCategory.TRITON_FUSED if 'triton' in name else OpCategory.ACTIVATION
            self.operations[name] = OpSpec(
                name=name,
                category=category,
                genesis_func=gfunc,
                torch_func=tfunc,
                test_shapes=self._get_test_shapes(category),
                description=f"{name} activation function"
            )
        
        # Reduction operations
        reduction_ops = [
            ("sum", lambda x: gF.sum(x), lambda x: x.sum()),
            ("summation", lambda x: gF.summation(x), lambda x: x.sum()),
            ("sum_axis0", lambda x: gF.sum(x, axis=0), lambda x: x.sum(dim=0)),
            ("summation_axis0", lambda x: gF.summation(x, axis=0), lambda x: x.sum(dim=0)),
            ("sum_axis1", lambda x: gF.sum(x, axis=1), lambda x: x.sum(dim=1)),
            ("summation_axis1", lambda x: gF.summation(x, axis=1), lambda x: x.sum(dim=1)),
            ("mean", lambda x: gF.mean(x), lambda x: x.mean()),
            ("mean_axis0", lambda x: gF.mean(x, axis=0), lambda x: x.mean(dim=0)),
            ("mean_axis1", lambda x: gF.mean(x, axis=1), lambda x: x.mean(dim=1)),
            ("max", lambda x: gF.max(x), lambda x: x.max()),
            ("max_axis0", lambda x: gF.max(x, axis=0), lambda x: x.max(dim=0)[0]),
            ("max_axis1", lambda x: gF.max(x, axis=1), lambda x: x.max(dim=1)[0]),
            ("logsumexp", lambda x: gF.logsumexp(x), lambda x: torch.logsumexp(x, dim=-1)),
        ]
        
        for name, gfunc, tfunc in reduction_ops:
            if hasattr(gF, name.split('_')[0]):  # Check if base function exists
                self.operations[name] = OpSpec(
                    name=name,
                    category=OpCategory.REDUCTION,
                    genesis_func=gfunc,
                    torch_func=tfunc,
                    test_shapes=self._get_test_shapes(OpCategory.REDUCTION),
                    description=f"{name} reduction operation"
                )
        
        # Shape operations
        shape_ops = [
            ("transpose", lambda x: gF.transpose(x), lambda x: x.transpose(-2, -1)),
            ("reshape", lambda x: gF.reshape(x, (-1,)), lambda x: x.reshape(-1)),
            ("flatten", lambda x: gF.flatten(x), lambda x: x.flatten()),
            ("view", lambda x: gF.view(x, (-1,)), lambda x: x.view(-1)),
            ("squeeze", lambda x: gF.squeeze(x.unsqueeze(0), 0), lambda x: x.unsqueeze(0).squeeze(0)),
            ("unsqueeze", lambda x: gF.unsqueeze(x, 0), lambda x: x.unsqueeze(0)),
        ]
        
        for name, gfunc, tfunc in shape_ops:
            if hasattr(gF, name):
                self.operations[name] = OpSpec(
                    name=name,
                    category=OpCategory.SHAPE,
                    genesis_func=gfunc,
                    torch_func=tfunc,
                    test_shapes=self._get_test_shapes(OpCategory.SHAPE),
                    description=f"{name} shape operation"
                )
        
        # Tensor manipulation operations  
        tensor_ops = [
            ("broadcast_to", 
             lambda x: gF.broadcast_to(x.unsqueeze(0), (3, *x.shape)), 
             lambda x: x.unsqueeze(0).expand(3, *x.shape)),
            ("stack", 
             lambda x: gF.stack([x, x], dim=0),
             lambda x: torch.stack([x, x], dim=0)),
            ("cat",
             lambda x: gF.cat([x, x], dim=0),
             lambda x: torch.cat([x, x], dim=0)),
        ]
        
        for name, gfunc, tfunc in tensor_ops:
            if hasattr(gF, name):
                self.operations[name] = OpSpec(
                    name=name,
                    category=OpCategory.TENSOR_MANIPULATION,
                    genesis_func=gfunc,
                    torch_func=tfunc,
                    test_shapes=self._get_test_shapes(OpCategory.TENSOR_MANIPULATION),
                    description=f"{name} tensor manipulation"
                )
        
        # Matrix operations
        self.operations["matmul"] = OpSpec(
            name="matmul",
            category=OpCategory.MATRIX_OPS,
            genesis_func=gF.matmul,
            torch_func=torch.matmul,
            test_shapes=self._get_test_shapes(OpCategory.MATRIX_OPS),
            requires_second_tensor=True,
            description="Matrix multiplication"
        )
    
    def get_operations_by_category(self, category: OpCategory) -> Dict[str, OpSpec]:
        """Get all operations in a specific category"""
        return {name: op for name, op in self.operations.items() 
                if op.category == category and op.skip_reason is None}
    
    def get_operation(self, name: str) -> Optional[OpSpec]:
        """Get a specific operation by name"""
        return self.operations.get(name)
    
    def list_categories(self) -> List[str]:
        """List all available categories"""
        return list(set(op.category.value for op in self.operations.values()))
    
    def list_operations(self) -> List[str]:
        """List all available operations"""
        return [name for name, op in self.operations.items() if op.skip_reason is None]
    
    def filter_operations_by_name(self, target_op: str) -> Dict[str, OpSpec]:
        """Filter operations by name (supports partial matching)"""
        filtered = {}
        for name, op in self.operations.items():
            if target_op.lower() in name.lower() and op.skip_reason is None:
                filtered[name] = op
        return filtered

def filter_outliers(times, method="iqr", iqr_factor=1.8):
    """
    Filter outliers from timing measurements using statistical methods optimized for GPU timing.
    
    Args:
        times: List of timing measurements in milliseconds
        method: 'iqr' (Interquartile Range), 'zscore' (Z-score), or 'gpu' (GPU-optimized)
        iqr_factor: Factor for IQR method (2.5 is more conservative for GPU timing)
    
    Returns:
        filtered_times: List of times with outliers removed
        outlier_info: Dict with outlier statistics
    """
    # 增加最小样本要求确保统计有效性  
    if len(times) < 10:
        return times, {"outliers_removed": 0, "original_count": len(times)}
    
    times = np.array(times)
    original_count = len(times)
    
    if method == "gpu":
        # GPU-optimized method: 更保守更稳健的异常值检测
        median_time = np.median(times)
        mad = np.median(np.abs(times - median_time))  # 中位绝对偏差，更稳健
        
        if mad == 0:  # 如果MAD为0，使用标准差
            std_time = np.std(times)
            lower_bound = median_time - 3 * std_time
            upper_bound = median_time + 3 * std_time
        else:
            # 基于MAD的更稳健的界限
            mad_scaled = mad * 1.4826  # 正态分布校正因子
            lower_bound = median_time - 3 * mad_scaled
            upper_bound = median_time + 3 * mad_scaled
        
        # 额外保护：确保不会过度过滤
        lower_bound = max(lower_bound, median_time * 0.2)  # 最多5倍差异
        upper_bound = min(upper_bound, median_time * 5.0)   # 最多5倍差异
        
        mask = (times >= lower_bound) & (times <= upper_bound)
        filtered_times = times[mask]
        
    elif method == "iqr":
        q1 = np.percentile(times, 25)
        q3 = np.percentile(times, 75)
        iqr = q3 - q1
        
        # Handle case where IQR is very small (common in GPU timing)
        if iqr < 0.001:  # If IQR < 0.001ms, use median-based filtering
            median_time = np.median(times)
            # Allow 5x variation around median for small timings
            lower_bound = median_time * 0.2
            upper_bound = median_time * 5.0
        else:
            lower_bound = q1 - iqr_factor * iqr
            upper_bound = q3 + iqr_factor * iqr
        
        # Filter outliers
        mask = (times >= lower_bound) & (times <= upper_bound)
        filtered_times = times[mask]
        
    elif method == "zscore":
        mean_time = np.mean(times)
        std_time = np.std(times)
        z_scores = np.abs((times - mean_time) / std_time) if std_time > 0 else np.zeros_like(times)
        
        # Remove measurements with |z-score| > 2.5 (more conservative)
        mask = z_scores <= 2.5
        filtered_times = times[mask]
    else:
        filtered_times = times
    
    outliers_removed = original_count - len(filtered_times)
    
    # If too many outliers removed, use original data (safety measure)
    if len(filtered_times) < max(3, original_count * 0.3):
        filtered_times = times
        outliers_removed = 0
    
    outlier_info = {
        "outliers_removed": outliers_removed,
        "original_count": original_count,
        "outlier_percentage": (outliers_removed / original_count) * 100 if original_count > 0 else 0
    }
    
    return filtered_times.tolist(), outlier_info


class BenchmarkTimer:
    """Professional benchmark timer with comprehensive metrics and outlier filtering"""
    
    def __init__(self, warmup_iters=50, test_iters=200, outlier_filter=True):
        self.warmup_iters = warmup_iters
        self.test_iters = test_iters
        self.outlier_filter = outlier_filter
        self.gpu_props = torch.cuda.get_device_properties(0)
        self.gpu_properties = self.gpu_props  # For backward compatibility
        self.theoretical_bandwidth_gb_s = self._get_theoretical_bandwidth()
        # Create CUDA events for precise timing
        self.start_event = torch.cuda.Event(enable_timing=True)
        self.end_event = torch.cuda.Event(enable_timing=True)
        
        # 稳定性改进: 预热状态跟踪和缓存清理
        self._warmup_done = False
    
    def _get_theoretical_bandwidth(self) -> float:
        """Get theoretical memory bandwidth for current GPU"""
        gpu_bandwidths = {
            "A100": 1555, "A800": 1555, "V100": 900, "RTX 4090": 1008,
            "RTX 3090": 936, "Tesla T4": 320, "RTX 3080": 760
        }
        gpu_name = self.gpu_properties.name
        for gpu_type, bandwidth in gpu_bandwidths.items():
            if gpu_type in gpu_name:
                return bandwidth
        return 500  # Conservative default
    
    def calculate_flops(self, operation: str, tensor_size: int) -> float:
        """Calculate FLOPs for different operations"""
        flop_counts = {
            # Element-wise binary operations
            "add": tensor_size, "sub": tensor_size, "multiply": tensor_size,
            "divide": tensor_size, "pow": tensor_size * 2,
            
            # Element-wise unary operations
            "negate": tensor_size, "sin": tensor_size * 4, "cos": tensor_size * 4,
            "log": tensor_size * 2, "exp": tensor_size * 2, "sqrt": tensor_size * 2,
            
            # Scalar operations
            "add_scalar": tensor_size, "mul_scalar": tensor_size,
            "divide_scalar": tensor_size, "pow_scalar": tensor_size * 2,
            
            # Activation functions
            "relu": tensor_size, "sigmoid": tensor_size * 4, "tanh": tensor_size * 6,
            "softmax_triton": tensor_size * 6, "safe_softmax_triton": tensor_size * 6,
            "dropout_triton": tensor_size,
            
            # Reduction operations
            "sum": tensor_size, "summation": tensor_size, "max": tensor_size,
            "logsumexp": tensor_size * 6,
            
            # Shape operations (memory-bound, minimal compute)
            "transpose": 0, "reshape": 0, "expand": 0, "view": 0, "flatten": 0,
            "broadcast_to": 0, "squeeze": 0, "unsqueeze": 0,
            
            # Tensor manipulation
            "stack": tensor_size, "cat": tensor_size, "split": tensor_size,
            
            # Matrix operations
            "matmul": tensor_size * 2,  # Simplified, actual depends on dimensions
        }
        return flop_counts.get(operation.lower(), tensor_size)
    
    def calculate_memory_bandwidth(
        self,
        operation: str,
        tensor_sizes: List[int],
        time_ms: float,
        dtype_bytes: int = 4,
    ) -> float:
        """Calculate memory bandwidth in GB/s with proper read/write modeling"""
        if not tensor_sizes or time_ms <= 0:
            return 0
        
        # Define read/write patterns for operations
        op_patterns = {
            # Binary element-wise: 2 reads, 1 write
            "add": (2, 1), "sub": (2, 1), "multiply": (2, 1), "divide": (2, 1),
            
            # Unary operations: 1 read, 1 write
            "pow": (1, 1), "negate": (1, 1), "sin": (1, 1), "cos": (1, 1),
            "log": (1, 1), "exp": (1, 1), "sqrt": (1, 1),
            "relu": (1, 1), "sigmoid": (1, 1), "tanh": (1, 1),
            
            # Scalar operations: 1 read, 1 write
            "add_scalar": (1, 1), "mul_scalar": (1, 1),
            "divide_scalar": (1, 1), "pow_scalar": (1, 1),
            
            # Triton fused operations
            "softmax_triton": (1, 1), "safe_softmax_triton": (1, 1), "dropout_triton": (1, 1),
            
            # Reductions: 1 read, smaller write
            "sum": (1, 0.1), "summation": (1, 0.1), "max": (1, 0.1), "logsumexp": (1, 0.1),
            
            # Shape ops: varies
            "transpose": (1, 1), "reshape": (0, 0), "view": (0, 0),
            "expand": (1, 1), "broadcast_to": (1, 1),
            "flatten": (0, 0), "squeeze": (0, 0), "unsqueeze": (0, 0),
            
            # Tensor manipulation
            "stack": (1, 1), "cat": (1, 1), "split": (1, 1),
            
            # Matrix operations
            "matmul": (2, 1),
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
    
    def benchmark(
        self,
        fn,
        operation: str = "unknown",
        tensor_sizes: List[int] = None,
        *args,
        **kwargs,
    ):
        """Run benchmark with CUDA events for precise timing"""
        tensor_sizes = tensor_sizes or [0]
        
        # 改进的Warmup: 清理缓存并充分预热
        if torch.cuda.is_available():
            torch.cuda.empty_cache()  # 清理GPU缓存确保一致环境
        
        # 充分warmup: 增加迭代次数并确保编译完成
        for i in range(self.warmup_iters):
            try:
                _ = fn(*args, **kwargs)
                # 每隔10次同步一次，避免过度同步开销
                if i % 10 == 9 and torch.cuda.is_available():
                    torch.cuda.synchronize()
            except Exception:
                # Silent failure during warmup
                break
        
        # 最终同步确保所有操作完成
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            # 再次清理缓存确保测试时的一致性
            torch.cuda.empty_cache()
        
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
            return {
                "mean": float("inf"),
                "std": 0,
                "min": float("inf"),
                "max": float("inf"),
                "median": float("inf"),
                "gflops": 0,
                "bandwidth_gb_s": 0,
                "reliability_score": 0.0,
                "outliers_removed": 0,
            }
        
        # Apply outlier filtering if enabled
        if self.outlier_filter and len(times) >= 5:
            # Use more conservative outlier detection for GPU timing
            # GPU measurements often have systematic timing variations
            filtered_times, outlier_info = filter_outliers(times, method="iqr", iqr_factor=2.5)
            # Print warning if too many outliers detected (raised threshold)
            if outlier_info["outlier_percentage"] > 30:
                print(f"  ⚠️  Warning: {outlier_info['outlier_percentage']:.1f}% outliers removed ({outlier_info['outliers_removed']}/{outlier_info['original_count']})")
        else:
            filtered_times = times
            outlier_info = {"outliers_removed": 0, "original_count": len(times)}
        
        if not filtered_times:
            return {
                "mean": float("inf"),
                "std": 0,
                "min": float("inf"),
                "max": float("inf"),
                "median": float("inf"),
                "gflops": 0,
                "bandwidth_gb_s": 0,
                "reliability_score": 0.0,
                "outliers_removed": 0,
            }
        
        # Use median instead of mean for more robust statistics
        median_time = np.median(filtered_times)
        mean_time = np.mean(filtered_times)
        std_time = np.std(filtered_times)
        
        # Calculate reliability score based on coefficient of variation and sample size
        cv = std_time / mean_time if mean_time > 0 else float("inf")
        reliability_score = max(0, 1.0 - min(cv, 1.0)) * min(1.0, len(filtered_times) / 10.0)
        
        tensor_size = tensor_sizes[0] if tensor_sizes else 0
        
        # Calculate professional metrics using median (more robust)
        flops = self.calculate_flops(operation, tensor_size)
        gflops = (flops / 1e9) / (median_time / 1000) if median_time > 0 else 0
        bandwidth = self.calculate_memory_bandwidth(operation, tensor_sizes, median_time, 4)  # Default to float32
        
        return {
            "mean": mean_time,
            "median": median_time,
            "std": std_time,
            "min": np.min(filtered_times),
            "max": np.max(filtered_times),
            "gflops": gflops,
            "bandwidth_gb_s": bandwidth,
            "reliability_score": reliability_score,
            "outliers_removed": outlier_info["outliers_removed"],
            "coefficient_of_variation": cv
        }

    def benchmark_pure_compute(
        self,
        fn,
        operation: str = "unknown",
        tensor_sizes: List[int] = None,
        *args,
        **kwargs,
    ):
        """Benchmark pure computational performance (batch timing with CUDA events)"""
        tensor_sizes = tensor_sizes or [0]
        
        tensor_size = tensor_sizes[0] if tensor_sizes else 0
        
        # Get iteration counts
        warmup_iters, test_iters = self._get_iterations(tensor_size)
        
        # 改进的Warmup策略与real timing一致
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # 充分warmup确保编译完成
        for i in range(warmup_iters):
            try:
                _ = fn(*args, **kwargs)
                # 每隔10次同步一次
                if i % 10 == 9 and torch.cuda.is_available():
                    torch.cuda.synchronize()
            except Exception:
                break
        
        # 最终同步和缓存清理
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        
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
            "mean": mean_time,
            "std": 0,  # No std for batch timing
            "min": mean_time,
            "max": mean_time,
            "median": mean_time,
            "gflops": gflops,
            "bandwidth_gb_s": bandwidth
        }


def get_comprehensive_shapes() -> Dict[str, List[Tuple[int, ...]]]:
    """Get comprehensive test shapes categorized by size"""
    return {
        "small": [
            (256, 256),          # 64K elements
            (512, 512),          # 256K elements  
            (1024, 1024),        # 1M elements
            (32, 32, 32),        # 32K elements (3D)
            (64, 64, 64),        # 256K elements (3D)
        ],
        "medium": [
            (2048, 2048),        # 4M elements
            (4096, 1024),        # 4M elements (non-square)
            (1024, 4096),        # 4M elements (non-square)
            (128, 128, 128),     # 2M elements (3D)
            (32, 256, 256),      # 2M elements (3D)
        ],
        "large": [
            (4096, 4096),        # 16M elements
            (8192, 2048),        # 16M elements (non-square)
            (2048, 8192),        # 16M elements (non-square)
            (256, 256, 256),     # 16M elements (3D)
            (64, 512, 512),      # 16M elements (3D)
        ],
        "very_large": [
            (8192, 8192),        # 64M elements
            (16384, 4096),       # 64M elements (non-square)
            (512, 512, 512),     # 128M elements (3D)
        ],
        "batch": [
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
    """Format benchmark results using median for more reliability"""
    median_time = results.get('median', results.get('mean', 0))
    std_time = results.get('std', 0)
    reliability = results.get('reliability_score', 1.0)
    
    # Use different formatting based on reliability
    if reliability > 0.8:
        status_indicator = ""
    elif reliability > 0.5:
        status_indicator = "~"
    else:
        status_indicator = "?"
    
    return f"{median_time:.3f}±{std_time:.3f}ms{status_indicator}"

def print_professional_header():
    """Print professional result table header"""
    print(f"{'Shape':<15} {'Size':<8} {'PyTorch':<10} {'Genesis(Real)':<12} {'Genesis(Pure)':<12} "
          f"{'Real Speedup':<12} {'Pure Speedup':<12} {'Efficiency':<12} {'Status':<15}")
    print("-" * 140)

def print_professional_row(
    shape: Tuple[int, ...], 
    pytorch_result: Dict, 
    genesis_real_result: Dict, 
    genesis_pure_result: Dict, 
    operation: str,
):
    """Print formatted result row with dual performance metrics and reliability indicators"""
    # Use median for more reliable comparison
    pytorch_time = pytorch_result.get("median", pytorch_result.get("mean", float("inf")))
    genesis_real_time = genesis_real_result.get("median", genesis_real_result.get("mean", float("inf")))
    genesis_pure_time = genesis_pure_result.get("median", genesis_pure_result.get("mean", float("inf")))
    
    real_speedup = pytorch_time / genesis_real_time if genesis_real_time > 0 else 0
    pure_speedup = pytorch_time / genesis_pure_time if genesis_pure_time > 0 else 0
    
    # Calculate relative efficiency vs PyTorch 
    relative_efficiency = real_speedup * 100  # Convert speedup to percentage
    
    status, _ = categorize_performance(real_speedup)  # Use real performance for status
    
    # Add reliability indicators
    pytorch_reliability = pytorch_result.get("reliability_score", 1.0)
    genesis_reliability = genesis_real_result.get("reliability_score", 1.0)
    
    # Add warning for low reliability
    if pytorch_reliability < 0.5 or genesis_reliability < 0.5:
        status = "⚠️  " + status
    
    shape_str = "×".join(map(str, shape))
    if len(shape_str) > 14:
        shape_str = shape_str[:11] + "..."
    
    tensor_size = np.prod(shape)
    
    print(f"{shape_str:<15} {format_size(tensor_size):<8} "
          f"{pytorch_time:.3f}ms{'':<2} {genesis_real_time:.3f}ms{'':<4} {genesis_pure_time:.3f}ms{'':<4} "
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
        genesis_x = genesis.tensor(np_x, device=genesis.device("cuda"))
        
        for act_name, torch_act, genesis_act in activations:
            torch_results = timer.benchmark(torch_act, torch_x)
            genesis_real_results = timer.benchmark(genesis_act, genesis_x)
            
            speedup = torch_results["mean"] / genesis_real_results["mean"]
            
            print(f"{act_name:<20} {format_results(torch_results):<15} "
                  f"{format_results(genesis_real_results):<15} {speedup:.2f}x")
        
        # Clean up
        del torch_x, genesis_x
        gc.collect()
        torch.cuda.empty_cache()

def benchmark_reduction_ops(shapes: List[Tuple[int, ...]], dtype=torch.float32):
    """Benchmark reduction operations with version comparison"""
    print(f"\n{'='*80}")
    print(f"Reduction Operations Benchmark ({dtype})")
    print(f"{'='*80}")
    
    timer = BenchmarkTimer(warmup_iters=10, test_iters=100)
    
    for shape in shapes:
        print(f"\nShape: {shape}")
        print(f"{'Operation':<18} {'PyTorch':<10} {'v1':<10} {'v2':<10} {'v3':<10} {'v1/PT':<6} {'v2/PT':<6} {'v3/PT':<6} {'v2/v1':<6} {'v3/v1':<6}")
        print(f"{'-'*108}")
        
        # Create test data
        np_x = np.random.randn(*shape).astype(np.float32 if dtype == torch.float32 else np.float16)
        
        torch_x = torch.from_numpy(np_x).cuda()
        genesis_x = genesis.tensor(np_x, device=genesis.device("cuda"))
        
        # Test different reduction operations
        reductions = [
            ("Sum (all)", lambda x: x.sum(), lambda x: x.sum()),
            ("Sum (axis=0)", lambda x: x.sum(dim=0), lambda x: x.sum(axis=0)),
            ("Sum (axis=-1)", lambda x: x.sum(dim=-1), lambda x: x.sum(axis=-1)),
            ("Max (all)", lambda x: x.max(), lambda x: x.max()),
            ("Max (axis=-1)", lambda x: x.max(dim=-1)[0], lambda x: x.max(axis=-1)),
        ]
        
        for op_name, torch_op, genesis_op in reductions:
            # Calculate tensor size for benchmark
            tensor_size = np.prod(shape)
            tensor_sizes = [tensor_size]
            
            # Benchmark PyTorch
            torch_results = timer.benchmark(torch_op, op_name, tensor_sizes, torch_x)
            
            # Benchmark Genesis v1 (default)
            os.environ.pop('GENESIS_REDUCTION_VERSION', None)  # Ensure v1 is used
            genesis_v1_results = timer.benchmark(genesis_op, op_name, tensor_sizes, genesis_x)
            
            # Benchmark Genesis v2 
            os.environ['GENESIS_REDUCTION_VERSION'] = 'v2'
            genesis_v2_results = timer.benchmark(genesis_op, op_name, tensor_sizes, genesis_x)
            
            # Benchmark Genesis v3
            os.environ['GENESIS_REDUCTION_VERSION'] = 'v3'
            genesis_v3_results = timer.benchmark(genesis_op, op_name, tensor_sizes, genesis_x)
            
            # Calculate speedups
            v1_speedup = torch_results["mean"] / genesis_v1_results["mean"]
            v2_speedup = torch_results["mean"] / genesis_v2_results["mean"]
            v3_speedup = torch_results["mean"] / genesis_v3_results["mean"]
            v2_vs_v1 = genesis_v1_results["mean"] / genesis_v2_results["mean"]
            v3_vs_v1 = genesis_v1_results["mean"] / genesis_v3_results["mean"]
            
            # Format results (shorter format for compact display)
            torch_str = f"{torch_results.get('median', torch_results.get('mean', 0)):.2f}ms"
            v1_str = f"{genesis_v1_results.get('median', genesis_v1_results.get('mean', 0)):.2f}ms"
            v2_str = f"{genesis_v2_results.get('median', genesis_v2_results.get('mean', 0)):.2f}ms"
            v3_str = f"{genesis_v3_results.get('median', genesis_v3_results.get('mean', 0)):.2f}ms"
            
            print(f"{op_name:<18} {torch_str:<10} {v1_str:<10} {v2_str:<10} {v3_str:<10} "
                  f"{v1_speedup:>4.2f}x {v2_speedup:>4.2f}x {v3_speedup:>4.2f}x {v2_vs_v1:>4.2f}x {v3_vs_v1:>4.2f}x")
        
        # Reset to default
        os.environ.pop('GENESIS_REDUCTION_VERSION', None)
        
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
        genesis_x = genesis.tensor(np_x, device=genesis.device("cuda"))
        
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
            
            speedup = torch_results["mean"] / genesis_real_results["mean"]
            
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
        genesis_a = genesis.tensor(np_a, device=genesis.device("cuda"))
        genesis_b = genesis.tensor(np_b, device=genesis.device("cuda"))
        
        # Benchmark addition with broadcasting
        torch_results = timer.benchmark(lambda: torch_a + torch_b)
        genesis_real_results = timer.benchmark(lambda: genesis_a + genesis_b)
        
        speedup = torch_results["mean"] / genesis_real_results["mean"]
        
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

def generate_professional_summary(
    arithmetic_results: Dict, activation_results: Dict,
    timer: BenchmarkTimer
):
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
                eff = result["efficiency"]
                if 0 < eff < float("inf"):
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
    
    if category_stats.get("small", []):
        small_avg = np.mean(category_stats["small"])
        if small_avg < 0.5:
            print("• Small tensor performance needs major optimization")
    
    if category_stats.get("large", []):
        large_avg = np.mean(category_stats["large"])
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
                    genesis_a = genesis.tensor(np_a, device=genesis.device("cuda"))
                    genesis_b = genesis.tensor(np_b, device=genesis.device("cuda")) if 'pow' not in op_key else None
                    
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
                    genesis_x = genesis.tensor(np_x.astype(np.float32), device=genesis.device("cuda"))  # Genesis uses float32
                    
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

class ComprehensiveBenchmarkRunner:
    """Main benchmark runner for all operations"""
    
    def __init__(self, warmup_iters: int = 20, test_iters: int = 100):
        self.timer = BenchmarkTimer(warmup_iters, test_iters)
        self.registry = ComprehensiveOpRegistry()
        self.results = []
    
    def _create_test_data(self, shape: Tuple[int, ...], dtype=torch.float32):
        """Create test data for benchmarking"""
        np_data = np.random.randn(*shape).astype(np.float32)
        torch_data = torch.from_numpy(np_data).to(dtype).cuda()
        genesis_data = genesis.tensor(np_data, device=genesis.device("cuda"))
        return np_data, torch_data, genesis_data
    
    def _benchmark_operation(self, op_spec: OpSpec, shape: Tuple[int, ...], 
                           dtype=torch.float32, timing_mode="real"):
        """Benchmark a single operation"""
        try:
            # Create test data
            np_data, torch_data, genesis_data = self._create_test_data(shape, dtype)
            
            # Create second tensor if needed
            if op_spec.requires_second_tensor:
                if op_spec.name == "matmul":
                    # Special case for matmul - ensure compatible shapes
                    # For 2D shapes, create a compatible second matrix (transpose-like shape)
                    if len(shape) == 2:
                        # Use a reasonable second dimension (not too large)
                        second_dim = min(shape[0], 4096)  # Cap at 4096 to avoid huge matrices
                        second_shape = (shape[-1], second_dim)
                    else:
                        second_shape = (shape[-1], shape[-1])
                else:
                    second_shape = shape
                    
                _, torch_data2, genesis_data2 = self._create_test_data(second_shape, dtype)
            
            tensor_size = np.prod(shape)
            tensor_sizes = [tensor_size]
            
            # Prepare arguments
            if op_spec.requires_second_tensor:
                torch_args = (torch_data, torch_data2)
                genesis_args = (genesis_data, genesis_data2)
            elif op_spec.requires_scalar:
                scalar = op_spec.special_args.get("scalar", 2.0)
                torch_args = (torch_data,)
                genesis_args = (genesis_data, scalar)
            else:
                torch_args = (torch_data,)
                genesis_args = (genesis_data,)
            
            # Run benchmarks
            pytorch_result = self.timer.benchmark(
                op_spec.torch_func, op_spec.name, tensor_sizes, *torch_args
            )
            
            if timing_mode == "pure":
                genesis_result = self.timer.benchmark_pure_compute(
                    op_spec.genesis_func, op_spec.name, tensor_sizes, *genesis_args
                )
            else:
                genesis_result = self.timer.benchmark(
                    op_spec.genesis_func, op_spec.name, tensor_sizes, *genesis_args
                )
            
            # Calculate metrics
            speedup = pytorch_result['mean'] / genesis_result['mean'] if genesis_result['mean'] > 0 else 0
            efficiency = min(speedup, 1.0)  # Cap at 100%
            
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
            
            return {
                'operation': op_spec.name,
                'category': op_spec.category.value,
                'shape': shape,
                'pytorch_time_ms': pytorch_result['mean'],
                'genesis_time_ms': genesis_result['mean'],
                'speedup': speedup,
                'efficiency': efficiency,
                'gflops': genesis_result['gflops'],
                'bandwidth_gb_s': genesis_result['bandwidth_gb_s'],
                'status': status,
                'error': None
            }
            
        except Exception as e:
            return {
                'operation': op_spec.name,
                'category': op_spec.category.value,
                'shape': shape,
                'pytorch_time_ms': float('inf'),
                'genesis_time_ms': float('inf'),
                'speedup': 0,
                'efficiency': 0,
                'gflops': 0,
                'bandwidth_gb_s': 0,
                'status': "❌ ERROR",
                'error': str(e)
            }
        finally:
            # Clean up
            gc.collect()
            torch.cuda.empty_cache()
    
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
    
    def _print_result_row(self, result):
        """Print a formatted result row"""
        shape_str = "×".join(map(str, result['shape']))
        if len(shape_str) > 18:
            shape_str = shape_str[:15] + "..."
        
        size_str = self._format_size(np.prod(result['shape']))
        
        if result['error']:
            print(f"  {shape_str:<20} {size_str:<8} ERROR: {result['error'][:50]}")
        else:
            print(f"  {shape_str:<20} {size_str:<8} "
                  f"PyTorch: {result['pytorch_time_ms']:.3f}ms | "
                  f"Genesis: {result['genesis_time_ms']:.3f}ms | "
                  f"Speedup: {result['speedup']:.2f}x | "
                  f"BW: {result['bandwidth_gb_s']:.1f}GB/s | {result['status']}")
    
    def benchmark_category(self, category_name: str, max_shapes_per_op: Optional[int] = None,
                          timing_mode: str = "real"):
        """Benchmark all operations in a category"""
        try:
            category = OpCategory(category_name)
        except ValueError:
            print(f"❌ Invalid category: {category_name}")
            return []
        
        operations = self.registry.get_operations_by_category(category)
        category_results = []
        
        print(f"\n{'='*100}")
        print(f"BENCHMARKING {category.value.upper().replace('_', ' ')} OPERATIONS ({timing_mode.upper()} TIMING)")
        print(f"{'='*100}")
        
        for op_name, op_spec in operations.items():
            print(f"\n🔧 Testing {op_name}: {op_spec.description}")
            
            shapes_to_test = op_spec.test_shapes
            if max_shapes_per_op:
                shapes_to_test = shapes_to_test[:max_shapes_per_op]
            
            for shape in shapes_to_test:
                result = self._benchmark_operation(op_spec, shape, timing_mode=timing_mode)
                category_results.append(result)
                self._print_result_row(result)
        
        self.results.extend(category_results)
        return category_results
    
    def benchmark_operation(self, op_name: str, max_shapes: Optional[int] = None,
                          timing_mode: str = "real"):
        """Benchmark a specific operation"""
        # Try exact match first
        op_spec = self.registry.get_operation(op_name)
        if not op_spec:
            # Try partial match
            matching_ops = self.registry.filter_operations_by_name(op_name)
            if not matching_ops:
                print(f"❌ Operation '{op_name}' not found")
                return []
            elif len(matching_ops) == 1:
                op_spec = list(matching_ops.values())[0]
                op_name = list(matching_ops.keys())[0]
            else:
                print(f"❌ Multiple operations match '{op_name}': {list(matching_ops.keys())}")
                return []
        
        print(f"\n{'='*100}")
        print(f"BENCHMARKING {op_name.upper()} OPERATION ({timing_mode.upper()} TIMING)")
        print(f"{'='*100}")
        print(f"Description: {op_spec.description}")
        print(f"Category: {op_spec.category.value}")
        
        shapes_to_test = op_spec.test_shapes
        if max_shapes:
            shapes_to_test = shapes_to_test[:max_shapes]
        
        op_results = []
        for shape in shapes_to_test:
            result = self._benchmark_operation(op_spec, shape, timing_mode=timing_mode)
            op_results.append(result)
            self._print_result_row(result)
        
        self.results.extend(op_results)
        return op_results
    
    def benchmark_all(self, max_shapes_per_op: Optional[int] = None,
                     timing_mode: str = "real"):
        """Benchmark all operations"""
        print(f"🚀 COMPREHENSIVE GENESIS OPERATIONS BENCHMARK ({timing_mode.upper()} TIMING)")
        print(f"GPU: {self.timer.gpu_props.name}")
        print(f"Memory: {self.timer.gpu_props.total_memory / 1024**3:.1f} GB")
        print(f"Theoretical Bandwidth: {self.timer.theoretical_bandwidth_gb_s:.0f} GB/s")
        
        all_results = []
        
        # Benchmark each category
        for category in OpCategory:
            try:
                category_results = self.benchmark_category(category.value, max_shapes_per_op, timing_mode)
                all_results.extend(category_results)
            except Exception as e:
                print(f"⚠️  Category {category.value} failed: {e}")
        
        return all_results
    
    def generate_comprehensive_summary(self, results=None):
        """Generate comprehensive benchmark summary"""
        if results is None:
            results = self.results
        
        if not results:
            print("❌ No results to summarize")
            return
        
        print(f"\n{'='*120}")
        print("COMPREHENSIVE OPERATIONS BENCHMARK SUMMARY")
        print(f"{'='*120}")
        
        # System information
        print(f"\n📊 SYSTEM INFORMATION")
        print(f"GPU: {self.timer.gpu_props.name}")
        print(f"Memory: {self.timer.gpu_props.total_memory / 1024**3:.1f} GB")
        print(f"Theoretical Bandwidth: {self.timer.theoretical_bandwidth_gb_s:.0f} GB/s")
        
        # Overall statistics
        valid_results = [r for r in results if r['error'] is None and r['speedup'] > 0]
        failed_results = [r for r in results if r['error'] is not None]
        
        if valid_results:
            speedups = [r['speedup'] for r in valid_results]
            bandwidths = [r['bandwidth_gb_s'] for r in valid_results if r['bandwidth_gb_s'] > 0]
            
            print(f"\n📈 OVERALL PERFORMANCE STATISTICS")
            print(f"Total Operations Tested: {len(results)}")
            print(f"Successful Tests: {len(valid_results)}")
            print(f"Failed Tests: {len(failed_results)}")
            print(f"Success Rate: {len(valid_results)/len(results)*100:.1f}%")
            print(f"Average Speedup: {np.mean(speedups):.2f}x")
            print(f"Median Speedup: {np.median(speedups):.2f}x")
            print(f"Best Speedup: {np.max(speedups):.2f}x")
            print(f"Worst Speedup: {np.min(speedups):.2f}x")
            if bandwidths:
                print(f"Average Bandwidth: {np.mean(bandwidths):.1f} GB/s")
                print(f"Peak Bandwidth: {np.max(bandwidths):.1f} GB/s")
                print(f"Bandwidth Efficiency: {np.mean(bandwidths)/self.timer.theoretical_bandwidth_gb_s*100:.1f}%")
        
        # Category breakdown
        print(f"\n📊 PERFORMANCE BY CATEGORY")
        print(f"{'Category':<20} {'Tests':<8} {'Success':<8} {'Avg Speedup':<12} {'Best':<8} {'Status':<15}")
        print("-" * 85)
        
        categories = set(r['category'] for r in results)
        for category in categories:
            cat_results = [r for r in results if r['category'] == category]
            cat_valid = [r for r in cat_results if r['error'] is None and r['speedup'] > 0]
            
            if cat_valid:
                speedups = [r['speedup'] for r in cat_valid]
                avg_speedup = np.mean(speedups)
                best_speedup = np.max(speedups)
                success_rate = len(cat_valid) / len(cat_results) * 100
                
                if avg_speedup >= 0.8:
                    status = "🟢 EXCELLENT"
                elif avg_speedup >= 0.6:
                    status = "🟡 GOOD"
                elif avg_speedup >= 0.4:
                    status = "🟠 FAIR"
                else:
                    status = "🔴 POOR"
                
                print(f"{category:<20} {len(cat_results):<8} {success_rate:5.1f}%{'':<2} "
                      f"{avg_speedup:.2f}x{'':<7} {best_speedup:.2f}x{'':<3} {status}")
            else:
                print(f"{category:<20} {len(cat_results):<8} {'0.0%':<7} {'N/A':<12} {'N/A':<8} {'❌ FAILED'}")
        
        # Top and bottom performers
        if valid_results:
            print(f"\n🏆 TOP 5 PERFORMERS")
            top_performers = sorted(valid_results, key=lambda x: x['speedup'], reverse=True)[:5]
            for i, result in enumerate(top_performers, 1):
                shape_str = "×".join(map(str, result['shape']))
                print(f"{i}. {result['operation']} ({shape_str}): {result['speedup']:.2f}x")
            
            print(f"\n⚠️  BOTTOM 5 PERFORMERS")
            bottom_performers = sorted(valid_results, key=lambda x: x['speedup'])[:5]
            for i, result in enumerate(bottom_performers, 1):
                shape_str = "×".join(map(str, result['shape']))
                print(f"{i}. {result['operation']} ({shape_str}): {result['speedup']:.2f}x - {result['status']}")
        
        print(f"\n✅ BENCHMARK COMPLETED")
        success_rate = len(valid_results) / len(results) * 100 if results else 0
        avg_performance = np.mean([r['speedup'] for r in valid_results]) if valid_results else 0
        print(f"Overall Efficiency: {avg_performance*100:.1f}%")
        print(f"Success Rate: {success_rate:.1f}%")


def generate_benchmark_markdown(results, runner, args):
    """Generate markdown documentation for docs directory"""
    try:
        import os
        from datetime import datetime
        
        # Create docs/benchmark directory if it doesn't exist
        docs_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs", "benchmark")
        os.makedirs(docs_dir, exist_ok=True)
        
        # Generate filename based on test parameters
        if args.op:
            filename = f"operations_{args.op}.md"
        elif args.category:
            filename = f"operations_{args.category}.md"
        else:
            filename = f"operations_comprehensive.md"
        
        filepath = os.path.join(docs_dir, filename)
        
        # Generate markdown content
        with open(filepath, 'w') as f:
            f.write(f"# Genesis Operations Benchmark Report\n\n")
            f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # System information
            gpu_props = runner.timer.gpu_props
            f.write(f"## System Information\n\n")
            f.write(f"- **GPU**: {gpu_props.name}\n")
            f.write(f"- **Memory**: {gpu_props.total_memory / 1024**3:.1f} GB\n")
            f.write(f"- **Theoretical Bandwidth**: {runner.timer.theoretical_bandwidth_gb_s:.0f} GB/s\n")
            f.write(f"- **Multi-processors**: {gpu_props.multi_processor_count}\n")
            
            # Test configuration
            f.write(f"\n## Test Configuration\n\n")
            f.write(f"- **Mode**: {'Fast' if args.fast else 'Comprehensive'}\n")
            f.write(f"- **Timing**: {args.timing}\n")
            f.write(f"- **Data Type**: {args.dtype}\n")
            if args.op:
                f.write(f"- **Operation**: {args.op}\n")
            if args.category:
                f.write(f"- **Category**: {args.category}\n")
            
            # Performance summary
            valid_results = [r for r in results if r['error'] is None and r['speedup'] > 0]
            failed_results = [r for r in results if r['error'] is not None]
            
            f.write(f"\n## Performance Summary\n\n")
            f.write(f"| Metric | Value |\n")
            f.write(f"|--------|-------|\n")
            f.write(f"| Total Tests | {len(results)} |\n")
            f.write(f"| Successful Tests | {len(valid_results)} |\n")
            f.write(f"| Failed Tests | {len(failed_results)} |\n")
            f.write(f"| Success Rate | {len(valid_results)/len(results)*100:.1f}% |\n")
            
            if valid_results:
                speedups = [r['speedup'] for r in valid_results]
                f.write(f"| Average Speedup | {np.mean(speedups):.2f}x |\n")
                f.write(f"| Median Speedup | {np.median(speedups):.2f}x |\n")
                f.write(f"| Best Speedup | {np.max(speedups):.2f}x |\n")
                f.write(f"| Worst Speedup | {np.min(speedups):.2f}x |\n")
            
            # Category performance breakdown
            f.write(f"\n## Performance by Category\n\n")
            f.write(f"| Category | Tests | Success Rate | Avg Speedup | Best Speedup | Status |\n")
            f.write(f"|----------|-------|--------------|-------------|--------------|--------|\n")
            
            categories = set(r['category'] for r in results)
            for category in sorted(categories):
                cat_results = [r for r in results if r['category'] == category]
                cat_valid = [r for r in cat_results if r['error'] is None and r['speedup'] > 0]
                
                if cat_valid:
                    speedups = [r['speedup'] for r in cat_valid]
                    avg_speedup = np.mean(speedups)
                    best_speedup = np.max(speedups)
                    success_rate = len(cat_valid) / len(cat_results) * 100
                    
                    if avg_speedup >= 0.8:
                        status = "🟢 Excellent"
                    elif avg_speedup >= 0.6:
                        status = "🟡 Good"
                    elif avg_speedup >= 0.4:
                        status = "🟠 Fair"
                    else:
                        status = "🔴 Poor"
                    
                    f.write(f"| {category} | {len(cat_results)} | {success_rate:.1f}% | {avg_speedup:.2f}x | {best_speedup:.2f}x | {status} |\n")
                else:
                    f.write(f"| {category} | {len(cat_results)} | 0.0% | N/A | N/A | ❌ Failed |\n")
            
            # Detailed results table
            f.write(f"\n## Detailed Results\n\n")
            f.write(f"| Operation | Category | Shape | PyTorch (ms) | Genesis (ms) | Speedup | Bandwidth (GB/s) | Status |\n")
            f.write(f"|-----------|----------|-------|--------------|--------------|---------|------------------|--------|\n")
            
            for result in sorted(results, key=lambda x: x.get('speedup', 0), reverse=True):
                if result['error']:
                    continue
                    
                shape_str = "×".join(map(str, result['shape']))
                f.write(f"| {result['operation']} | {result['category']} | {shape_str} | ")
                f.write(f"{result['pytorch_time_ms']:.3f} | {result['genesis_time_ms']:.3f} | ")
                f.write(f"{result['speedup']:.2f}x | {result['bandwidth_gb_s']:.1f} | {result['status']} |\n")
            
            # Performance charts (simple text-based)
            f.write(f"\n## Performance Distribution\n\n")
            if valid_results:
                excellent = sum(1 for r in valid_results if r['speedup'] >= 0.9)
                good = sum(1 for r in valid_results if 0.7 <= r['speedup'] < 0.9)
                fair = sum(1 for r in valid_results if 0.5 <= r['speedup'] < 0.7)
                poor = sum(1 for r in valid_results if 0.2 <= r['speedup'] < 0.5)
                critical = sum(1 for r in valid_results if r['speedup'] < 0.2)
                total = len(valid_results)
                
                f.write(f"- 🟢 **Excellent (≥90%)**: {excellent} tests ({excellent/total*100:.1f}%)\n")
                f.write(f"- 🟡 **Good (70-90%)**: {good} tests ({good/total*100:.1f}%)\n")
                f.write(f"- 🟠 **Fair (50-70%)**: {fair} tests ({fair/total*100:.1f}%)\n")
                f.write(f"- 🔴 **Poor (20-50%)**: {poor} tests ({poor/total*100:.1f}%)\n")
                f.write(f"- ❌ **Critical (<20%)**: {critical} tests ({critical/total*100:.1f}%)\n")
            
            # Top performers
            if valid_results:
                f.write(f"\n## Top 10 Performers\n\n")
                top_performers = sorted(valid_results, key=lambda x: x['speedup'], reverse=True)[:10]
                f.write(f"| Rank | Operation | Shape | Speedup | Status |\n")
                f.write(f"|------|-----------|-------|---------|--------|\n")
                for i, result in enumerate(top_performers, 1):
                    shape_str = "×".join(map(str, result['shape']))
                    f.write(f"| {i} | {result['operation']} | {shape_str} | {result['speedup']:.2f}x | {result['status']} |\n")
        
        print(f"\n📄 Markdown report generated: {filepath}")
        
    except Exception as e:
        print(f"⚠️  Failed to generate markdown report: {e}")

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Comprehensive Genesis Operations Benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python bench_ops.py                           # Full comprehensive benchmark
    python bench_ops.py --fast                    # Quick test mode
    python bench_ops.py --category element        # Test specific category
    python bench_ops.py --op add                  # Test specific operation
    python bench_ops.py --size medium             # Test medium-sized tensors
    python bench_ops.py --timing pure             # Use pure compute timing
    python bench_ops.py --op matmul --timing both # Test matmul with both timing modes
        """
    )
    
    parser.add_argument("--fast", action="store_true",
                       help="Quick mode with reduced iterations and shapes")
    parser.add_argument("--category", type=str, 
                       choices=[cat.value for cat in OpCategory],
                       help="Test specific category only")
    parser.add_argument("--op", type=str,
                       help="Test specific operation only (supports partial matching)")
    parser.add_argument("--size", type=str, choices=["small", "medium", "large", "batch"],
                       help="Test specific tensor size category")
    parser.add_argument("--timing", type=str, choices=["real", "pure", "both"], default="real",
                       help="Timing mode: real (with sync overhead), pure (batch timing), both")
    parser.add_argument("--dtype", type=str, choices=["float32", "float16", "bfloat16"], 
                       default="float32", help="Data type for testing")
    parser.add_argument("--list-categories", action="store_true",
                       help="List available categories and exit")
    parser.add_argument("--list-ops", action="store_true",
                       help="List available operations and exit")
    parser.add_argument("--list-sizes", action="store_true",
                       help="List available size categories and exit")
    parser.add_argument("--no-outlier-filter", action="store_true",
                       help="Disable outlier filtering (use all raw measurements)")
    parser.add_argument("--outlier-method", type=str, choices=["iqr", "zscore", "gpu"], 
                       default="iqr", help="Outlier detection method: iqr(conservative), zscore(statistical), gpu(extreme-only)")
    parser.add_argument("--min-iterations", type=int, default=None,
                       help="Minimum iterations for statistical reliability (overrides fast mode)")
    parser.add_argument("--show-reliability", action="store_true",
                       help="Show detailed reliability metrics in output")
    
    return parser.parse_args()


def main():
    """Main benchmark execution"""
    args = parse_args()
    
    # Create registry for listing
    registry = ComprehensiveOpRegistry()
    
    if args.list_categories:
        print("Available Categories:")
        print("=" * 60)
        for category in OpCategory:
            ops = registry.get_operations_by_category(category)
            print(f"\n{category.value.upper().replace('_', ' ')} ({len(ops)} operations):")
            for op_name in sorted(ops.keys()):
                print(f"  • {op_name}")
        return
    
    if args.list_ops:
        print("Available Operations:")
        print("=" * 60)
        all_ops = registry.list_operations()
        by_category = {}
        for op_name in all_ops:
            op_spec = registry.get_operation(op_name)
            if op_spec:
                cat = op_spec.category.value
                if cat not in by_category:
                    by_category[cat] = []
                by_category[cat].append(op_name)
        
        for category in sorted(by_category.keys()):
            print(f"\n{category.upper().replace('_', ' ')}:")
            for op in sorted(by_category[category]):
                print(f"  {op}")
        
        print(f"\nTotal: {len(all_ops)} operations")
        return
    
    if args.list_sizes:
        print("Available Size Categories:")
        print("=" * 50)
        shapes_dict = get_comprehensive_shapes()
        for category, shapes in shapes_dict.items():
            print(f"\n{category.upper()}:")
            for shape in shapes:
                size = np.prod(shape)
                print(f"  - {shape} ({format_size(size)} elements)")
        return
    
    # Configure benchmark
    if args.fast:
        warmup_iters, test_iters = 5, 20
        max_shapes = 2
        print("⚡ Fast mode: Reduced iterations and shapes")
    else:
        warmup_iters, test_iters = 20, 100
        max_shapes = None
    
    # Override with minimum iterations if specified
    if args.min_iterations:
        test_iters = max(test_iters, args.min_iterations)
        print(f"📊 Using minimum {test_iters} iterations for statistical reliability")
    
    # Configure outlier filtering
    outlier_filter = not args.no_outlier_filter
    if not outlier_filter:
        print("⚠️  Outlier filtering disabled - using all raw measurements")
    elif args.show_reliability:
        print(f"📈 Using {args.outlier_method.upper()} outlier detection with reliability metrics")
    
    # Print configuration
    mode = "FAST" if args.fast else "COMPREHENSIVE"
    print(f"🚀 Genesis {mode} Operations Benchmark")
    print("=" * 80)
    
    if args.op:
        print(f"🎯 Testing operation: {args.op.upper()}")
    if args.category:
        print(f"📏 Testing category: {args.category.upper()}")
    if args.timing != "real":
        print(f"⏱️  Timing mode: {args.timing.upper()}")
    if args.dtype != "float32":
        print(f"🔢 Data type: {args.dtype.upper()}")
    
    # Create benchmark runner
    runner = ComprehensiveBenchmarkRunner(warmup_iters, test_iters)
    # Update outlier filtering setting
    runner.timer.outlier_filter = outlier_filter
    
    # Run benchmarks
    all_results = []
    timing_modes = ["real", "pure"] if args.timing == "both" else [args.timing]
    
    for timing_mode in timing_modes:
        if args.op:
            results = runner.benchmark_operation(args.op, max_shapes, timing_mode)
        elif args.category:
            results = runner.benchmark_category(args.category, max_shapes, timing_mode)
        else:
            results = runner.benchmark_all(max_shapes, timing_mode)
        
        all_results.extend(results)
    
    # Generate summary
    runner.generate_comprehensive_summary(all_results)
    
    # Generate markdown documentation for docs directory
    generate_benchmark_markdown(all_results, runner, args)


if __name__ == "__main__":
    main()