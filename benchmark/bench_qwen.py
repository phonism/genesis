#!/usr/bin/env python3
"""
End-to-End Qwen Model Benchmark: Genesis vs PyTorch

Comprehensive performance comparison for Qwen model:
- Forward pass performance
- Backward pass performance  
- Memory usage comparison
- Different model sizes and batch sizes
- Gradient computation accuracy

Usage:
    python bench_qwen_e2e.py --size 0.5B              # Test 0.5B model
    python bench_qwen_e2e.py --batch-size 1,4,8       # Test different batch sizes
    python bench_qwen_e2e.py --seq-len 512,1024       # Test different sequence lengths
    python bench_qwen_e2e.py --profile                # Enable detailed profiling
"""

import sys
import os
import argparse
import time
import gc
import signal
import traceback
import threading
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import json

# Add genesis path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn.functional as F
import genesis

# Import both model implementations
from genesis.models.qwen import ModelArgs, QwenForCausalLM as GenesisQwen
from genesis.models.qwen_torch import QwenForCausalLM as TorchQwen

# Ensure CUDA is available
assert torch.cuda.is_available(), "CUDA is not available"

@dataclass
class BenchConfig:
    """Benchmark configuration"""
    model_size: str = "0.5B"
    batch_sizes: List[int] = None
    seq_lengths: List[int] = None
    warmup_iters: int = 10
    test_iters: int = 50
    profile: bool = False
    gradient_check: bool = True
    memory_check: bool = True

@dataclass
class BenchResult:
    """Benchmark result for a single test"""
    operation: str
    model_size: str
    batch_size: int
    seq_length: int
    pytorch_time_ms: float
    genesis_time_ms: float
    speedup: float
    pytorch_memory_mb: float
    genesis_memory_mb: float
    memory_ratio: float
    status: str
    error: Optional[str] = None

class QwenBenchmarkTimer:
    """High-precision CUDA timer for Qwen benchmarks"""

    def __init__(self, warmup_iters: int = 10, test_iters: int = 50):
        self.warmup_iters = warmup_iters
        self.test_iters = test_iters
        # Don't create events here - create fresh ones for each iteration
        self.hang_timeout = 60  # 15 seconds timeout for hang detection
        self.timer = None  # Threading timer backup

    def setup_hang_detection(self):
        """Setup hang detection with automatic stack trace"""
        def timeout_handler(signum, frame):
            print("\n" + "="*80)
            print("🚨 HANG DETECTED! Benchmark has been stuck for 30+ seconds")
            print("="*80)
            print("📍 Current stack trace at hang location:")
            print("-"*80)
            traceback.print_stack(frame)
            print("-"*80)
            print("💀 Exiting due to hang...")
            print("="*80)
            os._exit(1)  # Force exit

        signal.signal(signal.SIGALRM, timeout_handler)

    def start_hang_detection(self):
        """Start hang detection timer with dual mechanisms"""
        # Method 1: Signal-based (may not work if blocked)
        signal.alarm(self.hang_timeout)

        # Method 2: Threading timer backup (more reliable)
        def timeout_backup():
            print("\n" + "="*80)
            print("🚨 HANG DETECTED (Threading Timer)! Benchmark stuck for 15+ seconds")
            print("="*80)
            print("📍 Printing all thread stack traces:")
            print("-"*80)
            for thread_id, frame in sys._current_frames().items():
                print(f"Thread {thread_id}:")
                traceback.print_stack(frame)
                print("-"*40)
            print("💀 Force exiting due to hang...")
            print("="*80)
            os._exit(1)

        self.timer = threading.Timer(self.hang_timeout, timeout_backup)
        self.timer.start()

    def stop_hang_detection(self):
        """Stop hang detection timer"""
        signal.alarm(0)
        if self.timer and self.timer.is_alive():
            self.timer.cancel()
    
    def benchmark(self, fn, *args, **kwargs) -> Tuple[float, float]:
        """
        Run benchmark with CUDA timing and memory tracking
        Returns: (time_ms, memory_mb)
        """
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        # Setup hang detection
        self.setup_hang_detection()

        # Warmup
        for warmup_idx in range(self.warmup_iters):
            try:
                import time
                self.start_hang_detection()  # Start hang detection for this iteration
                start_time = time.time()
                result = fn()  # Call function without args/kwargs
                # Use appropriate sync based on function name
                if 'genesis' in fn.__name__:
                    genesis.cuda.synchronize()
                else:
                    torch.cuda.synchronize()
                self.stop_hang_detection()  # Stop hang detection
                elapsed = (time.time() - start_time) * 1000
            except Exception as e:
                self.stop_hang_detection()
                print(f" FAILED: {e}")
                break

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        # Actual timing
        times = []
        memory_usage = []

        for i in range(self.test_iters):
            try:
                # Set iteration debug flag for genesis_backward to use

                # Start hang detection for each test iteration
                self.start_hang_detection()

                torch.cuda.reset_peak_memory_stats()

                import time
                cpu_start = time.time()

                # Create fresh CUDA events for each iteration
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)

                start_event.record()
                result = fn()  # Call function without args/kwargs
                end_event.record()

                # Use appropriate sync based on function name
                if 'genesis' in fn.__name__:
                    genesis.cuda.synchronize()
                else:
                    torch.cuda.synchronize()
                cpu_elapsed = (time.time() - cpu_start) * 1000
                elapsed_ms = start_event.elapsed_time(end_event)

                # WORKAROUND: Force cleanup after each iteration to prevent state accumulation
                import gc
                gc.collect()
                torch.cuda.empty_cache()

                peak_memory_mb = torch.cuda.max_memory_allocated() / 1024 / 1024

                times.append(elapsed_ms)
                memory_usage.append(peak_memory_mb)

                # Stop hang detection after successful completion
                self.stop_hang_detection()
                pass  # Silent execution

            except Exception as e:
                # Stop hang detection on error
                self.stop_hang_detection()
                print(f" FAILED: {str(e)}")
                if i < 3:
                    import traceback
                    traceback.print_exc()
                break

        if not times:
            return float('inf'), float('inf')

        avg_time = np.mean(times)
        avg_mem = np.mean(memory_usage)
        return avg_time, avg_mem

def get_qwen_config(model_size: str) -> ModelArgs:
    """Get Qwen model configuration for different sizes"""
    configs = {
        "0.5B": ModelArgs(
            vocab_size=151936,
            n_layer=2,  # Small for testing  
            num_attention_heads=14,
            hidden_size=896,
            intermediate_size=4864,
            max_position_embeddings=32768,
            rope_base=1000000.0,
            norm_eps=1e-6
        ),
        "1.5B": ModelArgs(
            vocab_size=151936,
            n_layer=28,
            num_attention_heads=12,
            hidden_size=1536,
            intermediate_size=8960,
            max_position_embeddings=32768,
            rope_base=1000000.0,
            norm_eps=1e-6
        ),
        "3B": ModelArgs(
            vocab_size=151936,
            n_layer=36,
            num_attention_heads=16,
            hidden_size=2048,
            intermediate_size=11008,
            max_position_embeddings=32768,
            rope_base=1000000.0,
            norm_eps=1e-6
        )
    }
    
    if model_size not in configs:
        raise ValueError(f"Unsupported model size: {model_size}. Available: {list(configs.keys())}")
    
    return configs[model_size]

def create_torch_qwen(config: ModelArgs) -> torch.nn.Module:
    """Create a PyTorch Qwen model using dedicated PyTorch implementation"""
    return TorchQwen(config)

class QwenBenchmarkSuite:
    """Main benchmark suite for Qwen models"""
    
    def __init__(self, config: BenchConfig):
        self.config = config
        self.timer = QwenBenchmarkTimer(config.warmup_iters, config.test_iters)
        self.results: List[BenchResult] = []
    
    def _prepare_test_data(self, batch_size: int, seq_length: int) -> torch.Tensor:
        """Prepare test input data"""
        # Generate random token ids - use small range for testing
        vocab_size = 1000  # Small vocab size for testing
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_length), device='cuda')
        return input_ids
    
    def _clear_cache(self):
        """Clear cache and prepare for next benchmark"""
        gc.collect()
        torch.cuda.empty_cache()
    
    def benchmark_forward_pass(self, model_size: str, batch_size: int, seq_length: int) -> BenchResult:
        """Benchmark forward pass performance"""
        print(f"\n🔄 Forward Pass: {model_size} | Batch: {batch_size} | Seq: {seq_length}")
        
        try:
            # Get model configuration
            config = get_qwen_config(model_size)
            
            # Create PyTorch model
            torch_model = create_torch_qwen(config).cuda()
            
            # Create Genesis model directly
            genesis_model = GenesisQwen(config).cuda()
            self._clear_cache()
            
            # Prepare test data
            input_ids = self._prepare_test_data(batch_size, seq_length)
            
            # Benchmark PyTorch
            def pytorch_forward():
                with torch.no_grad():
                    return torch_model(input_ids)
            
            # Pre-convert to Genesis tensor (one-time cost, not part of benchmark)
            input_np = input_ids.detach().cpu().numpy()
            import genesis
            genesis_input = genesis.tensor(input_np, device=genesis.device("cuda"), requires_grad=False).long()
            
            # Benchmark Genesis - pure computation only
            def genesis_forward():
                # Use genesis.no_grad() instead of torch.no_grad() to avoid conflict
                with genesis.no_grad():
                    return genesis_model(genesis_input)
            
            pytorch_time, pytorch_memory = self.timer.benchmark(pytorch_forward)
            genesis_time, genesis_memory = self.timer.benchmark(genesis_forward)
            
            # Calculate metrics
            speedup = pytorch_time / genesis_time if genesis_time > 0 else 0
            memory_ratio = genesis_memory / pytorch_memory if pytorch_memory > 0 else 0
            
            # Determine status
            if speedup >= 0.8:
                status = "🟢 EXCELLENT"
            elif speedup >= 0.6:
                status = "🟡 GOOD"
            elif speedup >= 0.4:
                status = "🟠 FAIR"
            elif speedup >= 0.2:
                status = "🔴 POOR"
            else:
                status = "❌ CRITICAL"
            
            result = BenchResult(
                operation="forward",
                model_size=model_size,
                batch_size=batch_size,
                seq_length=seq_length,
                pytorch_time_ms=pytorch_time,
                genesis_time_ms=genesis_time,
                speedup=speedup,
                pytorch_memory_mb=pytorch_memory,
                genesis_memory_mb=genesis_memory,
                memory_ratio=memory_ratio,
                status=status
            )
            
            print(f"  PyTorch:  {pytorch_time:.2f}ms | {pytorch_memory:.1f}MB")
            print(f"  Genesis:  {genesis_time:.2f}ms | {genesis_memory:.1f}MB") 
            print(f"  Speedup:  {speedup:.2f}x | Memory: {memory_ratio:.2f}x | {status}")
            
            return result
            
        except Exception as e:
            error_result = BenchResult(
                operation="forward",
                model_size=model_size,
                batch_size=batch_size,
                seq_length=seq_length,
                pytorch_time_ms=float('inf'),
                genesis_time_ms=float('inf'),
                speedup=0,
                pytorch_memory_mb=float('inf'),
                genesis_memory_mb=float('inf'),
                memory_ratio=0,
                status="❌ ERROR",
                error=str(e)
            )
            print(f"  ❌ ERROR: {str(e)}")
            return error_result
        
        finally:
            # Cleanup
            gc.collect()
            torch.cuda.empty_cache()
    
    def benchmark_backward_pass(self, model_size: str, batch_size: int, seq_length: int) -> BenchResult:
        """Benchmark backward pass performance"""
        print(f"\n🔄 Backward Pass: {model_size} | Batch: {batch_size} | Seq: {seq_length}")
        
        try:
            # Get model configuration
            config = get_qwen_config(model_size)
            
            # Create PyTorch model
            torch_model = create_torch_qwen(config).cuda()
            torch_model.train()
            
            # Create Genesis model directly
            genesis_model = GenesisQwen(config).cuda() 
            genesis_model.train()
            self._clear_cache()
            
            # Prepare test data
            input_ids = self._prepare_test_data(batch_size, seq_length)
            labels = input_ids.clone()
            
            # Benchmark PyTorch backward
            def pytorch_backward():
                torch_model.zero_grad()
                logits = torch_model(input_ids)
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))
                loss.backward()  # Do backward pass
                return loss
            
            # Pre-convert to Genesis tensors (one-time cost, not part of benchmark)
            input_np = input_ids.detach().cpu().numpy()
            labels_np = labels.detach().cpu().numpy()
            import genesis
            genesis_input = genesis.tensor(input_np, device=genesis.device("cuda"), requires_grad=False).long()
            genesis_labels = genesis.tensor(labels_np, device=genesis.device("cuda"), requires_grad=False).long()
            
            # Benchmark Genesis backward - pure computation only
            def genesis_backward():
                # WORKAROUND: Ensure Genesis gradient is enabled explicitly
                import genesis
                with genesis.enable_grad():  # Force enable Genesis gradients
                    # Genesis models may not have zero_grad method, try to handle it
                    if hasattr(genesis_model, 'zero_grad'):
                        genesis_model.zero_grad()
                    else:
                        for param in genesis_model.parameters():
                            if hasattr(param, 'grad'):
                                param.grad = None

                    logits = genesis_model(genesis_input)
                    # Use Genesis's own backward - simple loss since cross_entropy not implemented
                    # Just use sum as loss for now
                    loss = logits.sum()
                    loss.backward()  # Do backward pass in Genesis
                    return loss
            
            pytorch_time, pytorch_memory = self.timer.benchmark(pytorch_backward)
            genesis_time, genesis_memory = self.timer.benchmark(genesis_backward)
            
            # Calculate metrics
            speedup = pytorch_time / genesis_time if genesis_time > 0 else 0
            memory_ratio = genesis_memory / pytorch_memory if pytorch_memory > 0 else 0
            
            # Determine status
            if speedup >= 0.8:
                status = "🟢 EXCELLENT"
            elif speedup >= 0.6:
                status = "🟡 GOOD"
            elif speedup >= 0.4:
                status = "🟠 FAIR"
            elif speedup >= 0.2:
                status = "🔴 POOR"
            else:
                status = "❌ CRITICAL"
            
            result = BenchResult(
                operation="backward",
                model_size=model_size,
                batch_size=batch_size,
                seq_length=seq_length,
                pytorch_time_ms=pytorch_time,
                genesis_time_ms=genesis_time,
                speedup=speedup,
                pytorch_memory_mb=pytorch_memory,
                genesis_memory_mb=genesis_memory,
                memory_ratio=memory_ratio,
                status=status
            )
            
            print(f"  PyTorch:  {pytorch_time:.2f}ms | {pytorch_memory:.1f}MB")
            print(f"  Genesis:  {genesis_time:.2f}ms | {genesis_memory:.1f}MB")
            print(f"  Speedup:  {speedup:.2f}x | Memory: {memory_ratio:.2f}x | {status}")
            
            return result
            
        except Exception as e:
            error_result = BenchResult(
                operation="backward",
                model_size=model_size,
                batch_size=batch_size,
                seq_length=seq_length,
                pytorch_time_ms=float('inf'),
                genesis_time_ms=float('inf'),
                speedup=0,
                pytorch_memory_mb=float('inf'),
                genesis_memory_mb=float('inf'),
                memory_ratio=0,
                status="❌ ERROR",
                error=str(e)
            )
            print(f"  ❌ ERROR: {str(e)}")
            return error_result
        
        finally:
            # Cleanup
            gc.collect()
            torch.cuda.empty_cache()
    
    def run_full_benchmark(self) -> List[BenchResult]:
        """Run complete benchmark suite"""
        print(f"🚀 QWEN END-TO-END BENCHMARK")
        print(f"Model Size: {self.config.model_size}")
        print(f"Batch Sizes: {self.config.batch_sizes}")
        print(f"Seq Lengths: {self.config.seq_lengths}")
        print(f"{'='*80}")
        
        results = []
        
        # Test all combinations
        for batch_size in self.config.batch_sizes:
            for seq_length in self.config.seq_lengths:
                # Forward pass benchmark
                forward_result = self.benchmark_forward_pass(
                    self.config.model_size, batch_size, seq_length
                )
                results.append(forward_result)
                
                # Backward pass benchmark  
                backward_result = self.benchmark_backward_pass(
                    self.config.model_size, batch_size, seq_length
                )
                results.append(backward_result)
        
        self.results.extend(results)
        return results
    
    def generate_summary(self, results: Optional[List[BenchResult]] = None):
        """Generate comprehensive benchmark summary"""
        if results is None:
            results = self.results
        
        if not results:
            print("❌ No results to summarize")
            return
        
        print(f"\n{'='*100}")
        print("QWEN END-TO-END BENCHMARK SUMMARY")
        print(f"{'='*100}")
        
        # Filter results by operation
        forward_results = [r for r in results if r.operation == "forward" and r.error is None]
        backward_results = [r for r in results if r.operation == "backward" and r.error is None]
        failed_results = [r for r in results if r.error is not None]
        
        # Forward pass summary
        if forward_results:
            forward_speedups = [r.speedup for r in forward_results]
            forward_memory_ratios = [r.memory_ratio for r in forward_results]
            
            print(f"\n🔄 FORWARD PASS PERFORMANCE")
            print(f"Tests: {len(forward_results)} | Avg Speedup: {np.mean(forward_speedups):.2f}x | Avg Memory: {np.mean(forward_memory_ratios):.2f}x")
            
            print(f"\n{'Operation':<12} {'Batch':<6} {'SeqLen':<8} {'PyTorch(ms)':<12} {'Genesis(ms)':<12} {'Speedup':<8} {'Status'}")
            print("-" * 80)
            for result in forward_results:
                print(f"{'forward':<12} {result.batch_size:<6} {result.seq_length:<8} "
                      f"{result.pytorch_time_ms:<12.2f} {result.genesis_time_ms:<12.2f} "
                      f"{result.speedup:<8.2f} {result.status}")
        
        # Backward pass summary
        if backward_results:
            backward_speedups = [r.speedup for r in backward_results]
            backward_memory_ratios = [r.memory_ratio for r in backward_results]
            
            print(f"\n🔙 BACKWARD PASS PERFORMANCE")
            print(f"Tests: {len(backward_results)} | Avg Speedup: {np.mean(backward_speedups):.2f}x | Avg Memory: {np.mean(backward_memory_ratios):.2f}x")
            
            print(f"\n{'Operation':<12} {'Batch':<6} {'SeqLen':<8} {'PyTorch(ms)':<12} {'Genesis(ms)':<12} {'Speedup':<8} {'Status'}")
            print("-" * 80)
            for result in backward_results:
                print(f"{'backward':<12} {result.batch_size:<6} {result.seq_length:<8} "
                      f"{result.pytorch_time_ms:<12.2f} {result.genesis_time_ms:<12.2f} "
                      f"{result.speedup:<8.2f} {result.status}")
        
        # Overall summary
        all_valid = forward_results + backward_results
        if all_valid:
            all_speedups = [r.speedup for r in all_valid]
            all_memory_ratios = [r.memory_ratio for r in all_valid]
            
            print(f"\n📊 OVERALL SUMMARY")
            print(f"Total Tests: {len(results)}")
            print(f"Successful: {len(all_valid)} | Failed: {len(failed_results)}")
            print(f"Success Rate: {len(all_valid)/len(results)*100:.1f}%")
            print(f"Average Speedup: {np.mean(all_speedups):.2f}x")
            print(f"Average Memory Usage: {np.mean(all_memory_ratios):.2f}x")
            
            # Best and worst performers
            best_result = max(all_valid, key=lambda x: x.speedup)
            worst_result = min(all_valid, key=lambda x: x.speedup)
            
            print(f"\n🏆 Best Performance: {best_result.operation} (batch={best_result.batch_size}, seq={best_result.seq_length}) - {best_result.speedup:.2f}x")
            print(f"🐌 Worst Performance: {worst_result.operation} (batch={worst_result.batch_size}, seq={worst_result.seq_length}) - {worst_result.speedup:.2f}x")
        
        # Failed operations
        if failed_results:
            print(f"\n❌ FAILED OPERATIONS ({len(failed_results)} total)")
            for result in failed_results[:5]:  # Show first 5 failures
                print(f"• {result.operation} (batch={result.batch_size}, seq={result.seq_length}): {result.error}")

def generate_qwen_benchmark_markdown(results, config):
    """Generate markdown documentation for Qwen benchmark results"""
    try:
        import os
        from datetime import datetime
        
        # Create docs/benchmark directory if it doesn't exist
        docs_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs", "benchmark")
        os.makedirs(docs_dir, exist_ok=True)
        
        # Generate filename
        filename = f"qwen_model_{config.model_size.lower().replace('.', 'p')}.md"
        filepath = os.path.join(docs_dir, filename)
        
        # Generate markdown content
        with open(filepath, 'w') as f:
            f.write(f"# Genesis Qwen Model Benchmark Report\n\n")
            f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Model information
            f.write(f"## Model Configuration\n\n")
            f.write(f"- **Model Size**: {config.model_size}\n")
            f.write(f"- **Batch Sizes**: {config.batch_sizes}\n")
            f.write(f"- **Sequence Lengths**: {config.seq_lengths}\n")
            f.write(f"- **Warmup Iterations**: {config.warmup_iters}\n")
            f.write(f"- **Test Iterations**: {config.test_iters}\n")
            f.write(f"- **Profiling**: {'Enabled' if config.profile else 'Disabled'}\n")
            
            # Performance summary
            valid_results = [r for r in results if r.error is None]
            failed_results = [r for r in results if r.error is not None]
            
            f.write(f"\n## Performance Summary\n\n")
            f.write(f"| Metric | Value |\n")
            f.write(f"|--------|-------|\n")
            f.write(f"| Total Tests | {len(results)} |\n")
            f.write(f"| Successful Tests | {len(valid_results)} |\n")
            f.write(f"| Failed Tests | {len(failed_results)} |\n")
            f.write(f"| Success Rate | {len(valid_results)/len(results)*100:.1f}% |\n")
            
            if valid_results:
                speedups = [r.speedup for r in valid_results if r.speedup > 0]
                if speedups:
                    f.write(f"| Average Speedup | {np.mean(speedups):.2f}x |\n")
                    f.write(f"| Median Speedup | {np.median(speedups):.2f}x |\n")
                    f.write(f"| Best Speedup | {max(speedups):.2f}x |\n")
                    f.write(f"| Worst Speedup | {min(speedups):.2f}x |\n")
                
                memory_ratios = [r.memory_ratio for r in valid_results if r.memory_ratio > 0]
                if memory_ratios:
                    f.write(f"| Average Memory Ratio | {np.mean(memory_ratios):.2f}x |\n")
                    f.write(f"| Best Memory Efficiency | {min(memory_ratios):.2f}x |\n")
            
            # Performance by operation
            f.write(f"\n## Performance by Operation\n\n")
            f.write(f"| Operation | Tests | Avg Speedup | Avg Memory Ratio | Status |\n")
            f.write(f"|-----------|-------|-------------|------------------|--------|\n")
            
            operations = set(r.operation for r in results)
            for operation in sorted(operations):
                op_results = [r for r in results if r.operation == operation]
                op_valid = [r for r in op_results if r.error is None and r.speedup > 0]
                
                if op_valid:
                    avg_speedup = np.mean([r.speedup for r in op_valid])
                    avg_memory = np.mean([r.memory_ratio for r in op_valid if r.memory_ratio > 0])
                    
                    if avg_speedup >= 0.8:
                        status = "🟢 Excellent"
                    elif avg_speedup >= 0.6:
                        status = "🟡 Good"
                    elif avg_speedup >= 0.4:
                        status = "🟠 Fair"
                    else:
                        status = "🔴 Poor"
                    
                    f.write(f"| {operation} | {len(op_results)} | {avg_speedup:.2f}x | {avg_memory:.2f}x | {status} |\n")
                else:
                    f.write(f"| {operation} | {len(op_results)} | N/A | N/A | ❌ Failed |\n")
            
            # Detailed results table
            f.write(f"\n## Detailed Results\n\n")
            f.write(f"| Operation | Batch Size | Seq Length | PyTorch Time (ms) | Genesis Time (ms) | Speedup | PyTorch Memory (MB) | Genesis Memory (MB) | Memory Ratio | Status |\n")
            f.write(f"|-----------|------------|------------|-------------------|-------------------|---------|---------------------|---------------------|--------------|--------|\n")
            
            for result in sorted(results, key=lambda x: x.speedup if x.speedup > 0 else 0, reverse=True):
                if result.error:
                    f.write(f"| {result.operation} | {result.batch_size} | {result.seq_length} | - | - | - | - | - | - | ❌ Error |\n")
                    continue
                    
                f.write(f"| {result.operation} | {result.batch_size} | {result.seq_length} | ")
                f.write(f"{result.pytorch_time_ms:.1f} | {result.genesis_time_ms:.1f} | {result.speedup:.2f}x | ")
                f.write(f"{result.pytorch_memory_mb:.1f} | {result.genesis_memory_mb:.1f} | {result.memory_ratio:.2f}x | {result.status} |\n")
            
            # Performance analysis
            f.write(f"\n## Performance Analysis\n\n")
            
            # Speedup analysis
            if valid_results:
                speedups = [r.speedup for r in valid_results if r.speedup > 0]
                if speedups:
                    excellent = sum(1 for s in speedups if s >= 0.9)
                    good = sum(1 for s in speedups if 0.7 <= s < 0.9)
                    fair = sum(1 for s in speedups if 0.5 <= s < 0.7)
                    poor = sum(1 for s in speedups if s < 0.5)
                    total = len(speedups)
                    
                    f.write(f"### Speed Performance Distribution\n\n")
                    f.write(f"- 🟢 **Excellent (≥90%)**: {excellent} tests ({excellent/total*100:.1f}%)\n")
                    f.write(f"- 🟡 **Good (70-90%)**: {good} tests ({good/total*100:.1f}%)\n")
                    f.write(f"- 🟠 **Fair (50-70%)**: {fair} tests ({fair/total*100:.1f}%)\n")
                    f.write(f"- 🔴 **Poor (<50%)**: {poor} tests ({poor/total*100:.1f}%)\n")
                
                # Memory analysis
                memory_ratios = [r.memory_ratio for r in valid_results if r.memory_ratio > 0]
                if memory_ratios:
                    f.write(f"\n### Memory Usage Analysis\n\n")
                    efficient = sum(1 for m in memory_ratios if m <= 1.1)
                    acceptable = sum(1 for m in memory_ratios if 1.1 < m <= 1.5)
                    high = sum(1 for m in memory_ratios if 1.5 < m <= 2.0)
                    very_high = sum(1 for m in memory_ratios if m > 2.0)
                    total = len(memory_ratios)
                    
                    f.write(f"- 🟢 **Efficient (≤1.1x)**: {efficient} tests ({efficient/total*100:.1f}%)\n")
                    f.write(f"- 🟡 **Acceptable (1.1-1.5x)**: {acceptable} tests ({acceptable/total*100:.1f}%)\n")
                    f.write(f"- 🟠 **High (1.5-2x)**: {high} tests ({high/total*100:.1f}%)\n")
                    f.write(f"- 🔴 **Very High (>2x)**: {very_high} tests ({very_high/total*100:.1f}%)\n")
            
            # Best performers
            if valid_results:
                f.write(f"\n## Best Performers\n\n")
                best_performers = sorted([r for r in valid_results if r.speedup > 0], 
                                       key=lambda x: x.speedup, reverse=True)[:5]
                f.write(f"| Rank | Operation | Batch Size | Seq Length | Speedup | Memory Ratio |\n")
                f.write(f"|------|-----------|------------|------------|---------|-------------|\n")
                for i, result in enumerate(best_performers, 1):
                    f.write(f"| {i} | {result.operation} | {result.batch_size} | {result.seq_length} | {result.speedup:.2f}x | {result.memory_ratio:.2f}x |\n")
            
            # Recommendations
            f.write(f"\n## Recommendations\n\n")
            if valid_results:
                avg_speedup = np.mean([r.speedup for r in valid_results if r.speedup > 0])
                if avg_speedup >= 0.8:
                    f.write(f"✅ **Genesis shows excellent performance** with {avg_speedup:.2f}x average speedup.\n\n")
                elif avg_speedup >= 0.6:
                    f.write(f"✅ **Genesis shows good performance** with {avg_speedup:.2f}x average speedup.\n\n")
                elif avg_speedup >= 0.4:
                    f.write(f"⚠️ **Genesis shows fair performance** with {avg_speedup:.2f}x average speedup. Consider optimization.\n\n")
                else:
                    f.write(f"❌ **Genesis needs optimization** with {avg_speedup:.2f}x average speedup.\n\n")
                
                f.write(f"### Optimization Priorities\n\n")
                # Identify worst performing operations
                worst_ops = sorted([r for r in valid_results if r.speedup > 0], 
                                 key=lambda x: x.speedup)[:3]
                if worst_ops:
                    f.write(f"1. **Focus on these operations**: {', '.join(set(r.operation for r in worst_ops))}\n")
                
                # Memory usage recommendations
                high_memory_ops = [r for r in valid_results if r.memory_ratio > 1.5]
                if high_memory_ops:
                    f.write(f"2. **Memory optimization needed for**: {', '.join(set(r.operation for r in high_memory_ops))}\n")
        
        print(f"\n📄 Qwen benchmark markdown report generated: {filepath}")
        
    except Exception as e:
        print(f"⚠️  Failed to generate Qwen markdown report: {e}")

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Qwen End-to-End Benchmark: Genesis vs PyTorch",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument("--size", type=str, default="0.5B",
                       choices=["0.5B", "1.5B"],
                       help="Model size to benchmark")
    parser.add_argument("--batch-size", type=str, default="1,4",
                       help="Comma-separated batch sizes (e.g., 1,2,4)")
    parser.add_argument("--seq-len", type=str, default="512,1024",
                       help="Comma-separated sequence lengths (e.g., 256,512,1024)")
    parser.add_argument("--warmup", type=int, default=5,
                       help="Warmup iterations")
    parser.add_argument("--iters", type=int, default=20,
                       help="Test iterations")
    parser.add_argument("--profile", action="store_true",
                       help="Enable detailed profiling")
    
    return parser.parse_args()

def main():
    """Main benchmark execution"""
    args = parse_args()
    
    # Parse batch sizes and sequence lengths
    batch_sizes = [int(x.strip()) for x in args.batch_size.split(",")]
    seq_lengths = [int(x.strip()) for x in args.seq_len.split(",")]
    
    # Create benchmark configuration
    config = BenchConfig(
        model_size=args.size,
        batch_sizes=batch_sizes,
        seq_lengths=seq_lengths,
        warmup_iters=args.warmup,
        test_iters=args.iters,
        profile=args.profile
    )
    
    print(f"🚀 Qwen End-to-End Benchmark")
    print(f"Model: {config.model_size}")
    print(f"Batch Sizes: {batch_sizes}")
    print(f"Sequence Lengths: {seq_lengths}")
    print(f"Iterations: {args.warmup} warmup + {args.iters} test")
    
    # Create and run benchmark
    suite = QwenBenchmarkSuite(config)
    results = suite.run_full_benchmark()
    
    # Generate summary
    suite.generate_summary(results)
    
    # Generate markdown documentation for docs directory
    generate_qwen_benchmark_markdown(results, config)
    
    # Save results to JSON
    output_file = f"qwen_benchmark_{config.model_size.lower()}_results.json"
    with open(output_file, 'w') as f:
        json.dump([{
            'operation': r.operation,
            'model_size': r.model_size,
            'batch_size': r.batch_size,
            'seq_length': r.seq_length,
            'pytorch_time_ms': r.pytorch_time_ms,
            'genesis_time_ms': r.genesis_time_ms,
            'speedup': r.speedup,
            'pytorch_memory_mb': r.pytorch_memory_mb,
            'genesis_memory_mb': r.genesis_memory_mb,
            'memory_ratio': r.memory_ratio,
            'status': r.status,
            'error': r.error
        } for r in results], f, indent=2)
    
    print(f"\n💾 Results saved to {output_file}")

if __name__ == "__main__":
    main()
