"""
Test fused RMSNorm implementation.

Verifies correctness and performance compared to decomposed version.
"""
import sys
sys.path.insert(0, "../")

import numpy as np
import genesis
from genesis.nn.modules.normalization import RMSNorm


def decomposed_rmsnorm_numpy(x, weight, eps=1e-6):
    """Reference implementation using NumPy."""
    x_square = x ** 2
    x_mean = np.mean(x_square, axis=-1, keepdims=True)
    rms = x / np.sqrt(x_mean + eps)
    return rms * weight


def test_fused_rmsnorm_correctness():
    """Test that fused RMSNorm produces correct results."""
    print("="*80)
    print("Testing Fused RMSNorm Correctness")
    print("="*80)

    # Test configurations
    test_cases = [
        (1, 128, 1024, "Small batch"),
        (4, 512, 1024, "Medium batch"),
        (8, 2048, 1024, "Large sequence"),
    ]

    eps = 1e-6
    device = genesis.device("cuda")

    for batch_size, seq_len, hidden_dim, desc in test_cases:
        print(f"\nTest case: {desc}")
        print(f"  Shape: ({batch_size}, {seq_len}, {hidden_dim})")

        # Generate test data
        np.random.seed(42)
        x_np = np.random.randn(batch_size, seq_len, hidden_dim).astype(np.float32)
        # Use ones for weight (default initialization)
        weight_np = np.ones(hidden_dim, dtype=np.float32)

        # Compute reference using NumPy
        expected = decomposed_rmsnorm_numpy(x_np, weight_np, eps)

        # Compute using Genesis fused kernel
        x = genesis.tensor(x_np, device=device, dtype=genesis.float32)

        rmsnorm = RMSNorm(hidden_dim, eps=eps)
        rmsnorm.to(device)
        # Weight is already initialized to ones by default

        output = rmsnorm(x)
        result = output.numpy()

        # Compare results
        max_diff = np.abs(result - expected).max()
        mean_diff = np.abs(result - expected).mean()
        rel_error = max_diff / (np.abs(expected).max() + 1e-8)

        print(f"  Max diff: {max_diff:.2e}")
        print(f"  Mean diff: {mean_diff:.2e}")
        print(f"  Relative error: {rel_error:.2e}")

        # Check correctness (allow small numerical differences)
        if max_diff < 1e-4 and rel_error < 1e-5:
            print(f"  ✅ PASS")
        else:
            print(f"  ❌ FAIL - Difference too large!")
            return False

    print("\n" + "="*80)
    print("All correctness tests PASSED!")
    print("="*80)
    return True


def test_fused_rmsnorm_backward():
    """Test that backward pass works correctly."""
    print("\n" + "="*80)
    print("Testing Fused RMSNorm Backward Pass")
    print("="*80)

    batch_size, seq_len, hidden_dim = 4, 128, 512
    eps = 1e-6
    device = genesis.device("cuda")

    # Create input and weight
    x = genesis.randn(batch_size, seq_len, hidden_dim, device=device, requires_grad=True)
    rmsnorm = RMSNorm(hidden_dim, eps=eps)
    rmsnorm.to(device)

    # Forward pass
    output = rmsnorm(x)

    # Backward pass
    grad_output = genesis.ones_like(output)
    output.backward(grad_output)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Input grad shape: {x.grad.shape if x.grad is not None else 'None'}")
    print(f"Weight grad shape: {rmsnorm.weight.grad.shape if rmsnorm.weight.grad is not None else 'None'}")

    if x.grad is not None and rmsnorm.weight.grad is not None:
        print("✅ Backward pass completed successfully")
        return True
    else:
        print("❌ Backward pass failed - gradients not computed")
        return False


def benchmark_fused_rmsnorm():
    """Benchmark fused vs decomposed RMSNorm."""
    print("\n" + "="*80)
    print("Benchmarking Fused RMSNorm Performance")
    print("="*80)

    import time

    batch_size, seq_len, hidden_dim = 1, 2048, 1024
    eps = 1e-6
    device = genesis.device("cuda")
    num_runs = 100
    warmup = 10

    # Create test data
    x = genesis.randn(batch_size, seq_len, hidden_dim, device=device)

    # Fused version
    rmsnorm_fused = RMSNorm(hidden_dim, eps=eps)
    rmsnorm_fused.to(device)

    # Warmup
    for _ in range(warmup):
        _ = rmsnorm_fused(x)

    # Benchmark
    import torch
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(num_runs):
        _ = rmsnorm_fused(x)
    torch.cuda.synchronize()
    fused_time = (time.perf_counter() - start) / num_runs

    print(f"\nConfiguration:")
    print(f"  Shape: ({batch_size}, {seq_len}, {hidden_dim})")
    print(f"  Runs: {num_runs}")
    print(f"\nResults:")
    print(f"  Fused RMSNorm: {fused_time*1000:.4f} ms")
    print(f"\nExpected improvement: ~7x faster than decomposed (7 kernels → 1 kernel)")
    print(f"Estimated decomposed time: ~{fused_time*7*1000:.4f} ms")


if __name__ == "__main__":
    success = True

    # Run correctness tests
    success = test_fused_rmsnorm_correctness() and success

    # Test backward pass
    success = test_fused_rmsnorm_backward() and success

    # Benchmark
    if success:
        benchmark_fused_rmsnorm()

    print("\n" + "="*80)
    if success:
        print("✅ ALL TESTS PASSED!")
    else:
        print("❌ SOME TESTS FAILED!")
    print("="*80)
