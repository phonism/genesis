"""
Random number generation operations for GPU backend.
"""
import triton
import triton.language as tl
from genesis.backends.cuda import CUDAStorage
from ...random import default_generator
from ..dispatcher import register_cuda


# =============================================================================
# TRITON RANDOM KERNELS  
# =============================================================================

@triton.jit
def randn_kernel(output_ptr, n_elements, seed, mean, std, BLOCK_SIZE: tl.constexpr):
    """
    Generate random numbers from normal distribution using Box-Muller transform.
    """
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Initialize random state for each thread
    random_seed = seed + offsets
    
    # Generate pairs of uniform random numbers using linear congruential generator
    # Based on Numerical Recipes parameters
    a = 1664525
    c = 1013904223
    m = 2**32
    
    # First uniform random number
    random_seed = (a * random_seed + c) % m
    u1 = random_seed.to(tl.float32) / m.to(tl.float32)
    
    # Second uniform random number  
    random_seed = (a * random_seed + c) % m
    u2 = random_seed.to(tl.float32) / m.to(tl.float32)
    
    # Box-Muller transform: convert uniform to normal
    # z0 = sqrt(-2 * ln(u1)) * cos(2*pi*u2)
    # Avoid log(0) by clamping u1
    u1 = tl.maximum(u1, 1e-7)
    
    magnitude = tl.sqrt(-2.0 * tl.log(u1))
    angle = 2.0 * 3.14159265359 * u2
    z0 = magnitude * tl.cos(angle)
    
    # Apply mean and std scaling
    result = z0 * std + mean
    
    tl.store(output_ptr + offsets, result, mask=mask)


@triton.jit
def rand_kernel(output_ptr, n_elements, seed, low, high, BLOCK_SIZE: tl.constexpr):
    """
    Generate random numbers from uniform distribution [low, high).
    """
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Initialize random state for each thread
    random_seed = seed + offsets

    # Linear congruential generator
    a = 1664525
    c = 1013904223
    m = 2**32

    random_seed = (a * random_seed + c) % m
    uniform = random_seed.to(tl.float32) / m.to(tl.float32)

    # Scale to [low, high) range
    result = uniform * (high - low) + low

    tl.store(output_ptr + offsets, result, mask=mask)


@triton.jit
def randint_kernel(output_ptr, n_elements, low, high, seed, BLOCK_SIZE: tl.constexpr):
    """Random integer generation kernel."""
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Simple linear congruential generator
    # Each thread gets different seed based on offset
    thread_seeds = seed + offsets

    # LCG: next = (a * current + c) % m
    a = 1664525
    c = 1013904223
    m = 2**32

    # Generate random number
    next_val = (a * thread_seeds + c) % m

    # Convert to range [low, high)
    # Use abs() to ensure positive values (Triton may treat as signed)
    range_size = high - low
    positive_val = tl.abs(next_val)
    result = (positive_val % range_size) + low

    tl.store(output_ptr + offsets, result, mask=mask)


# =============================================================================
# GPU OPERATIONS
# =============================================================================

@register_cuda("randn")
def randn(shape, dtype="float32", mean=0.0, std=1.0):
    """
    Create tensor with random normal distribution using Triton kernel.
    
    Args:
        shape: Tensor shape
        dtype: Data type
        mean: Mean of distribution  
        std: Standard deviation of distribution
        
    Returns:
        CUDAStorage: Random normal tensor
    """
    output = CUDAStorage(shape, dtype=dtype)
    n_elements = output.size
    
    # Use unified RNG system - respects genesis.manual_seed()
    seed = default_generator().next_seed()
    
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]), )
    randn_kernel[grid](output, n_elements, seed, mean, std, BLOCK_SIZE=1024)
    
    return output


@register_cuda("rand")
def rand(shape, dtype="float32", low=0.0, high=1.0):
    """
    Create tensor with random uniform distribution using Triton kernel.
    
    Args:
        shape: Tensor shape
        dtype: Data type
        
    Returns:
        CUDAStorage: Random uniform tensor
    """
    output = CUDAStorage(shape, dtype=dtype)
    n_elements = output.size
    
    # Use unified RNG system - respects genesis.manual_seed()
    seed = default_generator().next_seed()
    
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]), )
    rand_kernel[grid](output, n_elements, seed, low, high, BLOCK_SIZE=1024)

    return output


@register_cuda("randint")
def randint(shape, dtype_name, low=0, high=10):
    """Generate random integers in range [low, high)."""
    # Convert dtype name to numpy dtype
    if dtype_name == "int32":
        dtype = "int32"
    elif dtype_name == "int64":
        dtype = "int64"
    else:
        dtype = "int32"  # Default fallback

    # Calculate total elements
    if isinstance(shape, int):
        shape = (shape,)

    # Create output storage
    output = CUDAStorage(shape, dtype)
    n_elements = output.size

    # Use unified RNG system - respects genesis.manual_seed()
    seed = default_generator().next_seed()

    # Launch kernel
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]), )
    randint_kernel[grid](output, n_elements, low, high, seed, BLOCK_SIZE=1024)

    return output