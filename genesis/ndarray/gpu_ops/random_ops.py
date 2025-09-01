"""
Random number generation operations for GPU backend.
"""
import triton
import triton.language as tl
from ..cuda_storage import CUDAStorage
from ...random import default_generator


# =============================================================================
# TRITON RANDOM KERNELS  
# =============================================================================

@triton.jit
def randn_kernel(output_ptr, n_elements, seed, BLOCK_SIZE: tl.constexpr):
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
    
    tl.store(output_ptr + offsets, z0, mask=mask)


@triton.jit  
def rand_kernel(output_ptr, n_elements, seed, BLOCK_SIZE: tl.constexpr):
    """
    Generate random numbers from uniform distribution [0, 1).
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
    
    tl.store(output_ptr + offsets, uniform, mask=mask)


# =============================================================================
# GPU OPERATIONS
# =============================================================================

def randn(shape, dtype="float32"):
    """
    Create tensor with random normal distribution using Triton kernel.
    
    Args:
        shape: Tensor shape
        dtype: Data type
        
    Returns:
        CUDAStorage: Random normal tensor
    """
    output = CUDAStorage(shape, dtype=dtype)
    n_elements = output.size
    
    # Use unified RNG system - respects genesis.manual_seed()
    seed = default_generator().next_seed()
    
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]), )
    randn_kernel[grid](output, n_elements, seed, BLOCK_SIZE=1024)
    
    return output


def rand(shape, dtype="float32"):
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
    rand_kernel[grid](output, n_elements, seed, BLOCK_SIZE=1024)
    
    return output