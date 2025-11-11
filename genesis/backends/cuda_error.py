"""
CUDA error checking utilities.

This module is separate to avoid circular imports between cuda.py and cuda_kernels.py
"""

try:
    from cuda.bindings import driver as cuda
except ImportError:
    from cuda import cuda


def check_cuda_error(result):
    """Check CUDA operation result and raise error if failed."""
    if isinstance(result, tuple):
        err = result[0]
        if err != cuda.CUresult.CUDA_SUCCESS:
            error_name = cuda.cuGetErrorName(err)[1].decode() if len(cuda.cuGetErrorName(err)) > 1 else "Unknown"
            error_string = cuda.cuGetErrorString(err)[1].decode() if len(cuda.cuGetErrorString(err)) > 1 else "Unknown error"
            raise RuntimeError(f"CUDA error: {error_name} - {error_string}")
        return result[1:] if len(result) > 1 else None
    else:
        if result != cuda.CUresult.CUDA_SUCCESS:
            raise RuntimeError(f"CUDA error: {result}")