"""
Communication operations for distributed training.

This module provides the core collective communication operations
used in distributed training, abstracting away the underlying backend.
"""

import enum
from typing import Optional, List
import genesis


class ReduceOp(enum.Enum):
    """Reduction operations for collective communication."""
    SUM = "sum"
    PRODUCT = "prod" 
    MIN = "min"
    MAX = "max"
    BAND = "band"  # Bitwise AND
    BOR = "bor"    # Bitwise OR
    BXOR = "bxor"  # Bitwise XOR


# Global communication backend instance
_backend = None


def _get_backend():
    """Get the current communication backend."""
    global _backend
    if _backend is None:
        raise RuntimeError("Distributed not initialized. Call init_process_group() first.")
    return _backend


def _set_backend(backend):
    """Set the communication backend."""
    global _backend
    _backend = backend


def all_reduce(tensor: genesis.Tensor, op: ReduceOp = ReduceOp.SUM, async_op: bool = False):
    """
    Reduce the tensor data across all processes.
    
    Args:
        tensor: Input tensor to be reduced
        op: Reduction operation 
        async_op: Whether to perform asynchronous operation
        
    Returns:
        Work handle if async_op=True, otherwise None
    """
    backend = _get_backend()
    return backend.all_reduce(tensor, op, async_op)


def all_gather(tensor_list: List[genesis.Tensor], tensor: genesis.Tensor, async_op: bool = False):
    """
    Gather tensors from all processes.
    
    Args:
        tensor_list: Output list of tensors to store gathered results
        tensor: Input tensor to be gathered
        async_op: Whether to perform asynchronous operation
        
    Returns:
        Work handle if async_op=True, otherwise None
    """
    backend = _get_backend()
    return backend.all_gather(tensor_list, tensor, async_op)


def broadcast(tensor: genesis.Tensor, src: int, async_op: bool = False):
    """
    Broadcast tensor from source process to all other processes.
    
    Args:
        tensor: Tensor to broadcast (input on src, output on others)
        src: Source process rank
        async_op: Whether to perform asynchronous operation
        
    Returns:
        Work handle if async_op=True, otherwise None
    """
    backend = _get_backend()
    return backend.broadcast(tensor, src, async_op)


def reduce_scatter(output: genesis.Tensor, input_list: List[genesis.Tensor],
                  op: ReduceOp = ReduceOp.SUM, async_op: bool = False):
    """
    Reduce and scatter tensor chunks across processes.

    Args:
        output: Output tensor to store reduced result chunk
        input_list: List of input tensors to be reduced and scattered
        op: Reduction operation
        async_op: Whether to perform asynchronous operation

    Returns:
        Work handle if async_op=True, otherwise None
    """
    backend = _get_backend()
    return backend.reduce_scatter(output, input_list, op, async_op)


def barrier():
    """
    Synchronize all processes in the process group.

    This is a collective operation - all ranks must call it.
    Standard DDP pattern for ensuring all ranks reach the same point.
    """
    backend = _get_backend()
    return backend.barrier()