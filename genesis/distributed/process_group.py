"""
Process group management for distributed training.

This module handles initialization, management, and cleanup of 
distributed process groups across multiple devices and nodes.
"""

import os
import socket
import struct
from typing import Optional, List, Union
from .comm import _set_backend
from .nccl_backend import NCCLBackend


# Global process group state
_world_size = None
_rank = None
_local_rank = None
_initialized = False
_backend_instance = None


def init_process_group(backend: str = "nccl", 
                      init_method: Optional[str] = None,
                      world_size: Optional[int] = None,
                      rank: Optional[int] = None):
    """
    Initialize the distributed process group.
    
    Args:
        backend: Backend to use ("nccl", "mpi", "gloo")
        init_method: URL specifying how to initialize the process group
        world_size: Number of processes participating in the job
        rank: Rank of the current process
    """
    global _world_size, _rank, _local_rank, _initialized, _backend_instance
    
    if _initialized:
        return
    
    # Auto-detect parameters from environment if not provided
    if world_size is None:
        world_size = int(os.environ.get('WORLD_SIZE', 1))
    if rank is None:
        rank = int(os.environ.get('RANK', 0))
    
    # Set local rank for GPU device selection
    _local_rank = int(os.environ.get('LOCAL_RANK', 0))
    
    _world_size = world_size
    _rank = rank
    
    # Initialize the appropriate backend
    if backend.lower() == "nccl":
        _backend_instance = NCCLBackend()
        _backend_instance.init_process_group(init_method, world_size, rank)
    elif backend.lower() == "mpi":
        raise NotImplementedError("MPI backend not implemented yet")
    elif backend.lower() == "gloo":
        raise NotImplementedError("Gloo backend not implemented yet")  
    else:
        raise ValueError(f"Unsupported backend: {backend}")
    
    _set_backend(_backend_instance)
    _initialized = True
    
    print(f"Initialized distributed process group: rank={rank}, world_size={world_size}, local_rank={_local_rank}")


def destroy_process_group():
    """Clean up the distributed process group."""
    global _world_size, _rank, _local_rank, _initialized, _backend_instance
    
    if not _initialized:
        return
        
    if _backend_instance is not None:
        _backend_instance.destroy()
        
    _world_size = None
    _rank = None  
    _local_rank = None
    _initialized = False
    _backend_instance = None
    _set_backend(None)
    
    print("Destroyed distributed process group")


def get_world_size() -> int:
    """Get the number of processes in the current process group."""
    if not _initialized:
        raise RuntimeError("Process group not initialized")
    return _world_size


def get_rank() -> int:
    """Get the rank of the current process."""
    if not _initialized:
        raise RuntimeError("Process group not initialized") 
    return _rank


def get_local_rank() -> int:
    """Get the local rank of the current process."""
    if not _initialized:
        raise RuntimeError("Process group not initialized")
    return _local_rank


def is_initialized() -> bool:
    """Check if the distributed process group is initialized."""
    return _initialized


def barrier():
    """
    Synchronize all processes.
    
    This collective blocks processes until the whole group reaches this point.
    """
    if not _initialized:
        raise RuntimeError("Process group not initialized")
    _backend_instance.barrier()