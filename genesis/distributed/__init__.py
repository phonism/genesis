"""
Genesis Distributed Training Module

A high-performance distributed training framework using NCCL for GPU communication.
Provides scalable multi-GPU and multi-node training capabilities.
"""

from .process_group import (
    init_process_group,
    destroy_process_group, 
    get_world_size,
    get_rank,
    is_initialized,
    barrier
)

from .comm import (
    all_reduce,
    all_gather,
    broadcast,
    reduce_scatter,
    ReduceOp
)

from .parallel import DistributedDataParallel

# Convenience alias
DDP = DistributedDataParallel

__all__ = [
    'init_process_group',
    'destroy_process_group',
    'get_world_size', 
    'get_rank',
    'is_initialized',
    'barrier',
    'all_reduce',
    'all_gather', 
    'broadcast',
    'reduce_scatter',
    'ReduceOp',
    'DistributedDataParallel',
    'DDP'
]