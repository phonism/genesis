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

# Note: DistributedDataParallel is now exposed through genesis.nn.parallel
# to match PyTorch's API structure (torch.nn.parallel.DistributedDataParallel)
# It can still be imported directly from genesis.distributed.parallel if needed

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
]