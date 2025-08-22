# Distributed Training with Genesis

Learn how to scale your training across multiple GPUs and nodes using Genesis.

## Overview

Genesis provides built-in support for distributed training using multiple GPUs and distributed data parallel (DDP).

## Multi-GPU Training

```python
import genesis
import genesis.distributed as dist

# Initialize distributed training
dist.init_process_group(backend='nccl')

# Create model and wrap with DDP
model = MyModel()
model = dist.DistributedDataParallel(model)

# Distributed data loading
sampler = dist.DistributedSampler(dataset)
dataloader = DataLoader(dataset, sampler=sampler)

# Training loop remains similar
for batch in dataloader:
    # ... training code
```

## Multi-Node Setup

*This tutorial is under construction. Examples for multi-node distributed training will be added.*

## See Also

- [Advanced Features](../training/advanced-features.md) - Advanced training techniques
- [Performance Tuning](performance-tuning.md) - Optimizing distributed performance