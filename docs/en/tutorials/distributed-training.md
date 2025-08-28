# Distributed Training with Genesis

Learn how to scale your training across multiple GPUs and nodes using Genesis.

## Overview

Genesis provides complete distributed training support, including:

- **NCCL Backend** - High-performance GPU communication
- **DistributedDataParallel (DDP)** - Data parallel training wrapper
- **Collective Communication Operations** - all_reduce, broadcast, all_gather, etc.
- **Single-Process Testing** - Convenient for development and debugging

## Quick Start

### 1. Basic Distributed Training Setup

```python
import genesis
import genesis.distributed as dist
import genesis.nn as nn

# Initialize distributed process group
dist.init_process_group(backend='nccl', world_size=2, rank=0)  # Adjust rank per process

# Create model
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(512, 256)
        self.relu = nn.ReLU()
        self.output = nn.Linear(256, 10)
    
    def forward(self, x):
        x = self.relu(self.linear(x))
        return self.output(x)

model = MyModel()

# Wrap as distributed data parallel model
device = genesis.device('cuda')
ddp_model = dist.DistributedDataParallel(model, device_ids=[device.index])
```

### 2. Distributed Training Loop

```python
# Optimizer and loss function
optimizer = genesis.optim.Adam(ddp_model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# Training loop
ddp_model.train()
for epoch in range(num_epochs):
    for batch_idx, (data, targets) in enumerate(dataloader):
        # Move data to GPU
        data = data.to(device)
        targets = targets.to(device)
        
        # Forward pass
        outputs = ddp_model(data)
        loss = criterion(outputs, targets)
        
        # Backward pass (gradients automatically synchronized)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if batch_idx % 100 == 0:
            print(f'Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}')
```

### 3. Process Group Management

```python
# Check distributed status
if dist.is_initialized():
    print(f"Process group initialized")
    print(f"World size: {dist.get_world_size()}")
    print(f"Current rank: {dist.get_rank()}")

# Synchronize all processes
dist.barrier()

# Cleanup
dist.destroy_process_group()
```

## Advanced Features

### Collective Communication Operations

```python
import genesis

# Create test tensor
device = genesis.device('cuda')
tensor = genesis.ones([4], dtype=genesis.float32, device=device)

# all_reduce - Aggregate across all processes
dist.all_reduce(tensor, dist.ReduceOp.SUM)  # Sum
dist.all_reduce(tensor, dist.ReduceOp.MAX)  # Maximum
dist.all_reduce(tensor, dist.ReduceOp.MIN)  # Minimum

# broadcast - Broadcast operation
broadcast_tensor = genesis.randn([8], device=device)
dist.broadcast(broadcast_tensor, src=0)  # Broadcast from rank 0

# all_gather - Gather data from all processes
input_tensor = genesis.randn([4, 8], device=device)
output_list = [genesis.zeros_like(input_tensor) for _ in range(dist.get_world_size())]
dist.all_gather(output_list, input_tensor)
```

### Single-Process Testing Mode

```python
# Single-process mode for development and debugging
def test_single_process():
    # Initialize single-process distributed environment
    dist.init_process_group(backend="nccl", world_size=1, rank=0)
    
    # Create and test model
    model = MyModel()
    ddp_model = dist.DistributedDataParallel(model, device_ids=[0])
    
    # Test forward pass
    input_data = genesis.randn([8, 512], device='cuda')
    output = ddp_model(input_data)
    
    # Test backward pass
    loss = output.sum()
    loss.backward()
    
    print("Single-process distributed test successful!")
    dist.destroy_process_group()

# Run test
if __name__ == "__main__":
    test_single_process()
```

## Multi-GPU Training Scripts

### launcher.py

```python
#!/usr/bin/env python3
"""
Multi-GPU training launcher script
Usage: python launcher.py --gpus 2
"""

import argparse
import subprocess
import sys
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpus', type=int, default=1, help='Number of GPUs')
    parser.add_argument('--script', type=str, default='train.py', help='Training script')
    args = parser.parse_args()
    
    processes = []
    
    try:
        for rank in range(args.gpus):
            env = os.environ.copy()
            env['CUDA_VISIBLE_DEVICES'] = str(rank)
            env['RANK'] = str(rank)
            env['WORLD_SIZE'] = str(args.gpus)
            
            cmd = [sys.executable, args.script]
            proc = subprocess.Popen(cmd, env=env)
            processes.append(proc)
            
        # Wait for all processes to complete
        for proc in processes:
            proc.wait()
            
    except KeyboardInterrupt:
        print("Stopping training...")
        for proc in processes:
            proc.terminate()

if __name__ == "__main__":
    main()
```

### train.py

```python
#!/usr/bin/env python3
"""
Distributed training main script
"""

import os
import genesis
import genesis.distributed as dist
import genesis.nn as nn

def main():
    # Get distributed parameters from environment variables
    rank = int(os.environ.get('RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    
    # Initialize distributed training
    dist.init_process_group(
        backend='nccl',
        world_size=world_size,
        rank=rank
    )
    
    print(f"Process {rank}/{world_size} started")
    
    try:
        # Create model
        model = create_model()
        ddp_model = dist.DistributedDataParallel(
            model, 
            device_ids=[genesis.cuda.current_device()]
        )
        
        # Create optimizer
        optimizer = genesis.optim.Adam(ddp_model.parameters(), lr=0.001)
        
        # Training loop
        train_loop(ddp_model, optimizer, rank)
        
    finally:
        # Cleanup distributed environment
        dist.destroy_process_group()

def create_model():
    """Create model"""
    return nn.Sequential([
        nn.Linear(784, 512),
        nn.ReLU(),
        nn.Linear(512, 256), 
        nn.ReLU(),
        nn.Linear(256, 10)
    ])

def train_loop(model, optimizer, rank):
    """Training loop"""
    model.train()
    
    for epoch in range(10):
        # Simulate training data
        data = genesis.randn([32, 784], device='cuda')
        targets = genesis.randint(0, 10, [32], device='cuda')
        
        # Forward pass
        outputs = model(data)
        loss = nn.functional.cross_entropy(outputs, targets)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if rank == 0:  # Only print on main process
            print(f'Epoch {epoch}, Loss: {loss.item():.4f}')

if __name__ == "__main__":
    main()
```

## Error Handling and Debugging

### Common Issues

1. **NCCL Not Available**
```python
try:
    dist.init_process_group(backend="nccl", world_size=1, rank=0)
except RuntimeError as e:
    if "NCCL library not available" in str(e):
        print("NCCL library not available, please check CUDA and NCCL installation")
    else:
        raise
```

2. **Process Group Not Initialized**
```python
if not dist.is_initialized():
    print("Error: Distributed process group not initialized")
    print("Please call dist.init_process_group() first")
```

3. **Device Mismatch**
```python
# Ensure model and data are on the same device
device = genesis.device(f'cuda:{genesis.cuda.current_device()}')
model = model.to(device)
data = data.to(device)
```

## Performance Optimization Tips

### 1. Gradient Accumulation
```python
accumulation_steps = 4

for i, batch in enumerate(dataloader):
    outputs = ddp_model(batch['input'])
    loss = criterion(outputs, batch['target']) / accumulation_steps
    loss.backward()
    
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

### 2. Mixed Precision Training
```python
# Combine automatic mixed precision with distributed training
scaler = genesis.amp.GradScaler()

with genesis.amp.autocast():
    outputs = ddp_model(data)
    loss = criterion(outputs, targets)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

### 3. Communication Optimization
```python
# Enable gradient compression when creating DDP
ddp_model = dist.DistributedDataParallel(
    model,
    device_ids=[device.index],
    find_unused_parameters=False,  # Improve performance
    gradient_as_bucket_view=True   # Reduce memory usage
)
```

## See Also

- [Advanced Features](../training/advanced-features.md) - Advanced training techniques
- [Performance Tuning](performance-tuning.md) - Optimizing distributed performance