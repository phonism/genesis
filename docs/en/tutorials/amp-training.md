# Automatic Mixed Precision Training

Learn how to use AMP (Automatic Mixed Precision) training in Genesis for better performance and memory efficiency.

## Overview

Genesis supports mixed precision training with FP16 and BF16 data types to accelerate training and reduce memory usage.

## Basic AMP Training

```python
import genesis
import genesis.nn as nn
from genesis.amp import GradScaler, autocast

model = MyModel()
optimizer = genesis.optim.Adam(model.parameters())
scaler = GradScaler()

for batch in dataloader:
    optimizer.zero_grad()
    
    with autocast():
        output = model(batch.data)
        loss = criterion(output, batch.targets)
    
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

## Advanced Configuration

*This tutorial is under construction. More detailed examples and best practices will be added.*

## See Also

- [Mixed Precision Training](mixed-precision.md) - Detailed mixed precision guide
- [Performance Tuning](performance-tuning.md) - General performance optimization