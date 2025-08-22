# Model Definition in Genesis

Learn how to define neural network models using Genesis.

## Overview

Genesis provides PyTorch-like APIs for defining neural network architectures.

## Basic Model Definition

```python
import genesis
import genesis.nn as nn

class SimpleNet(nn.Module):
    def __init__(self, input_size, num_classes):
        super().__init__()
        self.linear1 = nn.Linear(input_size, 128)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(128, num_classes)
    
    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x

# Create model instance
model = SimpleNet(784, 10)
```

## Advanced Architectures

*This tutorial is under construction. More examples and patterns will be added.*

## See Also

- [Basic Training](basic-training.md) - Training your defined models
- [Custom Operations](custom-ops.md) - Creating custom layers