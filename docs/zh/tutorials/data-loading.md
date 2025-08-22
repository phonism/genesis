# Data Loading in Genesis

Learn how to efficiently load and preprocess data for training with Genesis.

## Overview

Genesis provides flexible data loading utilities compatible with various data sources and formats.

## Basic Data Loading

```python
import genesis
from genesis.data import DataLoader

# Create a simple dataset
dataset = MyDataset()  # Your custom dataset
loader = DataLoader(dataset, batch_size=32, shuffle=True)

for batch in loader:
    data, targets = batch
    # Process batch...
```

## Custom Datasets

*This tutorial is under construction. More examples and detailed documentation will be added soon.*

## See Also

- [Basic Training](basic-training.md) - Training loops with data loaders
- [Performance Tuning](performance-tuning.md) - Optimizing data loading performance