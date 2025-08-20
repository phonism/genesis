# Utilities (genesis.utils)

## Overview

The `genesis.utils` module provides essential utilities for development, debugging, and data handling. It includes profiling tools, data loading utilities, and helper functions to streamline the deep learning workflow.

## Core Components

### Performance Profiling
- Function and method execution time tracking
- Automatic profiling with decorators
- Performance analysis and reporting

### Data Loading
- Dataset abstraction for training data
- DataLoader with batching and shuffling
- Support for both map-style and iterable datasets

## Profiling Tools

### `@profile` Decorator

Automatic performance profiling for functions and classes.

```python
from genesis.utils import profile

@profile
def expensive_function(x):
    """
    Profile this function's execution time and call count.
    """
    # Your computation here
    return x * 2

@profile
class MyModel:
    """
    Profile all methods in this class.
    """
    def forward(self, x):
        return x + 1
    
    def backward(self, grad):
        return grad
```

The profiler automatically tracks:
- **Call Count**: Number of times each function is called
- **Total Time**: Cumulative execution time
- **Average Time**: Mean execution time per call

#### Usage Examples

```python
import genesis.utils as utils
import time

# Profile a function
@utils.profile
def matrix_multiply(a, b):
    """Dummy matrix multiplication."""
    time.sleep(0.01)  # Simulate computation
    return a @ b

# Profile a class
@utils.profile
class NeuralNetwork:
    def __init__(self):
        pass
    
    def forward(self, x):
        time.sleep(0.005)  # Simulate forward pass
        return x * 2
    
    def backward(self, grad):
        time.sleep(0.003)  # Simulate backward pass
        return grad

# Use the profiled functions
model = NeuralNetwork()
for i in range(100):
    x = matrix_multiply([[1, 2]], [[3], [4]])
    y = model.forward(x)
    model.backward([1, 1])

# Profile data is automatically printed at program exit
```

#### Profile Data Format

When the program exits, profiling data is automatically printed:

```
Program cost 2.1456 seconds!
__main__.matrix_multiply: 100 calls, 1.0234 total seconds
__main__.NeuralNetwork.forward: 100 calls, 0.5123 total seconds
__main__.NeuralNetwork.backward: 100 calls, 0.3089 total seconds
```

### Manual Profiling

For more granular control, you can access profiling data programmatically:

```python
from genesis.utils.profile import profile_data, print_profile_data

# Get current profile data
current_data = profile_data.copy()
print(f"Function calls so far: {sum(data['calls'] for data in current_data.values())}")

# Print profile summary manually
print_profile_data()
```

## Data Loading

### `Dataset`

Abstract base class for all datasets.

```python
from genesis.utils.data import Dataset

class Dataset:
    """
    Abstract dataset class.
    
    All subclasses must implement __len__ and __getitem__.
    """
    
    def __len__(self) -> int:
        """
        Return the size of the dataset.
        
        Returns:
            Number of samples in dataset
        """
        raise NotImplementedError
    
    def __getitem__(self, idx: int):
        """
        Retrieve a sample by index.
        
        Args:
            idx: Sample index
            
        Returns:
            Data sample at the given index
        """
        raise NotImplementedError
```

#### Custom Dataset Example

```python
import numpy as np
from genesis.utils.data import Dataset

class MNIST(Dataset):
    """Example MNIST dataset implementation."""
    
    def __init__(self, data_path, transform=None):
        """
        Initialize MNIST dataset.
        
        Args:
            data_path: Path to MNIST data files
            transform: Optional data transformation function
        """
        self.data = self._load_data(data_path)
        self.labels = self._load_labels(data_path)
        self.transform = transform
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        image = self.data[idx]
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label
    
    def _load_data(self, path):
        # Load your data here
        return np.random.randn(10000, 28, 28)  # Dummy data
    
    def _load_labels(self, path):
        # Load your labels here
        return np.random.randint(0, 10, 10000)  # Dummy labels
```

### `IterableDataset`

Base class for iterable-style datasets.

```python
from genesis.utils.data import IterableDataset

class IterableDataset(Dataset):
    """
    Base class for iterable datasets.
    
    Useful for streaming data or when random access is not feasible.
    """
    
    def __iter__(self):
        """
        Return an iterator over the dataset.
        
        Returns:
            Iterator that yields data samples
        """
        raise NotImplementedError
```

#### Iterable Dataset Example

```python
import random
from genesis.utils.data import IterableDataset

class RandomDataStream(IterableDataset):
    """Example streaming dataset."""
    
    def __init__(self, num_samples, feature_dim):
        """
        Initialize streaming dataset.
        
        Args:
            num_samples: Number of samples to generate
            feature_dim: Dimension of each sample
        """
        self.num_samples = num_samples
        self.feature_dim = feature_dim
    
    def __iter__(self):
        """Generate random samples on-the-fly."""
        for _ in range(self.num_samples):
            # Generate random data
            data = [random.random() for _ in range(self.feature_dim)]
            label = random.randint(0, 9)
            yield data, label
```

### `DataLoader`

Efficient data loading with batching and shuffling.

```python
from genesis.utils.data import DataLoader

class DataLoader:
    """
    Data loader for batching and shuffling datasets.
    
    Args:
        dataset: Dataset instance (Dataset or IterableDataset)
        batch_size: Number of samples per batch (default: 1)
        shuffle: Whether to shuffle data each epoch (default: False)
    """
    
    def __init__(
        self, 
        dataset, 
        batch_size: int = 1, 
        shuffle: bool = False
    ):
```

#### DataLoader Examples

```python
from genesis.utils.data import Dataset, DataLoader
import numpy as np

# Create a simple dataset
class SimpleDataset(Dataset):
    def __init__(self, size):
        self.data = np.random.randn(size, 10)
        self.labels = np.random.randint(0, 2, size)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

# Create dataset and dataloader
dataset = SimpleDataset(1000)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Training loop
for epoch in range(5):
    print(f"Epoch {epoch + 1}")
    for batch_idx, batch in enumerate(dataloader):
        # batch is a list of (data, label) tuples
        batch_data = [item[0] for item in batch]
        batch_labels = [item[1] for item in batch]
        
        # Convert to arrays if needed
        batch_data = np.array(batch_data)
        batch_labels = np.array(batch_labels)
        
        print(f"  Batch {batch_idx}: data shape {batch_data.shape}")
        
        # Your training code here
        pass
```

#### Advanced DataLoader Usage

```python
# Large dataset with shuffling
large_dataset = SimpleDataset(50000)
train_loader = DataLoader(large_dataset, batch_size=128, shuffle=True)

# Iterable dataset
stream_dataset = RandomDataStream(1000, 20)
stream_loader = DataLoader(stream_dataset, batch_size=16)

# Small batch for debugging
debug_loader = DataLoader(dataset, batch_size=4, shuffle=False)

# Training loop with multiple dataloaders
def train_model(model, train_loader, val_loader):
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        for batch in train_loader:
            # Training code
            pass
        
        # Validation phase
        model.eval()
        for batch in val_loader:
            # Validation code
            pass
```

## Integration with Genesis Training

### Complete Training Example

```python
import genesis
import genesis.nn as nn
import genesis.optim as optim
from genesis.utils.data import Dataset, DataLoader
from genesis.utils import profile
import numpy as np

# Custom dataset
class TrainingDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# Profiled model
@profile
class SimpleModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Generate dummy data
X = np.random.randn(1000, 20).astype(np.float32)
y = np.random.randint(0, 3, 1000)

# Create dataset and dataloader
dataset = TrainingDataset(X, y)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Model and optimizer
model = SimpleModel(20, 64, 3)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# Training loop with profiling
@profile
def train_epoch(model, dataloader, optimizer, criterion):
    """Train for one epoch."""
    total_loss = 0.0
    for batch in dataloader:
        # Extract batch data
        batch_x = [item[0] for item in batch]
        batch_y = [item[1] for item in batch]
        
        # Convert to Genesis tensors
        x = genesis.tensor(batch_x)
        y = genesis.tensor(batch_y)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(x)
        loss = criterion(outputs, y)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)

# Train the model
for epoch in range(10):
    avg_loss = train_epoch(model, dataloader, optimizer, criterion)
    print(f"Epoch {epoch + 1}, Average Loss: {avg_loss:.4f}")

# Profiling data will be printed automatically at program exit
```

## Best Practices

### Profiling Guidelines

1. **Use for Development**: Enable profiling during development to identify bottlenecks
2. **Disable for Production**: Remove profiling decorators in production code
3. **Selective Profiling**: Profile only the functions you suspect are slow
4. **Batch Profiling**: Profile entire training loops rather than individual operations

### Data Loading Guidelines

1. **Appropriate Batch Size**: Balance memory usage and training efficiency
2. **Shuffle Training Data**: Always shuffle training data between epochs
3. **Don't Shuffle Validation**: Keep validation data in consistent order
4. **Memory Considerations**: Use iterable datasets for very large datasets
5. **Data Preprocessing**: Apply transforms in the dataset's `__getitem__` method

## Performance Tips

### Efficient Data Loading

```python
# Good: Efficient batch processing
class EfficientDataset(Dataset):
    def __init__(self, data):
        # Pre-process data once
        self.data = self._preprocess(data)
    
    def _preprocess(self, data):
        # Expensive preprocessing done once
        return data * 2 + 1
    
    def __getitem__(self, idx):
        # Fast access
        return self.data[idx]

# Good: Large batch sizes when possible
train_loader = DataLoader(dataset, batch_size=128, shuffle=True)

# Good: Use appropriate data types
data = np.array(data, dtype=np.float32)  # Use float32 instead of float64
```

### Memory Management

```python
# Good: Delete large objects when done
del large_dataset
del temporary_data

# Good: Use generators for large datasets
def data_generator():
    for file in file_list:
        data = load_file(file)
        yield data

# Good: Limit memory usage with smaller batches if needed
small_batch_loader = DataLoader(dataset, batch_size=16)
```

## See Also

- [Neural Network Modules](../nn/modules.md) - Building models
- [Optimizers](../optim/optimizers.md) - Training algorithms  
- [Autograd](../autograd.md) - Automatic differentiation
- [Performance Guide](../../performance/) - Optimization techniques