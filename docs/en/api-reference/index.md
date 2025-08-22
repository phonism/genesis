# API Reference

Genesis deep learning framework provides complete API interfaces. This section provides detailed code-level documentation and usage examples.

## Core Module Structure

### Main Namespaces

- **`genesis`**: Core tensor and automatic differentiation system
- **`genesis.nn`**: Neural network modules and layers  
- **`genesis.optim`**: Optimizers and learning rate schedulers
- **`genesis.functional`**: Functional operation interfaces
- **`genesis.utils`**: Utility functions and helper classes

### Quick Navigation

| Module | Description | Main Classes/Functions |
|--------|-------------|------------------------|
| [genesis](genesis.md) | Core tensor system | `Tensor`, `autocast`, `no_grad` |
| [nn](nn.md) | Neural network layers | `Module`, `Linear`, `MultiHeadAttention` |
| [optim](optim.md) | Optimizers | `SGD`, `Adam`, `AdamW` |
| [functional](functional.md) | Functional operations | `relu`, `softmax`, `matmul` |
| [utils](utils.md) | Utility functions | `profile`, `DataLoader` |

## Code Conventions

### Import Standards
```python
import genesis
import genesis.nn as nn
import genesis.optim as optim
import genesis.nn.functional as F
```

### Device Management
```python
# Set default device
genesis.set_default_device(genesis.cuda())

# Check CUDA availability
if genesis.cuda.is_available():
    device = genesis.cuda()
else:
    device = genesis.cpu()
```

### Data Types
```python
# Supported data types
genesis.float32  # Default float type
genesis.float16  # Half precision float
genesis.int32    # 32-bit integer
genesis.bool     # Boolean type
```

## Quick Examples

### Basic Tensor Operations
```python
import genesis

# Create tensors
x = genesis.tensor([[1, 2], [3, 4]], dtype=genesis.float32)
y = genesis.randn(2, 2)

# Basic operations
z = x + y
result = genesis.matmul(x, y.T)

# Gradient computation
x.requires_grad_(True)
loss = (x ** 2).sum()
loss.backward()
print(x.grad)  # Print gradients
```

### Neural Network Model
```python
import genesis.nn as nn

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        return self.fc2(x)

# Use model
model = MLP(784, 256, 10)
x = genesis.randn(32, 784)
output = model(x)
```

### Training Loop
```python
import genesis.optim as optim

# Initialize
model = MLP(784, 256, 10)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# Training step
for epoch in range(100):
    for batch_idx, (data, target) in enumerate(dataloader):
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
```

## Performance Optimization Tips

### Mixed Precision Training
```python
# Enable automatic mixed precision
genesis.enable_autocast = True

with genesis.autocast():
    output = model(input_tensor)
    loss = criterion(output, target)
```

### GPU Memory Optimization
```python
# Use inplace operations to reduce memory usage
x.relu_()  # inplace ReLU
x.add_(y)  # inplace addition

# Release unnecessary gradients
with genesis.no_grad():
    inference_result = model(data)
```

### Batch Operation Optimization
```python
# Batch matrix multiplication
batch_result = genesis.bmm(batch_a, batch_b)

# Vectorized operations instead of loops
result = genesis.sum(tensor, dim=1, keepdim=True)
```