# Getting Started

Welcome to the Genesis deep learning framework! This guide will help you start using Genesis in just a few minutes.

## 🎯 Overview

Genesis is a lightweight deep learning framework designed specifically for learning and research. It provides:

- Simple and intuitive API design
- High-performance GPU-accelerated computing
- Complete neural network training capabilities
- Good compatibility with PyTorch ecosystem

## ⚡ 5-Minute Quick Start

### 1. Install Genesis

```bash
# Install core dependencies
pip install torch triton

# Clone source code
git clone https://github.com/phonism/genesis.git
cd genesis

# Install Genesis
pip install -e .
```

### 2. Your First Neural Network

```python
import genesis
import genesis.nn as nn

# Define a simple multi-layer perceptron
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.layer1(x))
        return self.layer2(x)

# Create model and data
model = MLP(784, 128, 10)
x = genesis.randn(32, 784)  # batch size 32, input dimension 784

# Forward pass
output = model(x)
print(f"Output shape: {output.shape}")  # torch.Size([32, 10])
```

### 3. Training Loop

```python
import genesis.optim as optim

# Create optimizer and loss function
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# Simulate training data
targets = genesis.randint(0, 10, (32,))

# Train one batch
optimizer.zero_grad()        # Zero gradients
output = model(x)           # Forward pass
loss = criterion(output, targets)  # Compute loss
loss.backward()             # Backward pass
optimizer.step()            # Update parameters

print(f"Loss value: {loss.item():.4f}")
```

## 📚 Core Concepts

### Tensor
The fundamental data structure in Genesis, supporting automatic differentiation:

```python
import genesis

# Create tensors
x = genesis.tensor([1.0, 2.0, 3.0], requires_grad=True)
y = genesis.tensor([4.0, 5.0, 6.0], requires_grad=True)

# Compute operations
z = x * y + x.sum()
z.backward(genesis.ones_like(z))

print(f"x gradients: {x.grad}")  # [5., 6., 7.]
print(f"y gradients: {y.grad}")  # [1., 2., 3.]
```

### Module
Base class for neural network components:

```python
import genesis.nn as nn

class CustomLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = genesis.randn(out_features, in_features, requires_grad=True)
        self.bias = genesis.zeros(out_features, requires_grad=True)
    
    def forward(self, x):
        return genesis.functional.linear(x, self.weight, self.bias)

# Use custom layer
layer = CustomLayer(10, 5)
input_tensor = genesis.randn(3, 10)
output = layer(input_tensor)
```

### Optimizer
Parameter update algorithms:

```python
import genesis.optim as optim

# Different optimizer choices
sgd_optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
adam_optimizer = optim.Adam(model.parameters(), lr=0.001)
adamw_optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
```

## 🛠️ Environment Setup

### Hardware Requirements

- **CPU**: Modern multi-core processor
- **Memory**: Minimum 8GB RAM, 16GB+ recommended
- **GPU**: NVIDIA GPU with CUDA support (recommended)
- **Storage**: At least 2GB available space

### Software Dependencies

```bash
# Python environment
Python >= 3.8

# Core dependencies
torch >= 2.0.0
triton >= 2.0.0
numpy >= 1.21.0
cuda-python >= 11.8.0  # GPU support

# Optional dependencies
matplotlib >= 3.5.0  # For visualization
tqdm >= 4.64.0      # Progress bars
wandb >= 0.13.0     # Experiment tracking
```

## 📖 Next Steps

Now that you understand the basics of Genesis, you can continue exploring:

### 🎓 Deep Learning
- [**Complete Installation Guide**](installation.md) - Detailed installation and configuration steps
- [**First Complete Program**](first-steps.md) - Build a complete training workflow
- [**Basic Training Tutorial**](../tutorials/basic-training.md) - Systematic training tutorials

### 🔍 Architecture Understanding
- [**Architecture Overview**](../architecture/index.md) - Understand Genesis's overall design
- [**Core Components**](../core-components/index.md) - Deep dive into internal implementation
- [**API Reference**](../api-reference/index.md) - Complete API documentation

### 🚀 Advanced Features
- [**Custom Operators**](../tutorials/custom-ops.md) - Implement custom operations
- [**Performance Optimization**](../tutorials/performance-tuning.md) - Training performance tuning
- [**Distributed Training**](../neural-networks/distributed.md) - Multi-GPU training

## ❓ Frequently Asked Questions

### Q: What's the difference between Genesis and PyTorch?
A: Genesis is education-oriented with cleaner, more understandable code, suitable for learning deep learning internals. PyTorch is better suited for production environments.

### Q: Can Genesis be used in production?
A: Genesis is primarily for education and research. While fully functional, we recommend more mature frameworks like PyTorch for production use.

### Q: How to get help?
A: You can get help through GitHub Issues, Discussions, or by consulting the detailed documentation.

---

## 🎉 Ready?

Let's start diving deep into Genesis!

[Detailed Installation Guide](installation.md){ .md-button .md-button--primary }
[Complete Tutorials](../tutorials/index.md){ .md-button }