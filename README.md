# Genesis

Genesis is a lightweight deep learning framework written from scratch in Python, with Triton as its backend for high-performance computing and PyTorch for GPU memory management.

## Features

### Core Components
- **Automatic Differentiation**: Full computational graph with backpropagation support
- **Tensor Operations**: Comprehensive tensor arithmetic with GPU acceleration
- **Neural Network Modules**: Complete layer implementations including attention mechanisms
- **Optimizers**: SGD, Adam, AdamW with learning rate scheduling
- **Mixed Precision Training**: AMP (Automatic Mixed Precision) support with float16/bfloat16
- **Distributed Training**: Data parallel training support
- **Serialization**: Model saving/loading with checkpoint management

### Architecture
```
genesis/
├── autograd.py         # Tensor class with automatic differentiation
├── backend.py          # Backend abstraction layer
├── functional.py       # Functional operations
├── nn/                 # Neural network components
│   ├── modules.py      # Base Module class and layers
│   ├── attention.py    # Multi-head attention implementation
│   ├── layer_norm.py   # Layer normalization
│   ├── functional.py   # NN functional operations
│   └── triton_ops/     # Triton-optimized kernels
├── optim/              # Optimizers and schedulers
├── utils/              # Utilities (profiling, data loading)
└── models/             # Pre-built model architectures
```

## Installation

### Requirements
- Python >= 3.6
- NumPy
- PyTorch
- Triton (for GPU acceleration)

### Setup
```bash
git clone https://github.com/phonism/genesis.git
cd genesis
pip install -r requirements.txt

# Install in development mode
pip install -e .

# Run tests
python -m pytest tests/
```

## Quick Start

### Basic Tensor Operations
```python
import genesis

# Create tensors
x = genesis.Tensor([[1, 2, 3], [4, 5, 6]], dtype=genesis.float32)
y = genesis.Tensor([[2, 0, 1], [1, 3, 2]], dtype=genesis.float32)

# Basic operations
z = x + y
result = genesis.matmul(x, y.T)
```

### Neural Networks
```python
import genesis
import genesis.nn as nn

class SimpleNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x

# Initialize model
model = SimpleNetwork(784, 128, 10)
input_tensor = genesis.randn(32, 784)  # Batch of 32

# Forward pass
output = model(input_tensor)
```

### Training Loop
```python
import genesis
import genesis.nn as nn
import genesis.optim as optim

# Model and optimizer setup
model = SimpleNetwork(784, 128, 10)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# Training step
def train_step(data, target):
    # Forward pass
    output = model(data)
    loss = criterion(output, target)
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    return loss.item()
```

### Mixed Precision Training
```python
import genesis

# Enable automatic mixed precision
genesis.enable_autocast = True

with genesis.autocast():
    output = model(input_tensor)
    loss = criterion(output, target)
```

## Examples and Applications

### Available Examples
- **Basic Usage**: `samples/sample.py` - Simple neural network example
- **Learning Rate Scheduling**: `samples/lr_scheduler.py` - LR scheduler usage
- **Gradient Clipping**: `samples/grad_norm.py` - Gradient norm clipping

### LLM Implementation
The `apps/llm/` directory contains a complete implementation of a 0.5B parameter language model:
- **Model Architecture**: Transformer-based LLM with attention mechanisms
- **Training Scripts**: Both standard and supervised fine-tuning (SFT) variants
- **Chat Interface**: Interactive chat functionality
- **Multiple Backends**: Support for both Genesis and PyTorch implementations

```bash
# Train LLM
cd apps/llm
bash train.sh

# Chat with model
python chat.py
```

## Performance Benchmarks

Performance comparison on A100 GPU (CUDA 12.3, PyTorch 2.4.1, Triton 3.0.0):

### Multi-Head Attention
| Implementation | Time (ms) | Speedup |
|---|---|---|
| PyTorch | 359.46 | 1.0x |
| Genesis (Triton) | 1723.07 | 0.2x |
| Genesis (Fused) | 584.51 | 0.6x |

### Layer Normalization
| Implementation | Time (ms) | Speedup |
|---|---|---|
| PyTorch | 179.89 | 1.0x |
| PyTorch (Fused) | 20.43 | 8.8x |
| Genesis (Triton) | 955.56 | 0.2x |
| Genesis (Fused) | 58.99 | 3.0x |

*Benchmarks available in `benchmark/` directory*

## Advanced Features

### Custom Operations with Triton
```python
# Genesis supports custom Triton kernels for high-performance operations
from genesis.nn.triton_ops import softmax, dropout

# Optimized operations
x = genesis.randn(1024, 1024)
output = softmax(x, dim=-1)
```

### Distributed Training
```python
# Multi-GPU training support
import genesis.nn.parallel as parallel

model = parallel.DataParallel(model)
```

### Model Serialization
```python
# Save model
genesis.save_checkpoint(model, optimizer, 'checkpoint.pth')

# Load model
model, optimizer = genesis.load_checkpoint('checkpoint.pth')
```

## Testing

Run the test suite:
```bash
# Run all tests
python -m pytest tests/

# Run specific test modules
python -m pytest tests/test_nn.py
python -m pytest tests/test_functional.py

# Run with coverage
python -m pytest --cov=genesis tests/
```

## Development

### Project Structure
- `genesis/`: Core framework code
- `tests/`: Test suite
- `benchmark/`: Performance benchmarks
- `samples/`: Usage examples
- `apps/`: Full applications (LLM implementation)

### Adding New Features
1. Implement core functionality in appropriate module
2. Add corresponding tests in `tests/`
3. Update documentation
4. Add benchmarks if applicable

## Contributing

We welcome contributions! Please:
1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
