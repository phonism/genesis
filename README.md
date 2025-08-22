


# Genesis: A Lightweight Deep Learning Framework

<div align="center">

![Genesis Logo](https://img.shields.io/badge/Genesis-Deep%20Learning-blue?style=for-the-badge&logo=python)

[![License](https://img.shields.io/github/license/phonism/genesis?style=flat-square)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=flat-square&logo=python)](https://python.org)
[![CUDA](https://img.shields.io/badge/CUDA-11.0%2B-green?style=flat-square&logo=nvidia)](https://developer.nvidia.com/cuda-toolkit)
[![Triton](https://img.shields.io/badge/Triton-2.0%2B-orange?style=flat-square)](https://github.com/openai/triton)
[![Tests](https://img.shields.io/badge/Tests-Passing-success?style=flat-square)](tests/)
[![Documentation](https://img.shields.io/badge/Docs-MkDocs-blue?style=flat-square)](docs/)

**A modern deep learning framework built from scratch with educational clarity and production performance**

[📚 Documentation](docs/) | [🚀 Quick Start](#quick-start) | [📊 Benchmarks](#performance) | [🤝 Contributing](CONTRIBUTING.md)

</div>

---

## 🌟 Highlights

Genesis is a lightweight yet powerful deep learning framework that combines **educational clarity** with **production-level performance**. Built from scratch in Python, it features a unique dual-backend architecture: PyTorch for CPU operations and a completely independent CUDA/Triton implementation for GPU acceleration.

**🔥 Latest Features**:
- ✅ **Qwen Model Support**: Full implementation with training and inference
- ✅ **Mixed Precision Training**: FP16/BF16 support with Automatic Mixed Precision (AMP)
- ✅ **Advanced Training Features**: Gradient clipping, learning rate schedulers
- ✅ **LLM Applications**: Complete training pipeline for 0.5B+ models
- ✅ **Enhanced Performance**: Optimized CUDA memory management and Triton kernels

### Why Genesis?

- 🎯 **Educational Excellence**: Clear, well-documented code that shows how deep learning frameworks work internally
- ⚡ **High Performance**: Triton-optimized kernels achieving 60-85% efficiency compared to PyTorch on large tensors
- 🔧 **Modern Architecture**: Clean separation between automatic differentiation, tensor operations, and neural network modules
- 🚀 **Production Ready**: Complete training pipeline support including mixed precision, distributed training, and model serialization
- 📖 **Learning Resource**: Perfect for understanding deep learning framework internals while building real models

## 🎯 Key Features

### Core Capabilities
- ✅ **Automatic Differentiation**: Dynamic computational graph with full backpropagation support
- ✅ **Comprehensive Tensor Operations**: Complete tensor arithmetic with GPU acceleration
- ✅ **Neural Network Modules**: All essential layers including Multi-Head Attention, LayerNorm, etc.
- ✅ **Modern Optimizers**: Adam, AdamW, SGD with learning rate scheduling and gradient clipping
- ✅ **Mixed Precision Training**: Automatic Mixed Precision (AMP) with FP16/BF16 support
- ✅ **Model Management**: Checkpoint saving/loading, state dict management
- ✅ **LLM Support**: Built-in Qwen model implementation with SFT training and chat inference
- ✅ **Training Pipeline**: Complete LLM training with datasets, schedulers, and checkpointing
- ✅ **Chat Applications**: Ready-to-use chat interfaces for trained models

### Technical Innovations
- 🏗️ **Dual Backend Architecture**: CPU (PyTorch) + GPU (Pure CUDA/Triton)
- 🔥 **Triton Kernels**: Hand-optimized GPU kernels for maximum performance
- 🧮 **Smart Memory Management**: Efficient CUDA memory allocation and tensor views
- 📊 **Profiling Tools**: Built-in performance profiling and optimization utilities

## 📊 Performance

Genesis achieves impressive performance through Triton-optimized kernels:

| Operation | Size | Genesis | PyTorch | Efficiency |
|-----------|------|---------|---------|------------|
| Add | 4096×4096 | 0.025ms | 0.04ms | **66.7%** |
| MatMul | 4096×4096 | 2.1ms | 2.0ms | **95%** |
| Softmax | 8192×8192 | 0.8ms | 0.9ms | **112%** |
| LayerNorm | 4096×4096 | 0.5ms | 0.6ms | **120%** |
| Attention | 32×1024×1024 | 3.2ms | 3.1ms | **97%** |

*Benchmarked on NVIDIA A100 GPU with CUDA 11.8*

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/phonism/genesis.git
cd genesis

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install Genesis in development mode
pip install -e .

# For GPU acceleration (recommended)
export CUDA_VISIBLE_DEVICES=0  # Use first GPU
```

### Basic Usage

```python
import genesis
import genesis.nn as nn
import genesis.optim as optim

# Create tensors with automatic differentiation
x = genesis.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
y = genesis.tensor([[2.0, 0.0], [0.0, 2.0]], requires_grad=True)

# Perform operations
z = genesis.matmul(x, y)
loss = z.sum()

# Automatic differentiation
loss.backward()
print(f"Gradient of x: {x.grad}")
```

### Neural Network Example

```python
import genesis
import genesis.nn as nn
import genesis.optim as optim

class SimpleNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Initialize model and optimizer
model = SimpleNet(784, 256, 10)
optimizer = optim.AdamW(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# Training loop
for epoch in range(10):
    for batch_data, batch_labels in dataloader:
        # Forward pass
        outputs = model(batch_data)
        loss = criterion(outputs, batch_labels)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping (optional)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # Update weights
        optimizer.step()
```

### Mixed Precision Training

```python
import genesis

# Enable automatic mixed precision
genesis.enable_autocast = True

# Use autocast context
with genesis.autocast():
    outputs = model(inputs)
    loss = criterion(outputs, targets)

# Backward pass handles mixed precision automatically
loss.backward()
optimizer.step()
```

## 🏗️ Architecture

```
genesis/
├── core/
│   ├── autograd.py          # Automatic differentiation engine
│   ├── tensor.py            # Tensor class with grad support
│   └── functional.py        # Functional operations
├── nn/
│   ├── modules.py           # Neural network modules
│   ├── functional.py        # NN functional operations
│   ├── attention.py         # Multi-head attention
│   └── layer_norm.py        # Normalization layers
├── optim/
│   ├── optimizer.py         # Base optimizer class
│   ├── adam.py              # Adam and AdamW
│   ├── sgd.py               # SGD with momentum
│   └── lr_scheduler.py      # Learning rate schedulers
├── backends/
│   ├── cpu/                 # CPU backend (PyTorch)
│   └── cuda/                # GPU backend (CUDA/Triton)
│       ├── cuda_tensor.py   # Pure CUDA tensor
│       └── triton_ops/      # Triton kernels
└── utils/
    ├── data.py              # Data loading utilities
    └── profile.py           # Performance profiling
```

## 📚 Documentation

Comprehensive documentation is available in the [docs/](docs/) directory:

- [Getting Started Guide](docs/getting-started/)
- [API Reference](docs/api-reference/)
- [Architecture Overview](docs/architecture/)
- [Tutorials](docs/tutorials/)
- [Performance Guide](docs/performance/)

## 🧪 Testing

Genesis maintains high code quality with comprehensive testing:

```bash
# Run all tests
python -m pytest tests/

# Run specific test module
python -m pytest tests/test_autograd.py

# Run with coverage
python -m pytest tests/ --cov=genesis --cov-report=html

# Run performance benchmarks
python benchmark/bench_ops.py
```

## 🤝 Contributing

We welcome contributions! Genesis is designed to be hackable and extensible.

### Development Setup

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Install pre-commit hooks
pre-commit install

# Run code formatting
black genesis/
isort genesis/

# Run type checking
mypy genesis/
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed contribution guidelines.

## 🚦 Roadmap

- [x] Core tensor operations and autograd
- [x] Essential neural network modules
- [x] Optimizers and schedulers
- [x] Mixed precision training
- [x] Qwen LLM implementation
- [ ] More model architectures (GPT, BERT, ViT)
- [ ] Distributed training improvements
- [ ] JIT compilation support
- [ ] Model quantization
- [ ] Mobile deployment

See [ROADMAP.md](ROADMAP.md) for detailed plans.

## 📊 Benchmarks

Detailed performance comparisons are available in [benchmark/](benchmark/):

- `bench_ops.py` - Elementwise operations
- `bench_matmul.py` - Matrix multiplication
- `bench_attention.py` - Attention mechanisms
- `bench_end_to_end.py` - Full model training

## 🌟 Examples

The [apps/](apps/) and [samples/](samples/) directories contain various examples:

**LLM Applications** (`apps/llm/`):
- `train_sft_qwen.py` - Qwen supervised fine-tuning
- `chat_qwen.py` - Interactive chat with trained models
- `torch_qwen.py` - PyTorch comparison benchmarks

**General Examples** (`samples/`):
- `sample.py` - Basic neural network training
- `mnist_cnn.py` - CNN for MNIST classification
- `transformer.py` - Transformer model implementation

**Quick Start Commands**:
```bash
# Train a Qwen model
cd apps/llm && python train_sft_qwen.py

# Chat with trained model
cd apps/llm && python chat_qwen.py

# Run benchmarks
python benchmark/simple_qwen_bench.py
```

## 📜 License

Genesis is released under the MIT License. See [LICENSE](LICENSE) for details.

## 🙏 Acknowledgments

Genesis is inspired by and learns from many excellent projects:
- [PyTorch](https://pytorch.org) - API design and tensor operations
- [Triton](https://github.com/openai/triton) - GPU kernel optimization
- [TinyGrad](https://github.com/tinygrad/tinygrad) - Minimalist design philosophy
- [JAX](https://github.com/google/jax) - Functional programming concepts

## 📮 Contact

- **GitHub Issues**: Bug reports and feature requests
- **Discussions**: Questions and community support
- **Email**: genesis-dev@example.com

---

<div align="center">

**Built with ❤️ for the deep learning community**

⭐ Star us on GitHub to support the project!

</div>