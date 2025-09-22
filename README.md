


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

Genesis is a lightweight yet powerful deep learning framework that combines **educational clarity** with **production-level performance**. Built from scratch in Python, it features a clean, modern architecture with modular backends for CPU and GPU operations.

**🚀 v2.0 - Clean Architecture Update**:
- ✅ **Modular Backend System**: Separated CPU and CUDA backends in `backends/` for better maintainability
- ✅ **Unified Device Abstraction**: Centralized device management in `genesis.device`
- ✅ **Advanced Memory Management**: High-performance CUDA memory manager with lazy initialization
- ✅ **Modern Dispatcher**: Clean operation dispatch system routing to device-specific implementations
- ✅ **Enhanced Stability**: Improved error handling and CUDA initialization
- ✅ **Production Ready**: Complete training pipeline with mixed precision and distributed support

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
- 🏗️ **Modular Backend System**: Clean separation of CPU and CUDA implementations in `backends/`
- 🎯 **Unified Operation Dispatch**: Central operation router automatically selects optimal backend
- 🔥 **Triton Kernels**: Hand-optimized GPU kernels for maximum performance
- 🧮 **Advanced Memory Management**: High-performance memory pooling with fragmentation control and statistics
- 🚀 **Lazy CUDA Initialization**: Reliable GPU initialization without import-time failures
- 📊 **Profiling Tools**: Built-in performance profiling, memory usage tracking, and optimization utilities
- 🎲 **Random State Management**: PyTorch-compatible RNG with thread-safe state handling
- 🏛️ **Device Abstraction**: Unified device interface supporting CPU, CUDA, and future backends

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

# Basic installation (CPU only)
pip install -e .

# Full installation with LLM support and development tools
pip install -e ".[llm,dev]"

# Verify installation
python verify_install.py

# For GPU acceleration (Linux/Windows only)
export CUDA_VISIBLE_DEVICES=0  # Use first GPU
```

**Installation Options:**
- `pip install -e .` - Core framework only
- `pip install -e ".[llm]"` - Add LLM support (transformers, safetensors)
- `pip install -e ".[dev]"` - Add development tools (pytest, black, mypy)
- `pip install -e ".[docs]"` - Add documentation tools (mkdocs)
- `pip install -e ".[all]"` - Everything included

See [INSTALLATION.md](INSTALLATION.md) for detailed platform-specific instructions.

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

### Random Number Generation

```python
import genesis

# Set global random seed for reproducibility
genesis.manual_seed(42)

# Create random tensors
x = genesis.rand(100, 100, device=genesis.device('cuda'))
y = genesis.randn(50, 50, device=genesis.device('cpu'))

# Advanced RNG state management
generator = genesis.Generator()
generator.manual_seed(12345)

# Save and restore RNG states
state = genesis.get_rng_state()
# ... some random operations ...
genesis.set_rng_state(state)  # Restore previous state

# Thread-safe random generation
with genesis.fork_rng():
    genesis.manual_seed(999)
    # Random operations in this context don't affect global state
```

### Memory Management and Profiling

```python
import genesis

# Monitor memory usage
device = genesis.device('cuda')
print(f"Memory allocated: {device.memory_allocated() / 1e6:.1f} MB")
print(f"Memory cached: {device.memory_cached() / 1e6:.1f} MB")

# Advanced memory statistics
stats = device.memory_stats()
print(f"Cache hit rate: {stats['cache_hit_rate']:.1%}")
print(f"Peak memory usage: {stats['peak_allocated'] / 1e9:.2f} GB")

# Memory profiling for optimization
with genesis.profiler.profile() as prof:
    x = genesis.rand(4096, 4096, device=device)
    y = genesis.matmul(x, x.T)
    
print(prof.memory_summary())
```

## 🏗️ Architecture

```
genesis/
├── tensor.py                # Core Tensor class with autograd support
├── function.py              # Automatic differentiation functions
├── device.py                # Unified device abstraction
├── storage.py               # Storage interface layer
├── backends/                # Device-specific implementations
│   ├── cpu.py               # CPU backend using PyTorch
│   ├── cuda.py              # CUDA tensor storage
│   ├── cuda_memory.py       # Advanced CUDA memory management
│   └── cuda_kernels.py      # Optimized CUDA kernels
├── ops/                     # Operation dispatch system
│   ├── dispatcher.py        # Central operation router
│   ├── cpu/                 # CPU operation implementations
│   └── cuda/                # CUDA operation implementations
├── nn/
│   ├── modules/             # Neural network modules (modularized)
│   │   ├── module.py        # Base Module class
│   │   ├── linear.py        # Linear layers
│   │   ├── activation.py    # Activation functions
│   │   ├── normalization.py # LayerNorm, BatchNorm, RMSNorm
│   │   ├── transformer.py   # Multi-head attention, transformers
│   │   └── loss.py          # Loss functions (CrossEntropy, MSE, etc.)
│   ├── functional.py        # Functional NN operations
│   └── triton_ops/          # Triton-accelerated operations
├── optim/
│   ├── optimizer.py         # Base optimizer and Adam/AdamW/SGD
│   └── lr_scheduler.py      # Learning rate schedulers
├── models/
│   └── qwen.py              # Qwen LLM implementation
├── distributed/             # Distributed training support
│   ├── parallel.py          # DDP implementation
│   └── nccl_backend.py      # NCCL communication
└── cuda/
    └── __init__.py          # CUDA utilities and initialization
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