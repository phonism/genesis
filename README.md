# Genesis

<div align="center">

![Genesis](https://img.shields.io/badge/Genesis-Deep%20Learning-blue?style=for-the-badge&logo=python)

[![License](https://img.shields.io/github/license/phonism/genesis?style=flat-square)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=flat-square&logo=python)](https://python.org)
[![CUDA](https://img.shields.io/badge/CUDA-11.0%2B-green?style=flat-square&logo=nvidia)](https://developer.nvidia.com/cuda-toolkit)

**A high-performance deep learning framework with educational clarity**

[📚 Documentation](docs/) | [🚀 Quick Start](#quick-start) | [📊 Performance](#performance)

</div>

---

## Overview

Genesis is a modern deep learning framework built from scratch, combining **production-level performance** with **educational transparency**. Featuring Triton-optimized kernels, automatic differentiation, and comprehensive neural network modules, Genesis serves both as a learning resource and a practical training framework.

## Key Features

**Core Capabilities**
- 🔥 **High Performance**: Triton-optimized GPU kernels achieving near-native performance
- ⚡ **Automatic Differentiation**: Dynamic computational graph with full gradient support
- 🧠 **Neural Networks**: Complete module library including transformers and attention mechanisms
- 🎯 **Mixed Precision**: AMP support with FP16/BF16 training
- 🚀 **Distributed Training**: Multi-GPU training with NCCL backend
- 📦 **Model Support**: Built-in LLM implementations (Qwen) with training pipelines

**Technical Highlights**
- Modular backend system with clean CPU/CUDA separation
- Advanced CUDA memory management with pooling and statistics
- Unified operation dispatch routing to optimal implementations
- Complete optimizer suite (Adam, AdamW, SGD) with schedulers
- Production-ready training pipeline with checkpointing

## Performance

Genesis delivers competitive performance through hand-optimized Triton kernels:

| Operation | Efficiency vs Reference |
|-----------|------------------------|
| Matrix Multiplication | ~95% |
| Softmax | ~112% |
| LayerNorm | ~120% |
| Multi-Head Attention | ~97% |

*Benchmarked on NVIDIA A100 GPU*

## Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/phonism/genesis.git
cd genesis

# Install (CPU only)
pip install -e .

# Install with LLM support
pip install -e ".[llm]"

# Verify installation
python -c "import genesis; print(genesis.__version__)"
```

### Basic Usage

```python
import genesis
import genesis.nn as nn
import genesis.optim as optim

# Define model
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 10)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = self.fc1(x).relu()
        x = self.dropout(x)
        return self.fc2(x)

# Training setup
model = Net()
optimizer = optim.AdamW(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

# Training loop
for data, target in dataloader:
    output = model(data)
    loss = criterion(output, target)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

### Mixed Precision Training

```python
from genesis.cuda import amp

scaler = amp.GradScaler()

for data, target in dataloader:
    with amp.autocast():
        output = model(data)
        loss = criterion(output, target)

    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad()
```

### Distributed Training

```bash
# Single command for multi-GPU training
torchrun --nproc_per_node=4 train.py
```

```python
import genesis.distributed as dist

# Initialize
dist.init_process_group(backend='nccl')

# Wrap model
from genesis.distributed import DistributedDataParallel as DDP
model = DDP(model)

# Train normally - gradients synchronized automatically
```

## Architecture

```
genesis/
├── tensor.py              # Core tensor with autograd
├── function.py            # Autodiff functions
├── backends/              # CPU/CUDA implementations
│   ├── cpu.py
│   ├── cuda.py
│   └── cuda_memory.py
├── ops/                   # Operation dispatch
├── nn/                    # Neural network modules
│   ├── modules/          # Layer implementations
│   ├── functional.py     # Functional operations
│   └── triton_ops/       # Optimized kernels
├── optim/                # Optimizers
├── distributed/          # Multi-GPU support
└── cuda/                 # CUDA utilities & AMP
```

## Examples

**Train Qwen LLM**
```bash
cd apps/llm
python train_sft_qwen.py --amp --dtype fp16
```

**Interactive Chat**
```bash
cd apps/llm
python chat_qwen.py --checkpoint model.pth
```

**Benchmarks**
```bash
python benchmark/bench_matmul.py
python benchmark/bench_qwen_training.py
```

## Documentation

- [Getting Started Guide](docs/getting-started/)
- [API Reference](docs/api-reference/)
- [Architecture Overview](docs/architecture/)
- [Performance Tuning](docs/performance/)

## Testing

```bash
# Run test suite
pytest tests/ -v

# With coverage
pytest tests/ --cov=genesis --cov-report=html
```

## Contributing

We welcome contributions! Genesis is designed to be hackable and educational.

```bash
# Development setup
pip install -e ".[dev]"
black genesis/ && isort genesis/
pytest tests/
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

MIT License - see [LICENSE](LICENSE) for details.

## Acknowledgments

Genesis builds on ideas from PyTorch, Triton, TinyGrad, and JAX. We thank these projects for their inspiration and the deep learning community for their support.

---

<div align="center">

**Built for deep learning researchers and practitioners**

⭐ Star us on GitHub if you find Genesis useful!

</div>
