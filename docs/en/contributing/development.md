# Development Environment Setup

This guide will help you set up a Genesis development environment, including code editing, debugging, testing, and other development workflows.

## 🛠️ System Requirements

### Hardware Requirements
- **CPU**: x86_64 architecture with AVX instruction set support
- **Memory**: Minimum 16GB, recommended 32GB+
- **GPU**: NVIDIA GPU with CUDA support (required for GPU operator development)
- **Storage**: 20GB available space

### Software Requirements
- **Operating System**: Linux (Ubuntu 20.04+ recommended), macOS 10.15+
- **Python**: 3.8, 3.9, 3.10, 3.11
- **Git**: Latest version
- **CUDA**: 11.8+ (required for GPU development)

## 🚀 Quick Start

### 1. Clone Repository

```bash
# Clone your fork (recommended)
git clone https://github.com/YOUR_USERNAME/genesis.git
cd genesis

# Or clone the main repository
git clone https://github.com/phonism/genesis.git
cd genesis

# Add upstream repository (if forked)
git remote add upstream https://github.com/phonism/genesis.git
```

### 2. Create Python Environment

=== "Using conda"
    ```bash
    # Create environment
    conda create -n genesis-dev python=3.9
    conda activate genesis-dev
    
    # Install base dependencies
    conda install numpy matplotlib ipython jupyter
    ```

=== "Using venv"
    ```bash
    # Create environment
    python -m venv genesis-dev
    source genesis-dev/bin/activate  # Linux/macOS
    # genesis-dev\\Scripts\\activate  # Windows
    
    # Upgrade pip
    pip install --upgrade pip setuptools wheel
    ```

### 3. Install Development Dependencies

```bash
# Install PyTorch (choose based on your CUDA version)
# CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# CPU version
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install Triton
pip install triton

# Install development tools
pip install -r requirements-dev.txt
```

### 4. Install Genesis (Development Mode)

```bash
# Development mode installation (recommended)
pip install -e .

# Verify installation
python -c "import genesis; print('Genesis development environment setup successful!')"
```

## 📦 Dependency Management

### Core Dependencies (requirements.txt)
```
torch>=2.0.0
triton>=2.0.0
numpy>=1.21.0
cuda-python>=11.8.0
```

### Development Dependencies (requirements-dev.txt)
```
pytest>=7.0.0
pytest-cov>=4.0.0
black>=22.0.0
flake8>=5.0.0
mypy>=1.0.0
isort>=5.0.0
pre-commit>=2.20.0
sphinx>=5.0.0
matplotlib>=3.5.0
jupyter>=1.0.0
ipython>=8.0.0
```

## 🔧 Development Tools Configuration

### 1. Git Configuration

```bash
# Configure user information
git config user.name "Your Name"
git config user.email "your.email@example.com"

# Configure commit template
echo "feat: brief description

More detailed explanation (optional)

- Change 1
- Change 2

Fixes #123" > ~/.gitmessage
git config commit.template ~/.gitmessage
```

### 2. Pre-commit Hooks

```bash
# Install pre-commit
pip install pre-commit

# Install hooks
pre-commit install

# Run checks manually
pre-commit run --all-files
```

### 3. IDE Configuration

=== "VS Code"
    Recommended extensions to install:
    
    ```json
    // .vscode/extensions.json
    {
        "recommendations": [
            "ms-python.python",
            "ms-python.black-formatter", 
            "ms-python.flake8",
            "ms-python.mypy-type-checker",
            "ms-toolsai.jupyter",
            "ms-vscode.cpptools"
        ]
    }
    ```
    
    Configuration file:
    ```json
    // .vscode/settings.json
    {
        "python.defaultInterpreterPath": "./genesis-dev/bin/python",
        "python.linting.enabled": true,
        "python.linting.flake8Enabled": true,
        "python.formatting.provider": "black",
        "python.formatting.blackArgs": ["--line-length=88"],
        "python.testing.pytestEnabled": true,
        "python.testing.pytestArgs": ["tests/"]
    }
    ```

=== "PyCharm"
    1. Open project settings (File -> Settings)
    2. Configure Python interpreter to point to virtual environment
    3. Enable code formatting tools (Black, isort)
    4. Configure test runner to pytest

### 4. Environment Variables

```bash
# Development environment variables
export GENESIS_DEV=1
export PYTHONPATH="${PWD}:${PYTHONPATH}"
export CUDA_VISIBLE_DEVICES=0  # Specify GPU device

# Add to ~/.bashrc or ~/.zshrc
echo 'export GENESIS_DEV=1' >> ~/.bashrc
```

## 🧪 Testing Framework

### Test Directory Structure
```
tests/
├── conftest.py              # pytest configuration
├── test_autograd.py         # Autograd tests
├── test_nn.py              # Neural network tests
├── test_cuda_tensor.py     # CUDA tensor tests
├── test_functional.py      # Functional interface tests
├── benchmarks/             # Performance tests
│   ├── bench_matmul.py
│   └── bench_attention.py
└── integration/            # Integration tests
    ├── test_training.py
    └── test_models.py
```

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_nn.py -v

# Run specific test function
pytest tests/test_nn.py::test_linear_layer -v

# Run tests with coverage
pytest tests/ --cov=genesis --cov-report=html

# Run performance tests
pytest tests/benchmarks/ -v --benchmark-only
```

### Writing Tests

```python
# tests/test_example.py
import pytest
import genesis
import genesis.nn as nn

class TestExample:
    """Example test class."""
    
    def setup_method(self):
        """Setup before each test method."""
        self.device = genesis.device('cuda' if genesis.cuda.is_available() else 'cpu')
        
    def test_basic_operation(self):
        """Test basic tensor operations."""
        x = genesis.randn(3, 4, device=self.device)
        y = genesis.randn(3, 4, device=self.device)
        z = x + y
        
        assert z.shape == (3, 4)
        assert z.device == self.device
        
    @pytest.mark.parametrize("input_size,output_size", [
        (10, 5),
        (128, 64),
        (512, 256)
    ])
    def test_linear_layers(self, input_size, output_size):
        """Test linear layers with different sizes."""
        layer = nn.Linear(input_size, output_size).to(self.device)
        x = genesis.randn(32, input_size, device=self.device)
        
        output = layer(x)
        assert output.shape == (32, output_size)
        
    @pytest.mark.cuda
    def test_cuda_specific(self):
        """Test CUDA-specific functionality."""
        if not genesis.cuda.is_available():
            pytest.skip("CUDA not available")
            
        x = genesis.randn(10, 10, device='cuda')
        assert x.is_cuda
```

## 📊 Performance Analysis

### 1. Built-in Profiler

```python
import genesis.utils.profile as profiler

# Using context manager
with profiler.profile() as prof:
    # Your code
    x = genesis.randn(1000, 1000)
    y = genesis.matmul(x, x)

# Print results
prof.print_stats()

# Save results
prof.export_chrome_trace("profile.json")
```

### 2. Memory Analysis

```python
import genesis

# Enable memory tracking
genesis.cuda.memory.enable_debug()

# Your code
x = genesis.randn(1000, 1000, device='cuda')
y = genesis.matmul(x, x)

# Check memory usage
print(f"Memory used: {genesis.cuda.memory_allocated() / 1024**2:.1f} MB")
print(f"Cached memory: {genesis.cuda.memory_cached() / 1024**2:.1f} MB")

# Memory snapshot
snapshot = genesis.cuda.memory.memory_snapshot()
```

### 3. Benchmarking

```python
# benchmark/bench_example.py
import time
import genesis
import torch

def benchmark_matmul():
    """Benchmark matrix multiplication."""
    sizes = [128, 256, 512, 1024]
    
    for size in sizes:
        # Genesis
        x_gen = genesis.randn(size, size, device='cuda')
        y_gen = genesis.randn(size, size, device='cuda')
        
        start_time = time.time()
        for _ in range(100):
            z_gen = genesis.matmul(x_gen, y_gen)
        genesis_time = time.time() - start_time
        
        # PyTorch
        x_torch = torch.randn(size, size, device='cuda')
        y_torch = torch.randn(size, size, device='cuda')
        
        start_time = time.time()
        for _ in range(100):
            z_torch = torch.matmul(x_torch, y_torch)
        torch_time = time.time() - start_time
        
        print(f"Size {size}x{size}:")
        print(f"  Genesis: {genesis_time:.4f}s")
        print(f"  PyTorch: {torch_time:.4f}s")
        print(f"  Ratio: {genesis_time/torch_time:.2f}x")

if __name__ == "__main__":
    benchmark_matmul()
```

## 🐛 Debugging Tips

### 1. Debug Environment Variables

```bash
# Enable debug mode
export GENESIS_DEBUG=1
export CUDA_LAUNCH_BLOCKING=1  # Synchronous CUDA execution
export PYTHONFAULTHANDLER=1    # Python error handling
```

### 2. Logging Configuration

```python
import logging
import genesis

# Configure logging
logging.basicConfig(level=logging.DEBUG)
genesis.set_log_level('DEBUG')

# Use logging
logger = logging.getLogger(__name__)
logger.debug("Debug information")
```

### 3. Breakpoint Debugging

```python
import pdb

def buggy_function(x):
    pdb.set_trace()  # Set breakpoint
    y = x * 2
    return y

# Or use ipdb (install with: pip install ipdb)
import ipdb
ipdb.set_trace()
```

## 📚 Documentation Development

### Building Documentation

```bash
# Install documentation dependencies
pip install -r docs/requirements.txt

# Local server
mkdocs serve

# Build static files
mkdocs build

# Deploy to GitHub Pages
mkdocs gh-deploy
```

### API Documentation Generation

```bash
# Auto-generate API documentation
python scripts/generate_api_docs.py

# Check docstring format
pydocstyle genesis/
```

## 🚀 Code Submission

### 1. Code Checks

```bash
# Format code
black genesis/ tests/
isort genesis/ tests/

# Type checking
mypy genesis/

# Code quality checks
flake8 genesis/ tests/

# Run tests
pytest tests/ -x
```

### 2. Submission Process

```bash
# 1. Sync latest code
git fetch upstream
git rebase upstream/main

# 2. Create feature branch
git checkout -b feature/your-feature

# 3. Development and testing
# ... your development work ...

# 4. Commit code
git add .
git commit -m "feat: add your feature"

# 5. Push branch
git push origin feature/your-feature

# 6. Create Pull Request
```

## ❓ Common Issues

### Q: CUDA-related errors?
A: Check CUDA version compatibility, ensure PyTorch and Triton versions match.

### Q: Test failures?
A: Run `pytest tests/ -v` to see detailed error information, check environment configuration.

### Q: Performance issues?
A: Use profiler to analyze bottlenecks, check if GPU acceleration is enabled.

### Q: Out of memory?
A: Reduce test case data size, enable CPU fallback mode.

---

!!! success "Development Environment Setup Complete"
    You can now start contributing code to Genesis!

[Next: Testing Guidelines](testing.md){ .md-button .md-button--primary }
[Back to Contributing Guide](index.md){ .md-button }