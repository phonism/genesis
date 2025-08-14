# Installation Guide

This guide will help you install the Genesis deep learning framework in different environments.

## 📋 System Requirements

### Hardware Requirements
- **CPU**: x86_64 architecture with AVX instruction set support
- **Memory**: Minimum 8GB, recommended 16GB+
- **GPU**: NVIDIA GPU with Compute Capability ≥ 6.0 (optional but recommended)
- **Storage**: 2GB available space

### Software Requirements
- **Operating System**: Linux (Ubuntu 20.04+), macOS (10.15+), Windows 10+
- **Python**: 3.8, 3.9, 3.10, 3.11
- **CUDA**: 11.0+ (required for GPU acceleration)

## 🚀 Quick Installation

### Method 1: Install from Source (Recommended)

```bash
# 1. Clone the repository
git clone https://github.com/phonism/genesis.git
cd genesis

# 2. Create virtual environment (recommended)
python -m venv genesis-env
source genesis-env/bin/activate  # Linux/macOS
# genesis-env\\Scripts\\activate  # Windows

# 3. Install dependencies
pip install -r genesis/requirements.txt

# 4. Install Genesis
pip install -e genesis/
```

### Method 2: Install via pip

```bash
# Install release version
pip install genesis-dl

# Install pre-release version
pip install --pre genesis-dl
```

## 🔧 Detailed Installation Steps

### Step 1: Prepare Python Environment

=== "Ubuntu/Debian"
    ```bash
    # Install Python and pip
    sudo apt update
    sudo apt install python3.9 python3.9-pip python3.9-venv
    
    # Create symbolic link (optional)
    sudo ln -sf /usr/bin/python3.9 /usr/bin/python
    ```

=== "CentOS/RHEL"
    ```bash
    # Install EPEL repository
    sudo yum install epel-release
    
    # Install Python
    sudo yum install python39 python39-pip
    ```

=== "macOS"
    ```bash
    # Install using Homebrew
    brew install python@3.9
    
    # Or use official installer
    # Download from https://python.org
    ```

=== "Windows"
    ```powershell
    # Download Python installer
    # https://python.org/downloads/windows/
    
    # Or use Chocolatey
    choco install python39
    ```

### Step 2: Install CUDA (GPU Acceleration)

!!! note "GPU Support Note"
    You can skip this step if you only need CPU version. However, installing CUDA is strongly recommended for optimal performance.

=== "Ubuntu/Debian"
    ```bash
    # Download CUDA Toolkit
    wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run
    sudo sh cuda_11.8.0_520.61.05_linux.run
    
    # Set environment variables
    echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
    echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
    source ~/.bashrc
    ```

=== "CentOS/RHEL"
    ```bash
    # Install NVIDIA driver repository
    sudo yum-config-manager --add-repo https://developer.download.nvidia.com/compute/cuda/repos/rhel8/x86_64/cuda-rhel8.repo
    
    # Install CUDA
    sudo yum install cuda-11-8
    ```

=== "Windows"
    ```powershell
    # Download CUDA installer
    # https://developer.nvidia.com/cuda-downloads
    
    # Run the installer and follow the prompts
    ```

### Step 3: Install Core Dependencies

```bash
# Create and activate virtual environment
python -m venv genesis-env
source genesis-env/bin/activate

# Upgrade pip
pip install --upgrade pip setuptools wheel

# Install PyTorch (choose based on your CUDA version)
# CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# CPU version
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install Triton
pip install triton

# Install other dependencies
pip install numpy matplotlib tqdm
```

### Step 4: Install Genesis

```bash
# Clone source code
git clone https://github.com/phonism/genesis.git
cd genesis

# View available versions
git tag

# Switch to stable version (optional)
git checkout v0.1.0

# Install Genesis
pip install -e genesis/
```

## ✅ Verify Installation

Run the following code to verify that the installation was successful:

```python
#!/usr/bin/env python3
"""Genesis installation verification script"""

def test_basic_import():
    """Test basic import"""
    try:
        import genesis
        print("✅ Genesis import successful")
        print(f"   Version: {genesis.__version__}")
    except ImportError as e:
        print(f"❌ Genesis import failed: {e}")
        return False
    return True

def test_tensor_operations():
    """Test tensor operations"""
    try:
        import genesis
        
        # Create tensors
        x = genesis.randn(3, 4)
        y = genesis.randn(3, 4)
        
        # Basic operations
        z = x + y
        print("✅ Tensor operations normal")
        print(f"   Tensor shape: {z.shape}")
    except Exception as e:
        print(f"❌ Tensor operations failed: {e}")
        return False
    return True

def test_neural_networks():
    """Test neural network modules"""
    try:
        import genesis.nn as nn
        
        # Create simple model
        model = nn.Sequential(
            nn.Linear(10, 5),
            nn.ReLU(),
            nn.Linear(5, 1)
        )
        
        # Test forward pass
        x = genesis.randn(2, 10)
        y = model(x)
        print("✅ Neural network modules normal")
        print(f"   Output shape: {y.shape}")
    except Exception as e:
        print(f"❌ Neural network modules failed: {e}")
        return False
    return True

def test_cuda_support():
    """Test CUDA support"""
    try:
        import genesis
        
        if genesis.cuda.is_available():
            device = genesis.device('cuda')
            x = genesis.randn(10, 10, device=device)
            print("✅ CUDA support normal")
            print(f"   GPU device count: {genesis.cuda.device_count()}")
            print(f"   GPU name: {genesis.cuda.get_device_name()}")
        else:
            print("⚠️  CUDA unavailable (will use CPU)")
    except Exception as e:
        print(f"❌ CUDA test failed: {e}")
        return False
    return True

def test_autograd():
    """Test automatic differentiation"""
    try:
        import genesis
        
        x = genesis.randn(5, requires_grad=True)
        y = (x ** 2).sum()
        y.backward()
        
        print("✅ Automatic differentiation normal")
        print(f"   Gradient shape: {x.grad.shape}")
    except Exception as e:
        print(f"❌ Automatic differentiation failed: {e}")
        return False
    return True

if __name__ == "__main__":
    print("🔍 Genesis Installation Verification\n")
    
    tests = [
        test_basic_import,
        test_tensor_operations,
        test_neural_networks,
        test_cuda_support,
        test_autograd
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print(f"📊 Test Results: {passed}/{total} passed")
    
    if passed == total:
        print("🎉 Congratulations! Genesis installation successful, all features working!")
    else:
        print("⚠️  Some features abnormal, please check installation steps")
```

Save the above code as `test_installation.py` and run:

```bash
python test_installation.py
```

## 🔧 Common Issues and Solutions

### Issue 1: CUDA Version Mismatch

**Error Message**:
```
RuntimeError: CUDA version mismatch
```

**Solution**:
```bash
# Check system CUDA version
nvidia-smi

# Check PyTorch CUDA version
python -c "import torch; print(torch.version.cuda)"

# Reinstall matching PyTorch version
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Issue 2: Triton Compilation Failure

**Error Message**:
```
Failed to compile Triton kernel
```

**Solution**:
```bash
# Upgrade Triton
pip install --upgrade triton

# Or install development version
pip install --pre triton
```

### Issue 3: Out of Memory

**Error Message**:
```
CUDA out of memory
```

**Solution**:
```python
import genesis

# Enable memory optimization
genesis.cuda.empty_cache()

# Reduce batch size
batch_size = 16  # Instead of 32

# Enable gradient checkpointing (if supported)
model.gradient_checkpointing = True
```

### Issue 4: Import Error

**Error Message**:
```
ModuleNotFoundError: No module named 'genesis'
```

**Solution**:
```bash
# Check virtual environment
which python
pip list | grep genesis

# Reinstall
pip uninstall genesis-dl
pip install -e genesis/
```

## 🐳 Docker Installation

If you encounter environment issues, you can use Docker:

```bash
# Download pre-built image
docker pull genesis/genesis:latest

# Or build your own image
git clone https://github.com/phonism/genesis.git
cd genesis
docker build -t genesis:local .

# Run container
docker run -it --gpus all genesis:local bash
```

Dockerfile contents:
```dockerfile
FROM nvidia/cuda:11.8-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PATH="/opt/conda/bin:$PATH"

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    wget git build-essential && \\
    rm -rf /var/lib/apt/lists/*

# Install Miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \\
    bash Miniconda3-latest-Linux-x86_64.sh -b -p /opt/conda && \\
    rm Miniconda3-latest-Linux-x86_64.sh

# Create environment and install dependencies
RUN conda create -n genesis python=3.9 -y
SHELL ["conda", "run", "-n", "genesis", "/bin/bash", "-c"]

RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 && \\
    pip install triton numpy matplotlib tqdm

# Copy and install Genesis
COPY . /workspace/genesis
WORKDIR /workspace/genesis
RUN pip install -e genesis/

# Set startup command
ENTRYPOINT ["conda", "run", "-n", "genesis"]
CMD ["bash"]
```

## 📊 Performance Optimization Tips

After installation, you can optimize performance with:

```bash
# Set environment variables
export CUDA_VISIBLE_DEVICES=0  # Specify GPU
export PYTHONPATH=$PWD:$PYTHONPATH

# Enable optimization options
export GENESIS_OPTIMIZE=1
export TRITON_CACHE_DIR=/tmp/triton_cache
```

## 🎯 Next Steps

After installation, it's recommended to:

1. [**Run your first program**](first-steps.md) - Verify installation and learn basic usage
2. [**Check tutorials**](../tutorials/basic-training.md) - Systematically learn Genesis usage
3. [**Read architecture documentation**](../architecture/index.md) - Understand framework design principles

---

If you encounter problems during installation, please check the [FAQ](../contributing/index.md#faq) or submit an issue on GitHub.