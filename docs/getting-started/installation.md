# 安装指南

本指南将帮助你在不同环境下安装Genesis深度学习框架。

## 📋 系统要求

### 硬件要求
- **CPU**: x86_64架构，支持AVX指令集
- **内存**: 最少8GB，推荐16GB+
- **GPU**: NVIDIA GPU with Compute Capability ≥ 6.0 (可选但推荐)
- **存储**: 2GB可用空间

### 软件要求
- **操作系统**: Linux (Ubuntu 20.04+), macOS (10.15+), Windows 10+
- **Python**: 3.8, 3.9, 3.10, 3.11
- **CUDA**: 11.0+ (GPU加速需要)

## 🚀 快速安装

### 方式一：从源码安装 (推荐)

```bash
# 1. 克隆仓库
git clone https://github.com/phonism/genesis.git
cd genesis

# 2. 创建虚拟环境 (推荐)
python -m venv genesis-env
source genesis-env/bin/activate  # Linux/macOS
# genesis-env\\Scripts\\activate  # Windows

# 3. 安装依赖
pip install -r genesis/requirements.txt

# 4. 安装Genesis
pip install -e genesis/
```

### 方式二：使用pip安装

```bash
# 安装发布版本
pip install genesis-dl

# 安装预发布版本
pip install --pre genesis-dl
```

## 🔧 详细安装步骤

### 第一步：准备Python环境

=== "Ubuntu/Debian"
    ```bash
    # 安装Python和pip
    sudo apt update
    sudo apt install python3.9 python3.9-pip python3.9-venv
    
    # 创建软链接 (可选)
    sudo ln -sf /usr/bin/python3.9 /usr/bin/python
    ```

=== "CentOS/RHEL"
    ```bash
    # 安装EPEL仓库
    sudo yum install epel-release
    
    # 安装Python
    sudo yum install python39 python39-pip
    ```

=== "macOS"
    ```bash
    # 使用Homebrew安装
    brew install python@3.9
    
    # 或使用官方安装包
    # 从 https://python.org 下载安装
    ```

=== "Windows"
    ```powershell
    # 下载Python安装包
    # https://python.org/downloads/windows/
    
    # 或使用Chocolatey
    choco install python39
    ```

### 第二步：安装CUDA (GPU加速)

!!! note "GPU支持说明"
    如果你只需要CPU版本，可以跳过此步骤。但强烈推荐安装CUDA以获得最佳性能。

=== "Ubuntu/Debian"
    ```bash
    # 下载CUDA Toolkit
    wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run
    sudo sh cuda_11.8.0_520.61.05_linux.run
    
    # 设置环境变量
    echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
    echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
    source ~/.bashrc
    ```

=== "CentOS/RHEL"
    ```bash
    # 安装NVIDIA驱动仓库
    sudo yum-config-manager --add-repo https://developer.download.nvidia.com/compute/cuda/repos/rhel8/x86_64/cuda-rhel8.repo
    
    # 安装CUDA
    sudo yum install cuda-11-8
    ```

=== "Windows"
    ```powershell
    # 下载CUDA安装包
    # https://developer.nvidia.com/cuda-downloads
    
    # 运行安装程序并按照提示操作
    ```

### 第三步：安装核心依赖

```bash
# 创建并激活虚拟环境
python -m venv genesis-env
source genesis-env/bin/activate

# 升级pip
pip install --upgrade pip setuptools wheel

# 安装PyTorch (根据你的CUDA版本选择)
# CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# CPU版本
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# 安装Triton
pip install triton

# 安装其他依赖
pip install numpy matplotlib tqdm
```

### 第四步：安装Genesis

```bash
# 克隆源码
git clone https://github.com/phonism/genesis.git
cd genesis

# 查看可用版本
git tag

# 切换到稳定版本 (可选)
git checkout v0.1.0

# 安装Genesis
pip install -e genesis/
```

## ✅ 验证安装

运行以下代码验证安装是否成功：

```python
#!/usr/bin/env python3
"""Genesis安装验证脚本"""

def test_basic_import():
    """测试基础导入"""
    try:
        import genesis
        print("✅ Genesis导入成功")
        print(f"   版本: {genesis.__version__}")
    except ImportError as e:
        print(f"❌ Genesis导入失败: {e}")
        return False
    return True

def test_tensor_operations():
    """测试张量操作"""
    try:
        import genesis
        
        # 创建张量
        x = genesis.randn(3, 4)
        y = genesis.randn(3, 4)
        
        # 基础运算
        z = x + y
        print("✅ 张量运算正常")
        print(f"   张量形状: {z.shape}")
    except Exception as e:
        print(f"❌ 张量运算失败: {e}")
        return False
    return True

def test_neural_networks():
    """测试神经网络模块"""
    try:
        import genesis.nn as nn
        
        # 创建简单模型
        model = nn.Sequential(
            nn.Linear(10, 5),
            nn.ReLU(),
            nn.Linear(5, 1)
        )
        
        # 测试前向传播
        x = genesis.randn(2, 10)
        y = model(x)
        print("✅ 神经网络模块正常")
        print(f"   输出形状: {y.shape}")
    except Exception as e:
        print(f"❌ 神经网络模块失败: {e}")
        return False
    return True

def test_cuda_support():
    """测试CUDA支持"""
    try:
        import genesis
        
        if genesis.cuda.is_available():
            device = genesis.device('cuda')
            x = genesis.randn(10, 10, device=device)
            print("✅ CUDA支持正常")
            print(f"   GPU设备数量: {genesis.cuda.device_count()}")
            print(f"   GPU名称: {genesis.cuda.get_device_name()}")
        else:
            print("⚠️  CUDA不可用 (将使用CPU)")
    except Exception as e:
        print(f"❌ CUDA测试失败: {e}")
        return False
    return True

def test_autograd():
    """测试自动微分"""
    try:
        import genesis
        
        x = genesis.randn(5, requires_grad=True)
        y = (x ** 2).sum()
        y.backward()
        
        print("✅ 自动微分正常")
        print(f"   梯度形状: {x.grad.shape}")
    except Exception as e:
        print(f"❌ 自动微分失败: {e}")
        return False
    return True

if __name__ == "__main__":
    print("🔍 Genesis安装验证\n")
    
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
    
    print(f"📊 测试结果: {passed}/{total} 通过")
    
    if passed == total:
        print("🎉 恭喜！Genesis安装成功，所有功能正常！")
    else:
        print("⚠️  部分功能异常，请检查安装步骤")
```

将上述代码保存为 `test_installation.py` 并运行：

```bash
python test_installation.py
```

## 🔧 常见问题解决

### 问题1：CUDA版本不匹配

**错误信息**：
```
RuntimeError: CUDA version mismatch
```

**解决方案**：
```bash
# 检查系统CUDA版本
nvidia-smi

# 检查PyTorch CUDA版本
python -c "import torch; print(torch.version.cuda)"

# 重新安装匹配版本的PyTorch
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 问题2：Triton编译失败

**错误信息**：
```
Failed to compile Triton kernel
```

**解决方案**：
```bash
# 升级Triton
pip install --upgrade triton

# 或安装开发版本
pip install --pre triton
```

### 问题3：内存不足

**错误信息**：
```
CUDA out of memory
```

**解决方案**：
```python
import genesis

# 启用内存优化
genesis.cuda.empty_cache()

# 减小批量大小
batch_size = 16  # 替代原来的32

# 启用梯度检查点 (如果支持)
model.gradient_checkpointing = True
```

### 问题4：导入错误

**错误信息**：
```
ModuleNotFoundError: No module named 'genesis'
```

**解决方案**：
```bash
# 检查虚拟环境
which python
pip list | grep genesis

# 重新安装
pip uninstall genesis-dl
pip install -e genesis/
```

## 🐳 Docker安装

如果你遇到环境问题，可以使用Docker：

```bash
# 下载预构建镜像
docker pull genesis/genesis:latest

# 或构建自己的镜像
git clone https://github.com/phonism/genesis.git
cd genesis
docker build -t genesis:local .

# 运行容器
docker run -it --gpus all genesis:local bash
```

Dockerfile内容：
```dockerfile
FROM nvidia/cuda:11.8-devel-ubuntu22.04

# 设置环境变量
ENV DEBIAN_FRONTEND=noninteractive
ENV PATH="/opt/conda/bin:$PATH"

# 安装系统依赖
RUN apt-get update && apt-get install -y \\
    wget git build-essential && \\
    rm -rf /var/lib/apt/lists/*

# 安装Miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \\
    bash Miniconda3-latest-Linux-x86_64.sh -b -p /opt/conda && \\
    rm Miniconda3-latest-Linux-x86_64.sh

# 创建环境并安装依赖
RUN conda create -n genesis python=3.9 -y
SHELL ["conda", "run", "-n", "genesis", "/bin/bash", "-c"]

RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 && \\
    pip install triton numpy matplotlib tqdm

# 复制并安装Genesis
COPY . /workspace/genesis
WORKDIR /workspace/genesis
RUN pip install -e genesis/

# 设置启动命令
ENTRYPOINT ["conda", "run", "-n", "genesis"]
CMD ["bash"]
```

## 📊 性能优化建议

安装完成后，可以通过以下方式优化性能：

```bash
# 设置环境变量
export CUDA_VISIBLE_DEVICES=0  # 指定GPU
export PYTHONPATH=$PWD:$PYTHONPATH

# 启用优化选项
export GENESIS_OPTIMIZE=1
export TRITON_CACHE_DIR=/tmp/triton_cache
```

## 🎯 下一步

安装完成后，建议：

1. [**运行第一个程序**](first-steps.md) - 验证安装并学习基础用法
2. [**查看教程**](../tutorials/basic-training.md) - 系统学习Genesis的使用
3. [**阅读架构文档**](../architecture/index.md) - 理解框架设计理念

---

如果在安装过程中遇到问题，请查看[FAQ](../contributing/index.md#faq)或在GitHub上提交issue。