# 开发环境配置

本指南将帮助你搭建Genesis开发环境，包括代码编辑、调试、测试等开发工作流程。

## 🛠️ 系统要求

### 硬件要求
- **CPU**: x86_64架构，支持AVX指令集
- **内存**: 最少16GB，推荐32GB+
- **GPU**: NVIDIA GPU with CUDA支持 (开发GPU算子时需要)
- **存储**: 20GB可用空间

### 软件要求
- **操作系统**: Linux (推荐Ubuntu 20.04+), macOS 10.15+
- **Python**: 3.8, 3.9, 3.10, 3.11
- **Git**: 最新版本
- **CUDA**: 11.8+ (GPU开发需要)

## 🚀 快速开始

### 1. 克隆仓库

```bash
# 克隆你的fork (推荐)
git clone https://github.com/YOUR_USERNAME/genesis.git
cd genesis

# 或克隆主仓库
git clone https://github.com/phonism/genesis.git
cd genesis

# 添加上游仓库 (如果fork的话)
git remote add upstream https://github.com/phonism/genesis.git
```

### 2. 创建Python环境

=== "使用conda"
    ```bash
    # 创建环境
    conda create -n genesis-dev python=3.9
    conda activate genesis-dev
    
    # 安装基础依赖
    conda install numpy matplotlib ipython jupyter
    ```

=== "使用venv"
    ```bash
    # 创建环境
    python -m venv genesis-dev
    source genesis-dev/bin/activate  # Linux/macOS
    # genesis-dev\\Scripts\\activate  # Windows
    
    # 升级pip
    pip install --upgrade pip setuptools wheel
    ```

### 3. 安装开发依赖

```bash
# 安装PyTorch (根据你的CUDA版本选择)
# CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# CPU版本
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# 安装Triton
pip install triton

# 安装开发工具
pip install -r requirements-dev.txt
```

### 4. 安装Genesis (开发模式)

```bash
# 开发模式安装 (推荐)
pip install -e .

# 验证安装
python -c "import genesis; print('Genesis开发环境配置成功！')"
```

## 📦 依赖管理

### 核心依赖 (requirements.txt)
```
torch>=2.0.0
triton>=2.0.0
numpy>=1.21.0
cuda-python>=11.8.0
```

### 开发依赖 (requirements-dev.txt)
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

## 🔧 开发工具配置

### 1. Git配置

```bash
# 配置用户信息
git config user.name "Your Name"
git config user.email "your.email@example.com"

# 配置提交模板
echo "feat: brief description

More detailed explanation (optional)

- Change 1
- Change 2

Fixes #123" > ~/.gitmessage
git config commit.template ~/.gitmessage
```

### 2. Pre-commit钩子

```bash
# 安装pre-commit
pip install pre-commit

# 安装钩子
pre-commit install

# 手动运行检查
pre-commit run --all-files
```

### 3. IDE配置

=== "VS Code"
    推荐安装以下扩展：
    
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
    
    配置文件：
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
    1. 打开项目设置 (File -> Settings)
    2. 配置Python解释器指向虚拟环境
    3. 启用代码格式化工具 (Black, isort)
    4. 配置测试运行器为pytest

### 4. 环境变量

```bash
# 开发环境变量
export GENESIS_DEV=1
export PYTHONPATH="${PWD}:${PYTHONPATH}"
export CUDA_VISIBLE_DEVICES=0  # 指定GPU设备

# 添加到 ~/.bashrc 或 ~/.zshrc
echo 'export GENESIS_DEV=1' >> ~/.bashrc
```

## 🧪 测试框架

### 测试目录结构
```
tests/
├── conftest.py              # pytest配置
├── test_autograd.py         # 自动微分测试
├── test_nn.py              # 神经网络测试
├── test_cuda_tensor.py     # CUDA张量测试
├── test_functional.py      # 函数式接口测试
├── benchmarks/             # 性能测试
│   ├── bench_matmul.py
│   └── bench_attention.py
└── integration/            # 集成测试
    ├── test_training.py
    └── test_models.py
```

### 运行测试

```bash
# 运行所有测试
pytest tests/ -v

# 运行特定测试文件
pytest tests/test_nn.py -v

# 运行特定测试函数
pytest tests/test_nn.py::test_linear_layer -v

# 运行带覆盖率的测试
pytest tests/ --cov=genesis --cov-report=html

# 运行性能测试
pytest tests/benchmarks/ -v --benchmark-only
```

### 编写测试

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

## 📊 性能分析

### 1. 内置profiler

```python
import genesis.utils.profile as profiler

# 使用context manager
with profiler.profile() as prof:
    # 你的代码
    x = genesis.randn(1000, 1000)
    y = genesis.matmul(x, x)

# 打印结果
prof.print_stats()

# 保存结果
prof.export_chrome_trace("profile.json")
```

### 2. 内存分析

```python
import genesis

# 启用内存跟踪
genesis.cuda.memory.enable_debug()

# 你的代码
x = genesis.randn(1000, 1000, device='cuda')
y = genesis.matmul(x, x)

# 查看内存使用
print(f"内存使用: {genesis.cuda.memory_allocated() / 1024**2:.1f} MB")
print(f"缓存内存: {genesis.cuda.memory_cached() / 1024**2:.1f} MB")

# 内存快照
snapshot = genesis.cuda.memory.memory_snapshot()
```

### 3. 基准测试

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

## 🐛 调试技巧

### 1. 调试环境变量

```bash
# 启用调试模式
export GENESIS_DEBUG=1
export CUDA_LAUNCH_BLOCKING=1  # 同步CUDA执行
export PYTHONFAULTHANDLER=1    # Python错误处理
```

### 2. 日志配置

```python
import logging
import genesis

# 配置日志
logging.basicConfig(level=logging.DEBUG)
genesis.set_log_level('DEBUG')

# 使用日志
logger = logging.getLogger(__name__)
logger.debug("调试信息")
```

### 3. 断点调试

```python
import pdb

def buggy_function(x):
    pdb.set_trace()  # 设置断点
    y = x * 2
    return y

# 或使用ipdb (需要安装: pip install ipdb)
import ipdb
ipdb.set_trace()
```

## 📚 文档开发

### 构建文档

```bash
# 安装文档依赖
pip install -r docs/requirements.txt

# 本地服务器
mkdocs serve

# 构建静态文件
mkdocs build

# 部署到GitHub Pages
mkdocs gh-deploy
```

### API文档生成

```bash
# 自动生成API文档
python scripts/generate_api_docs.py

# 检查docstring格式
pydocstyle genesis/
```

## 🚀 提交代码

### 1. 代码检查

```bash
# 格式化代码
black genesis/ tests/
isort genesis/ tests/

# 类型检查
mypy genesis/

# 代码质量检查
flake8 genesis/ tests/

# 运行测试
pytest tests/ -x
```

### 2. 提交流程

```bash
# 1. 同步最新代码
git fetch upstream
git rebase upstream/main

# 2. 创建功能分支
git checkout -b feature/your-feature

# 3. 开发和测试
# ... 你的开发工作 ...

# 4. 提交代码
git add .
git commit -m "feat: add your feature"

# 5. 推送分支
git push origin feature/your-feature

# 6. 创建Pull Request
```

## ❓ 常见问题

### Q: CUDA相关错误？
A: 检查CUDA版本兼容性，确保PyTorch和Triton版本匹配。

### Q: 测试失败？
A: 运行 `pytest tests/ -v` 查看详细错误信息，检查环境配置。

### Q: 性能问题？
A: 使用profiler分析瓶颈，检查是否启用了GPU加速。

### Q: 内存不足？
A: 减小测试用例的数据规模，启用CPU回退模式。

---

!!! success "开发环境配置完成"
    现在你可以开始为Genesis贡献代码了！

[下一步：了解测试规范](testing.md){ .md-button .md-button--primary }
[返回贡献指南](index.md){ .md-button }