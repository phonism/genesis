# 快速开始

欢迎使用 Genesis 深度学习框架！本指南将帮助您在几分钟内开始使用 Genesis。

## 🎯 概述

Genesis 是一个专为学习和研究设计的轻量级深度学习框架，它提供了：

- 简洁直观的API设计
- 高性能GPU加速计算
- 完整的神经网络训练功能
- 与PyTorch生态良好的兼容性
- 混合精度训练支持（FP16/BF16）
- 内置LLM模型如Qwen

## ⚡ 5分钟快速体验

### 1. 安装 Genesis

```bash
# 安装核心依赖
pip install torch triton numpy cuda-python

# 克隆代码仓库
git clone https://github.com/phonism/genesis.git
cd genesis

# 安装 Genesis
pip install -e .
```

### 2. 第一个神经网络

```python
import genesis
import genesis.nn as nn

# 定义一个简单的多层感知机
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.layer1(x))
        return self.layer2(x)

# 创建模型和数据
model = MLP(784, 128, 10)
x = genesis.randn(32, 784)  # 批次大小32，输入维度784

# 前向传播
output = model(x)
print(f"输出形状: {output.shape}")  # torch.Size([32, 10])
```

### 3. 训练循环

```python
import genesis.optim as optim

# 创建优化器和损失函数
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# 模拟训练数据
targets = genesis.randint(0, 10, (32,))

# 训练一个批次
optimizer.zero_grad()        # 清零梯度
output = model(x)           # 前向传播
loss = criterion(output, targets)  # 计算损失
loss.backward()             # 反向传播
optimizer.step()            # 更新参数

print(f"损失: {loss.item():.4f}")
```

## 📚 核心概念

### 张量 (Tensor)
Genesis中的基础数据结构，支持自动微分：

```python
import genesis

# 创建张量（自动数据类型推断）
x = genesis.tensor([1.0, 2.0, 3.0], requires_grad=True)  # → float32
y = genesis.tensor([4, 5, 6])                           # → int64
z = genesis.tensor([1, 2, 3], dtype=genesis.float32)     # 显式数据类型

# 基础操作
result = x * y.float() + x.sum()  # 广播和类型转换

# PyTorch风格的归约操作
total = x.sum()                      # 所有元素求和
mean_val = x.mean()                  # 所有元素平均值
max_val = x.max()                    # 最大元素

# 按维度操作
data = genesis.tensor([[1, 2, 3], [4, 5, 6]])
row_sums = data.sum(dim=1)                    # 按行求和
col_means = data.mean(dim=0, keepdim=True)   # 按列求平均值，保持维度

# 也支持NumPy风格（兼容性）
numpy_style = data.sum(axis=0, keepdims=True)

# 计算梯度
result.backward()
print(f"x的梯度: {x.grad}")  # 关于x的梯度
```

### 模块 (Module)
神经网络组件的基类：

```python
import genesis.nn as nn

class CustomLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = genesis.randn(out_features, in_features, requires_grad=True)
        self.bias = genesis.zeros(out_features, requires_grad=True)
    
    def forward(self, x):
        return genesis.functional.linear(x, self.weight, self.bias)

# 使用自定义层
layer = CustomLayer(10, 5)
input_tensor = genesis.randn(3, 10)
output = layer(input_tensor)
```

### 优化器 (Optimizer)
参数更新算法：

```python
import genesis.optim as optim

# 不同的优化器选择
sgd_optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
adam_optimizer = optim.Adam(model.parameters(), lr=0.001)
adamw_optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
```

## 🛠️ 环境配置

### 硬件要求

- **CPU**: 现代多核处理器
- **内存**: 最少8GB RAM，推荐16GB+
- **GPU**: NVIDIA GPU with CUDA支持 (推荐)
- **存储**: 至少2GB可用空间

### 软件依赖

```bash
# Python环境
Python >= 3.8

# 核心依赖
torch >= 2.0.0
triton >= 2.0.0
numpy >= 1.21.0
cuda-python >= 11.8.0  # GPU支持

# 可选依赖
matplotlib >= 3.5.0  # 用于可视化
tqdm >= 4.64.0      # 进度条
wandb >= 0.13.0     # 实验跟踪
```

## 📖 下一步

现在你已经了解了Genesis的基础用法，可以继续探索：

### 🎓 深入学习
- [**完整安装指南**](installation.md) - 详细的安装和配置步骤
- [**第一个完整程序**](first-steps.md) - 构建完整的训练流程
- [**基础训练教程**](../tutorials/basic-training.md) - 系统性的训练教程

### 🔍 架构理解
- [**架构概述**](../architecture/index.md) - 了解Genesis的整体设计
- [**核心组件**](../core-components/index.md) - 深入理解内部实现
- [**API参考**](../api-reference/index.md) - 完整的API文档

### 🚀 高级特性
- [**自定义算子**](../tutorials/custom-ops.md) - 实现自定义操作
- [**性能优化**](../tutorials/performance-tuning.md) - 训练性能调优
- [**分布式训练**](../neural-networks/distributed.md) - 多GPU训练

## ❓ 常见问题

### Q: Genesis与PyTorch有什么区别？
A: Genesis是教育导向的框架，代码更简洁易懂，适合学习深度学习的内部实现。PyTorch更适合生产环境使用。

### Q: 可以在生产环境中使用Genesis吗？
A: Genesis主要用于教育和研究，虽然功能完整，但建议生产环境使用更成熟的框架如PyTorch。

### Q: 如何获得帮助？
A: 可以通过GitHub Issues、Discussions或查看详细文档获得帮助。

---

## 🎉 准备好了吗？

让我们开始深入了解Genesis吧！

[详细安装指南](installation.md){ .md-button .md-button--primary }
[完整教程](../tutorials/index.md){ .md-button }