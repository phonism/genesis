# Genesis 深度学习框架

<div align="center">

**基于 Python + Triton + CUDA 从零构建的轻量级深度学习框架**

[![GitHub stars](https://img.shields.io/github/stars/phonism/genesis?style=social)](https://github.com/phonism/genesis/stargazers)
[![License](https://img.shields.io/github/license/phonism/genesis)](https://github.com/phonism/genesis/blob/main/LICENSE)
[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/downloads/)

[快速开始](getting-started/index.zh.md){ .md-button .md-button--primary }
[API 文档](api-reference/index.zh.md){ .md-button }
[GitHub](https://github.com/phonism/genesis){ .md-button }

</div>

---

## ✨ 特性

🚀 **高性能计算**  
基于Triton和CUDA的优化GPU核心，提供出色的计算性能

🔧 **简洁易用**  
PyTorch风格的API设计，学习成本低，上手容易

⚡ **轻量级架构**  
精简的核心设计，专注于深度学习的核心功能

🎯 **从零构建**  
完全自主实现的深度学习框架，深入理解每个组件

---

## 🏁 快速开始

### 安装

```bash
pip install genesis-dl
```

### 基础使用

```python
import genesis

# 创建张量
x = genesis.tensor([[1, 2], [3, 4]], dtype=genesis.float32, device=genesis.device('cuda'))
y = genesis.tensor([[5, 6], [7, 8]], dtype=genesis.float32, device=genesis.device('cuda'))

# 基本运算
z = x + y
print(z)
# 输出: [[6, 8], [10, 12]]

# 矩阵乘法
result = genesis.matmul(x, y)
print(result)
```

### 神经网络示例

```python
import genesis
import genesis.nn as nn

class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(784, 128)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(128, 10)
    
    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x

# 创建模型
model = SimpleNet()
model.cuda()  # 移至GPU

# 前向传播
x = genesis.randn(32, 784, device=genesis.device('cuda'))
output = model(x)
```

---

## 📚 文档导航

<div class="grid cards" markdown>

-   :material-rocket-launch:{ .lg .middle } __快速开始__

    ---

    快速了解Genesis框架的基本概念和使用方法

    [:octicons-arrow-right-24: 开始使用](getting-started/index.zh.md)

-   :material-book-open-page-variant:{ .lg .middle } __教程指南__

    ---

    详细的教程和示例，从基础到高级应用

    [:octicons-arrow-right-24: 查看教程](tutorials/index.zh.md)

-   :material-cogs:{ .lg .middle } __核心组件__

    ---

    深入了解Genesis的核心组件和架构设计

    [:octicons-arrow-right-24: 核心组件](core-components/index.zh.md)

-   :material-api:{ .lg .middle } __API参考__

    ---

    完整的API文档和参考资料

    [:octicons-arrow-right-24: API文档](api-reference/index.zh.md)

</div>

---

## 🛠️ 架构特点

### 双后端设计
- **CPU后端**: 基于PyTorch，确保功能完整性和正确性
- **GPU后端**: 基于Triton和CUDA，追求极致性能

### 现代化设计
- **自动微分系统**: 高效的梯度计算和反向传播
- **内存管理**: 优化的CUDA内存分配和管理策略  
- **算子优化**: 针对深度学习工作负载的专门优化

### 扩展性
- **模块化设计**: 便于添加新功能和算子
- **插件系统**: 支持自定义操作和扩展

---

## 🤝 贡献

我们欢迎所有形式的贡献！

- 🐛 [报告Bug](https://github.com/phonism/genesis/issues)
- 💡 [提出功能建议](https://github.com/phonism/genesis/issues)
- 📖 [改进文档](contributing/index.zh.md)
- 🔧 [贡献代码](contributing/development.zh.md)

---

## 📄 许可证

本项目采用 [MIT 许可证](https://github.com/phonism/genesis/blob/main/LICENSE)。