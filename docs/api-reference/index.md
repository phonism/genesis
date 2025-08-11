# API 参考文档

Genesis深度学习框架提供了完整的API接口，本节提供详细的代码级文档和使用示例。

## 核心模块结构

### 主要命名空间

- **`genesis`**: 核心张量和自动微分系统
- **`genesis.nn`**: 神经网络模块和层
- **`genesis.optim`**: 优化器和学习率调度器
- **`genesis.functional`**: 函数式操作接口
- **`genesis.utils`**: 工具函数和辅助类

### 快速导航

| 模块 | 描述 | 主要类/函数 |
|------|------|------------|
| [genesis](genesis.md) | 核心张量系统 | `Tensor`, `autocast`, `no_grad` |
| [nn](nn.md) | 神经网络层 | `Module`, `Linear`, `MultiHeadAttention` |
| [optim](optim.md) | 优化器 | `SGD`, `Adam`, `AdamW` |
| [functional](functional.md) | 函数式操作 | `relu`, `softmax`, `matmul` |
| [utils](utils.md) | 工具函数 | `profile`, `DataLoader` |

## 代码约定

### 导入规范
```python
import genesis
import genesis.nn as nn
import genesis.optim as optim
import genesis.nn.functional as F
```

### 设备管理
```python
# 设置默认设备
genesis.set_default_device(genesis.cuda())

# 检查CUDA可用性
if genesis.cuda.is_available():
    device = genesis.cuda()
else:
    device = genesis.cpu()
```

### 数据类型
```python
# 支持的数据类型
genesis.float32  # 默认浮点类型
genesis.float16  # 半精度浮点
genesis.int32    # 32位整数
genesis.bool     # 布尔类型
```

## 快速示例

### 基础张量操作
```python
import genesis

# 创建张量
x = genesis.tensor([[1, 2], [3, 4]], dtype=genesis.float32)
y = genesis.randn(2, 2)

# 基础运算
z = x + y
result = genesis.matmul(x, y.T)

# 梯度计算
x.requires_grad_(True)
loss = (x ** 2).sum()
loss.backward()
print(x.grad)  # 打印梯度
```

### 神经网络模型
```python
import genesis.nn as nn

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        return self.fc2(x)

# 使用模型
model = MLP(784, 256, 10)
x = genesis.randn(32, 784)
output = model(x)
```

### 训练循环
```python
import genesis.optim as optim

# 初始化
model = MLP(784, 256, 10)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# 训练步骤
for epoch in range(100):
    for batch_idx, (data, target) in enumerate(dataloader):
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
```

## 性能优化提示

### 混合精度训练
```python
# 启用自动混合精度
genesis.enable_autocast = True

with genesis.autocast():
    output = model(input_tensor)
    loss = criterion(output, target)
```

### GPU内存优化
```python
# 使用inplace操作减少内存使用
x.relu_()  # inplace ReLU
x.add_(y)  # inplace 加法

# 释放不需要的梯度
with genesis.no_grad():
    inference_result = model(data)
```

### 批量操作优化
```python
# 批量矩阵乘法
batch_result = genesis.bmm(batch_a, batch_b)

# 向量化操作替代循环
result = genesis.sum(tensor, dim=1, keepdim=True)
```