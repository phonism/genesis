# 优化器 (genesis.optim)

## 概述

`genesis.optim`模块为训练神经网络提供优化器。它实现了最先进的优化算法，支持参数组、梯度裁剪和混合精度训练。

## 核心概念

### 优化过程

优化器使用各种算法基于计算的梯度更新模型参数：
1. **梯度下降**: 使用梯度进行基本参数更新
2. **动量**: 使用移动平均加速收敛
3. **自适应学习率**: 每个参数的不同学习率
4. **正则化**: 权重衰减和梯度裁剪

### 参数组

参数可以组织成具有不同超参数的组：
- 不同层的不同学习率
- 选择性权重衰减应用
- 层特定的优化设置

## 基类

### `optim.Optimizer`

所有优化器的抽象基类。

```python
class Optimizer:
    """
    所有优化器的基类。
    
    参数:
        params: 参数的可迭代对象或定义参数组的字典
        defaults: 包含优化选项默认值的字典
    """
    
    def __init__(self, params, defaults: dict):
        """
        初始化优化器。
        
        参数:
            params: 模型参数或参数组
            defaults: 默认超参数值
        """
```

#### 核心方法

##### 优化步骤
```python
def step(self, closure: Optional[Callable] = None) -> Optional[float]:
    """
    执行单个优化步骤。
    
    参数:
        closure: 重新评估模型并返回损失的可选函数
        
    返回:
        如果提供closure则返回损失值
        
    示例:
        >>> optimizer.zero_grad()
        >>> loss = criterion(output, target)
        >>> loss.backward()
        >>> optimizer.step()
    """

def zero_grad(self, set_to_none: bool = True) -> None:
    """
    清除所有优化参数的梯度。
    
    参数:
        set_to_none: 如果为True，将梯度设置为None而不是零
        
    示例:
        >>> # 每个训练步骤前清除梯度
        >>> optimizer.zero_grad()
        >>> loss.backward()
        >>> optimizer.step()
    """
```

##### 状态管理
```python
def state_dict(self) -> Dict[str, Any]:
    """
    将优化器状态返回为字典。
    
    返回:
        包含优化器状态和参数组的字典
        
    示例:
        >>> # 保存优化器状态
        >>> state = optimizer.state_dict()
        >>> genesis.save(state, 'optimizer_checkpoint.pth')
    """

def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
    """
    从字典加载优化器状态。
    
    参数:
        state_dict: 优化器状态字典
        
    示例:
        >>> # 恢复优化器状态
        >>> state = genesis.load('optimizer_checkpoint.pth')
        >>> optimizer.load_state_dict(state)
    """
```

##### 参数组
```python
def add_param_group(self, param_group: Dict[str, Any]) -> None:
    """
    向优化器添加参数组。
    
    参数:
        param_group: 指定参数及其选项的字典
        
    示例:
        >>> # 添加具有不同学习率的新层
        >>> optimizer.add_param_group({
        ...     'params': new_layer.parameters(),
        ...     'lr': 0.001
        ... })
    """

@property
def param_groups(self) -> List[Dict[str, Any]]:
    """
    访问参数组。
    
    返回:
        参数组字典的列表
        
    示例:
        >>> # 手动调整学习率
        >>> for group in optimizer.param_groups:
        ...     group['lr'] *= 0.9
    """
```

## 优化器

### `optim.SGD`

带动量和权重衰减的随机梯度下降优化器。

```python
class SGD(Optimizer):
    """
    随机梯度下降优化器。
    
    参数:
        params: 要优化的参数的可迭代对象
        lr: 学习率（必需）
        momentum: 动量因子（默认: 0）
        dampening: 动量的阻尼（默认: 0）
        weight_decay: 权重衰减系数（默认: 0）
        nesterov: 是否使用Nesterov动量（默认: False）
        
    算法:
        v_t = momentum * v_{t-1} + g_t
        p_t = p_{t-1} - lr * v_t
        
    其中:
        g_t: 时刻t的梯度
        v_t: 时刻t的速度
        p_t: 时刻t的参数
    """
    
    def __init__(
        self,
        params,
        lr: float,
        momentum: float = 0,
        dampening: float = 0,
        weight_decay: float = 0,
        nesterov: bool = False
    ):
```

#### 使用示例

```python
import genesis.optim as optim

# 基本SGD
optimizer = optim.SGD(model.parameters(), lr=0.01)

# 带动量的SGD（大多数任务推荐）
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# 带权重衰减的SGD
optimizer = optim.SGD(model.parameters(), lr=0.01, 
                     momentum=0.9, weight_decay=1e-4)

# Nesterov加速梯度
optimizer = optim.SGD(model.parameters(), lr=0.01,
                     momentum=0.9, nesterov=True)

# 不同层的不同学习率
optimizer = optim.SGD([
    {'params': model.features.parameters(), 'lr': 0.001},
    {'params': model.classifier.parameters(), 'lr': 0.01}
], momentum=0.9)
```

### `optim.Adam`

结合RMSprop和动量的自适应矩估计优化器。

```python
class Adam(Optimizer):
    """
    Adam优化器。
    
    参数:
        params: 要优化的参数的可迭代对象
        lr: 学习率（默认: 1e-3）
        betas: 计算梯度及其平方的运行平均的系数
               （默认: (0.9, 0.999)）
        eps: 添加到分母以提高数值稳定性的项（默认: 1e-8）
        weight_decay: 权重衰减系数（默认: 0）
        amsgrad: 是否使用AMSGrad变体（默认: False）
        
    算法:
        m_t = β₁ * m_{t-1} + (1 - β₁) * g_t
        v_t = β₂ * v_{t-1} + (1 - β₂) * g_t²
        m̂_t = m_t / (1 - β₁ᵗ)
        v̂_t = v_t / (1 - β₂ᵗ)
        p_t = p_{t-1} - lr * m̂_t / (√v̂_t + ε)
        
    其中:
        g_t: 梯度
        m_t: 一阶矩估计（动量）
        v_t: 二阶矩估计（自适应学习率）
        m̂_t, v̂_t: 偏差修正的矩估计
    """
    
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0,
        amsgrad: bool = False
    ):
```

#### 状态变量

每个参数维护以下状态：
- `step`: 已执行的优化步数
- `exp_avg`: 梯度值的指数移动平均（动量）
- `exp_avg_sq`: 平方梯度值的指数移动平均
- `max_exp_avg_sq`: exp_avg_sq的最大值（仅AMSGrad）

#### 使用示例

```python
# 默认Adam（最常见）
optimizer = optim.Adam(model.parameters())

# 自定义学习率
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Transformer模型设置
optimizer = optim.Adam(model.parameters(), lr=0.0001,
                      betas=(0.9, 0.98), eps=1e-9)

# 带权重衰减
optimizer = optim.Adam(model.parameters(), lr=0.001,
                      weight_decay=1e-5)

# 不同学习率的微调
optimizer = optim.Adam([
    {'params': model.encoder.parameters(), 'lr': 1e-5},
    {'params': model.decoder.parameters(), 'lr': 1e-4},
    {'params': model.head.parameters(), 'lr': 1e-3}
])

# 使用AMSGrad变体
optimizer = optim.Adam(model.parameters(), lr=0.001, amsgrad=True)
```

### `optim.AdamW`

带解耦权重衰减的Adam优化器。

```python
class AdamW(Optimizer):
    """
    AdamW优化器（带解耦权重衰减的Adam）。
    
    参数:
        params: 要优化的参数的可迭代对象
        lr: 学习率（默认: 1e-3）
        betas: 计算运行平均的系数（默认: (0.9, 0.999)）
        eps: 数值稳定性项（默认: 1e-8）
        weight_decay: 权重衰减系数（默认: 0.01）
        amsgrad: 是否使用AMSGrad变体（默认: False）
        
    与Adam的区别:
        Adam: p_t = p_{t-1} - lr * (m̂_t / (√v̂_t + ε) + wd * p_{t-1})
        AdamW: p_t = p_{t-1} * (1 - lr * wd) - lr * m̂_t / (√v̂_t + ε)
        
    AdamW将权重衰减从梯度计算中解耦，直接应用到参数上
    以获得更好的正则化。
    """
    
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
        amsgrad: bool = False
    ):
```

#### 使用示例

```python
# 默认AdamW（Transformers推荐）
optimizer = optim.AdamW(model.parameters())

# BERT/GPT标准设置
optimizer = optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)

# 大模型训练
optimizer = optim.AdamW(model.parameters(), lr=1e-4,
                       betas=(0.9, 0.95), weight_decay=0.1)

# 从权重衰减中排除偏置和归一化
decay_params = []
no_decay_params = []
for name, param in model.named_parameters():
    if any(nd in name for nd in ['bias', 'norm', 'ln']):
        no_decay_params.append(param)
    else:
        decay_params.append(param)

optimizer = optim.AdamW([
    {'params': decay_params, 'weight_decay': 0.01},
    {'params': no_decay_params, 'weight_decay': 0.0}
], lr=1e-4)
```

### `optim.RMSprop`

均方根传播优化器。

```python
class RMSprop(Optimizer):
    """
    RMSprop优化器。
    
    参数:
        params: 要优化的参数的可迭代对象
        lr: 学习率（默认: 1e-2）
        alpha: 平滑常数（默认: 0.99）
        eps: 数值稳定性项（默认: 1e-8）
        weight_decay: 权重衰减系数（默认: 0）
        momentum: 动量因子（默认: 0）
        centered: 是否按中心化的二阶矩归一化（默认: False）
    """
```

## 梯度裁剪

防止梯度爆炸的实用工具。

```python
def clip_grad_norm_(
    parameters: Iterable[Tensor],
    max_norm: float,
    norm_type: float = 2.0,
    error_if_nonfinite: bool = False
) -> float:
    """
    按全局范数裁剪梯度。
    
    参数:
        parameters: 有梯度的参数的可迭代对象
        max_norm: 最大梯度范数
        norm_type: 范数类型（1, 2, 或 inf）
        error_if_nonfinite: 如果总范数非有限则报错
        
    返回:
        裁剪前梯度的总范数
        
    示例:
        >>> loss.backward()
        >>> # 裁剪梯度以防止爆炸
        >>> genesis.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        >>> optimizer.step()
    """

def clip_grad_value_(
    parameters: Iterable[Tensor],
    clip_value: float
) -> None:
    """
    按值裁剪梯度。
    
    参数:
        parameters: 有梯度的参数的可迭代对象
        clip_value: 裁剪阈值
        
    示例:
        >>> loss.backward()
        >>> # 将梯度限制在[-1, 1]范围内
        >>> genesis.nn.utils.clip_grad_value_(model.parameters(), clip_value=1.0)
        >>> optimizer.step()
    """
```

## 训练示例

### 基本训练循环

```python
import genesis
import genesis.nn as nn
import genesis.optim as optim

# 模型和优化器设置
model = MyModel()
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# 训练循环
for epoch in range(num_epochs):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        # 梯度清零
        optimizer.zero_grad()
        
        # 前向传播
        output = model(data)
        loss = criterion(output, target)
        
        # 后向传播
        loss.backward()
        
        # 梯度裁剪（可选）
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # 更新参数
        optimizer.step()
        
        if batch_idx % 100 == 0:
            print(f'Epoch: {epoch}, Batch: {batch_idx}, Loss: {loss.item():.4f}')
```

### 混合精度的高级训练

```python
import genesis
import genesis.nn as nn
import genesis.optim as optim

# 启用混合精度
genesis.enable_autocast = True

# 模型设置
model = TransformerModel()
optimizer = optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)

for epoch in range(num_epochs):
    for batch in train_loader:
        optimizer.zero_grad()
        
        # 使用autocast进行混合精度
        with genesis.autocast():
            outputs = model(batch['input'])
            loss = criterion(outputs, batch['target'])
        
        # 后向传播
        loss.backward()
        
        # 梯度裁剪
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # 优化器步骤
        optimizer.step()
```

### 学习率调度

```python
from genesis.optim.lr_scheduler import CosineAnnealingLR

# 优化器和调度器
optimizer = optim.AdamW(model.parameters(), lr=1e-3)
scheduler = CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-6)

for epoch in range(num_epochs):
    # 训练
    for batch in train_loader:
        optimizer.zero_grad()
        loss = compute_loss(model, batch)
        loss.backward()
        optimizer.step()
    
    # 更新学习率
    scheduler.step()
    print(f'Epoch {epoch}, LR: {scheduler.get_last_lr()[0]:.6f}')
```

### 梯度累积

```python
# 通过累积模拟更大的批大小
accumulation_steps = 4
optimizer.zero_grad()

for i, batch in enumerate(train_loader):
    # 前向传播
    outputs = model(batch['input'])
    loss = criterion(outputs, batch['target'])
    
    # 按累积步数归一化损失
    loss = loss / accumulation_steps
    loss.backward()
    
    # 每accumulation_steps更新权重
    if (i + 1) % accumulation_steps == 0:
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        optimizer.zero_grad()
```

## 优化器选择指南

### SGD
- **优点**: 简单、内存高效、泛化性好
- **缺点**: 收敛慢、对学习率敏感
- **适用于**:
  - 计算机视觉任务（ResNet、VGG）
  - 内存受限环境
  - 需要最佳泛化时
- **推荐设置**: `lr=0.1, momentum=0.9, weight_decay=1e-4`

### Adam
- **优点**: 收敛快、自适应、对超参数不敏感
- **缺点**: 内存使用较高、可能过拟合
- **适用于**:
  - NLP任务
  - 快速原型制作
  - 稀疏梯度
- **推荐设置**: `lr=1e-3, betas=(0.9, 0.999)`

### AdamW
- **优点**: 比Adam泛化性更好、对大模型优秀
- **缺点**: 内存使用较高
- **适用于**:
  - Transformer模型（BERT、GPT）
  - 大规模预训练
  - 需要强正则化时
- **推荐设置**: `lr=5e-5, weight_decay=0.01`

### RMSprop
- **优点**: 对非平稳目标函数好
- **缺点**: 高学习率时可能不稳定
- **适用于**:
  - RNN训练
  - 强化学习
  - 非平稳问题

## 性能提示

1. **梯度累积**: 内存有限时模拟更大批大小
2. **梯度裁剪**: 对RNN和Transformers必不可少
3. **参数组**: 对不同层使用不同学习率
4. **权重衰减**: AdamW通常比Adam + L2正则化表现更好
5. **学习率预热**: 大批量训练时使用预热
6. **混合精度**: 减少内存使用并加速训练

## 内存考虑

- 优化器维护每个参数的状态（Adam/AdamW使用2倍参数内存）
- 使用`zero_grad(set_to_none=True)`减少内存碎片
- 在设备间移动模型时考虑优化器状态
- 在检查点中保存优化器状态以恢复训练

## 最佳实践

1. **始终清除梯度** 在后向传播前
2. **使用梯度裁剪** 对RNN和Transformers
3. **监控学习率** 整个训练过程
4. **保存优化器状态** 在检查点中
5. **为模型类型使用适当的权重衰减**
6. **为大模型考虑混合精度**

## 另请参阅

- [学习率调度器](lr_scheduler.md) - 动态学习率调整
- [神经网络模块](../nn/modules.md) - 构建模型
- [自动微分](../autograd.md) - 自动微分
- [示例](../../../samples/) - 完整训练示例