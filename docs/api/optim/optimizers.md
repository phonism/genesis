# 优化器 (genesis.optim)

Genesis优化器模块提供了训练神经网络所需的各种优化算法。

## 模块概述

`genesis.optim`模块包含：
- 基础优化器类（Optimizer）
- 经典优化器（SGD、Adam、AdamW）
- 学习率调度器（在schedulers.md中详述）
- 梯度裁剪工具

## 基础类

### Optimizer

所有优化器的抽象基类。

```python
class Optimizer:
    """优化器基类"""
    
    def __init__(self, params, defaults):
        """
        初始化优化器
        
        参数:
            params: list - 参数列表或参数组
            defaults: dict - 默认超参数
        """
```

#### 核心方法

```python
def step(self, closure=None):
    """
    执行一步优化
    
    参数:
        closure: callable, optional - 重新计算损失的闭包函数
        
    返回:
        loss - 如果提供closure，返回损失值
        
    示例:
        >>> optimizer.zero_grad()
        >>> loss = criterion(output, target)
        >>> loss.backward()
        >>> optimizer.step()
    """

def zero_grad(self):
    """
    清零所有参数的梯度
    
    示例:
        >>> # 在每个训练步骤开始时清零梯度
        >>> optimizer.zero_grad()
        >>> output = model(input)
        >>> loss = criterion(output, target)
        >>> loss.backward()
        >>> optimizer.step()
    """

def state_dict(self) -> dict:
    """
    返回优化器状态字典
    
    返回:
        dict - 包含state和param_groups的字典
        
    示例:
        >>> # 保存优化器状态
        >>> state = optimizer.state_dict()
        >>> genesis.save(state, 'optimizer.pth')
    """

def load_state_dict(self, state_dict: dict):
    """
    加载优化器状态
    
    参数:
        state_dict: dict - 状态字典
        
    示例:
        >>> # 恢复优化器状态
        >>> state = genesis.load('optimizer.pth')
        >>> optimizer.load_state_dict(state)
    """

def add_param_group(self, param_group: dict):
    """
    添加参数组
    
    参数:
        param_group: dict - 新的参数组
        
    示例:
        >>> # 为新添加的层设置不同的学习率
        >>> optimizer.add_param_group({
        ...     'params': new_layer.parameters(),
        ...     'lr': 0.001
        ... })
    """

@property
def param_groups(self) -> List[dict]:
    """
    获取参数组列表
    
    返回:
        List[dict] - 每个字典包含'params'和其他超参数
        
    示例:
        >>> for group in optimizer.param_groups:
        ...     group['lr'] *= 0.95  # 手动调整学习率
    """
```

## SGD优化器

随机梯度下降优化器，支持动量和权重衰减。

```python
class SGD(Optimizer):
    """
    随机梯度下降优化器
    
    参数:
        params: iterable - 待优化参数的迭代器
        lr: float - 学习率，必需参数
        momentum: float - 动量因子，默认0
        dampening: float - 动量抑制，默认0
        weight_decay: float - 权重衰减（L2正则化），默认0
        nesterov: bool - 是否使用Nesterov动量，默认False
    """
    
    def __init__(self, params, lr, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False):
        """
        初始化SGD优化器
        
        算法:
            v_t = momentum * v_{t-1} + g_t
            p_t = p_{t-1} - lr * v_t
            
        其中g_t是梯度，v_t是速度，p_t是参数
        
        示例:
            >>> # 基础SGD
            >>> optimizer = optim.SGD(model.parameters(), lr=0.01)
            >>> 
            >>> # 带动量的SGD
            >>> optimizer = optim.SGD(model.parameters(), lr=0.01, 
            ...                      momentum=0.9)
            >>> 
            >>> # 带权重衰减的SGD
            >>> optimizer = optim.SGD(model.parameters(), lr=0.01,
            ...                      momentum=0.9, weight_decay=1e-4)
            >>> 
            >>> # Nesterov动量SGD
            >>> optimizer = optim.SGD(model.parameters(), lr=0.01,
            ...                      momentum=0.9, nesterov=True)
        """
    
    def step(self, closure=None):
        """
        执行一步SGD更新
        
        更新规则:
            如果使用动量:
                buf_t = momentum * buf_{t-1} + (1 - dampening) * g_t
                如果使用nesterov:
                    g_t = g_t + momentum * buf_t
                否则:
                    g_t = buf_t
            p_t = p_{t-1} - lr * g_t
        """
```

### 使用示例

```python
import genesis.optim as optim

# 基础SGD
optimizer = optim.SGD(model.parameters(), lr=0.1)

# 带动量的SGD（推荐用于大多数任务）
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# 不同参数组使用不同学习率
optimizer = optim.SGD([
    {'params': model.base.parameters(), 'lr': 0.001},
    {'params': model.head.parameters(), 'lr': 0.01}
], momentum=0.9)

# 训练循环
for epoch in range(num_epochs):
    for batch in dataloader:
        optimizer.zero_grad()
        output = model(batch['input'])
        loss = criterion(output, batch['target'])
        loss.backward()
        optimizer.step()
```

## Adam优化器

自适应矩估计优化器，结合了RMSprop和动量。

```python
class Adam(Optimizer):
    """
    Adam优化器
    
    参数:
        params: iterable - 待优化参数的迭代器
        lr: float - 学习率，默认1e-3
        betas: Tuple[float, float] - 用于计算梯度及其平方的移动平均的系数
                                     默认(0.9, 0.999)
        eps: float - 数值稳定性参数，默认1e-8
        weight_decay: float - 权重衰减，默认0
        amsgrad: bool - 是否使用AMSGrad变体，默认False
    """
    
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, amsgrad=False):
        """
        初始化Adam优化器
        
        算法:
            m_t = β1 * m_{t-1} + (1 - β1) * g_t
            v_t = β2 * v_{t-1} + (1 - β2) * g_t^2
            m̂_t = m_t / (1 - β1^t)
            v̂_t = v_t / (1 - β2^t)
            p_t = p_{t-1} - lr * m̂_t / (√v̂_t + ε)
            
        其中:
            g_t: 梯度
            m_t: 一阶矩估计（动量）
            v_t: 二阶矩估计（自适应学习率）
            m̂_t, v̂_t: 偏差修正的矩估计
            
        示例:
            >>> # 默认Adam
            >>> optimizer = optim.Adam(model.parameters())
            >>> 
            >>> # 自定义学习率
            >>> optimizer = optim.Adam(model.parameters(), lr=0.001)
            >>> 
            >>> # 调整beta参数
            >>> optimizer = optim.Adam(model.parameters(), lr=0.001,
            ...                       betas=(0.9, 0.98))
            >>> 
            >>> # 使用AMSGrad
            >>> optimizer = optim.Adam(model.parameters(), lr=0.001,
            ...                       amsgrad=True)
        """
    
    def step(self, closure=None):
        """
        执行一步Adam更新
        
        状态变量:
            - exp_avg: 梯度的指数移动平均（m_t）
            - exp_avg_sq: 梯度平方的指数移动平均（v_t）
            - max_exp_avg_sq: 最大的v_t（仅AMSGrad使用）
            - step: 当前步数（用于偏差修正）
        """
    
    @property
    def state(self) -> dict:
        """
        优化器状态
        
        每个参数的状态包含:
            - step: int - 更新步数
            - exp_avg: Tensor - 一阶矩估计
            - exp_avg_sq: Tensor - 二阶矩估计
            - max_exp_avg_sq: Tensor - 最大二阶矩（AMSGrad）
        """
```

### 使用示例

```python
# 默认Adam（最常用）
optimizer = optim.Adam(model.parameters())

# 自定义学习率
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# Transformer模型常用设置
optimizer = optim.Adam(model.parameters(), lr=0.0001, 
                      betas=(0.9, 0.98), eps=1e-9)

# 带权重衰减
optimizer = optim.Adam(model.parameters(), lr=0.001,
                      weight_decay=1e-5)

# 微调预训练模型（不同层不同学习率）
optimizer = optim.Adam([
    {'params': model.encoder.parameters(), 'lr': 1e-5},
    {'params': model.decoder.parameters(), 'lr': 1e-4},
    {'params': model.head.parameters(), 'lr': 1e-3}
])
```

## AdamW优化器

Adam优化器的改进版本，解耦权重衰减。

```python
class AdamW(Optimizer):
    """
    AdamW优化器（解耦权重衰减的Adam）
    
    参数:
        params: iterable - 待优化参数的迭代器
        lr: float - 学习率，默认1e-3
        betas: Tuple[float, float] - 动量系数，默认(0.9, 0.999)
        eps: float - 数值稳定性参数，默认1e-8
        weight_decay: float - 权重衰减系数，默认0.01
        amsgrad: bool - 是否使用AMSGrad，默认False
    """
    
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0.01, amsgrad=False):
        """
        初始化AdamW优化器
        
        与Adam的区别:
            Adam: p_t = p_{t-1} - lr * (m̂_t / (√v̂_t + ε) + wd * p_{t-1})
            AdamW: p_t = p_{t-1} * (1 - lr * wd) - lr * m̂_t / (√v̂_t + ε)
            
        AdamW将权重衰减从梯度计算中解耦，直接作用于参数
        
        示例:
            >>> # 默认AdamW（推荐用于Transformer）
            >>> optimizer = optim.AdamW(model.parameters())
            >>> 
            >>> # BERT/GPT常用设置
            >>> optimizer = optim.AdamW(model.parameters(), lr=5e-5,
            ...                        weight_decay=0.01)
            >>> 
            >>> # 大模型训练设置
            >>> optimizer = optim.AdamW(model.parameters(), lr=1e-4,
            ...                        betas=(0.9, 0.95),
            ...                        weight_decay=0.1)
        """
    
    def step(self, closure=None):
        """
        执行一步AdamW更新
        
        更新规则:
            1. 计算Adam更新（不包括权重衰减）
            2. 单独应用权重衰减: p = p * (1 - lr * weight_decay)
        """
```

### 使用示例

```python
# Transformer模型标准配置
optimizer = optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)

# GPT风格模型
optimizer = optim.AdamW(model.parameters(), lr=1e-4,
                       betas=(0.9, 0.95), weight_decay=0.1)

# 排除某些参数的权重衰减（如偏置和LayerNorm）
decay_params = []
no_decay_params = []
for name, param in model.named_parameters():
    if 'bias' in name or 'ln' in name or 'norm' in name:
        no_decay_params.append(param)
    else:
        decay_params.append(param)

optimizer = optim.AdamW([
    {'params': decay_params, 'weight_decay': 0.01},
    {'params': no_decay_params, 'weight_decay': 0.0}
], lr=1e-4)
```

## 梯度裁剪

防止梯度爆炸的实用函数。

```python
def clip_grad_norm_(parameters, max_norm: float, norm_type: float = 2.0):
    """
    裁剪梯度范数
    
    参数:
        parameters: iterable - 参数迭代器
        max_norm: float - 最大梯度范数
        norm_type: float - 范数类型（1、2或inf）
        
    返回:
        float - 裁剪前的总梯度范数
        
    示例:
        >>> loss.backward()
        >>> # 裁剪梯度，防止梯度爆炸
        >>> genesis.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        >>> optimizer.step()
    """

def clip_grad_value_(parameters, clip_value: float):
    """
    裁剪梯度值
    
    参数:
        parameters: iterable - 参数迭代器
        clip_value: float - 裁剪阈值
        
    示例:
        >>> loss.backward()
        >>> # 将梯度限制在[-1, 1]范围内
        >>> genesis.nn.utils.clip_grad_value_(model.parameters(), clip_value=1.0)
        >>> optimizer.step()
    """
```

## 完整训练示例

### 基础训练循环
```python
import genesis
import genesis.nn as nn
import genesis.optim as optim

# 模型和优化器
model = MyModel()
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# 训练循环
for epoch in range(num_epochs):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        # 清零梯度
        optimizer.zero_grad()
        
        # 前向传播
        output = model(data)
        loss = criterion(output, target)
        
        # 反向传播
        loss.backward()
        
        # 梯度裁剪（可选）
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # 更新参数
        optimizer.step()
        
        if batch_idx % 100 == 0:
            print(f'Epoch: {epoch}, Batch: {batch_idx}, Loss: {loss.item():.4f}')
```

### 带学习率调度的训练
```python
from genesis.optim.lr_scheduler import get_cosine_schedule_with_warmup

# 优化器和调度器
optimizer = optim.AdamW(model.parameters(), lr=5e-5)
num_training_steps = len(train_loader) * num_epochs
scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=num_training_steps * 0.1,
    num_training_steps=num_training_steps
)

# 训练循环
for epoch in range(num_epochs):
    for batch in train_loader:
        optimizer.zero_grad()
        loss = compute_loss(model, batch)
        loss.backward()
        optimizer.step()
        scheduler.step()  # 更新学习率
```

### 混合精度训练
```python
genesis.enable_autocast = True

# 使用自动混合精度
with genesis.autocast():
    output = model(data)
    loss = criterion(output, target)

loss.backward()
optimizer.step()
```

## 优化器选择指南

### SGD
- **优点**：简单、内存效率高、泛化性能好
- **缺点**：收敛慢、对学习率敏感
- **适用场景**：
  - 计算机视觉任务（ResNet、VGG等）
  - 内存受限环境
  - 需要最佳泛化性能时
- **推荐设置**：`lr=0.1, momentum=0.9, weight_decay=1e-4`

### Adam
- **优点**：收敛快、对学习率不敏感、适应性强
- **缺点**：内存占用大（需要存储一阶和二阶矩）、可能过拟合
- **适用场景**：
  - NLP任务
  - 快速原型开发
  - 稀疏梯度
- **推荐设置**：`lr=1e-3, betas=(0.9, 0.999)`

### AdamW
- **优点**：更好的泛化性能、适合大模型
- **缺点**：内存占用大
- **适用场景**：
  - Transformer模型（BERT、GPT等）
  - 大规模预训练
  - 需要强正则化时
- **推荐设置**：`lr=5e-5, weight_decay=0.01`

## 性能优化提示

1. **梯度累积**：小批量时累积多步梯度再更新
2. **梯度裁剪**：防止梯度爆炸，特别是RNN/Transformer
3. **参数组**：不同层使用不同学习率
4. **权重衰减**：AdamW通常比Adam+L2正则化效果好
5. **学习率预热**：大批量训练时使用warmup

## 注意事项

- 优化器状态会占用额外内存（Adam/AdamW是参数的2倍）
- 切换设备时需要将优化器状态也移到新设备
- 保存检查点时记得保存优化器状态
- 不同优化器的学习率范围差异很大
- 梯度裁剪应在optimizer.step()之前进行