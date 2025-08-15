# 学习率调度器

Genesis提供了学习率调度器来调整训练过程中的学习率，这对于在深度学习模型中实现最佳收敛至关重要。

## 概述

学习率调度是一种在整个训练过程中调整学习率的技术。Genesis提供了与PyTorch兼容的学习率调度器，可以显著改善模型收敛。

## 可用调度器

### LambdaLR

`LambdaLR`调度器允许你定义自定义函数来修改每个轮次的学习率。

```python
import genesis.optim as optim

class LambdaLR:
    def __init__(self, optimizer, lr_lambda, last_epoch=-1, verbose=False):
        """
        通过lr_lambda函数给出的因子乘以学习率。
        
        Args:
            optimizer: 包装的优化器
            lr_lambda: 计算乘法因子的函数或函数列表
            last_epoch: 最后一个轮次的索引
            verbose: 如果为True，为每次更新打印消息
        """
```

**使用示例：**
```python
import genesis
import genesis.nn as nn
import genesis.optim as optim

# 创建模型和优化器
model = nn.Linear(10, 1)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 定义学习率调度函数
def lr_lambda(epoch):
    # 每10个轮次将学习率衰减0.95倍
    return 0.95 ** (epoch // 10)

# 创建调度器
scheduler = optim.LambdaLR(optimizer, lr_lambda=lr_lambda)

# 训练循环
for epoch in range(100):
    # 这里是训练代码
    loss = train_one_epoch(model, dataloader, optimizer)
    
    # 调度器步进
    scheduler.step()
    print(f"轮次 {epoch}: lr={scheduler.get_last_lr()}")
```

### 余弦退火与预热

`get_cosine_schedule_with_warmup`函数创建一个带有线性预热的余弦退火调度。

```python
def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps):
    """
    创建具有线性预热和余弦衰减的调度。
    
    Args:
        optimizer: 包装的优化器
        num_warmup_steps: 预热阶段的步数
        num_training_steps: 总训练步数
        
    Returns:
        LambdaLR调度器对象
    """
```

**使用示例：**
```python
import genesis.optim as optim

# 训练配置
num_epochs = 100
steps_per_epoch = 1000
total_steps = num_epochs * steps_per_epoch
warmup_steps = total_steps // 10  # 10%预热

# 创建优化器和调度器
optimizer = optim.AdamW(model.parameters(), lr=5e-4)
scheduler = optim.get_cosine_schedule_with_warmup(
    optimizer, 
    num_warmup_steps=warmup_steps,
    num_training_steps=total_steps
)

# 逐步调度的训练循环
step = 0
for epoch in range(num_epochs):
    for batch in dataloader:
        # 前向传播和优化
        loss = model(batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # 每个批次调度器步进
        scheduler.step()
        step += 1
        
        if step % 100 == 0:
            print(f"步骤 {step}: lr={scheduler.get_last_lr():.6f}")
```

## 调度器方法

所有调度器提供以下方法：

### step()
```python
def step(self, epoch=None):
    """
    根据调度更新学习率。
    
    Args:
        epoch: 当前轮次（可选，如果为None使用内部计数器）
    """
```

### get_last_lr()
```python
def get_last_lr(self):
    """
    返回最后计算的学习率。
    
    Returns:
        当前学习率值
    """
```

### state_dict()
```python
def state_dict(self):
    """
    将调度器的状态作为字典返回。
    
    Returns:
        包含调度器状态的字典
    """
```

### load_state_dict()
```python
def load_state_dict(self, state_dict):
    """
    从字典加载调度器状态。
    
    Args:
        state_dict: 包含调度器状态的字典
    """
```

## 常见模式

### 指数衰减
```python
# 每个轮次学习率衰减0.95
scheduler = optim.LambdaLR(optimizer, lr_lambda=lambda epoch: 0.95 ** epoch)
```

### 阶梯衰减
```python
# 每30个轮次将学习率减半
def step_decay(epoch):
    return 0.5 ** (epoch // 30)

scheduler = optim.LambdaLR(optimizer, lr_lambda=step_decay)
```

### 多项式衰减
```python
# 多项式衰减到零
def poly_decay(epoch, total_epochs=100, power=0.9):
    return (1 - epoch / total_epochs) ** power

scheduler = optim.LambdaLR(optimizer, lr_lambda=lambda epoch: poly_decay(epoch))
```

### 余弦重启
```python
import math

def cosine_restart(epoch, restart_period=50):
    epoch_in_cycle = epoch % restart_period
    return 0.5 * (1 + math.cos(math.pi * epoch_in_cycle / restart_period))

scheduler = optim.LambdaLR(optimizer, lr_lambda=cosine_restart)
```

## 与训练集成

### 基础训练循环
```python
import genesis
import genesis.nn as nn
import genesis.optim as optim

# 设置
model = nn.Sequential(
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Linear(256, 10)
)

optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.get_cosine_schedule_with_warmup(
    optimizer, num_warmup_steps=1000, num_training_steps=10000
)

# 训练循环
for epoch in range(num_epochs):
    model.train()
    for batch_idx, (data, targets) in enumerate(train_loader):
        # 前向传播
        outputs = model(data)
        loss = nn.CrossEntropyLoss()(outputs, targets)
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()  # 更新学习率
        
        # 日志记录
        if batch_idx % 100 == 0:
            current_lr = scheduler.get_last_lr()
            print(f'轮次: {epoch}, 批次: {batch_idx}, LR: {current_lr:.6f}, 损失: {loss.item():.4f}')
```

### 检查点集成
```python
import genesis

# 保存调度器状态与模型检查点
checkpoint = {
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'scheduler_state_dict': scheduler.state_dict(),
    'epoch': epoch,
    'loss': loss
}
genesis.save_checkpoint(checkpoint, 'checkpoint.pth')

# 加载调度器状态
checkpoint = genesis.load_checkpoint('checkpoint.pth')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
```

## 最佳实践

1. **选择正确的调度**：
   - 大多数应用使用余弦退火
   - 为transformer模型添加预热
   - 微调时使用阶梯衰减

2. **预热阶段**：
   - 对大批量大小至关重要
   - 建议用于transformer架构
   - 通常为总训练步数的5-10%

3. **监控**：
   - 记录学习率值
   - 绘制学习率调度
   - 训练期间监控验证损失

4. **检查点保存**：
   - 始终保存调度器状态
   - 以正确的学习率恢复训练
   - 对长时间训练运行至关重要

## 示例

### Transformer训练调度
```python
# 典型的transformer训练调度
def get_transformer_schedule(optimizer, d_model=512, warmup_steps=4000):
    def lr_lambda(step):
        if step == 0:
            return 0
        return min(step ** -0.5, step * warmup_steps ** -1.5) * (d_model ** -0.5)
    
    return optim.LambdaLR(optimizer, lr_lambda=lr_lambda)

scheduler = get_transformer_schedule(optimizer, d_model=512, warmup_steps=4000)
```

### 学习率范围测试
```python
# 找到最佳学习率范围
def lr_range_test(model, optimizer, start_lr=1e-7, end_lr=10, num_it=100):
    lrs = []
    losses = []
    
    lr_lambda = lambda step: (end_lr / start_lr) ** (step / num_it)
    scheduler = optim.LambdaLR(optimizer, lr_lambda=lr_lambda)
    
    for i in range(num_it):
        # 训练步骤
        loss = train_step(model, batch)
        losses.append(loss)
        lrs.append(scheduler.get_last_lr())
        
        scheduler.step()
        
        if loss > 4 * min(losses):  # 如果损失爆炸则停止
            break
    
    return lrs, losses
```

## 从PyTorch迁移

Genesis学习率调度器设计为PyTorch调度器的直接替代：

```python
# PyTorch代码
import torch.optim as optim
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

# Genesis等效代码
import genesis.optim as optim
scheduler = optim.LambdaLR(
    optimizer, 
    lr_lambda=lambda epoch: 0.5 * (1 + math.cos(math.pi * epoch / 100))
)
```

API兼容，使得将现有PyTorch训练脚本迁移到Genesis变得容易。