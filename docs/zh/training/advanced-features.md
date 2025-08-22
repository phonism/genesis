# 高级训练特性

Genesis提供了多个高级特性来提升训练效率和模型性能。

## 🚀 混合精度训练 (AMP)

自动混合精度（AMP）允许你在适当的地方使用FP16/BF16计算来更快地训练模型，同时降低内存使用，并通过维持FP32主权重来保持数值稳定性。

### 基本用法

```python
import genesis
import genesis.nn as nn
import genesis.optim as optim
from genesis.amp import autocast, GradScaler

# 创建模型和优化器
model = nn.Linear(1024, 512)
optimizer = optim.Adam(model.parameters())

# 为混合精度初始化梯度缩放器
scaler = GradScaler()

# 使用AMP的训练循环
for data, target in dataloader:
    optimizer.zero_grad()
    
    # 使用autocast进行自动混合精度
    with autocast():
        output = model(data)
        loss = criterion(output, target)
    
    # 缩放损失并进行反向传播
    scaler.scale(loss).backward()
    
    # 反缩放并执行优化器步骤
    scaler.step(optimizer)
    scaler.update()
```

### 支持的数据类型

Genesis支持多种精度格式：

- **float16 (FP16)**: 半精度，在大多数GPU上最快
- **bfloat16 (BF16)**: 脑浮点数，比FP16有更好的数值范围
- **float32 (FP32)**: 单精度，主权重的默认类型

### 优势

- **速度**: 在现代GPU上训练速度提升高达2倍
- **内存**: 减少内存使用，允许更大的批次大小
- **精度**: 通过损失缩放保持模型精度

## ✂️ 梯度裁剪

梯度裁剪有助于防止深度网络中的梯度爆炸，提高训练稳定性，特别是对于RNN和Transformer。

### 梯度范数裁剪

当梯度的L2范数超过阈值时进行裁剪：

```python
import genesis.nn.utils as nn_utils

# 训练过程中
loss.backward()

# 按范数裁剪梯度（大多数情况推荐）
nn_utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

optimizer.step()
```

### 梯度值裁剪

将梯度值裁剪到特定范围：

```python
# 按值裁剪梯度
nn_utils.clip_grad_value_(model.parameters(), clip_value=0.5)
```

### 何时使用

- **梯度范数裁剪**: 推荐用于RNN、LSTM和Transformer
- **梯度值裁剪**: 当需要对梯度值进行硬限制时有用
- **典型值**: 大多数模型的max_norm在0.5到5.0之间

## 📈 学习率调度器

学习率调度器在训练过程中调整学习率，以改善收敛性和最终模型性能。

### StepLR

每step_size个epoch将学习率衰减gamma倍：

```python
from genesis.optim.lr_scheduler import StepLR

optimizer = optim.Adam(model.parameters(), lr=0.1)
scheduler = StepLR(optimizer, step_size=30, gamma=0.1)

for epoch in range(100):
    train(...)
    scheduler.step()  # 每30个epoch衰减学习率
```

### ExponentialLR

指数衰减学习率：

```python
from genesis.optim.lr_scheduler import ExponentialLR

scheduler = ExponentialLR(optimizer, gamma=0.95)

for epoch in range(100):
    train(...)
    scheduler.step()  # 每个epoch学习率 = 学习率 * 0.95
```

### CosineAnnealingLR

使用余弦退火调度：

```python
from genesis.optim.lr_scheduler import CosineAnnealingLR

# T_max: 最大迭代次数
scheduler = CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-6)

for epoch in range(100):
    train(...)
    scheduler.step()
```

### 自定义学习率调度

你也可以实现自定义调度：

```python
def custom_lr_lambda(epoch):
    # 前10个epoch预热，然后衰减
    if epoch < 10:
        return epoch / 10
    else:
        return 0.95 ** (epoch - 10)

scheduler = LambdaLR(optimizer, lr_lambda=custom_lr_lambda)
```

## 💾 检查点

在训练过程中保存和恢复模型状态，以实现容错和模型部署。

### 保存检查点

```python
import genesis

# 保存模型状态
genesis.save_checkpoint({
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': loss,
    'best_accuracy': best_acc
}, 'checkpoint_epoch_10.pth')
```

### 加载检查点

```python
# 加载检查点
checkpoint = genesis.load_checkpoint('checkpoint_epoch_10.pth')

# 恢复模型和优化器状态
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']
loss = checkpoint['loss']
```

### 最佳实践

1. **定期保存**: 每N个epoch保存检查点
2. **最佳模型跟踪**: 保留性能最好的模型
3. **元数据存储**: 包含训练配置和指标

```python
# 示例：在训练过程中保存最佳模型
best_loss = float('inf')

for epoch in range(num_epochs):
    val_loss = validate(model, val_loader)
    
    if val_loss < best_loss:
        best_loss = val_loss
        genesis.save_checkpoint({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'best_loss': best_loss
        }, 'best_model.pth')
```

## 🔧 完整训练示例

以下是结合所有高级特性的完整示例：

```python
import genesis
import genesis.nn as nn
import genesis.optim as optim
from genesis.amp import autocast, GradScaler
from genesis.optim.lr_scheduler import CosineAnnealingLR
import genesis.nn.utils as nn_utils

# 模型设置
model = YourModel()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = CosineAnnealingLR(optimizer, T_max=100)
scaler = GradScaler()

# 训练配置
max_grad_norm = 1.0
checkpoint_interval = 10

for epoch in range(num_epochs):
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        
        # 混合精度前向传播
        with autocast():
            output = model(data)
            loss = criterion(output, target)
        
        # 缩放的反向传播
        scaler.scale(loss).backward()
        
        # 梯度裁剪
        scaler.unscale_(optimizer)
        nn_utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        
        # 带缩放的优化器步骤
        scaler.step(optimizer)
        scaler.update()
    
    # 更新学习率
    scheduler.step()
    
    # 保存检查点
    if epoch % checkpoint_interval == 0:
        genesis.save_checkpoint({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'scaler_state_dict': scaler.state_dict(),
        }, f'checkpoint_epoch_{epoch}.pth')
```

## 📊 性能提示

### 内存优化
- 使用梯度累积获得更大的有效批次大小
- 为非常深的模型启用梯度检查点
- 使用混合精度训练减少内存使用

### 速度优化
- 使用适当的数据类型（FP16用于速度，BF16用于稳定性）
- 调整梯度累积步数
- 分析训练循环以识别瓶颈

### 收敛技巧
- 从学习率查找器开始识别最优学习率
- 对大批次训练使用预热
- 监控梯度范数以早期检测不稳定性

## 🔗 相关主题

- [基础训练教程](../tutorials/basic-training.md)
- [性能调优指南](../tutorials/performance-tuning.md)
- [模型架构指南](../core-components/index.md)
- [优化器文档](../api/optim/optimizers.md)