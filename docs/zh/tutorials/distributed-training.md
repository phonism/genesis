# Genesis 分布式训练

学习如何使用 Genesis 进行多GPU和多节点分布式训练。

## 概述

Genesis 提供完整的分布式训练支持，包括：

- **NCCL后端** - 高性能GPU间通信
- **DistributedDataParallel (DDP)** - 数据并行训练包装器
- **集体通信操作** - all_reduce, broadcast, all_gather等
- **单进程测试** - 方便开发和调试

## 快速开始

### 1. 基本分布式训练设置

```python
import genesis
import genesis.distributed as dist
import genesis.nn as nn

# 初始化分布式进程组
dist.init_process_group(backend='nccl', world_size=2, rank=0)  # rank根据进程调整

# 创建模型
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(512, 256)
        self.relu = nn.ReLU()
        self.output = nn.Linear(256, 10)
    
    def forward(self, x):
        x = self.relu(self.linear(x))
        return self.output(x)

model = MyModel()

# 包装为分布式数据并行模型
device = genesis.device('cuda')
ddp_model = dist.DistributedDataParallel(model, device_ids=[device.index])
```

### 2. 分布式训练循环

```python
# 优化器和损失函数
optimizer = genesis.optim.Adam(ddp_model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# 训练循环
ddp_model.train()
for epoch in range(num_epochs):
    for batch_idx, (data, targets) in enumerate(dataloader):
        # 数据移动到GPU
        data = data.to(device)
        targets = targets.to(device)
        
        # 前向传播
        outputs = ddp_model(data)
        loss = criterion(outputs, targets)
        
        # 反向传播（梯度会自动同步）
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if batch_idx % 100 == 0:
            print(f'Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}')
```

### 3. 进程组管理

```python
# 检查分布式状态
if dist.is_initialized():
    print(f"进程组已初始化")
    print(f"世界大小: {dist.get_world_size()}")
    print(f"当前rank: {dist.get_rank()}")

# 同步所有进程
dist.barrier()

# 清理
dist.destroy_process_group()
```

## 高级功能

### 集体通信操作

```python
import genesis

# 创建测试张量
device = genesis.device('cuda')
tensor = genesis.ones([4], dtype=genesis.float32, device=device)

# all_reduce - 所有进程聚合
dist.all_reduce(tensor, dist.ReduceOp.SUM)  # 求和
dist.all_reduce(tensor, dist.ReduceOp.MAX)  # 最大值
dist.all_reduce(tensor, dist.ReduceOp.MIN)  # 最小值

# broadcast - 广播操作
broadcast_tensor = genesis.randn([8], device=device)
dist.broadcast(broadcast_tensor, src=0)  # 从rank 0广播

# all_gather - 收集所有数据
input_tensor = genesis.randn([4, 8], device=device)
output_list = [genesis.zeros_like(input_tensor) for _ in range(dist.get_world_size())]
dist.all_gather(output_list, input_tensor)
```

### 单进程测试模式

```python
# 用于开发和调试的单进程模式
def test_single_process():
    # 初始化单进程分布式环境
    dist.init_process_group(backend="nccl", world_size=1, rank=0)
    
    # 创建和测试模型
    model = MyModel()
    ddp_model = dist.DistributedDataParallel(model, device_ids=[0])
    
    # 测试前向传播
    input_data = genesis.randn([8, 512], device='cuda')
    output = ddp_model(input_data)
    
    # 测试反向传播
    loss = output.sum()
    loss.backward()
    
    print("单进程分布式测试成功！")
    dist.destroy_process_group()

# 运行测试
if __name__ == "__main__":
    test_single_process()
```

## 多GPU训练脚本

### launcher.py

```python
#!/usr/bin/env python3
"""
多GPU训练启动脚本
使用方法: python launcher.py --gpus 2
"""

import argparse
import subprocess
import sys
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpus', type=int, default=1, help='GPU数量')
    parser.add_argument('--script', type=str, default='train.py', help='训练脚本')
    args = parser.parse_args()
    
    processes = []
    
    try:
        for rank in range(args.gpus):
            env = os.environ.copy()
            env['CUDA_VISIBLE_DEVICES'] = str(rank)
            env['RANK'] = str(rank)
            env['WORLD_SIZE'] = str(args.gpus)
            
            cmd = [sys.executable, args.script]
            proc = subprocess.Popen(cmd, env=env)
            processes.append(proc)
            
        # 等待所有进程完成
        for proc in processes:
            proc.wait()
            
    except KeyboardInterrupt:
        print("停止训练...")
        for proc in processes:
            proc.terminate()

if __name__ == "__main__":
    main()
```

### train.py

```python
#!/usr/bin/env python3
"""
分布式训练主脚本
"""

import os
import genesis
import genesis.distributed as dist
import genesis.nn as nn

def main():
    # 从环境变量获取分布式参数
    rank = int(os.environ.get('RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    
    # 初始化分布式训练
    dist.init_process_group(
        backend='nccl',
        world_size=world_size,
        rank=rank
    )
    
    print(f"进程 {rank}/{world_size} 启动")
    
    try:
        # 创建模型
        model = create_model()
        ddp_model = dist.DistributedDataParallel(
            model, 
            device_ids=[genesis.cuda.current_device()]
        )
        
        # 创建优化器
        optimizer = genesis.optim.Adam(ddp_model.parameters(), lr=0.001)
        
        # 训练循环
        train_loop(ddp_model, optimizer, rank)
        
    finally:
        # 清理分布式环境
        dist.destroy_process_group()

def create_model():
    """创建模型"""
    return nn.Sequential([
        nn.Linear(784, 512),
        nn.ReLU(),
        nn.Linear(512, 256), 
        nn.ReLU(),
        nn.Linear(256, 10)
    ])

def train_loop(model, optimizer, rank):
    """训练循环"""
    model.train()
    
    for epoch in range(10):
        # 模拟训练数据
        data = genesis.randn([32, 784], device='cuda')
        targets = genesis.randint(0, 10, [32], device='cuda')
        
        # 前向传播
        outputs = model(data)
        loss = nn.functional.cross_entropy(outputs, targets)
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if rank == 0:  # 只在主进程打印
            print(f'Epoch {epoch}, Loss: {loss.item():.4f}')

if __name__ == "__main__":
    main()
```

## 错误处理和调试

### 常见问题

1. **NCCL不可用**
```python
try:
    dist.init_process_group(backend="nccl", world_size=1, rank=0)
except RuntimeError as e:
    if "NCCL library not available" in str(e):
        print("NCCL库不可用，请检查CUDA和NCCL安装")
    else:
        raise
```

2. **进程组未初始化**
```python
if not dist.is_initialized():
    print("错误：分布式进程组未初始化")
    print("请先调用 dist.init_process_group()")
```

3. **设备不匹配**
```python
# 确保模型和数据在相同设备上
device = genesis.device(f'cuda:{genesis.cuda.current_device()}')
model = model.to(device)
data = data.to(device)
```

## 性能优化建议

### 1. 梯度累积
```python
accumulation_steps = 4

for i, batch in enumerate(dataloader):
    outputs = ddp_model(batch['input'])
    loss = criterion(outputs, batch['target']) / accumulation_steps
    loss.backward()
    
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

### 2. 混合精度训练
```python
# 结合自动混合精度使用分布式训练
scaler = genesis.amp.GradScaler()

with genesis.amp.autocast():
    outputs = ddp_model(data)
    loss = criterion(outputs, targets)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

### 3. 通信优化
```python
# 创建DDP时启用梯度压缩
ddp_model = dist.DistributedDataParallel(
    model,
    device_ids=[device.index],
    find_unused_parameters=False,  # 提高性能
    gradient_as_bucket_view=True   # 减少内存使用
)
```

## See Also

- [Advanced Features](../training/advanced-features.md) - Advanced training techniques
- [Performance Tuning](performance-tuning.md) - Optimizing distributed performance