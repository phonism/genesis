# 混合精度训练指南

混合精度训练是一种在训练过程中同时使用16位（半精度）和32位（单精度）浮点数的技术，用于减少内存使用并加速训练，同时保持模型精度。Genesis提供了全面的混合精度训练支持和自动混合精度（AMP）功能。

## 概述

### 混合精度训练的优势

- **内存效率**：减少约50%的内存使用
- **速度提升**：在带有Tensor Cores的现代GPU上训练更快
- **模型精度**：通过自动损失缩放保持训练稳定性
- **更大模型**：在同样硬件上训练更大的模型

### 支持的精度类型

Genesis支持多种精度格式：

- **float32 (FP32)**：标准单精度（默认）
- **float16 (FP16)**：IEEE半精度
- **bfloat16 (BF16)**：具有更大动态范围的Brain Float格式

## 数据类型系统

### 理解Genesis数据类型

```python
import genesis

# 可用的精度类型
print("可用的数据类型：")
print(f"FP32: {genesis.float32}")  # 标准精度
print(f"FP16: {genesis.float16}")  # 半精度
print(f"BF16: {genesis.bfloat16}") # Brain Float

# 检查数据类型属性
dtype = genesis.float16
print(f"名称: {dtype.name}")
print(f"大小: {dtype.itemsize} 字节")
print(f"是否浮点: {dtype.is_floating_point}")
print(f"NumPy类型: {dtype.numpy_dtype}")
```

### 创建混合精度张量

```python
import genesis

# 创建不同精度的张量
fp32_tensor = genesis.randn(1000, 1000, dtype=genesis.float32)
fp16_tensor = genesis.randn(1000, 1000, dtype=genesis.float16) 
bf16_tensor = genesis.randn(1000, 1000, dtype=genesis.bfloat16)

print(f"FP32内存: {fp32_tensor.numel() * 4} 字节")
print(f"FP16内存: {fp16_tensor.numel() * 2} 字节") 
print(f"BF16内存: {bf16_tensor.numel() * 2} 字节")

# 类型转换
fp16_from_fp32 = fp32_tensor.half()    # 转换为FP16
fp32_from_fp16 = fp16_tensor.float()   # 转换为FP32
```

## 自动混合精度（AMP）

### 基础AMP使用

Genesis通过`autocast`上下文和启用标志提供自动混合精度：

```python
import genesis
import genesis.nn as nn

# 全局启用自动混合精度
genesis.enable_autocast = True

# 创建模型和数据
model = nn.Linear(784, 10).cuda()
x = genesis.randn(32, 784, device='cuda')
labels = genesis.randint(0, 10, (32,), device='cuda')

# 使用自动类型转换的前向传播
outputs = model(x)  # 自动使用混合精度

# 损失计算（通常在FP32中进行）
criterion = nn.CrossEntropyLoss()
loss = criterion(outputs, labels)

print(f"输入数据类型: {x.dtype}")
print(f"输出数据类型: {outputs.dtype}")
print(f"损失数据类型: {loss.dtype}")
```

### 手动AMP控制

对于精细控制，使用`autocast`上下文管理器：

```python
import genesis

# 禁用全局autocast
genesis.enable_autocast = False

# 模型设置
model = nn.Sequential(
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Linear(256, 10)
).cuda()

x = genesis.randn(32, 784, device='cuda')

# 手动混合精度控制
with genesis.autocast():
    # 此块内的操作使用FP16/BF16
    hidden = model[0](x)  # Linear层使用FP16
    activated = model[1](hidden)  # ReLU使用FP16
    
# 块外操作使用默认精度
outputs = model[2](activated)  # 这将是FP32

print(f"隐藏层数据类型: {hidden.dtype}")
print(f"激活层数据类型: {activated.dtype}")
print(f"输出数据类型: {outputs.dtype}")
```

## 混合精度训练

### 简单混合精度训练循环

```python
import genesis
import genesis.nn as nn
import genesis.optim as optim

# 模型设置
model = nn.Sequential(
    nn.Linear(784, 512),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(512, 256),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(256, 10)
).cuda()

# 优化器
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 损失函数
criterion = nn.CrossEntropyLoss()

# 启用混合精度
genesis.enable_autocast = True

def train_epoch_amp(model, dataloader, optimizer, criterion):
    model.train()
    total_loss = 0.0
    
    for batch_idx, (data, targets) in enumerate(dataloader):
        data = data.cuda()
        targets = targets.cuda()
        
        # 清零梯度
        optimizer.zero_grad()
        
        # 使用混合精度的前向传播
        outputs = model(data)
        loss = criterion(outputs, targets)
        
        # 反向传播
        loss.backward()
        
        # 梯度裁剪（对稳定性很重要）
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # 优化器步骤
        optimizer.step()
        
        total_loss += loss.item()
        
        if batch_idx % 100 == 0:
            print(f'批次 {batch_idx}: loss={loss.item():.4f}')
    
    return total_loss / len(dataloader)

# 训练
for epoch in range(10):
    avg_loss = train_epoch_amp(model, train_loader, optimizer, criterion)
    print(f'轮次 {epoch}: 平均损失 = {avg_loss:.4f}')
```

### 带损失缩放的高级混合精度

为了训练稳定性，特别是使用FP16时，建议使用损失缩放：

```python
class GradScaler:
    """用于混合精度训练的梯度缩放器。"""
    
    def __init__(self, init_scale=2**16, growth_factor=2.0, backoff_factor=0.5, 
                 growth_interval=2000):
        self.scale = init_scale
        self.growth_factor = growth_factor
        self.backoff_factor = backoff_factor
        self.growth_interval = growth_interval
        self._growth_tracker = 0
    
    def scale_loss(self, loss):
        """缩放损失以防止梯度下溢。"""
        return loss * self.scale
    
    def unscale_gradients(self, optimizer):
        """在优化器步骤前反缩放梯度。"""
        for param_group in optimizer.param_groups:
            for param in param_group['params']:
                if param.grad is not None:
                    param.grad.data.div_(self.scale)
    
    def step(self, optimizer):
        """带梯度溢出检测的优化器步骤。"""
        # 检查梯度溢出
        has_overflow = self._check_overflow(optimizer)
        
        if has_overflow:
            # 跳过优化器步骤并减少缩放
            self.scale *= self.backoff_factor
            self.scale = max(self.scale, 1.0)
            self._growth_tracker = 0
            return False
        else:
            # 正常优化器步骤
            optimizer.step()
            
            # 定期增加缩放
            self._growth_tracker += 1
            if self._growth_tracker >= self.growth_interval:
                self.scale *= self.growth_factor
                self._growth_tracker = 0
            
            return True
    
    def _check_overflow(self, optimizer):
        """检查是否有梯度溢出。"""
        for param_group in optimizer.param_groups:
            for param in param_group['params']:
                if param.grad is not None:
                    if genesis.isnan(param.grad).any() or genesis.isinf(param.grad).any():
                        return True
        return False

# 带梯度缩放的训练
scaler = GradScaler()

def train_with_scaling(model, dataloader, optimizer, criterion, scaler):
    model.train()
    total_loss = 0.0
    successful_steps = 0
    
    for batch_idx, (data, targets) in enumerate(dataloader):
        data = data.cuda()
        targets = targets.cuda()
        
        optimizer.zero_grad()
        
        # 使用混合精度的前向传播
        with genesis.autocast():
            outputs = model(data)
            loss = criterion(outputs, targets)
        
        # 缩放损失以防止梯度下溢
        scaled_loss = scaler.scale_loss(loss)
        scaled_loss.backward()
        
        # 反缩放梯度并检查溢出
        scaler.unscale_gradients(optimizer)
        
        # 在反缩放梯度上进行梯度裁剪
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # 带溢出检测的优化器步骤
        if scaler.step(optimizer):
            successful_steps += 1
        
        total_loss += loss.item()
        
        if batch_idx % 100 == 0:
            print(f'批次 {batch_idx}: loss={loss.item():.4f}, scale={scaler.scale:.0f}')
    
    success_rate = successful_steps / len(dataloader)
    print(f'训练成功率: {success_rate:.1%}')
    
    return total_loss / len(dataloader)
```

## 精度特定考虑

### FP16（半精度）

```python
import genesis

# FP16特性
fp16_info = {
    'range': '±65,504',
    'precision': '约3-4位小数',
    'special_values': ['inf', '-inf', 'nan'],
    'benefits': ['在Tensor Cores上更快', '50%内存减少'],
    'challenges': ['有限的范围', '梯度下溢']
}

# FP16最佳实践
def create_fp16_model():
    model = nn.Sequential(
        nn.Linear(784, 256),
        nn.LayerNorm(256),  # LayerNorm在FP16下表现良好
        nn.ReLU(),
        nn.Linear(256, 10)
    )
    
    # 为FP16初始化适当的缩放
    for module in model.modules():
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight, gain=1.0)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    
    return model

# 监控FP16训练
def check_fp16_health(model):
    """检查FP16训练期间的模型健康状况。"""
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            param_norm = param.norm().item()
            
            print(f"{name}:")
            print(f"  参数范数: {param_norm:.2e}")
            print(f"  梯度范数: {grad_norm:.2e}")
            
            # 检查问题值
            if grad_norm < 1e-7:
                print(f"  警告: 检测到非常小的梯度!")
            if grad_norm > 1e4:
                print(f"  警告: 检测到非常大的梯度!")
```

### BF16（Brain Float）

```python
import genesis

# BF16优势
bf16_info = {
    'range': '与FP32相同 (±3.4×10^38)',
    'precision': '约2-3位小数', 
    'benefits': ['比FP16范围更大', '更稳定的训练'],
    'hardware': ['A100', 'H100', 'TPUs']
}

# BF16通常比FP16更稳定
def train_with_bf16():
    # 使用BF16创建模型
    model = nn.Linear(1000, 100).cuda()
    x = genesis.randn(32, 1000, dtype=genesis.bfloat16, device='cuda')
    
    # BF16前向传播
    output = model(x)
    print(f"输入: {x.dtype}, 输出: {output.dtype}")
    
    # BF16通常不需要损失缩放
    loss = output.sum()
    loss.backward()
    
    return model

# 比较精度
def compare_precisions():
    sizes = [100, 1000, 10000]
    
    for size in sizes:
        # 创建测试数据
        data_fp32 = genesis.randn(size, size)
        data_fp16 = data_fp32.half()
        data_bf16 = data_fp32.to(genesis.bfloat16)
        
        # 简单计算
        result_fp32 = genesis.matmul(data_fp32, data_fp32)
        result_fp16 = genesis.matmul(data_fp16, data_fp16)
        result_bf16 = genesis.matmul(data_bf16, data_bf16)
        
        # 比较精度
        error_fp16 = (result_fp32 - result_fp16.float()).abs().mean()
        error_bf16 = (result_fp32 - result_bf16.float()).abs().mean()
        
        print(f"大小 {size}x{size}:")
        print(f"  FP16误差: {error_fp16:.2e}")
        print(f"  BF16误差: {error_bf16:.2e}")
```

## 内存优化

### 内存使用分析

```python
import genesis

def analyze_memory_usage():
    """分析不同精度类型的内存使用。"""
    
    # 模型大小
    sizes = [(1000, 1000), (2000, 2000), (5000, 5000)]
    
    for h, w in sizes:
        print(f"\n张量大小: {h}x{w}")
        
        # 创建张量
        fp32_tensor = genesis.randn(h, w, dtype=genesis.float32, device='cuda')
        fp16_tensor = genesis.randn(h, w, dtype=genesis.float16, device='cuda')
        bf16_tensor = genesis.randn(h, w, dtype=genesis.bfloat16, device='cuda')
        
        # 内存使用
        fp32_memory = fp32_tensor.numel() * 4  # 每个float32 4字节
        fp16_memory = fp16_tensor.numel() * 2  # 每个float16 2字节
        bf16_memory = bf16_tensor.numel() * 2  # 每个bfloat16 2字节
        
        print(f"  FP32: {fp32_memory / 1e6:.1f} MB")
        print(f"  FP16: {fp16_memory / 1e6:.1f} MB ({fp16_memory/fp32_memory:.1%})")
        print(f"  BF16: {bf16_memory / 1e6:.1f} MB ({bf16_memory/fp32_memory:.1%})")
        
        # 清理
        del fp32_tensor, fp16_tensor, bf16_tensor
        genesis.cuda.empty_cache()

analyze_memory_usage()
```

### 梯度检查点与混合精度

```python
import genesis
import genesis.nn as nn

class CheckpointedModule(nn.Module):
    """支持梯度检查点的模块。"""
    
    def __init__(self, layers):
        super().__init__()
        self.layers = nn.ModuleList(layers)
        self.checkpoint = True
    
    def forward(self, x):
        def run_layers(x, layers):
            for layer in layers:
                x = layer(x)
            return x
        
        if self.training and self.checkpoint:
            # 使用梯度检查点节省内存
            return genesis.utils.checkpoint(run_layers, x, self.layers)
        else:
            return run_layers(x, self.layers)

# 创建内存高效模型
def create_checkpointed_model():
    layers = [
        nn.Linear(1024, 1024),
        nn.ReLU(),
        nn.Linear(1024, 1024),
        nn.ReLU(),
        nn.Linear(1024, 512),
        nn.ReLU(),
        nn.Linear(512, 10)
    ]
    
    return CheckpointedModule(layers)

# 使用检查点和混合精度进行训练
def train_memory_efficient():
    model = create_checkpointed_model().cuda()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 启用混合精度
    genesis.enable_autocast = True
    
    for epoch in range(10):
        for batch in dataloader:
            data, targets = batch
            data = data.cuda()
            targets = targets.cuda()
            
            optimizer.zero_grad()
            
            # 使用检查点和混合精度的前向传播
            outputs = model(data)
            loss = nn.CrossEntropyLoss()(outputs, targets)
            
            # 反向传播
            loss.backward()
            optimizer.step()
        
        print(f"轮次 {epoch} 完成")
```

## 性能基准测试

### 混合精度性能比较

```python
import genesis
import time

def benchmark_precision_performance():
    """基准测试不同精度格式。"""
    
    # 模型设置
    sizes = [512, 1024, 2048]
    batch_sizes = [16, 32, 64]
    
    results = {}
    
    for size in sizes:
        for batch_size in batch_sizes:
            print(f"\n基准测试: size={size}, batch_size={batch_size}")
            
            # 创建模型
            model_fp32 = nn.Linear(size, size).cuda()
            model_fp16 = nn.Linear(size, size).cuda().half()
            
            # 创建数据
            data_fp32 = genesis.randn(batch_size, size, device='cuda')
            data_fp16 = data_fp32.half()
            
            # 基准测试FP32
            torch.cuda.synchronize()
            start_time = time.time()
            
            for _ in range(100):
                output_fp32 = model_fp32(data_fp32)
            
            torch.cuda.synchronize()
            fp32_time = time.time() - start_time
            
            # 基准测试FP16
            torch.cuda.synchronize()
            start_time = time.time()
            
            for _ in range(100):
                output_fp16 = model_fp16(data_fp16)
            
            torch.cuda.synchronize()
            fp16_time = time.time() - start_time
            
            # 结果
            speedup = fp32_time / fp16_time
            print(f"  FP32时间: {fp32_time:.3f}s")
            print(f"  FP16时间: {fp16_time:.3f}s") 
            print(f"  加速比: {speedup:.2f}x")
            
            results[(size, batch_size)] = {
                'fp32_time': fp32_time,
                'fp16_time': fp16_time,
                'speedup': speedup
            }
    
    return results

# 运行基准测试
benchmark_results = benchmark_precision_performance()
```

## 最佳实践和故障排除

### 最佳实践

1. **从简单开始**：在手动控制之前先尝试自动混合精度
2. **监控训练**：关注梯度下溢/溢出
3. **使用损失缩放**：对FP16稳定性至关重要
4. **梯度裁剪**：有助于防止梯度爆炸
5. **分层精度**：某些层可能需要FP32（如批标准化）

### 常见问题和解决方案

```python
# 问题1: 梯度下溢
def handle_gradient_underflow():
    """处理FP16训练中的梯度下溢。"""
    
    # 解决方案1: 使用损失缩放
    scaler = GradScaler(init_scale=2**16)
    
    # 解决方案2: 跳过有问题的批次
    def safe_backward(loss, scaler):
        scaled_loss = scaler.scale_loss(loss)
        scaled_loss.backward()
        
        # 在优化器步骤前检查问题
        has_inf_or_nan = any(
            genesis.isinf(p.grad).any() or genesis.isnan(p.grad).any()
            for p in model.parameters() 
            if p.grad is not None
        )
        
        if has_inf_or_nan:
            print("由于inf/nan梯度跳过步骤")
            optimizer.zero_grad()
            return False
        
        return True

# 问题2: 模型发散
def prevent_model_divergence():
    """防止混合精度中的模型发散。"""
    
    # 解决方案1: 降低学习率
    optimizer = optim.Adam(model.parameters(), lr=0.0001)  # 更低的学习率
    
    # 解决方案2: 预热计划
    scheduler = optim.get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=1000, num_training_steps=10000
    )
    
    # 解决方案3: 密切监控损失
    def check_loss_stability(loss, loss_history):
        loss_history.append(loss.item())
        
        if len(loss_history) > 100:
            recent_losses = loss_history[-50:]
            if any(l > 10 * min(recent_losses) for l in recent_losses):
                print("警告: 检测到损失不稳定!")
                return False
        
        return True

# 问题3: 精度降低
def maintain_accuracy():
    """使用混合精度保持模型精度。"""
    
    # 解决方案1: 使用BF16而不是FP16
    genesis.enable_autocast = True
    default_dtype = genesis.bfloat16
    
    # 解决方案2: 保持关键层在FP32
    class MixedPrecisionModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.features = nn.Sequential(
                nn.Linear(784, 256),  # FP16/BF16
                nn.ReLU(),
                nn.Linear(256, 128),  # FP16/BF16
                nn.ReLU()
            )
            
            # 保持输出层在FP32以获得稳定性
            self.classifier = nn.Linear(128, 10).float()
        
        def forward(self, x):
            with genesis.autocast():
                features = self.features(x)
            
            # 输出层在FP32
            output = self.classifier(features.float())
            return output
```

### 调试混合精度训练

```python
def debug_mixed_precision():
    """调试混合精度训练问题。"""
    
    # 1. 检查整个模型的张量数据类型
    def print_tensor_info(tensor, name):
        print(f"{name}:")
        print(f"  形状: {tensor.shape}")
        print(f"  数据类型: {tensor.dtype}")
        print(f"  设备: {tensor.device}")
        print(f"  需要梯度: {tensor.requires_grad}")
        print(f"  最小/最大值: {tensor.min():.2e} / {tensor.max():.2e}")
        print()
    
    # 2. 监控梯度范数
    def check_gradient_norms(model):
        total_norm = 0.0
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                total_norm += grad_norm ** 2
                print(f"{name}: grad_norm = {grad_norm:.2e}")
        
        total_norm = total_norm ** 0.5
        print(f"总梯度范数: {total_norm:.2e}")
        return total_norm
    
    # 3. 验证数值稳定性
    def check_numerical_stability(tensor):
        """检查数值问题。"""
        has_nan = genesis.isnan(tensor).any()
        has_inf = genesis.isinf(tensor).any()
        
        if has_nan:
            print("警告: 检测到NaN值!")
        if has_inf:
            print("警告: 检测到Inf值!")
        
        return not (has_nan or has_inf)

# 在训练循环中使用
for epoch in range(num_epochs):
    for batch_idx, (data, targets) in enumerate(dataloader):
        # 前向传播
        outputs = model(data)
        loss = criterion(outputs, targets)
        
        # 调试信息
        if batch_idx % 100 == 0:
            print(f"轮次 {epoch}, 批次 {batch_idx}:")
            print_tensor_info(data, "输入")
            print_tensor_info(outputs, "输出") 
            print_tensor_info(loss, "损失")
            
            # 反向传播后检查梯度
            loss.backward()
            grad_norm = check_gradient_norms(model)
            
            if grad_norm > 10.0:
                print("警告: 检测到大梯度范数!")
```

这份全面指南涵盖了Genesis中混合精度训练的所有方面，从基础使用到高级优化技术和故障排除策略。