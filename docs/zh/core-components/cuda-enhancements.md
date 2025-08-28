# CUDA 增强功能

Genesis 在 CUDA 支持方面进行了重大改进，提供了更好的设备管理、内存操作和错误处理功能。

## 设备管理增强

### 新增的 CUDA 函数

#### `set_device(device)`
设置当前 CUDA 设备。

```python
import genesis.cuda as cuda

# 设置当前设备为 GPU 1
cuda.set_device(1)
print(f"当前设备: {cuda.current_device()}")
```

#### `device_count()`
获取可用 CUDA 设备数量。

```python
device_count = cuda.device_count()
print(f"可用GPU数量: {device_count}")
```

#### `get_device_name(device)`
获取指定设备的名称。

```python
device_name = cuda.get_device_name(0)
print(f"设备 0 名称: {device_name}")
```

#### `synchronize(device=None)`
同步 CUDA 操作，确保所有 CUDA 操作完成。

```python
# 同步当前设备
cuda.synchronize()

# 同步指定设备
cuda.synchronize(device=1)
```

## 设备属性增强

### Device 类新属性

```python
device = genesis.device('cuda:0')

# PyTorch 兼容的属性
print(f"设备类型: {device.type}")      # 'cuda'
print(f"设备索引: {device.index}")     # 0
```

这些属性提供了与 PyTorch 的完全兼容性，使迁移更加容易。

## 张量验证功能

### 数值有效性检查

Genesis 现在提供完整的张量数值有效性检查功能：

#### `isinf(tensor)`
检查张量中的无穷大值。

```python
import genesis

tensor = genesis.tensor([1.0, float('inf'), -float('inf'), 2.0], device='cuda')
inf_mask = genesis.isinf(tensor)
print(inf_mask)  # [False, True, True, False]

# 检查是否有无穷大值
has_inf = genesis.isinf(tensor).any()
if has_inf:
    print("张量包含无穷大值!")
```

#### `isnan(tensor)`
检查张量中的 NaN 值。

```python
tensor = genesis.tensor([1.0, float('nan'), 2.0, 3.0], device='cuda')
nan_mask = genesis.isnan(tensor)
print(nan_mask)  # [False, True, False, False]

# 检查是否有 NaN 值
has_nan = genesis.isnan(tensor).any()
if has_nan:
    print("张量包含 NaN 值!")
```

#### `isfinite(tensor)`
检查张量中的有限值。

```python
tensor = genesis.tensor([1.0, float('inf'), float('nan'), 2.0], device='cuda')
finite_mask = genesis.isfinite(tensor)
print(finite_mask)  # [True, False, False, True]

# 只保留有限值
finite_tensor = tensor[finite_mask]
```

### 在训练中的应用

```python
def check_model_gradients(model):
    """检查模型梯度的数值稳定性"""
    for name, param in model.named_parameters():
        if param.grad is not None:
            if genesis.isnan(param.grad).any():
                print(f"警告: {name} 的梯度包含 NaN!")
                return False
            if genesis.isinf(param.grad).any():
                print(f"警告: {name} 的梯度包含无穷大值!")
                return False
    return True

# 训练循环中使用
for batch in dataloader:
    optimizer.zero_grad()
    loss = model(batch)
    loss.backward()
    
    if not check_model_gradients(model):
        print("跳过此批次由于梯度异常")
        continue
    
    optimizer.step()
```

## 混合精度训练增强

### GradScaler 改进

```python
import genesis.amp as amp

# 创建梯度缩放器
scaler = amp.GradScaler()

# 训练循环
for batch in dataloader:
    with amp.autocast():
        outputs = model(batch['input'])
        loss = criterion(outputs, batch['target'])
    
    # 缩放损失并反向传播
    scaler.scale(loss).backward()
    
    # 检查梯度有效性（现在使用 genesis 原生函数）
    scaler.step(optimizer)  # 内部使用 genesis.isinf/isnan 检查
    scaler.update()
```

## CUDA 内存管理

### 改进的内存分配

```python
import genesis.cuda as cuda

# 检查 CUDA 可用性
if cuda.is_available():
    print(f"CUDA 可用，设备数量: {cuda.device_count()}")
    
    # 获取当前设备信息
    current_dev = cuda.current_device()
    device_name = cuda.get_device_name(current_dev)
    print(f"当前设备: {current_dev} ({device_name})")
    
    # 设置特定设备
    cuda.set_device(0)
    
    # 同步确保操作完成
    cuda.synchronize()
```

## Triton 内核优化

### 数值检查内核

Genesis 实现了高效的 Triton 内核用于数值检查：

```python
# 高性能的数值检查（内部实现）
@triton.jit
def isinf_kernel(input_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    """Triton kernel to check if elements are infinite"""
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input values
    x = tl.load(input_ptr + offsets, mask=mask)
    
    # Check for infinity
    finite_max = 3.4028235e+38  # Maximum finite float32 value
    is_pos_inf = x > finite_max
    is_neg_inf = x < -finite_max
    result = is_pos_inf | is_neg_inf
    
    # Store result as boolean
    tl.store(output_ptr + offsets, result.to(tl.int8), mask=mask)
```

## 错误处理改进

### CUDA 错误检查

```python
import genesis.cuda as cuda

try:
    # CUDA 操作
    cuda.set_device(0)
    cuda.synchronize()
except RuntimeError as e:
    if "CUDA" in str(e):
        print(f"CUDA 错误: {e}")
        # 处理 CUDA 错误
    else:
        raise
```

### 设备可用性检查

```python
def safe_cuda_operation():
    """安全的 CUDA 操作示例"""
    if not cuda.is_available():
        print("CUDA 不可用，使用 CPU")
        return genesis.device('cpu')
    
    try:
        device_count = cuda.device_count()
        if device_count == 0:
            print("没有可用的 CUDA 设备")
            return genesis.device('cpu')
        
        # 选择设备
        cuda.set_device(0)
        return genesis.device('cuda:0')
    
    except RuntimeError as e:
        print(f"CUDA 初始化失败: {e}")
        return genesis.device('cpu')
```

## 最佳实践

### 1. 数值稳定性检查

```python
def validate_tensor(tensor, name="tensor"):
    """验证张量数值稳定性"""
    if genesis.isnan(tensor).any():
        raise ValueError(f"{name} 包含 NaN 值")
    if genesis.isinf(tensor).any():
        raise ValueError(f"{name} 包含无穷大值")
    return True

# 在关键点使用
loss = criterion(outputs, targets)
validate_tensor(loss, "loss")

gradients = model.get_gradients()
for name, grad in gradients.items():
    validate_tensor(grad, f"gradient_{name}")
```

### 2. 设备管理

```python
class DeviceManager:
    """设备管理器"""
    
    def __init__(self):
        self.available_devices = []
        self._detect_devices()
    
    def _detect_devices(self):
        """检测可用设备"""
        if cuda.is_available():
            for i in range(cuda.device_count()):
                device_name = cuda.get_device_name(i)
                self.available_devices.append({
                    'index': i,
                    'name': device_name,
                    'type': 'cuda'
                })
        else:
            self.available_devices.append({
                'index': None,
                'name': 'CPU',
                'type': 'cpu'
            })
    
    def get_best_device(self):
        """获取最佳设备"""
        if self.available_devices and self.available_devices[0]['type'] == 'cuda':
            cuda.set_device(0)
            return genesis.device('cuda:0')
        return genesis.device('cpu')

# 使用示例
device_manager = DeviceManager()
device = device_manager.get_best_device()
model = model.to(device)
```

### 3. 内存管理

```python
def memory_safe_operation(tensor, operation):
    """内存安全的操作"""
    try:
        # 确保在正确设备上
        if tensor.device.type == 'cuda':
            cuda.set_device(tensor.device.index)
        
        # 执行操作
        result = operation(tensor)
        
        # 同步确保完成
        if tensor.device.type == 'cuda':
            cuda.synchronize()
        
        return result
    
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print("GPU 内存不足，尝试清理")
            # 可以添加内存清理逻辑
        raise

# 使用示例
result = memory_safe_operation(tensor, lambda x: x.matmul(weight))
```

## 性能监控

### CUDA 操作性能监控

```python
import time

def profile_cuda_operation(operation, tensor, name="operation"):
    """分析 CUDA 操作性能"""
    if tensor.device.type == 'cuda':
        cuda.synchronize()  # 确保之前操作完成
    
    start_time = time.time()
    result = operation(tensor)
    
    if tensor.device.type == 'cuda':
        cuda.synchronize()  # 确保操作完成
    
    end_time = time.time()
    print(f"{name} 耗时: {(end_time - start_time)*1000:.2f} ms")
    
    return result

# 使用示例
result = profile_cuda_operation(
    lambda x: genesis.matmul(x, weight),
    input_tensor,
    "矩阵乘法"
)
```

这些 CUDA 增强功能提供了更好的设备管理、数值稳定性检查和错误处理，使 Genesis 更加稳定和易用。