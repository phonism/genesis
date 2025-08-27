# 自动微分系统 (genesis.autograd)

## 概述

自动微分系统是Genesis的核心，提供动态计算图构建和自动梯度计算。它实现了反向模式自动微分（反向传播），支持复杂的计算图。

## 核心概念

### 计算图

Genesis在执行操作时构建动态计算图。每个操作在图中创建节点，用于追踪：
- 输入张量
- 执行的操作
- 输出张量
- 用于反向传播的梯度函数

### 梯度计算

梯度使用链式法则计算，从输出到输入反向遍历计算图。

## 主要类

### `genesis.Tensor`

Genesis中支持自动微分的基础数据结构。

```python
class Tensor:
    def __init__(
        self,
        array: Union[list, np.ndarray, NDArray],
        device: Optional[Device] = None,
        dtype: Optional[DType] = None,
        requires_grad: bool = False,
        **kwargs
    )
```

#### 参数

| 参数 | 类型 | 默认值 | 描述 |
|-----------|------|---------|-------------|
| `array` | array-like | required | 输入数据（列表、numpy数组或NDArray） |
| `device` | Device | `None` | 计算设备（cpu/cuda） |
| `dtype` | DType | `None` | 数据类型（如果为None则推断） |
| `requires_grad` | bool | `False` | 是否计算梯度 |
| `**kwargs` | dict | `{}` | 额外的NDArray参数 |

#### 数据类型推断

Genesis自动推断数据类型，遵循PyTorch约定：

```python
# 标量类型推断
genesis.tensor(42)        # → genesis.int64 (Python整数)
genesis.tensor(3.14)      # → genesis.float32 (Python浮点数) 
genesis.tensor(True)      # → genesis.bool (Python布尔值)

# 列表/数组推断
genesis.tensor([1, 2, 3])           # → genesis.int64 (整数列表)
genesis.tensor([1.0, 2.0, 3.0])     # → genesis.float32 (浮点数列表)
genesis.tensor(np.array([1, 2]))    # → 保持numpy数据类型（带转换）

# 显式数据类型指定
genesis.tensor([1, 2, 3], dtype=genesis.float32)  # → genesis.float32
```

**数据类型转换规则：**
- `numpy.float64` → `genesis.float32` （与PyTorch默认值保持一致）
- 整数类型被保留：`np.int32` → `genesis.int32`等
- 布尔类型被保留：`np.bool_` → `genesis.bool`

#### 属性

##### 形状和类型信息
```python
@property
def shape(self) -> Tuple[int, ...]:
    """返回张量的形状。"""

@property
def dtype(self) -> DType:
    """返回数据类型。"""

@property
def device(self) -> Device:
    """返回设备。"""

@property
def ndim(self) -> int:
    """返回维度数量。"""

@property
def size(self) -> int:
    """返回元素总数。"""
```

##### 梯度属性
```python
@property
def requires_grad(self) -> bool:
    """此张量是否需要梯度计算。"""

@property
def grad(self) -> Optional[Tensor]:
    """访问梯度张量。"""

@property
def is_leaf(self) -> bool:
    """是否是叶节点（用户创建的张量）。"""

@property
def grad_fn(self) -> Optional[Function]:
    """创建此张量的函数。"""
```

#### 核心方法

##### 梯度操作
```python
def backward(self, gradient: Optional[Tensor] = None) -> None:
    """
    通过反向传播计算梯度。
    
    参数:
        gradient: 输出梯度。对于标量默认为tensor([1.0])。
        
    示例:
        >>> x = genesis.tensor([1., 2., 3.], requires_grad=True)
        >>> y = (x ** 2).sum()
        >>> y.backward()
        >>> print(x.grad)  # tensor([2., 4., 6.])
    """

def detach(self) -> Tensor:
    """
    返回从计算图分离的新张量。
    
    返回:
        requires_grad=False的张量
        
    示例:
        >>> x = genesis.tensor([1., 2.], requires_grad=True)
        >>> y = x.detach()
        >>> print(y.requires_grad)  # False
    """

def retain_grad(self) -> None:
    """
    为非叶张量启用梯度保持。
    
    示例:
        >>> x = genesis.tensor([1., 2.], requires_grad=True)
        >>> y = x * 2  # 非叶张量
        >>> y.retain_grad()
        >>> z = y.sum()
        >>> z.backward()
        >>> print(y.grad)  # tensor([1., 1.])
    """

def zero_grad(self) -> None:
    """
    将梯度张量清零。
    
    示例:
        >>> x = genesis.tensor([1., 2.], requires_grad=True)
        >>> y = x.sum()
        >>> y.backward()
        >>> x.zero_grad()
        >>> print(x.grad)  # None
    """
```

##### 张量操作

所有标准数学操作都被支持并自动追踪梯度计算：

```python
# 算术操作
z = x + y          # 加法
z = x - y          # 减法
z = x * y          # 乘法
z = x / y          # 除法
z = x ** y         # 幂运算
z = x @ y          # 矩阵乘法

# 一元操作
z = -x             # 取负
z = x.abs()        # 绝对值
z = x.exp()        # 指数
z = x.log()        # 自然对数
z = x.sqrt()       # 平方根
z = x.sin()        # 正弦
z = x.cos()        # 余弦
z = x.tanh()       # 双曲正切

# 归约操作（PyTorch风格接口）
z = x.sum()              # 对所有元素求和
z = x.sum(dim=0)         # 沿第0维求和
z = x.sum(dim=1, keepdim=True)  # 沿第1维求和，保持维度

z = x.mean()             # 所有元素的平均值
z = x.mean(dim=0)        # 沿第0维求平均值
z = x.mean(dim=1, keepdim=True) # 沿第1维求平均值，保持维度

z = x.max()              # 最大元素
z = x.max(dim=0)         # 沿第0维求最大值
z = x.max(dim=1, keepdim=True)  # 沿第1维求最大值，保持维度

# 也支持NumPy风格参数（兼容性）
z = x.sum(axis=0, keepdims=True)    # NumPy风格（axis, keepdims）
z = x.mean(axis=1, keepdims=False)  # NumPy风格接口

# 形状操作
z = x.reshape(shape)      # 重塑
z = x.transpose(dims)     # 转置
z = x.squeeze()           # 移除单维度
z = x.unsqueeze(dim)      # 添加单维度
z = x.view(shape)         # 以不同形状查看
```

### `genesis.Function`

所有可微分操作的基类。

```python
class Function:
    """
    实现自定义可微分操作的基类。
    """
    
    @staticmethod
    def forward(ctx: Context, *args, **kwargs) -> Tensor:
        """
        前向传播实现。
        
        参数:
            ctx: 用于为后向传播保存信息的上下文对象
            *args: 输入张量
            **kwargs: 额外参数
            
        返回:
            输出张量
        """
        raise NotImplementedError
    
    @staticmethod
    def backward(ctx: Context, *grad_outputs) -> Tuple[Optional[Tensor], ...]:
        """
        后向传播实现。
        
        参数:
            ctx: 包含保存信息的上下文对象
            *grad_outputs: 相对于输出的梯度
            
        返回:
            相对于输入的梯度（对于不可微分输入为None）
        """
        raise NotImplementedError
    
    @classmethod
    def apply(cls, *args, **kwargs) -> Tensor:
        """
        应用函数并在计算图中注册。
        """
```

#### 自定义函数示例

```python
import genesis
from genesis import Function

class Exp(Function):
    @staticmethod
    def forward(ctx, x):
        # 为后向传播保存输入
        ctx.save_for_backward(x)
        return genesis.tensor(x.data.exp(), requires_grad=x.requires_grad)
    
    @staticmethod
    def backward(ctx, grad_output):
        # 检索保存的张量
        x, = ctx.saved_tensors
        # exp(x)的梯度是exp(x)
        return grad_output * x.exp()

# 使用
exp = Exp.apply
x = genesis.tensor([1., 2., 3.], requires_grad=True)
y = exp(x)
y.sum().backward()
print(x.grad)  # 通过自定义函数计算的梯度
```

## 上下文管理

### `genesis.no_grad()`

在推理时禁用梯度计算以提高效率的上下文管理器。

```python
with genesis.no_grad():
    # 这里的操作不会构建计算图
    y = model(x)  # 不计算梯度
```

### `genesis.enable_grad()`

启用梯度计算的上下文管理器（在no_grad上下文中有用）。

```python
with genesis.no_grad():
    # 大多数操作不需要梯度
    y = model(x)
    
    with genesis.enable_grad():
        # 这个特定操作需要梯度
        z = y.sum()
        z.backward()
```

### `genesis.set_grad_enabled(mode: bool)`

全局启用或禁用梯度计算。

```python
genesis.set_grad_enabled(False)  # 全局禁用
y = x * 2  # 无梯度

genesis.set_grad_enabled(True)   # 全局启用
z = x * 2  # 计算梯度
```

## 梯度钩子

### 前置和后置钩子

注册在后向传播期间调用的函数：

```python
def print_grad(grad):
    print(f"梯度: {grad}")
    return grad  # 可以在这里修改梯度

x = genesis.tensor([1., 2., 3.], requires_grad=True)
x.register_hook(print_grad)
y = (x ** 2).sum()
y.backward()  # 在后向传播期间打印梯度
```

## 内存管理

### 梯度累积

默认情况下，梯度会在多次后向传播中累积：

```python
x = genesis.tensor([1., 2.], requires_grad=True)

y1 = x.sum()
y1.backward()
print(x.grad)  # tensor([1., 1.])

y2 = (x * 2).sum()
y2.backward()
print(x.grad)  # tensor([3., 3.]) - 累积了！
```

### 清除梯度

```python
# 在新计算前清除梯度
x.grad = None  # 或 x.zero_grad()
```

## 最佳实践

### 1. 高效推理

在推理时始终使用`no_grad()`上下文：

```python
model.eval()
with genesis.no_grad():
    predictions = model(test_data)
```

### 2. 内存优化

当不需要梯度时，分离中间结果：

```python
# 不需要running_mean的梯度
running_mean = (alpha * running_mean.detach() + 
                (1 - alpha) * batch_mean)
```

### 3. 梯度裁剪

防止梯度爆炸：

```python
genesis.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

### 4. 混合精度训练

使用自动混合精度加速训练：

```python
genesis.enable_autocast = True
with genesis.autocast():
    output = model(input)
    loss = criterion(output, target)
```

## 常见模式

### 训练循环

```python
model = MyModel()
optimizer = genesis.optim.Adam(model.parameters())

for epoch in range(num_epochs):
    for batch in dataloader:
        # 前向传播
        outputs = model(batch.inputs)
        loss = criterion(outputs, batch.targets)
        
        # 后向传播
        optimizer.zero_grad()
        loss.backward()
        
        # 更新权重
        optimizer.step()
```

### 梯度检查点

通过重计算激活节省内存：

```python
# 未来版本中提供
from genesis.utils.checkpoint import checkpoint

def forward(self, x):
    # 检查点中间计算
    x = checkpoint(self.layer1, x)
    x = checkpoint(self.layer2, x)
    return self.layer3(x)
```

## 调试

### 梯度检查

使用数值微分验证梯度：

```python
from genesis.autograd import gradcheck

def func(x):
    return (x ** 2).sum()

x = genesis.tensor([1., 2., 3.], requires_grad=True)
gradcheck(func, x, eps=1e-6)  # 如果梯度正确返回True
```

### 检查计算图

```python
# 打印计算图结构
y = x * 2 + 3
print(y.grad_fn)  # <AddBackward>
print(y.grad_fn.next_functions)  # 连接的操作
```

## 性能提示

1. **重用张量**: 避免不必要地创建新张量
2. **原地操作**: 尽可能使用（如`x.add_(y)`）
3. **批量操作**: 同时处理多个样本
4. **禁用梯度**: 推理时使用`no_grad()`
5. **清除梯度**: 每次后向传播前将梯度清零

## 另请参阅

- [神经网络模块](nn/modules.md) - 使用Genesis构建模型
- [优化器](optim/optimizers.md) - 使用梯度下降训练
- [张量操作](../ndarray/index.md) - 低级张量操作
- [示例](../../../samples/) - 完整工作示例