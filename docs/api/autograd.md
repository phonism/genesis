# 自动微分系统 (genesis.autograd)

Genesis的自动微分系统是框架的核心，提供了张量操作和自动梯度计算功能。

## 模块概述

`genesis.autograd`模块实现了动态计算图和反向传播算法，支持：
- 自动梯度计算
- 混合精度训练
- 梯度钩子和累积
- 计算图构建和优化

## 核心类

### Tensor

Genesis框架的核心数据结构，支持自动微分的多维数组。

```python
class Tensor:
    def __init__(self, array, device=None, dtype=None, requires_grad=False, **kwargs)
```

#### 参数
- `array`: array-like - 输入数据，可以是list、numpy数组或NDArray
- `device`: Device, optional - 设备对象(cpu/cuda)，默认使用default_device
- `dtype`: DType, optional - 数据类型，默认从输入数据推断
- `requires_grad`: bool - 是否需要计算梯度，默认False
- `**kwargs`: 其他NDArray构造参数

#### 属性

##### 基础属性
```python
@property
def shape(self) -> Tuple[int, ...]:
    """返回张量形状"""
    
@property
def dtype(self) -> DType:
    """返回数据类型"""
    
@property
def device(self) -> Device:
    """返回设备对象"""
    
@property
def ndim(self) -> int:
    """返回张量维度数"""
    
@property
def size(self) -> int:
    """返回张量元素总数"""
```

##### 梯度相关属性
```python
@property
def requires_grad(self) -> bool:
    """是否需要计算梯度"""
    
@property
def grad(self) -> Optional[Tensor]:
    """访问梯度张量"""
    
@grad.setter
def grad(self, value: Optional[Tensor]):
    """设置梯度张量"""
    
@property
def is_leaf(self) -> bool:
    """是否为叶子节点（用户创建的张量）"""
```

#### 核心方法

##### 梯度操作
```python
def backward(self, gradient=None):
    """
    执行反向传播，计算所有需要梯度的张量的梯度
    
    参数:
        gradient: Tensor, optional - 输出梯度，默认为1.0（标量情况）
        
    示例:
        >>> x = genesis.tensor([1., 2., 3.], requires_grad=True)
        >>> y = (x ** 2).sum()
        >>> y.backward()
        >>> print(x.grad)  # tensor([2., 4., 6.])
    """

def detach(self) -> Tensor:
    """
    返回一个新张量，与计算图分离
    
    返回:
        Tensor - 分离的张量，requires_grad=False
        
    示例:
        >>> x = genesis.tensor([1., 2.], requires_grad=True)
        >>> y = x.detach()
        >>> print(y.requires_grad)  # False
    """

def requires_grad_(self, requires_grad=True) -> Tensor:
    """
    原地修改requires_grad属性
    
    参数:
        requires_grad: bool - 是否需要梯度
        
    返回:
        self - 返回自身以支持链式调用
    """

def zero_grad(self):
    """清零梯度"""

def register_hook(self, hook):
    """
    注册梯度钩子函数
    
    参数:
        hook: callable - 钩子函数，接收梯度作为参数
        
    示例:
        >>> def print_grad(grad):
        ...     print(f"Gradient: {grad}")
        >>> x.register_hook(print_grad)
    """
```

##### 设备和类型转换
```python
def cpu(self) -> Tensor:
    """将张量移到CPU"""
    
def cuda(self, device_id=0) -> Tensor:
    """
    将张量移到CUDA设备
    
    参数:
        device_id: int - GPU设备ID
    """
    
def to(self, device=None, dtype=None) -> Tensor:
    """
    转换设备或数据类型
    
    参数:
        device: Device, optional - 目标设备
        dtype: DType, optional - 目标数据类型
        
    示例:
        >>> x = genesis.tensor([1, 2, 3])
        >>> x_gpu = x.to(genesis.cuda())
        >>> x_fp16 = x.to(dtype=genesis.float16)
    """

def float(self) -> Tensor:
    """转换为float32类型"""
    
def half(self) -> Tensor:
    """转换为float16类型"""
    
def long(self) -> Tensor:
    """转换为int64类型"""
    
def int(self) -> Tensor:
    """转换为int32类型"""
    
def bool(self) -> Tensor:
    """转换为bool类型"""
```

##### 形状操作
```python
def reshape(self, *shape) -> Tensor:
    """
    改变张量形状
    
    参数:
        *shape: int - 新形状维度
        
    返回:
        Tensor - 改变形状后的张量
        
    示例:
        >>> x = genesis.tensor([[1, 2], [3, 4]])
        >>> y = x.reshape(4)  # [1, 2, 3, 4]
        >>> z = x.reshape(1, 4)  # [[1, 2, 3, 4]]
    """

def view(self, *shape) -> Tensor:
    """reshape的别名，与PyTorch兼容"""
    
def transpose(self, dim0=None, dim1=None) -> Tensor:
    """
    转置张量
    
    参数:
        dim0, dim1: int, optional - 要交换的维度，默认转置最后两维
        
    示例:
        >>> x = genesis.randn(2, 3, 4)
        >>> y = x.transpose(0, 2)  # shape: (4, 3, 2)
    """

def permute(self, *dims) -> Tensor:
    """
    按指定顺序重排维度
    
    参数:
        *dims: int - 新的维度顺序
        
    示例:
        >>> x = genesis.randn(2, 3, 4)
        >>> y = x.permute(2, 0, 1)  # shape: (4, 2, 3)
    """

def squeeze(self, dim=None) -> Tensor:
    """移除大小为1的维度"""
    
def unsqueeze(self, dim) -> Tensor:
    """在指定位置插入大小为1的维度"""
    
def expand(self, *shape) -> Tensor:
    """扩展张量到新形状（通过广播）"""
```

##### 数学运算
```python
# 算术运算
def __add__(self, other) -> Tensor
def __sub__(self, other) -> Tensor  
def __mul__(self, other) -> Tensor
def __truediv__(self, other) -> Tensor
def __pow__(self, other) -> Tensor
def __matmul__(self, other) -> Tensor

# 原地运算
def add_(self, other) -> Tensor
def sub_(self, other) -> Tensor
def mul_(self, other) -> Tensor
def div_(self, other) -> Tensor

# 聚合运算
def sum(self, axis=None, keepdims=False) -> Tensor:
    """
    求和
    
    参数:
        axis: int or tuple, optional - 求和的轴
        keepdims: bool - 是否保持维度
    """

def mean(self, axis=None, keepdims=False) -> Tensor:
    """求平均值"""
    
def max(self, axis=None, keepdims=False) -> Tensor:
    """求最大值"""
    
def min(self, axis=None, keepdims=False) -> Tensor:
    """求最小值"""

# 数学函数
def exp(self) -> Tensor
def log(self) -> Tensor
def sqrt(self) -> Tensor
def sin(self) -> Tensor
def cos(self) -> Tensor
def tanh(self) -> Tensor
def sigmoid(self) -> Tensor
def relu(self) -> Tensor
```

##### 索引和切片
```python
def __getitem__(self, key) -> Tensor:
    """
    张量索引和切片
    
    参数:
        key: int, slice, tuple - 索引键
        
    示例:
        >>> x = genesis.randn(3, 4, 5)
        >>> y = x[0]  # 获取第一个元素
        >>> z = x[:, 1:3, :]  # 切片操作
        >>> w = x[..., -1]  # 使用省略号
    """

def __setitem__(self, key, value):
    """张量赋值"""
```

##### 实用方法
```python
def item(self) -> float:
    """返回标量张量的Python数值"""
    
def numpy(self) -> numpy.ndarray:
    """转换为numpy数组"""
    
def contiguous(self) -> Tensor:
    """返回内存连续的张量"""
    
def clone(self) -> Tensor:
    """深拷贝张量"""
    
def data_ptr(self) -> int:
    """返回数据指针（用于Triton）"""
    
def stride(self, dim=None):
    """返回步长信息"""
    
def is_contiguous(self) -> bool:
    """检查是否内存连续"""
```

### Function

所有可微分操作的基类。

```python
class Function:
    """自定义可微分操作的基类"""
    
    @staticmethod
    def forward(ctx: Context, *args, **kwargs):
        """
        前向传播计算
        
        参数:
            ctx: Context - 用于保存中间结果的上下文
            *args: 输入参数
            **kwargs: 关键字参数
            
        返回:
            输出张量
        """
        raise NotImplementedError()
    
    @staticmethod
    def backward(ctx: Context, *grad_outputs):
        """
        反向传播计算
        
        参数:
            ctx: Context - 前向传播保存的上下文
            *grad_outputs: 输出梯度
            
        返回:
            tuple - 对应每个输入的梯度
        """
        raise NotImplementedError()
    
    @classmethod
    def apply(cls, *args, **kwargs):
        """
        应用函数，自动处理前向和反向传播
        
        示例:
            >>> class Exp(Function):
            ...     @staticmethod
            ...     def forward(ctx, x):
            ...         y = x.exp()
            ...         ctx.save_for_backward(y)
            ...         return y
            ...     
            ...     @staticmethod
            ...     def backward(ctx, grad_output):
            ...         y, = ctx.saved_tensors
            ...         return grad_output * y
            >>> 
            >>> y = Exp.apply(x)
        """
```

### Context

用于在前向和反向传播之间传递信息的上下文对象。

```python
class Context:
    """操作上下文，用于保存中间结果"""
    
    def save_for_backward(self, *tensors):
        """
        保存张量用于反向传播
        
        参数:
            *tensors: Tensor - 需要保存的张量
            
        示例:
            >>> def forward(ctx, x, y):
            ...     z = x * y
            ...     ctx.save_for_backward(x, y)
            ...     return z
        """
    
    @property
    def saved_tensors(self) -> List[Tensor]:
        """获取保存的张量列表"""
```

## 全局函数

### 梯度管理

```python
@contextmanager
def no_grad():
    """
    上下文管理器，禁用梯度计算
    
    示例:
        >>> with genesis.no_grad():
        ...     y = x * 2  # 不会构建计算图
    """

@contextmanager
def enable_grad():
    """
    上下文管理器，启用梯度计算
    
    示例:
        >>> with genesis.enable_grad():
        ...     y = x * 2  # 会构建计算图
    """

def set_grad_enabled(enabled: bool):
    """
    设置梯度计算开关
    
    参数:
        enabled: bool - 是否启用梯度计算
    """
```

## 使用示例

### 基础张量操作
```python
import genesis

# 创建张量
x = genesis.tensor([[1., 2.], [3., 4.]], requires_grad=True)
y = genesis.tensor([[2., 0.], [0., 2.]], requires_grad=True)

# 前向计算
z = genesis.matmul(x, y)
loss = z.sum()

# 反向传播
loss.backward()

print(x.grad)  # 梯度: [[2., 2.], [2., 2.]]
print(y.grad)  # 梯度: [[4., 4.], [6., 6.]]
```

### 自定义Function
```python
class CustomReLU(genesis.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return genesis.maximum(x, 0)
    
    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        grad_input = grad_output * (x > 0)
        return grad_input

# 使用自定义函数
x = genesis.randn(10, requires_grad=True)
y = CustomReLU.apply(x)
y.sum().backward()
```

### 梯度钩子
```python
def grad_hook(grad):
    print(f"Gradient norm: {grad.norm()}")
    return grad * 0.1  # 缩放梯度

x = genesis.randn(10, requires_grad=True)
x.register_hook(grad_hook)

y = (x ** 2).sum()
y.backward()  # 会打印梯度范数并缩放梯度
```

### 混合精度训练
```python
genesis.enable_autocast = True

with genesis.autocast():
    # 自动转换为FP16进行计算
    x = genesis.randn(1000, 1000)
    y = genesis.matmul(x, x)
    loss = y.mean()
    
loss.backward()  # 梯度计算自动处理精度转换
```

## 性能优化提示

1. **使用no_grad进行推理**：在不需要梯度的场景下使用`no_grad()`减少内存消耗
2. **及时清理梯度**：使用`zero_grad()`清理不需要的梯度
3. **使用detach分离计算图**：避免不必要的梯度传播
4. **利用原地操作**：使用`add_()`, `mul_()`等原地操作减少内存分配
5. **混合精度训练**：启用autocast提升训练速度

## 注意事项

- 张量的`requires_grad`属性只能在叶子节点上设置
- 原地操作可能会破坏计算图，谨慎使用
- 梯度默认会累积，需要手动清零
- 视图操作（reshape, transpose等）共享内存，修改会影响原张量