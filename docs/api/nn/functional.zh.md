# 函数式操作接口 (genesis.nn.functional)

Genesis的函数式接口提供了无状态的张量操作函数，可以直接在张量上调用而无需创建模块实例。

## 模块概述

`genesis.nn.functional`（通常导入为`F`）包含：
- **基础算术运算**（加、减、乘、除）
- **数学函数**（sin、cos、log、exp、sqrt、power）
- **张量形状操作**（transpose、reshape、expand、view、flatten）
- **张量索引和切片**（getitem、setitem、broadcast_to）
- **聚合操作**（sum、max、logsumexp）
- **矩阵操作**（matmul、stack、cat、squeeze、unsqueeze）
- **基础激活函数**（relu）
- **高级操作**（softmax、dropout来自triton_ops）

## 基础算术运算

### add
```python
def add(a: Tensor, b: Tensor) -> Tensor:
    """
    两个张量的逐元素加法。
    
    参数:
        a: Tensor - 第一个输入张量
        b: Tensor - 第二个输入张量
        
    返回:
        Tensor - 逐元素和 a + b
        
    示例:
        >>> x = genesis.tensor([[1.0, 2.0], [3.0, 4.0]])
        >>> y = genesis.tensor([[2.0, 1.0], [1.0, 2.0]])
        >>> z = F.add(x, y)
        >>> # 结果: [[3.0, 3.0], [4.0, 6.0]]
    """
```

### sub
```python
def sub(a: Tensor, b: Tensor) -> Tensor:
    """
    两个张量的逐元素减法。
    
    参数:
        a: Tensor - 第一个输入张量（被减数）
        b: Tensor - 第二个输入张量（减数）
        
    返回:
        Tensor - 逐元素差 a - b
        
    示例:
        >>> x = genesis.tensor([5.0, 3.0, 8.0])
        >>> y = genesis.tensor([2.0, 1.0, 3.0])
        >>> z = F.sub(x, y)
        >>> # 结果: [3.0, 2.0, 5.0]
    """
```

### multiply
```python
def multiply(a: Tensor, b: Tensor) -> Tensor:
    """
    两个张量的逐元素乘法。
    
    参数:
        a: Tensor - 第一个输入张量
        b: Tensor - 第二个输入张量
        
    返回:
        Tensor - 逐元素积 a * b
        
    示例:
        >>> x = genesis.tensor([2.0, 3.0, 4.0])
        >>> y = genesis.tensor([1.5, 2.0, 0.5])
        >>> z = F.multiply(x, y)
        >>> # 结果: [3.0, 6.0, 2.0]
    """
```

### divide
```python
def divide(a: Tensor, b: Tensor) -> Tensor:
    """
    两个张量的逐元素除法。
    
    参数:
        a: Tensor - 被除数张量
        b: Tensor - 除数张量
        
    返回:
        Tensor - 逐元素商 a / b
        
    示例:
        >>> x = genesis.tensor([6.0, 8.0, 9.0])
        >>> y = genesis.tensor([2.0, 4.0, 3.0])
        >>> z = F.divide(x, y)
        >>> # 结果: [3.0, 2.0, 3.0]
    """
```

## 标量运算

### add_scalar, mul_scalar, divide_scalar, pow_scalar
```python
def add_scalar(a: Tensor, scalar: float) -> Tensor:
def mul_scalar(a: Tensor, scalar: float) -> Tensor:
def divide_scalar(a: Tensor, scalar: float, reverse: bool = False) -> Tensor:
def pow_scalar(a: Tensor, scalar: float, reverse: bool = False) -> Tensor:
    """
    张量与标量之间的逐元素运算。
    
    参数:
        a: Tensor - 输入张量
        scalar: float - 标量值
        reverse: bool - 如果为True，则执行 scalar op tensor（用于divide/pow）
        
    返回:
        Tensor - 张量-标量运算结果
        
    示例:
        >>> x = genesis.tensor([1.0, 2.0, 3.0])
        >>> y1 = F.add_scalar(x, 5.0)      # [6.0, 7.0, 8.0]
        >>> y2 = F.mul_scalar(x, 2.0)      # [2.0, 4.0, 6.0]
        >>> y3 = F.pow_scalar(x, 2.0)      # [1.0, 4.0, 9.0]
    """
```

## 数学函数

### sin, cos, log, exp, sqrt
```python
def sin(a: Tensor) -> Tensor:
def cos(a: Tensor) -> Tensor:
def log(a: Tensor) -> Tensor:
def exp(a: Tensor) -> Tensor:
def sqrt(a: Tensor) -> Tensor:
    """
    逐元素数学函数。
    
    参数:
        a: Tensor - 输入张量
        
    返回:
        Tensor - 数学函数结果
        
    示例:
        >>> x = genesis.tensor([0.0, 1.0, 2.0])
        >>> y1 = F.sin(x)   # [0.0, 0.841, 0.909]
        >>> y2 = F.exp(x)   # [1.0, 2.718, 7.389]
        >>> y3 = F.sqrt(genesis.tensor([4.0, 9.0, 16.0]))  # [2.0, 3.0, 4.0]
    """
```

### negate
```python
def negate(a: Tensor) -> Tensor:
    """
    逐元素取负: -a
    
    参数:
        a: Tensor - 输入张量
        
    返回:
        Tensor - 取负后的张量
        
    示例:
        >>> x = genesis.tensor([1.0, -2.0, 3.0])
        >>> y = F.negate(x)
        >>> # 结果: [-1.0, 2.0, -3.0]
    """
```

## 形状操作

### transpose
```python
def transpose(a: Tensor, axis: tuple = None) -> Tensor:
    """
    转置张量维度。
    
    参数:
        a: Tensor - 输入张量
        axis: tuple - 要交换的维度对（默认：最后两个维度）
        
    返回:
        Tensor - 转置后的张量
        
    示例:
        >>> x = genesis.randn(3, 4, 5)
        >>> y1 = F.transpose(x)           # 交换最后两个维度: (3, 5, 4)
        >>> y2 = F.transpose(x, (0, 2))   # 交换维度0,2: (5, 4, 3)
    """
```

### reshape
```python
def reshape(a: Tensor, shape: tuple) -> Tensor:
    """
    重新塑形张量。
    
    参数:
        a: Tensor - 输入张量
        shape: tuple - 新形状（总元素数必须相同）
        
    返回:
        Tensor - 重塑形后的张量
        
    示例:
        >>> x = genesis.randn(2, 6)
        >>> y = F.reshape(x, (3, 4))
        >>> # 形状从 (2, 6) 变为 (3, 4)
    """
```

### view, expand, flatten
```python
def view(a: Tensor, shape: tuple) -> Tensor:
def expand(a: Tensor, shape: tuple) -> Tensor:
def flatten(a: Tensor, start_dim: int = 0, end_dim: int = None) -> Tensor:
    """
    张量视图和形状操作。
    
    参数:
        a: Tensor - 输入张量
        shape: tuple - 目标形状
        start_dim, end_dim: int - 要展平的维度
        
    返回:
        Tensor - 变换后的张量
        
    示例:
        >>> x = genesis.randn(2, 3, 4)
        >>> y1 = F.view(x, (6, 4))         # 视图为 (6, 4)
        >>> y2 = F.expand(x, (2, 3, 4, 5)) # 扩展最后一个维度
        >>> y3 = F.flatten(x, 1)           # 从维度1开始展平: (2, 12)
    """
```

## 张量操作

### matmul
```python
def matmul(a: Tensor, b: Tensor) -> Tensor:
    """
    矩阵乘法。
    
    参数:
        a: Tensor - 左矩阵
        b: Tensor - 右矩阵
        
    返回:
        Tensor - 矩阵乘积
        
    示例:
        >>> x = genesis.randn(3, 4)
        >>> y = genesis.randn(4, 5)
        >>> z = F.matmul(x, y)  # 形状: (3, 5)
    """
```

### stack, cat
```python
def stack(tensors: list, dim: int = 0) -> Tensor:
def cat(tensors: list, dim: int = 0) -> Tensor:
    """
    沿指定维度堆叠或连接张量。
    
    参数:
        tensors: list - 要组合的张量列表
        dim: int - 堆叠/连接的维度
        
    返回:
        Tensor - 组合后的张量
        
    示例:
        >>> x = genesis.randn(2, 3)
        >>> y = genesis.randn(2, 3)
        >>> z1 = F.stack([x, y], dim=0)  # 形状: (2, 2, 3)
        >>> z2 = F.cat([x, y], dim=0)    # 形状: (4, 3)
    """
```

### squeeze, unsqueeze
```python
def squeeze(tensor: Tensor, dim: int) -> Tensor:
def unsqueeze(tensor: Tensor, dim: int) -> Tensor:
    """
    移除或添加单一维度。
    
    参数:
        tensor: Tensor - 输入张量
        dim: int - 要squeeze/unsqueeze的维度
        
    返回:
        Tensor - 修改维度后的张量
        
    示例:
        >>> x = genesis.randn(1, 3, 1, 4)
        >>> y1 = F.squeeze(x, 0)    # 形状: (3, 1, 4)
        >>> y2 = F.unsqueeze(x, 2)  # 形状: (1, 3, 1, 1, 4)
    """
```

## 聚合操作

### sum
```python
def sum(a: Tensor, axis: int = None, keepdims: bool = False) -> Tensor:
    """
    沿指定维度求和张量元素。
    
    参数:
        a: Tensor - 输入张量
        axis: int - 求和的维度（None表示所有维度）
        keepdims: bool - 是否保持缩减的维度
        
    返回:
        Tensor - 求和后的张量
        
    示例:
        >>> x = genesis.randn(3, 4)
        >>> y1 = F.sum(x)           # 求所有元素的和：标量
        >>> y2 = F.sum(x, axis=0)   # 沿行求和：形状 (4,)
        >>> y3 = F.sum(x, axis=1, keepdims=True)  # 形状: (3, 1)
    """
```

### max, logsumexp
```python
def max(a: Tensor, axis: int = None, keepdims: bool = False) -> Tensor:
def logsumexp(a: Tensor, axis: int = None) -> Tensor:
    """
    最大值和log-sum-exp操作。
    
    参数:
        a: Tensor - 输入张量
        axis: int - 缩减的维度
        keepdims: bool - 是否保持缩减的维度
        
    返回:
        Tensor - 结果张量
        
    示例:
        >>> x = genesis.randn(3, 4)
        >>> y1 = F.max(x, axis=1)      # 沿行求最大值
        >>> y2 = F.logsumexp(x, axis=0) # 沿列LogSumExp
    """
```

## 激活函数

### relu
```python
def relu(a: Tensor) -> Tensor:
    """
    ReLU激活函数: f(x) = max(0, x)
    
    参数:
        a: Tensor - 输入张量
        
    返回:
        Tensor - ReLU激活后的张量
        
    示例:
        >>> x = genesis.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
        >>> y = F.relu(x)
        >>> # 结果: [0.0, 0.0, 0.0, 1.0, 2.0]
    """
```

## 高级操作（来自triton_ops）

### softmax
```python
# 从 genesis.nn.triton_ops 导入
from genesis.nn.triton_ops import softmax

def softmax(x: Tensor, dim: int = -1) -> Tensor:
    """
    使用优化的Triton内核的Softmax函数。
    
    参数:
        x: Tensor - 输入张量
        dim: int - 应用softmax的维度
        
    返回:
        Tensor - Softmax输出（沿dim维度和为1）
        
    示例:
        >>> x = genesis.randn(2, 3)
        >>> y = softmax(x, dim=1)
        >>> # 每行和为1
    """
```

### dropout
```python
# 从 genesis.nn.triton_ops 导入
from genesis.nn.triton_ops import dropout

def dropout(x: Tensor, p: float = 0.5, training: bool = True) -> Tensor:
    """
    使用Triton内核的Dropout正则化。
    
    参数:
        x: Tensor - 输入张量
        p: float - Dropout概率
        training: bool - 是否处于训练模式
        
    返回:
        Tensor - 应用dropout后的张量
        
    示例:
        >>> x = genesis.randn(100, 50)
        >>> y = dropout(x, p=0.2, training=True)
        >>> # 20%的元素设为0，其他元素按1/(1-p)缩放
    """
```

## 索引和广播

### getitem, setitem, broadcast_to
```python
def getitem(a: Tensor, index) -> Tensor:
def setitem(a: Tensor, index, value) -> Tensor:
def broadcast_to(a: Tensor, shape: tuple) -> Tensor:
    """
    张量索引和广播操作。
    
    参数:
        a: Tensor - 输入张量
        index: Various - 索引（int、slice、list、Tensor）
        value: Tensor/scalar - 要设置的值
        shape: tuple - 目标广播形状
        
    返回:
        Tensor - 索引/广播后的张量
        
    示例:
        >>> x = genesis.randn(3, 4)
        >>> y1 = F.getitem(x, [0, 2])      # 选择第0和第2行
        >>> y2 = F.broadcast_to(x, (2, 3, 4))  # 广播到 (2, 3, 4)
    """
```

## 性能说明

- **GPU加速**：当张量在CUDA设备上时，操作自动使用GPU
- **Triton优化**：Softmax和dropout使用优化的Triton内核
- **内存效率**：view操作在可能时共享内存
- **混合精度**：启用时函数支持自动混合精度

## 常用模式

```python
import genesis
import genesis.nn.functional as F

# 基础操作
x = genesis.randn(100, 784)
y = F.relu(F.matmul(x, weights) + bias)

# 形状操作
x = genesis.randn(32, 3, 224, 224)
x_flat = F.flatten(x, start_dim=1)  # (32, 150528)

# 聚合
logits = genesis.randn(32, 10)
probs = F.softmax(logits, dim=1)
max_vals = F.max(logits, axis=1)

# 高级索引
indices = genesis.tensor([0, 2, 4])
selected = F.getitem(x, indices)
```

## 未来功能（路线图）

以下函数计划在未来版本中实现：
- 高级激活函数（gelu、silu、swish）
- 损失函数（cross_entropy、mse_loss、l1_loss）
- 归一化函数（layer_norm、batch_norm）
- 卷积操作（conv1d、conv2d）
- 注意力机制（scaled_dot_product_attention）

要跟踪这些功能的进展，请查看GitHub上的项目路线图。