# 神经网络模块 (genesis.nn)

Genesis的神经网络模块提供了构建深度学习模型所需的所有基础组件。

## 模块概述

`genesis.nn`模块包含：
- 基础模块类（Module、Parameter）
- 线性层和容器（Linear、Sequential、ModuleList）
- 激活函数（ReLU、SiLU、Softmax）
- 正则化层（Dropout、BatchNorm、LayerNorm）
- 高级组件（Embedding、Attention）

## 核心基类

### Module

所有神经网络模块的基类，提供参数管理、前向传播、状态保存等功能。

```python
class Module:
    """神经网络模块基类"""
    
    def __init__(self):
        """初始化模块"""
```

#### 核心方法

##### 前向传播
```python
def forward(self, *args, **kwargs):
    """
    定义前向传播逻辑（子类必须实现）
    
    示例:
        >>> class MyModule(nn.Module):
        ...     def forward(self, x):
        ...         return x * 2
    """
    raise NotImplementedError()

def __call__(self, *args, **kwargs):
    """
    调用模块，执行前向传播
    
    注意: 直接调用模块而不是forward方法
    示例:
        >>> model = MyModule()
        >>> output = model(input)  # 正确
        >>> output = model.forward(input)  # 不推荐
    """
```

##### 参数管理
```python
def parameters(self) -> List[Tensor]:
    """
    返回模块的所有参数
    
    返回:
        List[Tensor] - 参数列表
        
    示例:
        >>> model = nn.Linear(10, 5)
        >>> params = model.parameters()
        >>> print(len(params))  # 2 (weight和bias)
    """

def named_parameters(self) -> List[Tuple[str, Tensor]]:
    """
    返回参数及其名称
    
    返回:
        List[Tuple[str, Tensor]] - (名称, 参数)对列表
        
    示例:
        >>> for name, param in model.named_parameters():
        ...     print(f"{name}: {param.shape}")
    """

def add_module(self, name: str, module: Optional[Module]):
    """
    添加子模块
    
    参数:
        name: str - 子模块名称
        module: Module - 子模块实例
        
    示例:
        >>> model = nn.Module()
        >>> model.add_module('fc', nn.Linear(10, 5))
    """

def modules(self) -> Iterator[Module]:
    """返回所有子模块的迭代器（包括自身）"""

def children(self) -> Iterator[Module]:
    """返回直接子模块的迭代器"""
```

##### 状态管理
```python
def state_dict(self, destination=None, prefix='', keep_vars=False) -> dict:
    """
    返回模块状态字典
    
    参数:
        destination: dict, optional - 目标字典
        prefix: str - 参数名前缀
        keep_vars: bool - 是否保持张量的梯度信息
        
    返回:
        dict - 参数名到张量的映射
        
    示例:
        >>> state = model.state_dict()
        >>> genesis.save(state, 'model.pth')
    """

def load_state_dict(self, state_dict: dict, strict: bool = True):
    """
    加载模块状态
    
    参数:
        state_dict: dict - 状态字典
        strict: bool - 是否严格匹配参数名
        
    示例:
        >>> state = genesis.load('model.pth')
        >>> model.load_state_dict(state)
    """
```

##### 训练模式控制
```python
def train(self, mode: bool = True) -> Module:
    """
    设置训练模式
    
    参数:
        mode: bool - True为训练模式，False为评估模式
        
    返回:
        self - 支持链式调用
        
    示例:
        >>> model.train()  # 训练模式
        >>> model.train(False)  # 评估模式
    """

def eval(self) -> Module:
    """
    设置评估模式（等价于train(False)）
    
    示例:
        >>> model.eval()
        >>> with genesis.no_grad():
        ...     output = model(input)
    """

@property
def training(self) -> bool:
    """返回是否处于训练模式"""
```

##### 设备管理
```python
def to(self, device=None, dtype=None) -> Module:
    """
    移动模块到指定设备或转换数据类型
    
    参数:
        device: Device - 目标设备
        dtype: DType - 目标数据类型
        
    示例:
        >>> model = model.to(genesis.cuda())
        >>> model = model.to(dtype=genesis.float16)
    """

def cpu(self) -> Module:
    """移动到CPU"""

def cuda(self, device_id: int = 0) -> Module:
    """移动到CUDA设备"""
```

##### 实用方法
```python
def apply(self, fn: Callable[[Module], None]) -> Module:
    """
    递归应用函数到所有子模块
    
    参数:
        fn: callable - 应用到每个模块的函数
        
    示例:
        >>> def init_weights(m):
        ...     if isinstance(m, nn.Linear):
        ...         m.weight.data.normal_(0, 0.01)
        >>> model.apply(init_weights)
    """

def zero_grad(self):
    """清零所有参数的梯度"""

def extra_repr(self) -> str:
    """返回模块的额外字符串表示（子类可重写）"""
```

### Parameter

模型参数的特殊张量类型。

```python
class Parameter(Tensor):
    """
    模型参数
    
    参数:
        data: Tensor - 参数数据
        requires_grad: bool - 是否需要梯度，默认True
        
    示例:
        >>> weight = nn.Parameter(genesis.randn(10, 5))
        >>> bias = nn.Parameter(genesis.zeros(5))
    """
    
    def __init__(self, data: Tensor, requires_grad: bool = True):
        """初始化参数"""
```

## 基础层

### Linear

全连接层（线性变换）。

```python
class Linear(Module):
    """
    线性层: y = xW^T + b
    
    参数:
        in_features: int - 输入特征维度
        out_features: int - 输出特征维度
        bias: bool - 是否使用偏置，默认True
        device: Device - 设备
        dtype: DType - 数据类型
    """
    
    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 device=None, dtype=None):
        """
        初始化线性层
        
        示例:
            >>> layer = nn.Linear(784, 128)
            >>> x = genesis.randn(32, 784)
            >>> y = layer(x)  # shape: (32, 128)
        """
    
    def forward(self, x: Tensor) -> Tensor:
        """
        前向传播
        
        参数:
            x: Tensor - 输入张量，shape: (..., in_features)
            
        返回:
            Tensor - 输出张量，shape: (..., out_features)
        """
    
    @property
    def weight(self) -> Parameter:
        """权重参数，shape: (out_features, in_features)"""
    
    @property
    def bias(self) -> Optional[Parameter]:
        """偏置参数，shape: (out_features,)"""
```

### Embedding

嵌入层，将离散索引映射到连续向量。

```python
class Embedding(Module):
    """
    嵌入层
    
    参数:
        num_embeddings: int - 嵌入字典大小
        embedding_dim: int - 嵌入向量维度
        padding_idx: int, optional - 填充索引
        device: Device - 设备
        dtype: DType - 数据类型
    """
    
    def __init__(self, num_embeddings: int, embedding_dim: int, 
                 padding_idx: Optional[int] = None, device=None, dtype=None):
        """
        初始化嵌入层
        
        示例:
            >>> embed = nn.Embedding(10000, 128)  # 词汇表大小10000，嵌入维度128
            >>> indices = genesis.tensor([1, 2, 3, 4])
            >>> vectors = embed(indices)  # shape: (4, 128)
        """
    
    def forward(self, input: Tensor) -> Tensor:
        """
        前向传播
        
        参数:
            input: Tensor - 索引张量，dtype必须是整数
            
        返回:
            Tensor - 嵌入向量，shape: (*input.shape, embedding_dim)
        """
    
    @property  
    def weight(self) -> Parameter:
        """嵌入权重矩阵，shape: (num_embeddings, embedding_dim)"""
```

## 容器模块

### Sequential

顺序容器，按顺序执行子模块。

```python
class Sequential(Module):
    """
    顺序容器
    
    参数:
        *args: Module - 按顺序执行的模块
        
    示例:
        >>> model = nn.Sequential(
        ...     nn.Linear(784, 256),
        ...     nn.ReLU(),
        ...     nn.Linear(256, 128),
        ...     nn.ReLU(),
        ...     nn.Linear(128, 10)
        ... )
        >>> output = model(input)
    """
    
    def __init__(self, *args):
        """初始化顺序容器"""
    
    def forward(self, x: Tensor) -> Tensor:
        """依次通过所有子模块"""
    
    def append(self, module: Module) -> Sequential:
        """添加模块到末尾"""
    
    def __getitem__(self, idx: int) -> Module:
        """通过索引访问子模块"""
    
    def __len__(self) -> int:
        """返回子模块数量"""
```

### ModuleList

模块列表容器。

```python
class ModuleList(Module):
    """
    模块列表
    
    参数:
        modules: list, optional - 模块列表
        
    示例:
        >>> layers = nn.ModuleList([
        ...     nn.Linear(10, 10) for _ in range(5)
        ... ])
        >>> for layer in layers:
        ...     x = layer(x)
    """
    
    def __init__(self, modules: Optional[List[Module]] = None):
        """初始化模块列表"""
    
    def append(self, module: Module) -> ModuleList:
        """添加模块"""
    
    def extend(self, modules: List[Module]) -> ModuleList:
        """扩展模块列表"""
    
    def __getitem__(self, idx: int) -> Module:
        """索引访问"""
    
    def __len__(self) -> int:
        """返回模块数量"""
```

## 激活函数

### ReLU

线性整流单元激活函数。

```python
class ReLU(Module):
    """
    ReLU激活函数: f(x) = max(0, x)
    
    参数:
        inplace: bool - 是否原地操作，默认False
        
    示例:
        >>> relu = nn.ReLU()
        >>> x = genesis.randn(10)
        >>> y = relu(x)
    """
    
    def __init__(self, inplace: bool = False):
        """初始化ReLU"""
    
    def forward(self, x: Tensor) -> Tensor:
        """应用ReLU激活"""
```

### SiLU

Sigmoid线性单元（Swish激活函数）。

```python
class SiLU(Module):
    """
    SiLU激活函数: f(x) = x * sigmoid(x)
    
    示例:
        >>> silu = nn.SiLU()
        >>> x = genesis.randn(10)
        >>> y = silu(x)
    """
    
    def forward(self, x: Tensor) -> Tensor:
        """应用SiLU激活"""
```

### Softmax

Softmax激活函数。

```python
class Softmax(Module):
    """
    Softmax激活函数
    
    参数:
        dim: int - 计算softmax的维度，默认-1
        
    示例:
        >>> softmax = nn.Softmax(dim=-1)
        >>> x = genesis.randn(10, 5)
        >>> y = softmax(x)  # 每行和为1
    """
    
    def __init__(self, dim: int = -1):
        """初始化Softmax"""
    
    def forward(self, x: Tensor) -> Tensor:
        """应用Softmax"""
```

## 正则化层

### Dropout

随机失活正则化。

```python
class Dropout(Module):
    """
    Dropout正则化
    
    参数:
        p: float - 失活概率，默认0.5
        inplace: bool - 是否原地操作，默认False
        
    示例:
        >>> dropout = nn.Dropout(p=0.2)
        >>> model.train()  # 训练模式下应用dropout
        >>> y = dropout(x)
        >>> model.eval()  # 评估模式下不应用dropout
    """
    
    def __init__(self, p: float = 0.5, inplace: bool = False):
        """初始化Dropout"""
    
    def forward(self, x: Tensor) -> Tensor:
        """
        应用Dropout
        
        注意: 训练模式下随机失活，评估模式下直接返回输入
        """
```

### BatchNorm1d

一维批量归一化。

```python
class BatchNorm1d(Module):
    """
    批量归一化（用于全连接层）
    
    参数:
        num_features: int - 特征维度
        eps: float - 数值稳定性参数，默认1e-5
        momentum: float - 移动平均动量，默认0.1
        device: Device - 设备
        dtype: DType - 数据类型
        
    示例:
        >>> bn = nn.BatchNorm1d(128)
        >>> x = genesis.randn(32, 128)  # batch_size=32, features=128
        >>> y = bn(x)
    """
    
    def __init__(self, num_features: int, eps: float = 1e-5, 
                 momentum: float = 0.1, device=None, dtype=None):
        """初始化BatchNorm1d"""
    
    def forward(self, x: Tensor) -> Tensor:
        """
        应用批量归一化
        
        参数:
            x: Tensor - 输入张量，shape: (N, C) 或 (N, C, L)
            
        返回:
            Tensor - 归一化后的张量
        """
    
    @property
    def weight(self) -> Parameter:
        """缩放参数γ，shape: (num_features,)"""
    
    @property
    def bias(self) -> Parameter:
        """偏移参数β，shape: (num_features,)"""
    
    @property
    def running_mean(self) -> Tensor:
        """移动平均均值"""
    
    @property
    def running_var(self) -> Tensor:
        """移动平均方差"""
```

### LayerNorm

层归一化。

```python
class LayerNorm(Module):
    """
    层归一化
    
    参数:
        normalized_shape: int or tuple - 归一化形状
        eps: float - 数值稳定性参数，默认1e-5
        elementwise_affine: bool - 是否使用可学习参数，默认True
        device: Device - 设备
        dtype: DType - 数据类型
        
    示例:
        >>> ln = nn.LayerNorm(128)
        >>> x = genesis.randn(32, 10, 128)
        >>> y = ln(x)  # 在最后一维归一化
    """
    
    def __init__(self, normalized_shape, eps: float = 1e-5,
                 elementwise_affine: bool = True, device=None, dtype=None):
        """初始化LayerNorm"""
    
    def forward(self, x: Tensor) -> Tensor:
        """应用层归一化"""
    
    @property
    def weight(self) -> Optional[Parameter]:
        """缩放参数"""
    
    @property
    def bias(self) -> Optional[Parameter]:
        """偏移参数"""
```

### RMSNorm

RMS归一化（Root Mean Square Normalization）。

```python
class RMSNorm(Module):
    """
    RMS归一化
    
    参数:
        dim: int - 归一化维度
        eps: float - 数值稳定性参数，默认1e-6
        device: Device - 设备
        dtype: DType - 数据类型
        
    示例:
        >>> rms_norm = nn.RMSNorm(128)
        >>> x = genesis.randn(32, 10, 128)
        >>> y = rms_norm(x)
    """
    
    def __init__(self, dim: int, eps: float = 1e-6, device=None, dtype=None):
        """初始化RMSNorm"""
    
    def forward(self, x: Tensor) -> Tensor:
        """应用RMS归一化"""
```

## 其他模块

### Flatten

展平层。

```python
class Flatten(Module):
    """
    展平层，将多维输入展平为二维
    
    参数:
        start_dim: int - 开始展平的维度，默认1
        end_dim: int - 结束展平的维度，默认-1
        
    示例:
        >>> flatten = nn.Flatten()
        >>> x = genesis.randn(32, 3, 28, 28)
        >>> y = flatten(x)  # shape: (32, 2352)
    """
    
    def __init__(self, start_dim: int = 1, end_dim: int = -1):
        """初始化Flatten"""
    
    def forward(self, x: Tensor) -> Tensor:
        """展平张量"""
```

### Residual

残差连接包装器。

```python
class Residual(Module):
    """
    残差连接: output = x + fn(x)
    
    参数:
        fn: Module - 要应用的函数/模块
        
    示例:
        >>> residual = nn.Residual(
        ...     nn.Sequential(
        ...         nn.Linear(128, 128),
        ...         nn.ReLU()
        ...     )
        ... )
        >>> y = residual(x)  # y = x + fn(x)
    """
    
    def __init__(self, fn: Module):
        """初始化残差模块"""
    
    def forward(self, x: Tensor) -> Tensor:
        """应用残差连接"""
```

## 使用示例

### 构建简单的MLP
```python
import genesis.nn as nn

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.dropout(self.relu(self.fc2(x)))
        x = self.fc3(x)
        return x

# 使用模型
model = MLP(784, 256, 10)
x = genesis.randn(32, 784)
output = model(x)
print(output.shape)  # (32, 10)
```

### 使用Sequential构建模型
```python
model = nn.Sequential(
    nn.Linear(784, 256),
    nn.BatchNorm1d(256),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(256, 128),
    nn.BatchNorm1d(128),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(128, 10)
)

# 训练模式
model.train()
output = model(x)

# 评估模式
model.eval()
with genesis.no_grad():
    output = model(x)
```

### 参数初始化
```python
def init_weights(m):
    if isinstance(m, nn.Linear):
        # Xavier初始化
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Embedding):
        nn.init.normal_(m.weight, mean=0, std=0.02)

model.apply(init_weights)
```

### 模型保存和加载
```python
# 保存模型状态
genesis.save(model.state_dict(), 'model.pth')

# 加载模型状态
model = MLP(784, 256, 10)
state_dict = genesis.load('model.pth')
model.load_state_dict(state_dict)

# 保存完整检查点
checkpoint = {
    'epoch': 100,
    'model_state': model.state_dict(),
    'optimizer_state': optimizer.state_dict(),
    'loss': loss.item()
}
genesis.save_checkpoint(checkpoint, 'checkpoint.pth')
```

## 性能优化提示

1. **使用eval模式进行推理**：调用`model.eval()`禁用dropout和批量归一化的训练行为
2. **参数共享**：使用同一个模块实例可以共享参数
3. **原地操作**：使用`inplace=True`减少内存使用（注意梯度计算）
4. **批量处理**：尽可能使用较大的批量大小提升GPU利用率
5. **混合精度**：结合autocast使用半精度训练

## 注意事项

- Module的forward方法必须被子类实现
- 使用`model(x)`而不是`model.forward(x)`调用模型
- 训练和评估模式会影响Dropout和BatchNorm的行为
- Parameter会自动注册为模块参数，普通Tensor不会
- 模块的设备和数据类型需要与输入一致