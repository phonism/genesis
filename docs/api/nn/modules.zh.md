# 神经网络模块 (genesis.nn)

## 概述

`genesis.nn`模块提供了创建深度学习模型所需的所有构建块。它采用模块化设计，通过组合更简单的组件来构建复杂模型。

## 核心概念

### 模块系统

所有神经网络组件都继承自`nn.Module`，它提供：
- 参数管理
- 设备和dtype处理  
- 状态序列化
- 前向传播定义

### 参数

参数是自动跟踪并在训练期间更新的张量：
- 分配为模块属性时自动注册
- 包含在`module.parameters()`中供优化器使用
- 与模型状态一起保存/加载

## 基类

### `nn.Module`

所有神经网络模块的基类。

```python
class Module:
    """所有神经网络模块的基类。"""
    
    def __init__(self):
        """初始化模块。"""
        self._modules = {}
        self._parameters = {}
        self._buffers = {}
        self.training = True
```

#### 核心方法

##### 前向传播
```python
def forward(self, *args, **kwargs) -> Tensor:
    """
    定义前向传播计算。
    必须由子类重写。
    
    示例:
        >>> class MyModule(nn.Module):
        ...     def forward(self, x):
        ...         return x * 2
    """
    raise NotImplementedError

def __call__(self, *args, **kwargs) -> Tensor:
    """
    使模块可调用。内部调用forward()。
    
    注意: 始终使用module(input)而不是module.forward(input)
    """
```

##### 参数管理
```python
def parameters(self) -> List[Tensor]:
    """
    返回模块中的所有参数。
    
    返回:
        参数张量列表
        
    示例:
        >>> model = nn.Linear(10, 5)
        >>> params = model.parameters()
        >>> print(len(params))  # 2 (权重和偏置)
    """

def named_parameters(self) -> List[Tuple[str, Tensor]]:
    """
    返回带名称的参数。
    
    返回:
        (名称, 参数)元组列表
        
    示例:
        >>> for name, param in model.named_parameters():
        ...     print(f"{name}: {param.shape}")
    """

def zero_grad(self) -> None:
    """
    将所有参数的梯度清零。
    
    示例:
        >>> model.zero_grad()  # 清除所有梯度
    """
```

##### 模块层次结构
```python
def add_module(self, name: str, module: Optional[Module]) -> None:
    """
    添加子模块。
    
    参数:
        name: 子模块的名称
        module: 要添加的模块实例
        
    示例:
        >>> model = nn.Module()
        >>> model.add_module('fc', nn.Linear(10, 5))
    """

def modules(self) -> Iterator[Module]:
    """返回所有模块（包括自身）的迭代器。"""

def children(self) -> Iterator[Module]:
    """返回直接子模块的迭代器。"""

def named_modules(self) -> Iterator[Tuple[str, Module]]:
    """返回所有模块及其名称的迭代器。"""
```

##### 训练模式
```python
def train(self, mode: bool = True) -> Module:
    """
    将模块设置为训练模式。
    
    参数:
        mode: 是否启用训练模式
        
    返回:
        self
        
    示例:
        >>> model.train()  # 启用训练模式
        >>> model.train(False)  # 等价于model.eval()
    """

def eval(self) -> Module:
    """
    将模块设置为评估模式。
    
    返回:
        self
        
    示例:
        >>> model.eval()  # 禁用dropout，使用BN的运行统计
    """
```

##### 状态管理
```python
def state_dict(self) -> Dict[str, Tensor]:
    """
    返回包含所有参数和缓冲区的状态字典。
    
    返回:
        参数名称到张量的映射字典
        
    示例:
        >>> state = model.state_dict()
        >>> genesis.save(state, 'model.pth')
    """

def load_state_dict(self, state_dict: Dict[str, Tensor]) -> None:
    """
    从状态字典加载参数。
    
    参数:
        state_dict: 参数字典
        
    示例:
        >>> state = genesis.load('model.pth')
        >>> model.load_state_dict(state)
    """
```

### `nn.Parameter`

自动注册为模块参数的特殊张量。

```python
class Parameter(Tensor):
    """
    自动注册为模块参数的张量。
    
    参数:
        data: 张量数据
        requires_grad: 是否计算梯度（默认: True）
        
    示例:
        >>> class MyModule(nn.Module):
        ...     def __init__(self):
        ...         super().__init__()
        ...         self.weight = nn.Parameter(genesis.randn(10, 5))
    """
```

## 层类型

### 线性层

#### `nn.Linear`

执行线性变换的全连接层。

```python
class Linear(Module):
    """
    线性变换: y = xW^T + b
    
    参数:
        in_features: 输入特征大小
        out_features: 输出特征大小
        bias: 是否包含偏置项（默认: True）
        
    形状:
        - 输入: (*, in_features)
        - 输出: (*, out_features)
        
    示例:
        >>> layer = nn.Linear(20, 30)
        >>> x = genesis.randn(128, 20)
        >>> output = layer(x)  # 形状: (128, 30)
    """
```

### 卷积层

#### `nn.Conv2d`

用于图像处理的2D卷积层。

```python
class Conv2d(Module):
    """
    输入信号的2D卷积。
    
    参数:
        in_channels: 输入通道数
        out_channels: 输出通道数
        kernel_size: 卷积核大小
        stride: 卷积步长（默认: 1）
        padding: 两侧添加的零填充（默认: 0）
        bias: 是否添加偏置（默认: True）
        
    形状:
        - 输入: (N, C_in, H, W)
        - 输出: (N, C_out, H_out, W_out)
        
    示例:
        >>> conv = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        >>> x = genesis.randn(32, 3, 224, 224)
        >>> output = conv(x)  # 形状: (32, 64, 224, 224)
    """
```

### 激活函数

#### `nn.ReLU`

修正线性单元激活。

```python
class ReLU(Module):
    """
    ReLU激活: f(x) = max(0, x)
    
    参数:
        inplace: 是否原地修改输入（默认: False）
        
    示例:
        >>> relu = nn.ReLU()
        >>> x = genesis.randn(10)
        >>> output = relu(x)
    """
```

#### `nn.Sigmoid`

Sigmoid激活函数。

```python
class Sigmoid(Module):
    """
    Sigmoid激活: f(x) = 1 / (1 + exp(-x))
    
    示例:
        >>> sigmoid = nn.Sigmoid()
        >>> x = genesis.randn(10)
        >>> output = sigmoid(x)  # 值在(0, 1)范围内
    """
```

#### `nn.Tanh`

双曲正切激活。

```python
class Tanh(Module):
    """
    Tanh激活: f(x) = tanh(x)
    
    示例:
        >>> tanh = nn.Tanh()
        >>> x = genesis.randn(10)
        >>> output = tanh(x)  # 值在(-1, 1)范围内
    """
```

#### `nn.SiLU` (Swish)

Sigmoid线性单元激活。

```python
class SiLU(Module):
    """
    SiLU/Swish激活: f(x) = x * sigmoid(x)
    
    示例:
        >>> silu = nn.SiLU()
        >>> x = genesis.randn(10)
        >>> output = silu(x)
    """
```

#### `nn.GELU`

高斯误差线性单元激活。

```python
class GELU(Module):
    """
    GELU激活: f(x) = x * Φ(x)
    其中Φ(x)是标准高斯分布的累积分布函数。
    
    示例:
        >>> gelu = nn.GELU()
        >>> x = genesis.randn(10)
        >>> output = gelu(x)
    """
```

#### `nn.Softmax`

多类分类的Softmax激活。

```python
class Softmax(Module):
    """
    Softmax激活: softmax(x_i) = exp(x_i) / Σ exp(x_j)
    
    参数:
        dim: 应用softmax的维度
        
    示例:
        >>> softmax = nn.Softmax(dim=-1)
        >>> x = genesis.randn(10, 5)
        >>> output = softmax(x)  # 每行和为1
    """
```

### 归一化层

#### `nn.BatchNorm1d`

1D或2D输入的批量归一化。

```python
class BatchNorm1d(Module):
    """
    2D或3D输入的批量归一化。
    
    参数:
        num_features: 特征数量（[N, C]或[N, C, L]中的C）
        eps: 数值稳定性的小值（默认: 1e-5）
        momentum: 运行统计的动量（默认: 0.1）
        
    形状:
        - 输入: (N, C)或(N, C, L)
        - 输出: 与输入相同
        
    示例:
        >>> bn = nn.BatchNorm1d(100)
        >>> x = genesis.randn(20, 100)
        >>> output = bn(x)
    """
```

#### `nn.LayerNorm`

层归一化。

```python
class LayerNorm(Module):
    """
    最后维度的层归一化。
    
    参数:
        normalized_shape: 要归一化的维度形状
        eps: 数值稳定性的小值（默认: 1e-5）
        
    形状:
        - 输入: (*, normalized_shape)
        - 输出: 与输入相同
        
    示例:
        >>> ln = nn.LayerNorm([768])
        >>> x = genesis.randn(32, 100, 768)
        >>> output = ln(x)  # 在最后一个维度上归一化
    """
```

### Dropout层

#### `nn.Dropout`

正则化的Dropout。

```python
class Dropout(Module):
    """
    随机将元素置零进行正则化。
    
    参数:
        p: 将元素置零的概率（默认: 0.5）
        inplace: 是否原地修改输入（默认: False）
        
    示例:
        >>> dropout = nn.Dropout(p=0.2)
        >>> x = genesis.randn(20, 16)
        >>> output = dropout(x)  # 训练模式：随机将20%的元素置零
    """
```

### 池化层

#### `nn.MaxPool2d`

2D最大池化。

```python
class MaxPool2d(Module):
    """
    2D输入的最大池化。
    
    参数:
        kernel_size: 池化窗口大小
        stride: 池化步长（默认: kernel_size）
        padding: 零填充（默认: 0）
        
    形状:
        - 输入: (N, C, H, W)
        - 输出: (N, C, H_out, W_out)
        
    示例:
        >>> pool = nn.MaxPool2d(kernel_size=2, stride=2)
        >>> x = genesis.randn(1, 16, 32, 32)
        >>> output = pool(x)  # 形状: (1, 16, 16, 16)
    """
```

#### `nn.AvgPool2d`

2D平均池化。

```python
class AvgPool2d(Module):
    """
    2D输入的平均池化。
    
    参数:
        kernel_size: 池化窗口大小
        stride: 池化步长（默认: kernel_size）
        padding: 零填充（默认: 0）
        
    示例:
        >>> pool = nn.AvgPool2d(kernel_size=2, stride=2)
        >>> x = genesis.randn(1, 16, 32, 32)
        >>> output = pool(x)  # 形状: (1, 16, 16, 16)
    """
```

### 嵌入层

#### `nn.Embedding`

嵌入查找表。

```python
class Embedding(Module):
    """
    嵌入查找表。
    
    参数:
        num_embeddings: 词汇大小
        embedding_dim: 嵌入维度
        
    形状:
        - 输入: (*)包含索引
        - 输出: (*, embedding_dim)
        
    示例:
        >>> embed = nn.Embedding(10000, 300)  # 10k词汇，300维嵌入
        >>> indices = genesis.tensor([1, 2, 3, 4])
        >>> output = embed(indices)  # 形状: (4, 300)
    """
```

### 注意力层

#### `nn.MultiheadAttention`

多头注意力机制。

```python
class MultiheadAttention(Module):
    """
    多头注意力机制。
    
    参数:
        embed_dim: 嵌入维度
        num_heads: 注意力头数
        dropout: Dropout概率（默认: 0.0）
        bias: 是否添加偏置（默认: True）
        
    形状:
        - Query: (L, N, E)或(N, L, E)
        - Key: (S, N, E)或(N, S, E)
        - Value: (S, N, E)或(N, S, E)
        - Output: (L, N, E)或(N, L, E)
        
    示例:
        >>> attn = nn.MultiheadAttention(embed_dim=512, num_heads=8)
        >>> x = genesis.randn(10, 32, 512)  # (seq_len, batch, embed_dim)
        >>> output, weights = attn(x, x, x)
    """
```

## 容器模块

### `nn.Sequential`

模块的序列容器。

```python
class Sequential(Module):
    """
    按顺序运行模块的序列容器。
    
    参数:
        *modules: 要应用的模块序列
        
    示例:
        >>> model = nn.Sequential(
        ...     nn.Linear(784, 256),
        ...     nn.ReLU(),
        ...     nn.Linear(256, 10)
        ... )
        >>> x = genesis.randn(32, 784)
        >>> output = model(x)  # 形状: (32, 10)
    """
```

### `nn.ModuleList`

模块的列表容器。

```python
class ModuleList(Module):
    """
    正确注册的模块列表。
    
    参数:
        modules: 可选的模块列表
        
    示例:
        >>> layers = nn.ModuleList([
        ...     nn.Linear(10, 10) for _ in range(5)
        ... ])
        >>> x = genesis.randn(32, 10)
        >>> for layer in layers:
        ...     x = layer(x)
    """
```

### `nn.ModuleDict`

模块的字典容器。

```python
class ModuleDict(Module):
    """
    带字符串键的模块字典。
    
    参数:
        modules: 可选的模块字典
        
    示例:
        >>> layers = nn.ModuleDict({
        ...     'fc1': nn.Linear(10, 20),
        ...     'fc2': nn.Linear(20, 10)
        ... })
        >>> x = genesis.randn(32, 10)
        >>> x = layers['fc1'](x)
        >>> x = layers['fc2'](x)
    """
```

## 损失函数

### `nn.MSELoss`

均方误差损失。

```python
class MSELoss(Module):
    """
    均方误差损失: L = mean((y_pred - y_true)^2)
    
    参数:
        reduction: 'mean', 'sum', 或 'none'（默认: 'mean'）
        
    示例:
        >>> loss_fn = nn.MSELoss()
        >>> pred = genesis.randn(32, 10)
        >>> target = genesis.randn(32, 10)
        >>> loss = loss_fn(pred, target)
    """
```

### `nn.CrossEntropyLoss`

分类的交叉熵损失。

```python
class CrossEntropyLoss(Module):
    """
    多类分类的交叉熵损失。
    
    参数:
        weight: 每个类的手动重缩放权重
        reduction: 'mean', 'sum', 或 'none'（默认: 'mean'）
        
    形状:
        - 输入: (N, C) 其中C是类别数
        - 目标: (N,) 包含类别索引
        
    示例:
        >>> loss_fn = nn.CrossEntropyLoss()
        >>> logits = genesis.randn(32, 10)  # 32个样本，10个类别
        >>> targets = genesis.randint(0, 10, (32,))
        >>> loss = loss_fn(logits, targets)
    """
```

### `nn.BCELoss`

二元交叉熵损失。

```python
class BCELoss(Module):
    """
    二元交叉熵损失。
    
    参数:
        reduction: 'mean', 'sum', 或 'none'（默认: 'mean'）
        
    形状:
        - 输入: (N, *) 其中*表示任意数量的维度
        - 目标: 与输入相同形状
        
    示例:
        >>> loss_fn = nn.BCELoss()
        >>> pred = genesis.sigmoid(genesis.randn(32, 1))
        >>> target = genesis.randint(0, 2, (32, 1)).float()
        >>> loss = loss_fn(pred, target)
    """
```

## 工具

### 权重初始化

```python
def init_weights(module: Module, init_type: str = 'xavier'):
    """
    初始化模块权重。
    
    参数:
        module: 要初始化的模块
        init_type: 'xavier', 'kaiming', 'normal', 'uniform'
        
    示例:
        >>> model = nn.Linear(10, 5)
        >>> init_weights(model, 'xavier')
    """
```

### 梯度裁剪

```python
def clip_grad_norm_(parameters, max_norm: float, norm_type: float = 2.0):
    """
    按范数裁剪梯度。
    
    参数:
        parameters: 参数的可迭代对象
        max_norm: 最大范数值
        norm_type: 范数类型（默认: 2.0）
        
    示例:
        >>> nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    """

def clip_grad_value_(parameters, clip_value: float):
    """
    按值裁剪梯度。
    
    参数:
        parameters: 参数的可迭代对象
        clip_value: 最大绝对值
        
    示例:
        >>> nn.utils.clip_grad_value_(model.parameters(), clip_value=0.5)
    """
```

## 构建自定义模块

### 示例：自定义层

```python
class CustomLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        # 参数自动跟踪
        self.weight = nn.Parameter(genesis.randn(out_features, in_features))
        self.bias = nn.Parameter(genesis.zeros(out_features))
        
        # 子模块自动跟踪
        self.activation = nn.ReLU()
        
    def forward(self, x):
        # 定义前向传播
        x = genesis.matmul(x, self.weight.T) + self.bias
        x = self.activation(x)
        return x

# 使用
layer = CustomLayer(10, 5)
x = genesis.randn(32, 10)
output = layer(x)
```

### 示例：自定义模型

```python
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        residual = x
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x = x + residual  # 跳跃连接
        x = self.relu(x)
        return x

class ResNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        
        # 残差块
        self.layer1 = nn.Sequential(*[ResidualBlock(64) for _ in range(3)])
        self.layer2 = nn.Sequential(*[ResidualBlock(64) for _ in range(4)])
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, num_classes)
        
    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
```

## 最佳实践

1. **始终重写`forward()`**: 在forward方法中定义计算
2. **使用`module(input)`**: 绝不直接调用forward()
3. **注册参数**: 对可学习参数使用nn.Parameter
4. **跟踪子模块**: 将模块分配为属性以自动跟踪
5. **处理训练/评估**: 在训练和评估中使用不同行为
6. **初始化权重**: 适当的初始化改善收敛

## 另请参阅

- [函数式API](functional.md) - 函数式操作
- [优化器](../optim/optimizers.md) - 训练优化器
- [自动微分](../autograd.md) - 自动微分
- [示例](../../../samples/) - 完整示例