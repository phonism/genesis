# 函数式操作接口 (genesis.nn.functional)

Genesis的函数式接口提供了无状态的张量操作函数，可以直接在张量上调用而无需创建模块实例。

## 模块概述

`genesis.nn.functional`（通常导入为`F`）包含：
- 激活函数（relu、sigmoid、softmax等）
- 损失函数（cross_entropy、mse_loss等）
- 张量操作（matmul、transpose、reshape等）
- 归一化函数（layer_norm、batch_norm等）
- 注意力机制（scaled_dot_product_attention等）

## 激活函数

### relu
```python
def relu(x: Tensor, inplace: bool = False) -> Tensor:
    """
    ReLU激活函数: f(x) = max(0, x)
    
    参数:
        x: Tensor - 输入张量
        inplace: bool - 是否原地操作
        
    返回:
        Tensor - 激活后的张量
        
    示例:
        >>> x = genesis.randn(10)
        >>> y = F.relu(x)
        >>> # 原地操作
        >>> F.relu(x, inplace=True)
    """
```

### sigmoid
```python
def sigmoid(x: Tensor) -> Tensor:
    """
    Sigmoid激活函数: f(x) = 1 / (1 + exp(-x))
    
    参数:
        x: Tensor - 输入张量
        
    返回:
        Tensor - 范围在(0, 1)的输出
        
    示例:
        >>> x = genesis.randn(10)
        >>> y = F.sigmoid(x)
    """
```

### tanh
```python
def tanh(x: Tensor) -> Tensor:
    """
    双曲正切激活函数: f(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
    
    参数:
        x: Tensor - 输入张量
        
    返回:
        Tensor - 范围在(-1, 1)的输出
        
    示例:
        >>> x = genesis.randn(10)
        >>> y = F.tanh(x)
    """
```

### softmax
```python
def softmax(x: Tensor, dim: int = -1) -> Tensor:
    """
    Softmax函数，将输入转换为概率分布
    
    参数:
        x: Tensor - 输入张量
        dim: int - 计算softmax的维度
        
    返回:
        Tensor - 概率分布，在指定维度上和为1
        
    示例:
        >>> logits = genesis.randn(32, 10)
        >>> probs = F.softmax(logits, dim=-1)
        >>> print(probs.sum(dim=-1))  # 全为1
    """
```

### log_softmax
```python
def log_softmax(x: Tensor, dim: int = -1) -> Tensor:
    """
    Log-Softmax函数: log(softmax(x))
    
    参数:
        x: Tensor - 输入张量
        dim: int - 计算的维度
        
    返回:
        Tensor - log概率
        
    注意: 数值上比先softmax再log更稳定
        
    示例:
        >>> logits = genesis.randn(32, 10)
        >>> log_probs = F.log_softmax(logits, dim=-1)
    """
```

### silu
```python
def silu(x: Tensor) -> Tensor:
    """
    SiLU/Swish激活函数: f(x) = x * sigmoid(x)
    
    参数:
        x: Tensor - 输入张量
        
    返回:
        Tensor - 激活后的张量
        
    示例:
        >>> x = genesis.randn(10)
        >>> y = F.silu(x)
    """
```

### gelu
```python
def gelu(x: Tensor) -> Tensor:
    """
    GELU激活函数: f(x) = x * Φ(x)
    其中Φ(x)是标准正态分布的累积分布函数
    
    参数:
        x: Tensor - 输入张量
        
    返回:
        Tensor - 激活后的张量
        
    示例:
        >>> x = genesis.randn(10)
        >>> y = F.gelu(x)
    """
```

## 损失函数

### cross_entropy
```python
def cross_entropy(input: Tensor, target: Tensor, 
                  reduction: str = 'mean') -> Tensor:
    """
    交叉熵损失函数
    
    参数:
        input: Tensor - 预测logits，shape: (N, C)
        target: Tensor - 目标标签，shape: (N,) 或 (N, C)
        reduction: str - 'none'|'mean'|'sum'
        
    返回:
        Tensor - 损失值
        
    示例:
        >>> logits = genesis.randn(32, 10)  # 32个样本，10个类别
        >>> labels = genesis.randint(0, 10, (32,))
        >>> loss = F.cross_entropy(logits, labels)
    """
```

### mse_loss
```python
def mse_loss(input: Tensor, target: Tensor, 
             reduction: str = 'mean') -> Tensor:
    """
    均方误差损失函数
    
    参数:
        input: Tensor - 预测值
        target: Tensor - 目标值
        reduction: str - 'none'|'mean'|'sum'
        
    返回:
        Tensor - MSE损失
        
    示例:
        >>> pred = genesis.randn(32, 10)
        >>> target = genesis.randn(32, 10)
        >>> loss = F.mse_loss(pred, target)
    """
```

### nll_loss
```python
def nll_loss(input: Tensor, target: Tensor, 
             reduction: str = 'mean') -> Tensor:
    """
    负对数似然损失函数
    
    参数:
        input: Tensor - log概率，shape: (N, C)
        target: Tensor - 目标类别索引，shape: (N,)
        reduction: str - 'none'|'mean'|'sum'
        
    返回:
        Tensor - NLL损失
        
    示例:
        >>> log_probs = F.log_softmax(logits, dim=-1)
        >>> loss = F.nll_loss(log_probs, labels)
    """
```

### binary_cross_entropy
```python
def binary_cross_entropy(input: Tensor, target: Tensor,
                        reduction: str = 'mean') -> Tensor:
    """
    二元交叉熵损失函数
    
    参数:
        input: Tensor - 预测概率，范围[0, 1]
        target: Tensor - 二元目标，0或1
        reduction: str - 'none'|'mean'|'sum'
        
    返回:
        Tensor - BCE损失
        
    示例:
        >>> probs = F.sigmoid(logits)
        >>> loss = F.binary_cross_entropy(probs, targets)
    """
```

## 张量操作

### matmul
```python
def matmul(a: Tensor, b: Tensor) -> Tensor:
    """
    矩阵乘法
    
    参数:
        a: Tensor - 第一个张量
        b: Tensor - 第二个张量
        
    返回:
        Tensor - 矩阵乘积
        
    支持:
        - 向量×向量: 点积
        - 矩阵×向量: 矩阵向量乘法
        - 矩阵×矩阵: 矩阵乘法
        - 批量矩阵乘法: 广播规则
        
    示例:
        >>> # 矩阵乘法
        >>> a = genesis.randn(3, 4)
        >>> b = genesis.randn(4, 5)
        >>> c = F.matmul(a, b)  # shape: (3, 5)
        >>> 
        >>> # 批量矩阵乘法
        >>> a = genesis.randn(10, 3, 4)
        >>> b = genesis.randn(10, 4, 5)
        >>> c = F.matmul(a, b)  # shape: (10, 3, 5)
    """
```

### transpose
```python
def transpose(x: Tensor, dim0: int = -2, dim1: int = -1) -> Tensor:
    """
    转置张量的两个维度
    
    参数:
        x: Tensor - 输入张量
        dim0: int - 第一个维度
        dim1: int - 第二个维度
        
    返回:
        Tensor - 转置后的张量
        
    示例:
        >>> x = genesis.randn(2, 3, 4)
        >>> y = F.transpose(x, 0, 2)  # shape: (4, 3, 2)
    """
```

### reshape
```python
def reshape(x: Tensor, *shape) -> Tensor:
    """
    改变张量形状
    
    参数:
        x: Tensor - 输入张量
        *shape: int - 新形状，-1表示自动推断
        
    返回:
        Tensor - 新形状的张量
        
    示例:
        >>> x = genesis.randn(2, 3, 4)
        >>> y = F.reshape(x, 6, 4)  # shape: (6, 4)
        >>> z = F.reshape(x, -1)  # shape: (24,)
    """
```

### flatten
```python
def flatten(x: Tensor, start_dim: int = 0, end_dim: int = -1) -> Tensor:
    """
    展平张量维度
    
    参数:
        x: Tensor - 输入张量
        start_dim: int - 开始展平的维度
        end_dim: int - 结束展平的维度
        
    返回:
        Tensor - 展平后的张量
        
    示例:
        >>> x = genesis.randn(32, 3, 28, 28)
        >>> y = F.flatten(x, 1)  # shape: (32, 2352)
    """
```

### squeeze
```python
def squeeze(x: Tensor, dim: Optional[int] = None) -> Tensor:
    """
    移除大小为1的维度
    
    参数:
        x: Tensor - 输入张量
        dim: int, optional - 指定要移除的维度
        
    返回:
        Tensor - 压缩后的张量
        
    示例:
        >>> x = genesis.randn(1, 3, 1, 4)
        >>> y = F.squeeze(x)  # shape: (3, 4)
        >>> z = F.squeeze(x, dim=0)  # shape: (3, 1, 4)
    """
```

### unsqueeze
```python
def unsqueeze(x: Tensor, dim: int) -> Tensor:
    """
    插入大小为1的维度
    
    参数:
        x: Tensor - 输入张量
        dim: int - 插入维度的位置
        
    返回:
        Tensor - 扩展后的张量
        
    示例:
        >>> x = genesis.randn(3, 4)
        >>> y = F.unsqueeze(x, 0)  # shape: (1, 3, 4)
        >>> z = F.unsqueeze(x, -1)  # shape: (3, 4, 1)
    """
```

### cat
```python
def cat(tensors: List[Tensor], dim: int = 0) -> Tensor:
    """
    连接张量列表
    
    参数:
        tensors: List[Tensor] - 张量列表
        dim: int - 连接的维度
        
    返回:
        Tensor - 连接后的张量
        
    示例:
        >>> x1 = genesis.randn(2, 3)
        >>> x2 = genesis.randn(2, 3)
        >>> y = F.cat([x1, x2], dim=0)  # shape: (4, 3)
    """
```

### stack
```python
def stack(tensors: List[Tensor], dim: int = 0) -> Tensor:
    """
    堆叠张量列表（新增维度）
    
    参数:
        tensors: List[Tensor] - 张量列表，形状必须相同
        dim: int - 新维度的位置
        
    返回:
        Tensor - 堆叠后的张量
        
    示例:
        >>> x1 = genesis.randn(2, 3)
        >>> x2 = genesis.randn(2, 3)
        >>> y = F.stack([x1, x2], dim=0)  # shape: (2, 2, 3)
    """
```

### split
```python
def split(x: Tensor, split_size_or_sections, dim: int = 0) -> List[Tensor]:
    """
    分割张量
    
    参数:
        x: Tensor - 输入张量
        split_size_or_sections: int or list - 分割大小或各部分大小
        dim: int - 分割的维度
        
    返回:
        List[Tensor] - 分割后的张量列表
        
    示例:
        >>> x = genesis.randn(10, 3)
        >>> # 等分为5份
        >>> parts = F.split(x, 2, dim=0)  # 5个(2, 3)张量
        >>> # 指定各部分大小
        >>> parts = F.split(x, [3, 3, 4], dim=0)
    """
```

## 归约操作

### sum
```python
def sum(x: Tensor, dim: Optional[Union[int, Tuple[int]]] = None, 
        keepdim: bool = False) -> Tensor:
    """
    求和
    
    参数:
        x: Tensor - 输入张量
        dim: int or tuple, optional - 求和的维度
        keepdim: bool - 是否保持维度
        
    返回:
        Tensor - 求和结果
        
    示例:
        >>> x = genesis.randn(3, 4, 5)
        >>> y = F.sum(x)  # 标量
        >>> y = F.sum(x, dim=1)  # shape: (3, 5)
        >>> y = F.sum(x, dim=(1, 2))  # shape: (3,)
        >>> y = F.sum(x, dim=1, keepdim=True)  # shape: (3, 1, 5)
    """
```

### mean
```python
def mean(x: Tensor, dim: Optional[Union[int, Tuple[int]]] = None,
         keepdim: bool = False) -> Tensor:
    """
    求平均值
    
    参数:
        x: Tensor - 输入张量
        dim: int or tuple, optional - 求平均的维度
        keepdim: bool - 是否保持维度
        
    返回:
        Tensor - 平均值
        
    示例:
        >>> x = genesis.randn(3, 4, 5)
        >>> y = F.mean(x, dim=1)  # shape: (3, 5)
    """
```

### max
```python
def max(x: Tensor, dim: Optional[int] = None, 
        keepdim: bool = False) -> Union[Tensor, Tuple[Tensor, Tensor]]:
    """
    求最大值
    
    参数:
        x: Tensor - 输入张量
        dim: int, optional - 求最大值的维度
        keepdim: bool - 是否保持维度
        
    返回:
        Tensor or (Tensor, Tensor) - 最大值（和索引，如果指定dim）
        
    示例:
        >>> x = genesis.randn(3, 4)
        >>> # 全局最大值
        >>> max_val = F.max(x)
        >>> # 沿维度求最大值
        >>> max_vals, indices = F.max(x, dim=1)
    """
```

### min
```python
def min(x: Tensor, dim: Optional[int] = None,
        keepdim: bool = False) -> Union[Tensor, Tuple[Tensor, Tensor]]:
    """
    求最小值
    
    参数:
        x: Tensor - 输入张量
        dim: int, optional - 求最小值的维度
        keepdim: bool - 是否保持维度
        
    返回:
        Tensor or (Tensor, Tensor) - 最小值（和索引，如果指定dim）
    """
```

## 数学函数

### exp
```python
def exp(x: Tensor) -> Tensor:
    """
    指数函数: e^x
    
    参数:
        x: Tensor - 输入张量
        
    返回:
        Tensor - 指数值
        
    示例:
        >>> x = genesis.tensor([0., 1., 2.])
        >>> y = F.exp(x)  # [1., 2.718, 7.389]
    """
```

### log
```python
def log(x: Tensor) -> Tensor:
    """
    自然对数: ln(x)
    
    参数:
        x: Tensor - 输入张量（必须为正）
        
    返回:
        Tensor - 对数值
        
    示例:
        >>> x = genesis.tensor([1., 2.718, 7.389])
        >>> y = F.log(x)  # [0., 1., 2.]
    """
```

### sqrt
```python
def sqrt(x: Tensor) -> Tensor:
    """
    平方根函数
    
    参数:
        x: Tensor - 输入张量（必须非负）
        
    返回:
        Tensor - 平方根
        
    示例:
        >>> x = genesis.tensor([0., 1., 4., 9.])
        >>> y = F.sqrt(x)  # [0., 1., 2., 3.]
    """
```

### pow
```python
def pow(x: Tensor, exponent: Union[float, Tensor]) -> Tensor:
    """
    幂函数: x^exponent
    
    参数:
        x: Tensor - 底数
        exponent: float or Tensor - 指数
        
    返回:
        Tensor - 幂值
        
    示例:
        >>> x = genesis.tensor([1., 2., 3.])
        >>> y = F.pow(x, 2)  # [1., 4., 9.]
    """
```

### sin
```python
def sin(x: Tensor) -> Tensor:
    """
    正弦函数
    
    参数:
        x: Tensor - 输入张量（弧度）
        
    返回:
        Tensor - 正弦值
    """
```

### cos
```python
def cos(x: Tensor) -> Tensor:
    """
    余弦函数
    
    参数:
        x: Tensor - 输入张量（弧度）
        
    返回:
        Tensor - 余弦值
    """
```

## 归一化函数

### layer_norm
```python
def layer_norm(x: Tensor, normalized_shape: Union[int, List[int]],
               weight: Optional[Tensor] = None, bias: Optional[Tensor] = None,
               eps: float = 1e-5) -> Tensor:
    """
    层归一化
    
    参数:
        x: Tensor - 输入张量
        normalized_shape: int or list - 归一化的形状
        weight: Tensor, optional - 缩放参数
        bias: Tensor, optional - 偏移参数
        eps: float - 数值稳定性参数
        
    返回:
        Tensor - 归一化后的张量
        
    示例:
        >>> x = genesis.randn(32, 10, 128)
        >>> y = F.layer_norm(x, 128)
    """
```

### batch_norm
```python
def batch_norm(x: Tensor, running_mean: Optional[Tensor], 
               running_var: Optional[Tensor], weight: Optional[Tensor],
               bias: Optional[Tensor], training: bool = True,
               momentum: float = 0.1, eps: float = 1e-5) -> Tensor:
    """
    批量归一化
    
    参数:
        x: Tensor - 输入张量
        running_mean: Tensor - 移动平均均值
        running_var: Tensor - 移动平均方差
        weight: Tensor - 缩放参数
        bias: Tensor - 偏移参数
        training: bool - 是否训练模式
        momentum: float - 移动平均动量
        eps: float - 数值稳定性参数
        
    返回:
        Tensor - 归一化后的张量
    """
```

## 注意力机制

### scaled_dot_product_attention
```python
def scaled_dot_product_attention(query: Tensor, key: Tensor, value: Tensor,
                                 attn_mask: Optional[Tensor] = None,
                                 dropout_p: float = 0.0,
                                 is_causal: bool = False) -> Tensor:
    """
    缩放点积注意力
    
    参数:
        query: Tensor - 查询张量，shape: (*, seq_len, d_k)
        key: Tensor - 键张量，shape: (*, seq_len, d_k)
        value: Tensor - 值张量，shape: (*, seq_len, d_v)
        attn_mask: Tensor, optional - 注意力掩码
        dropout_p: float - dropout概率
        is_causal: bool - 是否使用因果掩码
        
    返回:
        Tensor - 注意力输出，shape: (*, seq_len, d_v)
        
    示例:
        >>> q = genesis.randn(32, 10, 64)  # batch=32, seq=10, dim=64
        >>> k = genesis.randn(32, 10, 64)
        >>> v = genesis.randn(32, 10, 64)
        >>> output = F.scaled_dot_product_attention(q, k, v)
    """
```

## 其他函数

### dropout
```python
def dropout(x: Tensor, p: float = 0.5, training: bool = True,
           inplace: bool = False) -> Tensor:
    """
    Dropout正则化
    
    参数:
        x: Tensor - 输入张量
        p: float - 失活概率
        training: bool - 是否训练模式
        inplace: bool - 是否原地操作
        
    返回:
        Tensor - dropout后的张量
        
    示例:
        >>> x = genesis.randn(100)
        >>> # 训练时应用dropout
        >>> y = F.dropout(x, p=0.2, training=True)
        >>> # 推理时不应用
        >>> y = F.dropout(x, p=0.2, training=False)
    """
```

### embedding
```python
def embedding(input: Tensor, weight: Tensor, padding_idx: Optional[int] = None) -> Tensor:
    """
    嵌入查找
    
    参数:
        input: Tensor - 索引张量
        weight: Tensor - 嵌入矩阵，shape: (num_embeddings, embedding_dim)
        padding_idx: int, optional - 填充索引
        
    返回:
        Tensor - 嵌入向量
        
    示例:
        >>> weight = genesis.randn(10000, 128)  # 10000个词，128维
        >>> indices = genesis.tensor([1, 2, 3, 4])
        >>> embeddings = F.embedding(indices, weight)  # shape: (4, 128)
    """
```

### one_hot
```python
def one_hot(x: Tensor, num_classes: int) -> Tensor:
    """
    One-hot编码
    
    参数:
        x: Tensor - 类别索引张量
        num_classes: int - 类别总数
        
    返回:
        Tensor - one-hot编码张量
        
    示例:
        >>> labels = genesis.tensor([0, 1, 2, 1])
        >>> one_hot = F.one_hot(labels, num_classes=3)
        >>> # [[1, 0, 0],
        >>> #  [0, 1, 0],
        >>> #  [0, 0, 1],
        >>> #  [0, 1, 0]]
    """
```

## 使用示例

### 构建前向传播
```python
import genesis.nn.functional as F

def forward(x, weight1, bias1, weight2, bias2):
    # 第一层
    x = F.matmul(x, weight1.T) + bias1
    x = F.relu(x)
    x = F.dropout(x, p=0.5, training=True)
    
    # 第二层
    x = F.matmul(x, weight2.T) + bias2
    x = F.softmax(x, dim=-1)
    
    return x
```

### 计算损失
```python
# 分类任务
logits = model(x)
loss = F.cross_entropy(logits, labels)

# 回归任务
predictions = model(x)
loss = F.mse_loss(predictions, targets)

# 自定义损失
log_probs = F.log_softmax(logits, dim=-1)
loss = F.nll_loss(log_probs, labels)
```

### 数据预处理
```python
# 归一化
x = F.layer_norm(x, x.shape[-1])

# 数据增强
x = F.dropout(x, p=0.1, training=True)

# 形状变换
x = F.flatten(x, start_dim=1)
x = F.reshape(x, batch_size, -1)
```

## 性能优化提示

1. **使用fused操作**：如`fused_layer_norm`比单独的操作更快
2. **避免小批量操作**：批量处理比逐个处理效率高
3. **使用原地操作**：`inplace=True`减少内存分配
4. **合并操作**：如使用`cross_entropy`而不是`softmax`+`nll_loss`
5. **注意数值稳定性**：使用`log_softmax`而不是`log(softmax())`

## 注意事项

- 函数式接口是无状态的，不保存参数
- 某些函数在训练和评估模式下行为不同（如dropout）
- 原地操作可能破坏梯度计算，谨慎使用
- 注意张量形状的广播规则
- GPU操作可能使用Triton优化内核