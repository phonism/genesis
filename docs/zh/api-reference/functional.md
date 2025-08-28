# 函数式API参考

Genesis操作的函数式接口 - 神经网络操作的无状态函数。

## 核心函数

### 激活函数
- `F.relu(x)` - ReLU激活
- `F.softmax(x, dim=-1)` - Softmax
- `F.gelu(x)` - GELU激活

### 损失函数
- `F.mse_loss(input, target)` - 均方误差
- `F.cross_entropy(input, target)` - 交叉熵损失

### 卷积操作
- `F.conv2d(input, weight, bias)` - 2D卷积
- `F.linear(input, weight, bias)` - 线性变换

### 张量操作函数

#### `sort(input, dim=-1, descending=False, stable=False)`
沿维度排序元素。

```python
values, indices = genesis.sort(tensor, dim=1, descending=False)
```

**参数:**
- `input`: 输入张量
- `dim`: 排序的维度
- `descending`: 如果为True，降序排序
- `stable`: 如果为True，稳定排序（保持相等元素的顺序）

**返回:** (值, 索引) 张量元组

#### `topk(input, k, dim=-1, largest=True, sorted=True)`
返回沿维度的k个最大/最小元素。

```python
values, indices = genesis.topk(tensor, k=3, dim=1, largest=True)
```

**参数:**
- `input`: 输入张量
- `k`: 返回的top值数量
- `dim`: 查找top-k值的维度
- `largest`: 如果为True，返回最大值；如果为False，返回最小值
- `sorted`: 如果为True，返回排序后的值

**返回:** (值, 索引) 张量元组

#### `argsort(input, dim=-1, descending=False)`
返回沿维度排序张量的索引。

```python
indices = genesis.argsort(tensor, dim=1, descending=False)
```

**参数:**
- `input`: 输入张量
- `dim`: 排序的维度
- `descending`: 如果为True，降序排序

**返回:** 索引张量

#### `gather(input, dim, index)`
沿由index指定的轴收集值。

```python
output = genesis.gather(tensor, dim=1, index=indices)
```

**参数:**
- `input`: 输入张量
- `dim`: 收集的维度
- `index`: 与输入具有相同维数的索引张量

**返回:** 包含收集值的张量

#### `scatter_add(input, dim, index, src)`
将src中的值添加到input在index指定位置处。

```python
genesis.scatter_add(tensor, dim=1, index=indices, src=values)
```

**参数:**
- `input`: 输入张量（就地修改）
- `dim`: 散布的维度
- `index`: 索引张量
- `src`: 包含要添加值的源张量

**返回:** 修改后的输入张量

#### `bincount(input, weights=None, minlength=0)`
统计整数张量中每个值的出现次数。

```python
counts = genesis.bincount(tensor, minlength=10)
```

**参数:**
- `input`: 1D整数张量
- `weights`: 可选的权重张量
- `minlength`: 输出的最小长度

**返回:** 包含计数的张量

### 实用函数

#### `allclose(input, other, rtol=1e-05, atol=1e-08, equal_nan=False)`
测试输入和其他张量的所有元素是否接近。

```python
result = genesis.allclose(tensor1, tensor2, rtol=1e-5, atol=1e-8)
```

**参数:**
- `input`: 第一个张量
- `other`: 第二个张量
- `rtol`: 相对容差
- `atol`: 绝对容差
- `equal_nan`: 是否将NaN值视为相等

**返回:** 布尔标量张量

### 张量验证函数

#### `isinf(input)`
测试每个元素是否为无穷大（正无穷或负无穷）。

```python
inf_mask = genesis.isinf(tensor)
```

**参数:**
- `input`: 输入张量

**返回:** 与输入形状相同的布尔张量

#### `isnan(input)`
测试每个元素是否为NaN（非数值）。

```python
nan_mask = genesis.isnan(tensor)
```

**参数:**
- `input`: 输入张量

**返回:** 与输入形状相同的布尔张量

#### `isfinite(input)`
测试每个元素是否为有限值（非无穷且非NaN）。

```python
finite_mask = genesis.isfinite(tensor)
```

**参数:**
- `input`: 输入张量

**返回:** 与输入形状相同的布尔张量

### 分布式训练函数

#### `genesis.distributed.init_process_group(backend, world_size, rank)`
初始化分布式进程组。

```python
import genesis.distributed as dist
dist.init_process_group(backend='nccl', world_size=2, rank=0)
```

**参数:**
- `backend`: 通信后端（'nccl' 用于GPU）
- `world_size`: 进程总数
- `rank`: 当前进程的rank

#### `genesis.distributed.DistributedDataParallel(model, device_ids=None)`
分布式数据并行包装器。

```python
ddp_model = dist.DistributedDataParallel(model, device_ids=[0])
```

**参数:**
- `model`: 要包装的模型
- `device_ids`: GPU设备ID列表

**返回:** DDP包装的模型

### 创建函数

#### `eye(n, m=None, device=None, dtype=genesis.float32)`
生成单位矩阵。

```python
identity = genesis.eye(5)  # 5x5单位矩阵
rect_matrix = genesis.eye(3, 5)  # 3x5矩阵
```

#### `ones_like(tensor, dtype=None, device=None)`
生成与输入形状相同的全1张量。

```python
ones_tensor = genesis.ones_like(input_tensor)
```

#### `from_numpy(array, device=None, dtype=None)`
从numpy数组创建张量。

```python
np_array = np.array([1, 2, 3])
tensor = genesis.from_numpy(np_array)
```

*此参考涵盖了主要的函数式操作。完整的API详情请参阅源码文档。*