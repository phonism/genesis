# 实用工具 (genesis.utils)

## 概述

`genesis.utils`模块为开发、调试和数据处理提供必要的实用工具。它包括性能分析工具、数据加载实用程序和帮助函数，以简化深度学习工作流程。

## 核心组件

### 性能分析
- 函数和方法执行时间跟踪
- 装饰器自动分析
- 性能分析和报告

### 数据加载
- 训练数据的数据集抽象
- 具有批处理和洗牌的DataLoader
- 支持映射式和可迭代数据集

## 性能分析工具

### `@profile` 装饰器

函数和类的自动性能分析。

```python
from genesis.utils import profile

@profile
def expensive_function(x):
    """
    分析此函数的执行时间和调用次数。
    """
    # 你的计算在这里
    return x * 2

@profile
class MyModel:
    """
    分析此类中的所有方法。
    """
    def forward(self, x):
        return x + 1
    
    def backward(self, grad):
        return grad
```

分析器自动跟踪：
- **调用次数**: 每个函数被调用的次数
- **总时间**: 累积执行时间
- **平均时间**: 每次调用的平均执行时间

#### 使用示例

```python
import genesis.utils as utils
import time

# 分析函数
@utils.profile
def matrix_multiply(a, b):
    """虚拟矩阵乘法。"""
    time.sleep(0.01)  # 模拟计算
    return a @ b

# 分析类
@utils.profile
class NeuralNetwork:
    def __init__(self):
        pass
    
    def forward(self, x):
        time.sleep(0.005)  # 模拟前向传播
        return x * 2
    
    def backward(self, grad):
        time.sleep(0.003)  # 模拟后向传播
        return grad

# 使用被分析的函数
model = NeuralNetwork()
for i in range(100):
    x = matrix_multiply([[1, 2]], [[3], [4]])
    y = model.forward(x)
    model.backward([1, 1])

# 分析数据在程序退出时自动打印
```

#### 分析数据格式

程序退出时，分析数据自动打印：

```
程序花费了 2.1456 秒!
__main__.matrix_multiply: 100次调用, 1.0234总秒数
__main__.NeuralNetwork.forward: 100次调用, 0.5123总秒数
__main__.NeuralNetwork.backward: 100次调用, 0.3089总秒数
```

### 手动分析

为了更精细的控制，你可以编程式地访问分析数据：

```python
from genesis.utils.profile import profile_data, print_profile_data

# 获取当前分析数据
current_data = profile_data.copy()
print(f"到目前为止的函数调用: {sum(data['calls'] for data in current_data.values())}")

# 手动打印分析摘要
print_profile_data()
```

## 数据加载

### `Dataset`

所有数据集的抽象基类。

```python
from genesis.utils.data import Dataset

class Dataset:
    """
    抽象数据集类。
    
    所有子类必须实现 __len__ 和 __getitem__。
    """
    
    def __len__(self) -> int:
        """
        返回数据集的大小。
        
        返回:
            数据集中样本的数量
        """
        raise NotImplementedError
    
    def __getitem__(self, idx: int):
        """
        按索引检索样本。
        
        参数:
            idx: 样本索引
            
        返回:
            给定索引处的数据样本
        """
        raise NotImplementedError
```

#### 自定义数据集示例

```python
import numpy as np
from genesis.utils.data import Dataset

class MNIST(Dataset):
    """MNIST数据集实现示例。"""
    
    def __init__(self, data_path, transform=None):
        """
        初始化MNIST数据集。
        
        参数:
            data_path: MNIST数据文件路径
            transform: 可选的数据变换函数
        """
        self.data = self._load_data(data_path)
        self.labels = self._load_labels(data_path)
        self.transform = transform
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        image = self.data[idx]
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label
    
    def _load_data(self, path):
        # 在这里加载你的数据
        return np.random.randn(10000, 28, 28)  # 虚拟数据
    
    def _load_labels(self, path):
        # 在这里加载你的标签
        return np.random.randint(0, 10, 10000)  # 虚拟标签
```

### `IterableDataset`

可迭代式数据集的基类。

```python
from genesis.utils.data import IterableDataset

class IterableDataset(Dataset):
    """
    可迭代数据集的基类。
    
    对于流式数据或随机访问不可行时很有用。
    """
    
    def __iter__(self):
        """
        返回数据集的迭代器。
        
        返回:
            产生数据样本的迭代器
        """
        raise NotImplementedError
```

#### 可迭代数据集示例

```python
import random
from genesis.utils.data import IterableDataset

class RandomDataStream(IterableDataset):
    """流式数据集示例。"""
    
    def __init__(self, num_samples, feature_dim):
        """
        初始化流式数据集。
        
        参数:
            num_samples: 要生成的样本数
            feature_dim: 每个样本的维度
        """
        self.num_samples = num_samples
        self.feature_dim = feature_dim
    
    def __iter__(self):
        """即时生成随机样本。"""
        for _ in range(self.num_samples):
            # 生成随机数据
            data = [random.random() for _ in range(self.feature_dim)]
            label = random.randint(0, 9)
            yield data, label
```

### `DataLoader`

具有批处理和洗牌的高效数据加载。

```python
from genesis.utils.data import DataLoader

class DataLoader:
    """
    用于批处理和洗牌数据集的数据加载器。
    
    参数:
        dataset: 数据集实例（Dataset或IterableDataset）
        batch_size: 每批样本数（默认: 1）
        shuffle: 是否在每个epoch洗牌数据（默认: False）
    """
    
    def __init__(
        self, 
        dataset, 
        batch_size: int = 1, 
        shuffle: bool = False
    ):
```

#### DataLoader示例

```python
from genesis.utils.data import Dataset, DataLoader
import numpy as np

# 创建简单数据集
class SimpleDataset(Dataset):
    def __init__(self, size):
        self.data = np.random.randn(size, 10)
        self.labels = np.random.randint(0, 2, size)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

# 创建数据集和数据加载器
dataset = SimpleDataset(1000)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# 训练循环
for epoch in range(5):
    print(f"Epoch {epoch + 1}")
    for batch_idx, batch in enumerate(dataloader):
        # batch是(data, label)元组的列表
        batch_data = [item[0] for item in batch]
        batch_labels = [item[1] for item in batch]
        
        # 如需要转换为数组
        batch_data = np.array(batch_data)
        batch_labels = np.array(batch_labels)
        
        print(f"  Batch {batch_idx}: data shape {batch_data.shape}")
        
        # 你的训练代码在这里
        pass
```

#### 高级DataLoader使用

```python
# 带洗牌的大数据集
large_dataset = SimpleDataset(50000)
train_loader = DataLoader(large_dataset, batch_size=128, shuffle=True)

# 可迭代数据集
stream_dataset = RandomDataStream(1000, 20)
stream_loader = DataLoader(stream_dataset, batch_size=16)

# 调试用小批量
debug_loader = DataLoader(dataset, batch_size=4, shuffle=False)

# 多数据加载器的训练循环
def train_model(model, train_loader, val_loader):
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        for batch in train_loader:
            # 训练代码
            pass
        
        # 验证阶段
        model.eval()
        for batch in val_loader:
            # 验证代码
            pass
```

## 与Genesis训练的集成

### 完整训练示例

```python
import genesis
import genesis.nn as nn
import genesis.optim as optim
from genesis.utils.data import Dataset, DataLoader
from genesis.utils import profile
import numpy as np

# 自定义数据集
class TrainingDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# 被分析的模型
@profile
class SimpleModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# 生成虚拟数据
X = np.random.randn(1000, 20).astype(np.float32)
y = np.random.randint(0, 3, 1000)

# 创建数据集和数据加载器
dataset = TrainingDataset(X, y)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# 模型和优化器
model = SimpleModel(20, 64, 3)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# 带分析的训练循环
@profile
def train_epoch(model, dataloader, optimizer, criterion):
    """训练一个epoch。"""
    total_loss = 0.0
    for batch in dataloader:
        # 提取批数据
        batch_x = [item[0] for item in batch]
        batch_y = [item[1] for item in batch]
        
        # 转换为Genesis张量
        x = genesis.tensor(batch_x)
        y = genesis.tensor(batch_y)
        
        # 前向传播
        optimizer.zero_grad()
        outputs = model(x)
        loss = criterion(outputs, y)
        
        # 后向传播
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)

# 训练模型
for epoch in range(10):
    avg_loss = train_epoch(model, dataloader, optimizer, criterion)
    print(f"Epoch {epoch + 1}, 平均损失: {avg_loss:.4f}")

# 分析数据将在程序退出时自动打印
```

## 最佳实践

### 分析指南

1. **用于开发**: 在开发过程中启用分析以识别瓶颈
2. **生产环境禁用**: 在生产代码中删除分析装饰器
3. **选择性分析**: 只分析你怀疑慢的函数
4. **批量分析**: 分析整个训练循环而不是单个操作

### 数据加载指南

1. **适当的批大小**: 平衡内存使用和训练效率
2. **洗牌训练数据**: 在epochs之间始终洗牌训练数据
3. **不洗牌验证**: 保持验证数据一致的顺序
4. **内存考虑**: 对非常大的数据集使用可迭代数据集
5. **数据预处理**: 在数据集的`__getitem__`方法中应用变换

## 性能提示

### 高效数据加载

```python
# 好：高效批处理
class EfficientDataset(Dataset):
    def __init__(self, data):
        # 预处理数据一次
        self.data = self._preprocess(data)
    
    def _preprocess(self, data):
        # 昂贵的预处理只做一次
        return data * 2 + 1
    
    def __getitem__(self, idx):
        # 快速访问
        return self.data[idx]

# 好：尽可能使用大批大小
train_loader = DataLoader(dataset, batch_size=128, shuffle=True)

# 好：使用适当的数据类型
data = np.array(data, dtype=np.float32)  # 使用float32而不是float64
```

### 内存管理

```python
# 好：完成后删除大对象
del large_dataset
del temporary_data

# 好：对大数据集使用生成器
def data_generator():
    for file in file_list:
        data = load_file(file)
        yield data

# 好：如需要用较小批量限制内存使用
small_batch_loader = DataLoader(dataset, batch_size=16)
```

## 另请参阅

- [神经网络模块](../nn/modules.md) - 构建模型
- [优化器](../optim/optimizers.md) - 训练算法  
- [自动微分](../autograd.md) - 自动微分
- [性能指南](../../performance/) - 优化技术