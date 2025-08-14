# Your First Complete Program

After completing the installation, let's learn Genesis basics through a complete example. We will implement an image classifier to demonstrate the complete deep learning workflow.

## 🎯 Project Goals

Build a handwritten digit recognizer (MNIST-like) to learn Genesis core concepts:

- Data loading and preprocessing
- Model definition and initialization
- Training loop and validation
- Model saving and loading

## 📊 Project Structure

Create the project directory structure:

```
first_project/
├── data/                # Data directory
├── models/              # Model save directory
├── train.py            # Training script
├── model.py            # Model definition
├── dataset.py          # Data loading
└── utils.py            # Utility functions
```

## 📁 1. Data Processing (`dataset.py`)

```python
"""数据加载和预处理模块"""
import genesis
import numpy as np
from typing import Tuple, List
import pickle
import os

class SimpleDataset:
    """简单的数据集类"""
    
    def __init__(self, data: np.ndarray, labels: np.ndarray, transform=None):
        """
        初始化数据集
        
        Args:
            data: 输入数据 (N, H, W) 或 (N, D)
            labels: 标签 (N,)
            transform: 数据变换函数
        """
        self.data = data.astype(np.float32)
        self.labels = labels.astype(np.int64)
        self.transform = transform
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Tuple[genesis.Tensor, genesis.Tensor]:
        """获取单个样本"""
        x = self.data[idx]
        y = self.labels[idx]
        
        if self.transform:
            x = self.transform(x)
        
        return genesis.tensor(x), genesis.tensor(y)

class DataLoader:
    """简单的数据加载器"""
    
    def __init__(self, dataset: SimpleDataset, batch_size: int = 32, shuffle: bool = True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self._reset_indices()
    
    def _reset_indices(self):
        """重置索引"""
        self.indices = np.arange(len(self.dataset))
        if self.shuffle:
            np.random.shuffle(self.indices)
        self.current = 0
    
    def __iter__(self):
        self._reset_indices()
        return self
    
    def __next__(self):
        if self.current >= len(self.dataset):
            raise StopIteration
        
        # 获取当前批次的索引
        end_idx = min(self.current + self.batch_size, len(self.dataset))
        batch_indices = self.indices[self.current:end_idx]
        
        # 收集批次数据
        batch_data = []
        batch_labels = []
        
        for idx in batch_indices:
            x, y = self.dataset[idx]
            batch_data.append(x)
            batch_labels.append(y)
        
        self.current = end_idx
        
        # 堆叠成批次
        batch_x = genesis.stack(batch_data, dim=0)
        batch_y = genesis.stack(batch_labels, dim=0)
        
        return batch_x, batch_y

def create_synthetic_data(n_samples: int = 1000, n_features: int = 784, n_classes: int = 10) -> Tuple[np.ndarray, np.ndarray]:
    """创建合成数据用于演示"""
    np.random.seed(42)
    
    # 生成随机数据
    data = np.random.randn(n_samples, n_features).astype(np.float32)
    
    # 为每个类别添加一些模式
    labels = np.random.randint(0, n_classes, n_samples)
    for i in range(n_classes):
        mask = labels == i
        # 给每个类别添加特定的偏置
        data[mask] += np.random.randn(n_features) * 0.5
    
    return data, labels

def load_data() -> Tuple[DataLoader, DataLoader]:
    """加载训练和验证数据"""
    print("🔄 加载数据...")
    
    # 创建合成数据 (实际项目中替换为真实数据)
    train_data, train_labels = create_synthetic_data(800, 784, 10)
    val_data, val_labels = create_synthetic_data(200, 784, 10)
    
    # 数据标准化
    def normalize(x):
        return (x - x.mean()) / (x.std() + 1e-8)
    
    # 创建数据集
    train_dataset = SimpleDataset(train_data, train_labels, transform=normalize)
    val_dataset = SimpleDataset(val_data, val_labels, transform=normalize)
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    print(f"✅ 数据加载完成 - 训练集: {len(train_dataset)}, 验证集: {len(val_dataset)}")
    
    return train_loader, val_loader
```

## 🧠 2. 模型定义 (`model.py`)

```python
"""神经网络模型定义"""
import genesis
import genesis.nn as nn
import genesis.nn.functional as F

class MLP(nn.Module):
    """多层感知机分类器"""
    
    def __init__(self, input_dim: int = 784, hidden_dims: list = None, num_classes: int = 10, dropout_rate: float = 0.2):
        """
        初始化MLP模型
        
        Args:
            input_dim: 输入维度
            hidden_dims: 隐藏层维度列表
            num_classes: 分类数量
            dropout_rate: Dropout比率
        """
        super().__init__()
        
        if hidden_dims is None:
            hidden_dims = [512, 256, 128]
        
        # 构建网络层
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim
        
        # 输出层
        layers.append(nn.Linear(prev_dim, num_classes))
        
        self.network = nn.Sequential(*layers)
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """权重初始化"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Xavier初始化
                std = (2.0 / (module.in_features + module.out_features)) ** 0.5
                module.weight.data.normal_(0, std)
                if module.bias is not None:
                    module.bias.data.zero_()
    
    def forward(self, x: genesis.Tensor) -> genesis.Tensor:
        """前向传播"""
        # 展平输入 (如果是图像数据)
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        
        return self.network(x)

class CNN(nn.Module):
    """卷积神经网络分类器 (如果需要处理图像)"""
    
    def __init__(self, num_classes: int = 10):
        super().__init__()
        
        # 卷积层
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        # 池化层
        self.pool = nn.MaxPool2d(2, 2)
        
        # 全连接层
        self.fc1 = nn.Linear(128 * 3 * 3, 512)  # 假设输入是28x28
        self.fc2 = nn.Linear(512, num_classes)
        
        # Dropout
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x: genesis.Tensor) -> genesis.Tensor:
        # 卷积 + 池化
        x = self.pool(F.relu(self.conv1(x)))  # 28x28 -> 14x14
        x = self.pool(F.relu(self.conv2(x)))  # 14x14 -> 7x7  
        x = self.pool(F.relu(self.conv3(x)))  # 7x7 -> 3x3
        
        # 展平
        x = x.view(x.size(0), -1)
        
        # 全连接层
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

def create_model(model_type: str = "mlp", **kwargs) -> nn.Module:
    """工厂函数：创建模型"""
    if model_type.lower() == "mlp":
        return MLP(**kwargs)
    elif model_type.lower() == "cnn":
        return CNN(**kwargs)
    else:
        raise ValueError(f"未知的模型类型: {model_type}")
```

## 🛠️ 3. 工具函数 (`utils.py`)

```python
"""工具函数模块"""
import genesis
import time
import os
from typing import Dict, Any
import json

class AverageMeter:
    """平均值计算器"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class Timer:
    """计时器"""
    
    def __init__(self):
        self.start_time = None
        self.end_time = None
    
    def start(self):
        self.start_time = time.time()
    
    def stop(self):
        self.end_time = time.time()
        return self.end_time - self.start_time
    
    def elapsed(self):
        if self.start_time is None:
            return 0
        return time.time() - self.start_time

def accuracy(output: genesis.Tensor, target: genesis.Tensor, topk: tuple = (1,)) -> list:
    """计算准确率"""
    with genesis.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        
        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        
        return res

def save_checkpoint(model: genesis.nn.Module, optimizer: genesis.optim.Optimizer, 
                   epoch: int, loss: float, accuracy: float, filepath: str):
    """保存检查点"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'accuracy': accuracy
    }
    
    # 确保目录存在
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    genesis.save(checkpoint, filepath)
    print(f"💾 检查点已保存: {filepath}")

def load_checkpoint(filepath: str, model: genesis.nn.Module, optimizer: genesis.optim.Optimizer = None) -> Dict[str, Any]:
    """加载检查点"""
    checkpoint = genesis.load(filepath)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    print(f"📁 检查点已加载: {filepath}")
    print(f"   轮次: {checkpoint['epoch']}, 损失: {checkpoint['loss']:.4f}, 准确率: {checkpoint['accuracy']:.2f}%")
    
    return checkpoint

def save_training_history(history: Dict[str, list], filepath: str):
    """保存训练历史"""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    with open(filepath, 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f"📊 训练历史已保存: {filepath}")

def print_model_summary(model: genesis.nn.Module, input_shape: tuple):
    """打印模型摘要"""
    print("🏗️  模型架构:")
    print("=" * 50)
    
    # 计算参数数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"总参数量: {total_params:,}")
    print(f"可训练参数: {trainable_params:,}")
    print(f"输入形状: {input_shape}")
    
    # 测试前向传播
    dummy_input = genesis.randn(*input_shape)
    try:
        with genesis.no_grad():
            output = model(dummy_input)
        print(f"输出形状: {output.shape}")
    except Exception as e:
        print(f"前向传播测试失败: {e}")
    
    print("=" * 50)
```

## 🚂 4. 训练脚本 (`train.py`)

<function_calls>
<invoke name="TodoWrite">
<parameter name="todos">[{"id": "1", "content": "\u521b\u5efa\u6587\u6863\u76ee\u5f55\u7ed3\u6784\u548c\u914d\u7f6e\u6587\u4ef6", "status": "completed"}, {"id": "2", "content": "\u7f16\u5199\u9996\u9875\u548c\u5feb\u901f\u5f00\u59cb\u6587\u6863", "status": "completed"}, {"id": "3", "content": "\u7f16\u5199\u67b6\u6784\u8bbe\u8ba1\u6587\u6863", "status": "pending"}, {"id": "4", "content": "\u7f16\u5199\u6838\u5fc3\u7ec4\u4ef6\u6587\u6863", "status": "pending"}, {"id": "5", "content": "\u7f16\u5199\u795e\u7ecf\u7f51\u7edc\u6a21\u5757\u6587\u6863", "status": "pending"}, {"id": "6", "content": "\u7f16\u5199API\u53c2\u8003\u6587\u6863", "status": "pending"}]