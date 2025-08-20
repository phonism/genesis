# 基础训练教程

本教程将带你从零开始，使用Genesis深度学习框架构建和训练你的第一个神经网络。我们将通过一个完整的图像分类项目来学习Genesis的核心概念和用法。

## 🎯 学习目标

通过本教程，你将学会：
- Genesis的基本API和数据结构
- 如何定义和训练神经网络模型
- 数据加载和预处理
- 训练循环的构建和优化
- 模型评估和保存

## 🛠️ 环境准备

### 安装依赖

```bash
# 确保已安装Genesis
pip install torch triton numpy matplotlib tqdm
git clone https://github.com/phonism/genesis.git
cd genesis
pip install -e .
```

### 验证安装

```python
import genesis
import genesis.nn as nn
import genesis.optim as optim

# 测试基本功能
x = genesis.randn(2, 3)
print(f"Genesis张量已创建: {x.shape}")
print(f"Genesis模块可用: {dir(nn)}")
```

## 📊 项目：手写数字识别

我们将构建一个手写数字识别系统，使用简单的全连接神经网络和合成数据来演示Genesis的功能。

### 1. 数据准备

由于Genesis还没有内置的数据加载工具，我们将创建模仿MNIST结构的合成数据：

```python
import genesis
import genesis.nn as nn
import genesis.optim as optim
import numpy as np
import matplotlib.pyplot as plt

class SimpleDataset:
    """演示用的简单数据集类"""
    
    def __init__(self, num_samples=1000, input_dim=784, num_classes=10):
        # 生成类似展平MNIST的合成数据
        self.data = genesis.randn(num_samples, input_dim)
        
        # 基于数据模式创建标签（合成）
        labels = genesis.randn(num_samples, num_classes)
        self.labels = genesis.functional.max(labels, axis=1, keepdims=False)
        
        self.num_samples = num_samples
        
    def __len__(self):
        return self.num_samples
    
    def get_batch(self, batch_size=32, start_idx=0):
        """获取一批数据"""
        end_idx = min(start_idx + batch_size, self.num_samples)
        return (self.data[start_idx:end_idx], 
                self.labels[start_idx:end_idx])

# 创建数据集
train_dataset = SimpleDataset(num_samples=800, input_dim=784, num_classes=10)
test_dataset = SimpleDataset(num_samples=200, input_dim=784, num_classes=10)

print(f"训练集大小: {len(train_dataset)}")
print(f"测试集大小: {len(test_dataset)}")
print(f"输入维度: 784 (28x28展平)")
print(f"类别数量: 10")
```

### 2. 模型定义

我们将使用Genesis模块构建一个简单但有效的全连接神经网络：

```python
class MNISTNet(nn.Module):
    """数字识别的简单全连接网络"""
    
    def __init__(self, input_dim=784, hidden_dim=128, num_classes=10):
        super(MNISTNet, self).__init__()
        
        # 使用实际的Genesis模块定义层
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, num_classes)
        
        # 激活函数和正则化
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        # 如果需要，展平输入
        if len(x.shape) > 2:
            x = x.view(x.shape[0], -1)
        
        # 第一个隐藏层
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        # 第二个隐藏层
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        # 输出层
        x = self.fc3(x)
        
        return x

# 创建模型实例
model = MNISTNet(input_dim=784, hidden_dim=128, num_classes=10)

print("模型结构:")
print(f"层1: {model.fc1}")
print(f"层2: {model.fc2}")
print(f"层3: {model.fc3}")
print(f"参数总数: {sum(p.data.size for p in model.parameters())}")
```

### 3. 损失函数和优化器

```python
# 使用Genesis定义损失函数和优化器
criterion = nn.SoftmaxLoss()  # 使用Genesis的SoftmaxLoss
optimizer = optim.Adam(model.parameters(), lr=0.001)

print(f"损失函数: {criterion}")
print(f"优化器: {optimizer}")
print(f"学习率: 0.001")
```

### 4. 训练循环

```python
def train_epoch(model, dataset, criterion, optimizer, batch_size=32):
    """训练一个epoch"""
    model.train()  # 设置为训练模式
    
    total_loss = 0.0
    num_batches = len(dataset) // batch_size
    
    for i in range(num_batches):
        # 获取批数据
        start_idx = i * batch_size
        batch_data, batch_labels = dataset.get_batch(batch_size, start_idx)
        
        # 前向传播
        outputs = model(batch_data)
        loss = criterion(outputs, batch_labels)
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        
        # 应用梯度裁剪（可选）
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # 更新权重
        optimizer.step()
        
        total_loss += loss.data.item() if hasattr(loss.data, 'item') else float(loss.data)
    
    return total_loss / num_batches

def evaluate(model, dataset, criterion, batch_size=32):
    """评估模型性能"""
    model.eval()  # 设置为评估模式
    
    total_loss = 0.0
    correct = 0
    total = 0
    num_batches = len(dataset) // batch_size
    
    for i in range(num_batches):
        start_idx = i * batch_size
        batch_data, batch_labels = dataset.get_batch(batch_size, start_idx)
        
        # 前向传播（不需要梯度）
        outputs = model(batch_data)
        loss = criterion(outputs, batch_labels)
        
        # 计算准确率
        predicted = genesis.functional.max(outputs, axis=1, keepdims=False)
        total += batch_labels.shape[0]
        correct += (predicted == batch_labels).sum().data
        
        total_loss += loss.data.item() if hasattr(loss.data, 'item') else float(loss.data)
    
    accuracy = correct / total
    avg_loss = total_loss / num_batches
    
    return avg_loss, accuracy

# 训练配置
num_epochs = 10
batch_size = 32

print("开始训练...")
print(f"轮数: {num_epochs}")
print(f"批量大小: {batch_size}")
print("-" * 50)

# 训练循环
train_losses = []
test_losses = []
test_accuracies = []

for epoch in range(num_epochs):
    # 训练一个epoch
    train_loss = train_epoch(model, train_dataset, criterion, optimizer, batch_size)
    
    # 在测试集上评估
    test_loss, test_accuracy = evaluate(model, test_dataset, criterion, batch_size)
    
    # 记录指标
    train_losses.append(train_loss)
    test_losses.append(test_loss)
    test_accuracies.append(test_accuracy)
    
    # 打印进度
    print(f"Epoch {epoch+1}/{num_epochs}")
    print(f"  训练损失: {train_loss:.4f}")
    print(f"  测试损失: {test_loss:.4f}")
    print(f"  测试准确率: {test_accuracy:.4f}")
    print("-" * 30)

print("训练完成！")
```

### 5. 模型评估和可视化

```python
# 绘制训练进度
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 4))

# 绘制损失
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='训练损失')
plt.plot(test_losses, label='测试损失')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('训练和测试损失')
plt.legend()
plt.grid(True)

# 绘制准确率
plt.subplot(1, 2, 2)
plt.plot(test_accuracies, label='测试准确率')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('测试准确率')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# 最终评估
final_test_loss, final_test_accuracy = evaluate(model, test_dataset, criterion, batch_size)
print(f"\n最终结果:")
print(f"测试损失: {final_test_loss:.4f}")
print(f"测试准确率: {final_test_accuracy:.4f}")
```

### 6. 模型保存和加载

```python
# 使用Genesis序列化保存模型
model_path = "mnist_model.pkl"
genesis.save(model.state_dict(), model_path)
print(f"模型已保存到 {model_path}")

# 加载模型
model_new = MNISTNet(input_dim=784, hidden_dim=128, num_classes=10)
model_new.load_state_dict(genesis.load(model_path))
print("模型加载成功！")

# 验证加载的模型是否工作
test_loss, test_accuracy = evaluate(model_new, test_dataset, criterion, batch_size)
print(f"加载模型的准确率: {test_accuracy:.4f}")
```

## 🎓 学到的关键概念

### 1. Genesis张量操作
- 使用`genesis.randn()`, `genesis.tensor()`创建张量
- 基本操作如矩阵乘法和逐元素操作
- 使用`requires_grad`进行自动微分

### 2. 神经网络模块
- 通过继承`nn.Module`定义模型
- 使用内置层：`nn.Linear`, `nn.ReLU`, `nn.Dropout`
- 理解前向传播实现

### 3. 训练过程
- 设置损失函数和优化器
- 实现训练和评估循环
- 使用梯度裁剪和正则化

### 4. 模型管理
- 使用Genesis序列化保存和加载模型状态
- 管理模型参数和优化状态

## 🚀 下一步

完成本教程后，你可以：

1. **探索更复杂的模型** - 尝试具有更多层的不同架构
2. **学习高级特性** - 探索混合精度训练和学习率调度
3. **处理真实数据** - 当数据加载工具可用时与实际数据集集成
4. **性能优化** - 了解GPU加速和Triton内核使用

## 📚 其他资源

- [Genesis API参考](../api-reference/index.md) - 完整的API文档
- [高级训练特性](../training/advanced-features.md) - 混合精度、调度器等
- [性能优化](performance-tuning.md) - 更快训练的技巧

## 🐛 故障排除

### 常见问题

1. **导入错误**：确保使用`pip install -e .`正确安装Genesis
2. **形状不匹配**：检查前向传播中的张量维度
3. **内存问题**：如果遇到内存不足错误，减少批量大小
4. **训练缓慢**：在可用时启用GPU支持

### 获取帮助

- 查看[Genesis文档](../index.md)
- 在[GitHub Issues](https://github.com/phonism/genesis/issues)报告问题
- 在社区论坛加入讨论