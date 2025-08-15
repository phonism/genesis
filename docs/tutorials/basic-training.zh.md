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
pip install torch triton
git clone https://github.com/phonism/genesis.git
cd genesis
pip install -e .

# 安装额外依赖
pip install matplotlib torchvision tqdm
```

### 验证安装

```python
import genesis
import genesis.nn as nn
import genesis.optim as optim

print(f"Genesis版本: {genesis.__version__}")
print(f"CUDA可用: {genesis.cuda.is_available()}")
```

## 📊 项目：手写数字识别

我们将构建一个手写数字识别系统，使用经典的MNIST数据集。

### 1. 数据准备

```python
import genesis
import genesis.nn as nn
import genesis.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np

# 数据预处理
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# 加载MNIST数据集
train_dataset = datasets.MNIST('data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST('data', train=False, transform=transform)

# 创建数据加载器
batch_size = 64
train_loader = genesis.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = genesis.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

print(f"训练集大小: {len(train_dataset)}")
print(f"测试集大小: {len(test_dataset)}")
```

### 2. 模型定义

我们将构建一个简单但有效的卷积神经网络：

```python
class MNISTNet(nn.Module):
    """MNIST手写数字识别网络"""
    
    def __init__(self, num_classes=10):
        super(MNISTNet, self).__init__()
        
        # 卷积层
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        
        # 全连接层
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, num_classes)
        
        # 激活函数和Dropout
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        # 卷积块1
        x = self.pool(self.relu(self.conv1(x)))  # 28x28 -> 14x14
        
        # 卷积块2  
        x = self.pool(self.relu(self.conv2(x)))  # 14x14 -> 7x7
        
        # 展平
        x = x.view(x.size(0), -1)  # [batch_size, 64*7*7]
        
        # 全连接层
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

# 创建模型实例
device = genesis.device('cuda' if genesis.cuda.is_available() else 'cpu')
model = MNISTNet().to(device)

print("模型结构:")
print(model)
print(f"\\n参数总数: {sum(p.numel() for p in model.parameters()):,}")
```

### 3. 训练配置

```python
# 损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 学习率调度器
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.8)

# 训练参数
num_epochs = 10
print_every = 100  # 每100个batch打印一次

print(f"设备: {device}")
print(f"批量大小: {batch_size}")
print(f"训练轮数: {num_epochs}")
print(f"学习率: {optimizer.param_groups[0]['lr']}")
```

### 4. 训练循环

```python
def train_epoch(model, train_loader, criterion, optimizer, epoch):
    """训练一个epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        # 数据移到设备
        data, target = data.to(device), target.to(device)
        
        # 前向传播
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        
        # 反向传播
        loss.backward()
        optimizer.step()
        
        # 统计
        running_loss += loss.item()
        _, predicted = genesis.max(output, dim=1)
        total += target.size(0)
        correct += (predicted == target).sum().item()
        
        # 打印进度
        if batch_idx % print_every == 0:
            print(f'Epoch {epoch+1}/{num_epochs}, '
                  f'Batch {batch_idx}/{len(train_loader)}, '
                  f'Loss: {loss.item():.4f}, '
                  f'Acc: {100*correct/total:.2f}%')
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100 * correct / total
    
    return epoch_loss, epoch_acc

def validate(model, test_loader, criterion):
    """验证模型"""
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    
    with genesis.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            
            test_loss += criterion(output, target).item()
            _, predicted = genesis.max(output, dim=1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    
    avg_loss = test_loss / len(test_loader)
    accuracy = 100 * correct / total
    
    return avg_loss, accuracy

# 开始训练
print("开始训练...")
train_losses, train_accs = [], []
val_losses, val_accs = [], []

for epoch in range(num_epochs):
    # 训练
    train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, epoch)
    
    # 验证
    val_loss, val_acc = validate(model, test_loader, criterion)
    
    # 学习率调度
    scheduler.step()
    
    # 记录结果
    train_losses.append(train_loss)
    train_accs.append(train_acc)
    val_losses.append(val_loss)
    val_accs.append(val_acc)
    
    print(f"Epoch {epoch+1}/{num_epochs}:")
    print(f"  训练 - Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%")
    print(f"  验证 - Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%")
    print(f"  学习率: {optimizer.param_groups[0]['lr']:.6f}")
    print("-" * 50)

print("训练完成！")
```

### 5. 结果可视化

```python
# 绘制训练曲线
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

# 损失曲线
ax1.plot(train_losses, label='训练损失', color='blue')
ax1.plot(val_losses, label='验证损失', color='red')
ax1.set_title('损失曲线')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.legend()
ax1.grid(True)

# 准确率曲线
ax2.plot(train_accs, label='训练准确率', color='blue')
ax2.plot(val_accs, label='验证准确率', color='red')
ax2.set_title('准确率曲线')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Accuracy (%)')
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.show()

print(f"最终测试准确率: {val_accs[-1]:.2f}%")
```

### 6. 模型保存和加载

```python
# 保存模型
model_path = 'mnist_model.pth'
genesis.save_checkpoint({
    'epoch': num_epochs,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'train_loss': train_losses[-1],
    'val_loss': val_losses[-1],
    'val_acc': val_accs[-1]
}, model_path)

print(f"模型已保存到: {model_path}")

# 加载模型
def load_model(model_path, model_class, num_classes=10):
    """加载训练好的模型"""
    checkpoint = genesis.load_checkpoint(model_path)
    
    model = model_class(num_classes)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"模型加载成功，验证准确率: {checkpoint['val_acc']:.2f}%")
    return model

# 测试加载
loaded_model = load_model(model_path, MNISTNet)
```

### 7. 单张图片预测

```python
def predict_single_image(model, image, class_names=None):
    """对单张图片进行预测"""
    model.eval()
    
    if class_names is None:
        class_names = [str(i) for i in range(10)]
    
    with genesis.no_grad():
        if image.dim() == 3:  # 添加batch维度
            image = image.unsqueeze(0)
        
        image = image.to(device)
        output = model(image)
        probabilities = genesis.softmax(output, dim=1)
        
        confidence, predicted = genesis.max(probabilities, dim=1)
        
    return predicted.item(), confidence.item()

# 测试预测
test_iter = iter(test_loader)
images, labels = next(test_iter)

# 预测前5张图片
fig, axes = plt.subplots(1, 5, figsize=(15, 3))
for i in range(5):
    image = images[i]
    true_label = labels[i].item()
    
    predicted, confidence = predict_single_image(model, image)
    
    # 显示图片
    axes[i].imshow(image.squeeze(), cmap='gray')
    axes[i].set_title(f'真实: {true_label}\\n预测: {predicted}\\n置信度: {confidence:.3f}')
    axes[i].axis('off')

plt.tight_layout()
plt.show()
```

## 📈 性能对比

让我们比较Genesis与PyTorch的性能：

```python
import time

def benchmark_training(model, train_loader, criterion, optimizer, device, num_batches=100):
    """训练性能基准测试"""
    model.train()
    start_time = time.time()
    
    for batch_idx, (data, target) in enumerate(train_loader):
        if batch_idx >= num_batches:
            break
            
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
    
    elapsed_time = time.time() - start_time
    return elapsed_time

# 运行基准测试
print("性能基准测试 (100个batch):")
genesis_time = benchmark_training(model, train_loader, criterion, optimizer, device)
print(f"Genesis训练时间: {genesis_time:.2f} 秒")
print(f"平均每个batch: {genesis_time/100*1000:.1f} ms")
```

## 🎯 关键概念总结

### 1. 张量操作
```python
# 创建张量
x = genesis.randn(3, 4, requires_grad=True)
y = genesis.ones(3, 4)

# 基础运算
z = x + y
w = genesis.matmul(x, y.T)

# 梯度计算
z.sum().backward()
print(x.grad)  # x的梯度
```

### 2. 模型定义最佳实践
```python
class BestPracticeNet(nn.Module):
    def __init__(self):
        super().__init__()
        # 使用nn.Sequential简化定义
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, 10)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)
```

### 3. 训练技巧
```python
# 梯度裁剪
genesis.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

# 权重初始化
def init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        genesis.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            genesis.nn.init.zeros_(m.bias)

model.apply(init_weights)
```

## 🚀 下一步

恭喜！你已经完成了第一个Genesis训练项目。接下来可以探索：

1. **[混合精度训练](amp-training.zh.md)** - 加速训练并节省显存
2. **[自定义算子](custom-ops.zh.md)** - 实现专用的神经网络操作
3. **[性能调优](performance-tuning.zh.md)** - 优化训练性能
4. **[分布式训练](distributed-training.zh.md)** - 多GPU并行训练

## ❓ 常见问题

**Q: 训练速度比预期慢？**
A: 检查是否启用了CUDA，确保数据预处理不是瓶颈，考虑调整batch_size。

**Q: 内存不足错误？**
A: 减小batch_size，启用梯度检查点，或使用混合精度训练。

**Q: 模型不收敛？**
A: 检查学习率设置，确认数据预处理正确，尝试不同的初始化方法。

---

!!! success "完成了基础教程！"
    你现在已经掌握了Genesis的核心概念。继续探索更高级的特性吧！

[下一教程：自定义算子](custom-ops.zh.md){ .md-button .md-button--primary }
[返回教程目录](index.zh.md){ .md-button }