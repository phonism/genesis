# 模型序列化与检查点

Genesis提供了强大的模型序列化和检查点功能，用于保存和加载模型状态、优化器状态和训练进度。这对于长时间训练、模型部署和实验可重现性至关重要。

## 概述

Genesis中的序列化系统处理：
- 模型状态字典（参数和缓冲区）
- 优化器状态（动量、运行平均值等）
- 训练元数据（epoch、损失、指标）
- 原子写操作和安全备份

## 核心函数

### save()
```python
import genesis

def save(state_dict, file_path):
    """
    将状态字典保存到文件，使用原子写操作。
    
    Args:
        state_dict (dict): 包含要保存状态的字典
        file_path (str): 保存文件的路径
        
    Features:
        - 原子写操作与备份
        - 成功时自动清理
        - 失败时回滚
        - 保存后内存清理
    """
```

### load()
```python
def load(file_path):
    """
    从文件加载状态字典。
    
    Args:
        file_path (str): 保存文件的路径
        
    Returns:
        dict: 加载的状态字典
        
    Raises:
        FileNotFoundError: 如果文件不存在
        pickle.UnpicklingError: 如果文件损坏
    """
```

### save_checkpoint()
```python
def save_checkpoint(model_state_dict, optimizer_state_dict, file_path):
    """
    保存模型和优化器检查点。
    
    Args:
        model_state_dict (dict): 模型状态字典
        optimizer_state_dict (dict): 优化器状态字典
        file_path (str): 保存检查点的路径
        
    创建包含以下内容的检查点：
        - model_state_dict: 模型参数和缓冲区
        - optimizer_state_dict: 优化器状态
    """
```

### load_checkpoint()
```python
def load_checkpoint(file_path):
    """
    加载模型和优化器检查点。
    
    Args:
        file_path (str): 检查点文件路径
        
    Returns:
        tuple: (model_state_dict, optimizer_state_dict)
        
    Example:
        >>> model_state, optimizer_state = genesis.load_checkpoint('checkpoint.pth')
        >>> model.load_state_dict(model_state)
        >>> optimizer.load_state_dict(optimizer_state)
    """
```

## 基本用法

### 保存简单模型
```python
import genesis
import genesis.nn as nn

# 创建和训练模型
model = nn.Linear(784, 10)

# 保存模型状态
genesis.save(model.state_dict(), 'model.pth')

# 加载模型状态
state_dict = genesis.load('model.pth')
model.load_state_dict(state_dict)
```

### 训练检查点
```python
import genesis
import genesis.nn as nn
import genesis.optim as optim

# 设置模型和优化器
model = nn.Sequential(
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Linear(256, 10)
)

optimizer = optim.Adam(model.parameters(), lr=0.001)

# 带检查点的训练循环
for epoch in range(100):
    # 训练代码...
    train_loss = train_one_epoch(model, train_loader, optimizer)
    
    # 每10个epoch保存检查点
    if epoch % 10 == 0:
        genesis.save_checkpoint(
            model.state_dict(),
            optimizer.state_dict(), 
            f'checkpoint_epoch_{epoch}.pth'
        )
        print(f"检查点已保存在epoch {epoch}")

# 加载检查点恢复训练
model_state, optimizer_state = genesis.load_checkpoint('checkpoint_epoch_90.pth')
model.load_state_dict(model_state)
optimizer.load_state_dict(optimizer_state)
```

## 高级检查点

### 完整训练状态
```python
import genesis

def save_training_checkpoint(model, optimizer, scheduler, epoch, loss, metrics, file_path):
    """保存完整训练状态。"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'loss': loss,
        'metrics': metrics,
        'model_config': {
            'input_size': model.input_size,
            'hidden_size': model.hidden_size,
            'num_classes': model.num_classes
        }
    }
    genesis.save(checkpoint, file_path)

def load_training_checkpoint(file_path):
    """加载完整训练状态。"""
    return genesis.load(file_path)

# 用法
save_training_checkpoint(
    model, optimizer, scheduler, 
    epoch=50, loss=0.234, 
    metrics={'accuracy': 0.94, 'f1': 0.91},
    file_path='complete_checkpoint.pth'
)

# 恢复训练
checkpoint = load_training_checkpoint('complete_checkpoint.pth')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
if checkpoint['scheduler_state_dict']:
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

start_epoch = checkpoint['epoch'] + 1
print(f"从epoch {start_epoch}恢复训练，损失: {checkpoint['loss']}")
```

### 最佳模型跟踪
```python
import genesis

class ModelCheckpointer:
    def __init__(self, save_dir='checkpoints'):
        self.save_dir = save_dir
        self.best_loss = float('inf')
        self.best_accuracy = 0.0
        
    def save_checkpoint(self, model, optimizer, epoch, loss, accuracy, is_best=False):
        """保存检查点并跟踪最佳模型。"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            'accuracy': accuracy
        }
        
        # 保存常规检查点
        checkpoint_path = f'{self.save_dir}/checkpoint_epoch_{epoch}.pth'
        genesis.save(checkpoint, checkpoint_path)
        
        # 基于损失保存最佳模型
        if loss < self.best_loss:
            self.best_loss = loss
            best_loss_path = f'{self.save_dir}/best_loss_model.pth'
            genesis.save(checkpoint, best_loss_path)
            print(f"新的最佳损失: {loss:.4f}")
            
        # 基于准确率保存最佳模型
        if accuracy > self.best_accuracy:
            self.best_accuracy = accuracy
            best_acc_path = f'{self.save_dir}/best_accuracy_model.pth'
            genesis.save(checkpoint, best_acc_path)
            print(f"新的最佳准确率: {accuracy:.4f}")
    
    def load_best_model(self, model, metric='loss'):
        """加载最佳模型。"""
        if metric == 'loss':
            path = f'{self.save_dir}/best_loss_model.pth'
        elif metric == 'accuracy':
            path = f'{self.save_dir}/best_accuracy_model.pth'
        else:
            raise ValueError("metric必须是'loss'或'accuracy'")
            
        checkpoint = genesis.load(path)
        model.load_state_dict(checkpoint['model_state_dict'])
        return checkpoint

# 用法
checkpointer = ModelCheckpointer()

for epoch in range(100):
    train_loss = train_one_epoch(model, train_loader, optimizer)
    val_loss, val_accuracy = validate(model, val_loader)
    
    checkpointer.save_checkpoint(
        model, optimizer, epoch, val_loss, val_accuracy
    )
```

## 模型部署

### 推理模型保存
```python
import genesis

def save_for_inference(model, file_path, model_config=None):
    """保存优化的推理模型。"""
    model.eval()  # 设置为评估模式
    
    inference_state = {
        'model_state_dict': model.state_dict(),
        'model_config': model_config,
        'genesis_version': genesis.__version__,
        'inference_only': True
    }
    
    genesis.save(inference_state, file_path)

def load_for_inference(file_path, model_class):
    """加载推理模型。"""
    checkpoint = genesis.load(file_path)
    
    # 创建模型实例
    if 'model_config' in checkpoint and checkpoint['model_config']:
        model = model_class(**checkpoint['model_config'])
    else:
        model = model_class()
    
    # 加载状态
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model

# 保存训练好的模型用于部署
model_config = {
    'input_size': 784,
    'hidden_size': 256, 
    'num_classes': 10
}

save_for_inference(model, 'deployed_model.pth', model_config)

# 在生产环境中加载
deployed_model = load_for_inference('deployed_model.pth', MyModelClass)
```

### 模型版本管理
```python
import genesis
import time
from datetime import datetime

class VersionedCheckpoint:
    def __init__(self, base_path='models'):
        self.base_path = base_path
        
    def save_version(self, model, optimizer, epoch, metrics, version_name=None):
        """保存带有版本信息的模型。"""
        if version_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            version_name = f"v_{timestamp}"
            
        checkpoint = {
            'version': version_name,
            'timestamp': time.time(),
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': metrics,
            'genesis_version': genesis.__version__
        }
        
        file_path = f'{self.base_path}/{version_name}.pth'
        genesis.save(checkpoint, file_path)
        
        # 更新最新版本链接
        latest_path = f'{self.base_path}/latest.pth'
        genesis.save(checkpoint, latest_path)
        
        return version_name
    
    def load_version(self, version_name='latest'):
        """加载特定版本的模型。"""
        file_path = f'{self.base_path}/{version_name}.pth'
        return genesis.load(file_path)
    
    def list_versions(self):
        """列出可用的模型版本。"""
        import os
        versions = []
        for file in os.listdir(self.base_path):
            if file.endswith('.pth') and file != 'latest.pth':
                versions.append(file[:-4])  # 移除.pth扩展名
        return sorted(versions)

# 用法
versioner = VersionedCheckpoint()

# 保存新版本
version = versioner.save_version(
    model, optimizer, epoch=100, 
    metrics={'accuracy': 0.95, 'loss': 0.15},
    version_name='model_v1.2'
)

# 加载特定版本
checkpoint = versioner.load_version('model_v1.2')

# 加载最新版本
latest = versioner.load_version('latest')
```

## 错误处理和安全性

### 稳健的检查点加载
```python
import genesis
import os

def safe_load_checkpoint(file_path, model, optimizer=None):
    """安全加载检查点，带错误处理。"""
    try:
        if not os.path.exists(file_path):
            print(f"警告: 检查点 {file_path} 未找到")
            return False
            
        checkpoint = genesis.load(file_path)
        
        # 验证检查点结构
        required_keys = ['model_state_dict']
        for key in required_keys:
            if key not in checkpoint:
                print(f"错误: 检查点中缺少键 '{key}'")
                return False
        
        # 加载模型状态
        try:
            model.load_state_dict(checkpoint['model_state_dict'])
            print("模型状态加载成功")
        except Exception as e:
            print(f"加载模型状态时出错: {e}")
            return False
            
        # 如果提供了优化器，则加载优化器状态
        if optimizer and 'optimizer_state_dict' in checkpoint:
            try:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                print("优化器状态加载成功")
            except Exception as e:
                print(f"警告: 无法加载优化器状态: {e}")
        
        # 返回附加信息
        epoch = checkpoint.get('epoch', 0)
        loss = checkpoint.get('loss', 'unknown')
        print(f"从epoch {epoch}加载检查点，损失: {loss}")
        
        return True
        
    except Exception as e:
        print(f"加载检查点时出错: {e}")
        return False

# 用法
success = safe_load_checkpoint('checkpoint.pth', model, optimizer)
if success:
    print("检查点加载成功")
else:
    print("加载检查点失败，从头开始")
```

### 检查点验证
```python
import genesis

def validate_checkpoint(file_path):
    """验证检查点文件完整性。"""
    try:
        checkpoint = genesis.load(file_path)
        
        # 基本结构验证
        if not isinstance(checkpoint, dict):
            return False, "检查点不是字典类型"
            
        if 'model_state_dict' not in checkpoint:
            return False, "缺少model_state_dict"
        
        # 检查模型状态结构
        model_state = checkpoint['model_state_dict']
        if not isinstance(model_state, dict):
            return False, "model_state_dict不是字典类型"
            
        # 检查空状态
        if len(model_state) == 0:
            return False, "model_state_dict为空"
        
        # 验证张量形状（基本检查）
        for key, tensor in model_state.items():
            if not hasattr(tensor, 'shape'):
                return False, f"键 '{key}' 的张量无效"
        
        return True, "检查点有效"
        
    except Exception as e:
        return False, f"验证检查点时出错: {e}"

# 用法
is_valid, message = validate_checkpoint('checkpoint.pth')
print(f"检查点验证: {message}")
```

## 最佳实践

### 1. 检查点策略
- 定期保存检查点（每N个epoch）
- 保留多个最近的检查点
- 单独保存最佳模型
- 包含训练元数据

### 2. 文件组织
```python
# 推荐的目录结构
checkpoints/
├── latest.pth                    # 最新检查点
├── best_model.pth               # 最佳性能模型
├── epoch_000010.pth            # 常规检查点
├── epoch_000020.pth
└── deployed/
    └── production_model.pth     # 生产就绪模型
```

### 3. 内存管理
```python
import genesis
import gc

def efficient_checkpoint_save(model, optimizer, file_path):
    """带内存优化的检查点保存。"""
    # 创建检查点字典
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }
    
    # 保存检查点
    genesis.save(checkpoint, file_path)
    
    # 从内存中清除检查点字典
    del checkpoint
    gc.collect()
    
    print(f"检查点已保存到 {file_path}")
```

### 4. 跨设备兼容性
```python
def save_device_agnostic(model, file_path):
    """保存可在任何设备上加载的模型。"""
    # 保存前移动到CPU
    model.cpu()
    genesis.save(model.state_dict(), file_path)
    
def load_to_device(file_path, model, device):
    """将检查点加载到指定设备。"""
    # 加载检查点
    state_dict = genesis.load(file_path)
    
    # 加载到模型
    model.load_state_dict(state_dict)
    
    # 移动到目标设备
    model.to(device)
```

## 迁移和兼容性

### 从PyTorch迁移
```python
import genesis
import torch

def convert_pytorch_checkpoint(pytorch_file, genesis_file):
    """将PyTorch检查点转换为Genesis格式。"""
    # 加载PyTorch检查点
    torch_checkpoint = torch.load(pytorch_file, map_location='cpu')
    
    # 转换为Genesis格式（如需要）
    genesis_checkpoint = {
        'model_state_dict': torch_checkpoint['model_state_dict'],
        'optimizer_state_dict': torch_checkpoint.get('optimizer_state_dict', {}),
        'epoch': torch_checkpoint.get('epoch', 0),
        'converted_from_pytorch': True
    }
    
    # 以Genesis格式保存
    genesis.save(genesis_checkpoint, genesis_file)
    print(f"已转换 {pytorch_file} -> {genesis_file}")
```

Genesis序列化系统为生产级深度学习工作流程提供了强大、高效和安全的模型检查点功能。