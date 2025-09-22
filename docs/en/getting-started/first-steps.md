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
"""Data loading and preprocessing module"""
import genesis
import numpy as np
from typing import Tuple, List
import pickle
import os

class SimpleDataset:
    """Simple dataset class"""
    
    def __init__(self, data: np.ndarray, labels: np.ndarray, transform=None):
        """
        Initialize the dataset

        Args:
            data: Input data (N, H, W) or (N, D)
            labels: Labels (N,)
            transform: Data transformation function
        """
        self.data = data.astype(np.float32)
        self.labels = labels.astype(np.int64)
        self.transform = transform
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Tuple[genesis.Tensor, genesis.Tensor]:
        """Get a single sample"""
        x = self.data[idx]
        y = self.labels[idx]
        
        if self.transform:
            x = self.transform(x)
        
        return genesis.tensor(x), genesis.tensor(y)

class DataLoader:
    """Simple data loader"""
    
    def __init__(self, dataset: SimpleDataset, batch_size: int = 32, shuffle: bool = True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self._reset_indices()
    
    def _reset_indices(self):
        """Reset indices"""
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
        
        # Get the current batch indices
        end_idx = min(self.current + self.batch_size, len(self.dataset))
        batch_indices = self.indices[self.current:end_idx]
        
        # Collect batch data
        batch_data = []
        batch_labels = []
        
        for idx in batch_indices:
            x, y = self.dataset[idx]
            batch_data.append(x)
            batch_labels.append(y)
        
        self.current = end_idx
        
        # Stack into batches
        batch_x = genesis.stack(batch_data, dim=0)
        batch_y = genesis.stack(batch_labels, dim=0)
        
        return batch_x, batch_y

def create_synthetic_data(n_samples: int = 1000, n_features: int = 784, n_classes: int = 10) -> Tuple[np.ndarray, np.ndarray]:
    """Create synthetic data for demonstration"""
    np.random.seed(42)
    
    # Generate random data
    data = np.random.randn(n_samples, n_features).astype(np.float32)
    
    # Add some patterns for each class
    labels = np.random.randint(0, n_classes, n_samples)
    for i in range(n_classes):
        mask = labels == i
        # Add specific bias to each class
        data[mask] += np.random.randn(n_features) * 0.5
    
    return data, labels

def load_data() -> Tuple[DataLoader, DataLoader]:
    """Load training and validation data"""
    print("🔄 Loading data...")
    
    # Create synthetic data (replace with real data in actual projects)
    train_data, train_labels = create_synthetic_data(800, 784, 10)
    val_data, val_labels = create_synthetic_data(200, 784, 10)
    
    # Data normalization
    def normalize(x):
        return (x - x.mean()) / (x.std() + 1e-8)
    
    # Create datasets
    train_dataset = SimpleDataset(train_data, train_labels, transform=normalize)
    val_dataset = SimpleDataset(val_data, val_labels, transform=normalize)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    print(f"✅ Data loading complete - Training set: {len(train_dataset)}, Validation set: {len(val_dataset)}")
    
    return train_loader, val_loader
```

## 🧠 2. Model Definition (`model.py`)

```python
"""Neural network model definition"""
import genesis
import genesis.nn as nn
import genesis.nn.functional as F

class MLP(nn.Module):
    """Multilayer Perceptron Classifier"""
    
    def __init__(self, input_dim: int = 784, hidden_dims: list = None, num_classes: int = 10, dropout_rate: float = 0.2):
        """
        Initialize MLP model

        Args:
            input_dim: Input dimension
            hidden_dims: List of hidden layer dimensions
            num_classes: Number of classes
            dropout_rate: Dropout ratio
        """
        super().__init__()
        
        if hidden_dims is None:
            hidden_dims = [512, 256, 128]
        
        # Build network layers
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, num_classes))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Weight initialization"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Xavier initialization
                std = (2.0 / (module.in_features + module.out_features)) ** 0.5
                module.weight.data.normal_(0, std)
                if module.bias is not None:
                    module.bias.data.zero_()
    
    def forward(self, x: genesis.Tensor) -> genesis.Tensor:
        """Forward propagation"""
        # Flatten input (if image data)
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        
        return self.network(x)

class CNN(nn.Module):
    """Convolutional Neural Network Classifier (if processing images)"""
    
    def __init__(self, num_classes: int = 10):
        super().__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        # Pooling layers
        self.pool = nn.MaxPool2d(2, 2)
        
        # Fully connected layers
        self.fc1 = nn.Linear(128 * 3 * 3, 512)  # Assuming input is 28x28
        self.fc2 = nn.Linear(512, num_classes)
        
        # Dropout
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x: genesis.Tensor) -> genesis.Tensor:
        # Convolution + pooling
        x = self.pool(F.relu(self.conv1(x)))  # 28x28 -> 14x14
        x = self.pool(F.relu(self.conv2(x)))  # 14x14 -> 7x7  
        x = self.pool(F.relu(self.conv3(x)))  # 7x7 -> 3x3
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

def create_model(model_type: str = "mlp", **kwargs) -> nn.Module:
    """Factory function: create model"""
    if model_type.lower() == "mlp":
        return MLP(**kwargs)
    elif model_type.lower() == "cnn":
        return CNN(**kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
```

## 🛠️ 3. Utility Functions (`utils.py`)

```python
"""Utility functions module"""
import genesis
import time
import os
from typing import Dict, Any
import json

class AverageMeter:
    """Average calculator"""
    
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
    """Timer"""
    
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
    """Calculate accuracy"""
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
    """Save checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'accuracy': accuracy
    }
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    genesis.save(checkpoint, filepath)
    print(f"💾 Checkpoint saved: {filepath}")

def load_checkpoint(filepath: str, model: genesis.nn.Module, optimizer: genesis.optim.Optimizer = None) -> Dict[str, Any]:
    """Load checkpoint"""
    checkpoint = genesis.load(filepath)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    print(f"📁 Checkpoint loaded: {filepath}")
    print(f"   Epoch: {checkpoint['epoch']}, Loss: {checkpoint['loss']:.4f}, Accuracy: {checkpoint['accuracy']:.2f}%")
    
    return checkpoint

def save_training_history(history: Dict[str, list], filepath: str):
    """Save training history"""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    with open(filepath, 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f"📊 Training history saved: {filepath}")

def print_model_summary(model: genesis.nn.Module, input_shape: tuple):
    """Print model summary"""
    print("🏗️  Model architecture:")
    print("=" * 50)
    
    # Calculate parameter count
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Input shape: {input_shape}")
    
    # Test forward propagation
    dummy_input = genesis.randn(*input_shape)
    try:
        with genesis.no_grad():
            output = model(dummy_input)
        print(f"Output shape: {output.shape}")
    except Exception as e:
        print(f"Forward propagation test failed: {e}")
    
    print("=" * 50)
```

## 🚂 4. Training Script (`train.py`)

<function_calls>
<invoke name="TodoWrite">
<parameter name="todos">[{"id": "1", "content": "\u521b\u5efa\u6587\u6863\u76ee\u5f55\u7ed3\u6784\u548c\u914d\u7f6e\u6587\u4ef6", "status": "completed"}, {"id": "2", "content": "\u7f16\u5199\u9996\u9875\u548c\u5feb\u901f\u5f00\u59cb\u6587\u6863", "status": "completed"}, {"id": "3", "content": "\u7f16\u5199\u67b6\u6784\u8bbe\u8ba1\u6587\u6863", "status": "pending"}, {"id": "4", "content": "\u7f16\u5199\u6838\u5fc3\u7ec4\u4ef6\u6587\u6863", "status": "pending"}, {"id": "5", "content": "\u7f16\u5199\u795e\u7ecf\u7f51\u7edc\u6a21\u5757\u6587\u6863", "status": "pending"}, {"id": "6", "content": "\u7f16\u5199API\u53c2\u8003\u6587\u6863", "status": "pending"}]