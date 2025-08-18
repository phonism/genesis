# Basic Training Tutorial

This tutorial will take you from zero to building and training your first neural network using the Genesis deep learning framework. We will learn Genesis core concepts and usage through a complete image classification project.

## 🎯 Learning Objectives

Through this tutorial, you will learn:
- Genesis basic APIs and data structures
- How to define and train neural network models
- Data loading and preprocessing
- Building and optimizing training loops
- Model evaluation and saving

## 🛠️ Environment Setup

### Install Dependencies

```bash
# Ensure Genesis is installed
pip install torch triton numpy matplotlib tqdm
git clone https://github.com/phonism/genesis.git
cd genesis
pip install -e .
```

### Verify Installation

```python
import genesis
import genesis.nn as nn
import genesis.optim as optim

# Test basic functionality
x = genesis.randn(2, 3)
print(f"Genesis tensor created: {x.shape}")
print(f"Genesis modules available: {dir(nn)}")
```

## 📊 Project: Handwritten Digit Recognition

We will build a handwritten digit recognition system using a simple fully connected neural network on synthetic data to demonstrate Genesis capabilities.

### 1. Data Preparation

Since Genesis doesn't have built-in data loading utilities yet, we'll create synthetic data that mimics the MNIST structure:

```python
import genesis
import genesis.nn as nn
import genesis.optim as optim
import numpy as np
import matplotlib.pyplot as plt

class SimpleDataset:
    """Simple dataset class for demonstration"""
    
    def __init__(self, num_samples=1000, input_dim=784, num_classes=10):
        # Generate synthetic data similar to flattened MNIST
        self.data = genesis.randn(num_samples, input_dim)
        
        # Create labels based on data patterns (synthetic)
        labels = genesis.randn(num_samples, num_classes)
        self.labels = genesis.functional.max(labels, axis=1, keepdims=False)
        
        self.num_samples = num_samples
        
    def __len__(self):
        return self.num_samples
    
    def get_batch(self, batch_size=32, start_idx=0):
        """Get a batch of data"""
        end_idx = min(start_idx + batch_size, self.num_samples)
        return (self.data[start_idx:end_idx], 
                self.labels[start_idx:end_idx])

# Create datasets
train_dataset = SimpleDataset(num_samples=800, input_dim=784, num_classes=10)
test_dataset = SimpleDataset(num_samples=200, input_dim=784, num_classes=10)

print(f"Training set size: {len(train_dataset)}")
print(f"Test set size: {len(test_dataset)}")
print(f"Input dimension: 784 (28x28 flattened)")
print(f"Number of classes: 10")
```

### 2. Model Definition

We'll build a simple but effective fully connected neural network using Genesis modules:

```python
class MNISTNet(nn.Module):
    """Simple fully connected network for digit recognition"""
    
    def __init__(self, input_dim=784, hidden_dim=128, num_classes=10):
        super(MNISTNet, self).__init__()
        
        # Define layers using actual Genesis modules
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, num_classes)
        
        # Activation and regularization
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        # Flatten input if needed
        if len(x.shape) > 2:
            x = x.view(x.shape[0], -1)
        
        # First hidden layer
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        # Second hidden layer
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        # Output layer
        x = self.fc3(x)
        
        return x

# Create model instance
model = MNISTNet(input_dim=784, hidden_dim=128, num_classes=10)

print("Model structure:")
print(f"Layer 1: {model.fc1}")
print(f"Layer 2: {model.fc2}")
print(f"Layer 3: {model.fc3}")
print(f"Total parameters: {sum(p.data.size for p in model.parameters())}")
```

### 3. Loss Function and Optimizer

```python
# Define loss function and optimizer using Genesis
criterion = nn.SoftmaxLoss()  # Use Genesis SoftmaxLoss
optimizer = optim.Adam(model.parameters(), lr=0.001)

print(f"Loss function: {criterion}")
print(f"Optimizer: {optimizer}")
print(f"Learning rate: 0.001")
```

### 4. Training Loop

```python
def train_epoch(model, dataset, criterion, optimizer, batch_size=32):
    """Train for one epoch"""
    model.train()  # Set to training mode
    
    total_loss = 0.0
    num_batches = len(dataset) // batch_size
    
    for i in range(num_batches):
        # Get batch data
        start_idx = i * batch_size
        batch_data, batch_labels = dataset.get_batch(batch_size, start_idx)
        
        # Forward pass
        outputs = model(batch_data)
        loss = criterion(outputs, batch_labels)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Apply gradient clipping (optional)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # Update weights
        optimizer.step()
        
        total_loss += loss.data.item() if hasattr(loss.data, 'item') else float(loss.data)
    
    return total_loss / num_batches

def evaluate(model, dataset, criterion, batch_size=32):
    """Evaluate model performance"""
    model.eval()  # Set to evaluation mode
    
    total_loss = 0.0
    correct = 0
    total = 0
    num_batches = len(dataset) // batch_size
    
    for i in range(num_batches):
        start_idx = i * batch_size
        batch_data, batch_labels = dataset.get_batch(batch_size, start_idx)
        
        # Forward pass (no gradients needed)
        outputs = model(batch_data)
        loss = criterion(outputs, batch_labels)
        
        # Calculate accuracy
        predicted = genesis.functional.max(outputs, axis=1, keepdims=False)
        total += batch_labels.shape[0]
        correct += (predicted == batch_labels).sum().data
        
        total_loss += loss.data.item() if hasattr(loss.data, 'item') else float(loss.data)
    
    accuracy = correct / total
    avg_loss = total_loss / num_batches
    
    return avg_loss, accuracy

# Training configuration
num_epochs = 10
batch_size = 32

print("Starting training...")
print(f"Epochs: {num_epochs}")
print(f"Batch size: {batch_size}")
print("-" * 50)

# Training loop
train_losses = []
test_losses = []
test_accuracies = []

for epoch in range(num_epochs):
    # Train for one epoch
    train_loss = train_epoch(model, train_dataset, criterion, optimizer, batch_size)
    
    # Evaluate on test set
    test_loss, test_accuracy = evaluate(model, test_dataset, criterion, batch_size)
    
    # Record metrics
    train_losses.append(train_loss)
    test_losses.append(test_loss)
    test_accuracies.append(test_accuracy)
    
    # Print progress
    print(f"Epoch {epoch+1}/{num_epochs}")
    print(f"  Train Loss: {train_loss:.4f}")
    print(f"  Test Loss: {test_loss:.4f}")
    print(f"  Test Accuracy: {test_accuracy:.4f}")
    print("-" * 30)

print("Training completed!")
```

### 5. Model Evaluation and Visualization

```python
# Plot training progress
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 4))

# Plot losses
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss')
plt.plot(test_losses, label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Test Loss')
plt.legend()
plt.grid(True)

# Plot accuracy
plt.subplot(1, 2, 2)
plt.plot(test_accuracies, label='Test Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Test Accuracy')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# Final evaluation
final_test_loss, final_test_accuracy = evaluate(model, test_dataset, criterion, batch_size)
print(f"\nFinal Results:")
print(f"Test Loss: {final_test_loss:.4f}")
print(f"Test Accuracy: {final_test_accuracy:.4f}")
```

### 6. Model Saving and Loading

```python
# Save model using Genesis serialization
model_path = "mnist_model.pkl"
genesis.save(model.state_dict(), model_path)
print(f"Model saved to {model_path}")

# Load model
model_new = MNISTNet(input_dim=784, hidden_dim=128, num_classes=10)
model_new.load_state_dict(genesis.load(model_path))
print("Model loaded successfully!")

# Verify loaded model works
test_loss, test_accuracy = evaluate(model_new, test_dataset, criterion, batch_size)
print(f"Loaded model accuracy: {test_accuracy:.4f}")
```

## 🎓 Key Concepts Learned

### 1. Genesis Tensor Operations
- Creating tensors with `genesis.randn()`, `genesis.tensor()`
- Basic operations like matrix multiplication and element-wise operations
- Automatic differentiation with `requires_grad`

### 2. Neural Network Modules
- Defining models by inheriting from `nn.Module`
- Using built-in layers: `nn.Linear`, `nn.ReLU`, `nn.Dropout`
- Understanding forward pass implementation

### 3. Training Process
- Setting up loss functions and optimizers
- Implementing training and evaluation loops
- Using gradient clipping and regularization

### 4. Model Management
- Saving and loading model state with Genesis serialization
- Managing model parameters and optimization state

## 🚀 Next Steps

After completing this tutorial, you can:

1. **Explore more complex models** - Try different architectures with more layers
2. **Learn advanced features** - Explore mixed precision training and learning rate scheduling
3. **Work with real data** - Integrate with actual datasets when data loading utilities are available
4. **Performance optimization** - Learn about GPU acceleration and Triton kernel usage

## 📚 Additional Resources

- [Genesis API Reference](../api-reference/index.md) - Complete API documentation
- [Advanced Training Features](../training/advanced-features.md) - Mixed precision, schedulers, etc.
- [Performance Optimization](performance-tuning.md) - Tips for faster training

## 🐛 Troubleshooting

### Common Issues

1. **Import errors**: Ensure Genesis is properly installed with `pip install -e .`
2. **Shape mismatches**: Check tensor dimensions in forward pass
3. **Memory issues**: Reduce batch size if encountering out-of-memory errors
4. **Slow training**: Enable GPU support when available

### Getting Help

- Check the [Genesis Documentation](../index.md)
- Report issues on [GitHub Issues](https://github.com/phonism/genesis/issues)
- Join discussions in the community forums