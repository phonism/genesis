# Model Serialization and Checkpointing

Genesis provides robust model serialization and checkpointing functionality to save and load model states, optimizer states, and training progress. This is essential for long training runs, model deployment, and experiment reproducibility.

## Overview

The serialization system in Genesis handles:
- Model state dictionaries (parameters and buffers)
- Optimizer state (momentum, running averages, etc.)
- Training metadata (epoch, loss, metrics)
- Atomic write operations with backup for safety

## Core Functions

### save()
```python
import genesis

def save(state_dict, file_path):
    """
    Save state dictionary to file with atomic write operation.
    
    Args:
        state_dict (dict): Dictionary containing state to save
        file_path (str): Path where to save the file
        
    Features:
        - Atomic write with backup
        - Automatic cleanup on success
        - Rollback on failure
        - Memory cleanup after save
    """
```

### load()
```python
def load(file_path):
    """
    Load state dictionary from file.
    
    Args:
        file_path (str): Path to the saved file
        
    Returns:
        dict: Loaded state dictionary
        
    Raises:
        FileNotFoundError: If file doesn't exist
        pickle.UnpicklingError: If file is corrupted
    """
```

### save_checkpoint()
```python
def save_checkpoint(model_state_dict, optimizer_state_dict, file_path):
    """
    Save model and optimizer checkpoint.
    
    Args:
        model_state_dict (dict): Model state dictionary
        optimizer_state_dict (dict): Optimizer state dictionary  
        file_path (str): Path where to save checkpoint
        
    Creates checkpoint containing:
        - model_state_dict: Model parameters and buffers
        - optimizer_state_dict: Optimizer state
    """
```

### load_checkpoint()
```python
def load_checkpoint(file_path):
    """
    Load model and optimizer checkpoint.
    
    Args:
        file_path (str): Path to checkpoint file
        
    Returns:
        tuple: (model_state_dict, optimizer_state_dict)
        
    Example:
        >>> model_state, optimizer_state = genesis.load_checkpoint('checkpoint.pth')
        >>> model.load_state_dict(model_state)
        >>> optimizer.load_state_dict(optimizer_state)
    """
```

## Basic Usage

### Saving a Simple Model
```python
import genesis
import genesis.nn as nn

# Create and train model
model = nn.Linear(784, 10)

# Save model state
genesis.save(model.state_dict(), 'model.pth')

# Load model state
state_dict = genesis.load('model.pth')
model.load_state_dict(state_dict)
```

### Training Checkpoints
```python
import genesis
import genesis.nn as nn
import genesis.optim as optim

# Setup model and optimizer
model = nn.Sequential(
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Linear(256, 10)
)

optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop with checkpointing
for epoch in range(100):
    # Training code...
    train_loss = train_one_epoch(model, train_loader, optimizer)
    
    # Save checkpoint every 10 epochs
    if epoch % 10 == 0:
        genesis.save_checkpoint(
            model.state_dict(),
            optimizer.state_dict(), 
            f'checkpoint_epoch_{epoch}.pth'
        )
        print(f"Checkpoint saved at epoch {epoch}")

# Load checkpoint to resume training
model_state, optimizer_state = genesis.load_checkpoint('checkpoint_epoch_90.pth')
model.load_state_dict(model_state)
optimizer.load_state_dict(optimizer_state)
```

## Advanced Checkpointing

### Complete Training State
```python
import genesis

def save_training_checkpoint(model, optimizer, scheduler, epoch, loss, metrics, file_path):
    """Save complete training state."""
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
    """Load complete training state."""
    return genesis.load(file_path)

# Usage
save_training_checkpoint(
    model, optimizer, scheduler, 
    epoch=50, loss=0.234, 
    metrics={'accuracy': 0.94, 'f1': 0.91},
    file_path='complete_checkpoint.pth'
)

# Resume training
checkpoint = load_training_checkpoint('complete_checkpoint.pth')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
if checkpoint['scheduler_state_dict']:
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

start_epoch = checkpoint['epoch'] + 1
print(f"Resuming from epoch {start_epoch}, loss: {checkpoint['loss']}")
```

### Best Model Tracking
```python
import genesis

class ModelCheckpointer:
    def __init__(self, save_dir='checkpoints'):
        self.save_dir = save_dir
        self.best_loss = float('inf')
        self.best_accuracy = 0.0
        
    def save_checkpoint(self, model, optimizer, epoch, loss, accuracy, is_best=False):
        """Save checkpoint and track best model."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            'accuracy': accuracy
        }
        
        # Save regular checkpoint
        checkpoint_path = f'{self.save_dir}/checkpoint_epoch_{epoch}.pth'
        genesis.save(checkpoint, checkpoint_path)
        
        # Save best model based on loss
        if loss < self.best_loss:
            self.best_loss = loss
            best_loss_path = f'{self.save_dir}/best_loss_model.pth'
            genesis.save(checkpoint, best_loss_path)
            print(f"New best loss: {loss:.4f}")
            
        # Save best model based on accuracy  
        if accuracy > self.best_accuracy:
            self.best_accuracy = accuracy
            best_acc_path = f'{self.save_dir}/best_accuracy_model.pth'
            genesis.save(checkpoint, best_acc_path)
            print(f"New best accuracy: {accuracy:.4f}")
    
    def load_best_model(self, model, metric='loss'):
        """Load the best model."""
        if metric == 'loss':
            path = f'{self.save_dir}/best_loss_model.pth'
        elif metric == 'accuracy':
            path = f'{self.save_dir}/best_accuracy_model.pth'
        else:
            raise ValueError("metric must be 'loss' or 'accuracy'")
            
        checkpoint = genesis.load(path)
        model.load_state_dict(checkpoint['model_state_dict'])
        return checkpoint

# Usage
checkpointer = ModelCheckpointer()

for epoch in range(100):
    train_loss = train_one_epoch(model, train_loader, optimizer)
    val_loss, val_accuracy = validate(model, val_loader)
    
    checkpointer.save_checkpoint(
        model, optimizer, epoch, val_loss, val_accuracy
    )
```

## Model Deployment

### Saving for Inference
```python
import genesis

def save_for_inference(model, file_path, model_config=None):
    """Save model optimized for inference."""
    model.eval()  # Set to evaluation mode
    
    inference_state = {
        'model_state_dict': model.state_dict(),
        'model_config': model_config,
        'genesis_version': genesis.__version__,
        'inference_only': True
    }
    
    genesis.save(inference_state, file_path)

def load_for_inference(file_path, model_class):
    """Load model for inference."""
    checkpoint = genesis.load(file_path)
    
    # Create model instance
    if 'model_config' in checkpoint and checkpoint['model_config']:
        model = model_class(**checkpoint['model_config'])
    else:
        model = model_class()
    
    # Load state
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model

# Save trained model for deployment
model_config = {
    'input_size': 784,
    'hidden_size': 256, 
    'num_classes': 10
}

save_for_inference(model, 'deployed_model.pth', model_config)

# Load in production
deployed_model = load_for_inference('deployed_model.pth', MyModelClass)
```

### Model Versioning
```python
import genesis
import time
from datetime import datetime

class VersionedCheckpoint:
    def __init__(self, base_path='models'):
        self.base_path = base_path
        
    def save_version(self, model, optimizer, epoch, metrics, version_name=None):
        """Save model with version information."""
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
        
        # Update latest symlink
        latest_path = f'{self.base_path}/latest.pth'
        genesis.save(checkpoint, latest_path)
        
        return version_name
    
    def load_version(self, version_name='latest'):
        """Load specific model version."""
        file_path = f'{self.base_path}/{version_name}.pth'
        return genesis.load(file_path)
    
    def list_versions(self):
        """List available model versions."""
        import os
        versions = []
        for file in os.listdir(self.base_path):
            if file.endswith('.pth') and file != 'latest.pth':
                versions.append(file[:-4])  # Remove .pth extension
        return sorted(versions)

# Usage
versioner = VersionedCheckpoint()

# Save new version
version = versioner.save_version(
    model, optimizer, epoch=100, 
    metrics={'accuracy': 0.95, 'loss': 0.15},
    version_name='model_v1.2'
)

# Load specific version
checkpoint = versioner.load_version('model_v1.2')

# Load latest version
latest = versioner.load_version('latest')
```

## Error Handling and Safety

### Robust Checkpoint Loading
```python
import genesis
import os

def safe_load_checkpoint(file_path, model, optimizer=None):
    """Safely load checkpoint with error handling."""
    try:
        if not os.path.exists(file_path):
            print(f"Warning: Checkpoint {file_path} not found")
            return False
            
        checkpoint = genesis.load(file_path)
        
        # Validate checkpoint structure
        required_keys = ['model_state_dict']
        for key in required_keys:
            if key not in checkpoint:
                print(f"Error: Missing key '{key}' in checkpoint")
                return False
        
        # Load model state
        try:
            model.load_state_dict(checkpoint['model_state_dict'])
            print("Model state loaded successfully")
        except Exception as e:
            print(f"Error loading model state: {e}")
            return False
            
        # Load optimizer state if provided
        if optimizer and 'optimizer_state_dict' in checkpoint:
            try:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                print("Optimizer state loaded successfully")
            except Exception as e:
                print(f"Warning: Could not load optimizer state: {e}")
        
        # Return additional info
        epoch = checkpoint.get('epoch', 0)
        loss = checkpoint.get('loss', 'unknown')
        print(f"Loaded checkpoint from epoch {epoch}, loss: {loss}")
        
        return True
        
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return False

# Usage
success = safe_load_checkpoint('checkpoint.pth', model, optimizer)
if success:
    print("Checkpoint loaded successfully")
else:
    print("Failed to load checkpoint, starting from scratch")
```

### Checkpoint Validation
```python
import genesis

def validate_checkpoint(file_path):
    """Validate checkpoint file integrity."""
    try:
        checkpoint = genesis.load(file_path)
        
        # Basic structure validation
        if not isinstance(checkpoint, dict):
            return False, "Checkpoint is not a dictionary"
            
        if 'model_state_dict' not in checkpoint:
            return False, "Missing model_state_dict"
        
        # Check model state structure
        model_state = checkpoint['model_state_dict']
        if not isinstance(model_state, dict):
            return False, "model_state_dict is not a dictionary"
            
        # Check for empty state
        if len(model_state) == 0:
            return False, "model_state_dict is empty"
        
        # Validate tensor shapes (basic check)
        for key, tensor in model_state.items():
            if not hasattr(tensor, 'shape'):
                return False, f"Invalid tensor for key '{key}'"
        
        return True, "Checkpoint is valid"
        
    except Exception as e:
        return False, f"Error validating checkpoint: {e}"

# Usage
is_valid, message = validate_checkpoint('checkpoint.pth')
print(f"Checkpoint validation: {message}")
```

## Best Practices

### 1. Checkpoint Strategy
- Save checkpoints regularly (every N epochs)
- Keep multiple recent checkpoints
- Save best model separately  
- Include training metadata

### 2. File Organization
```python
# Recommended directory structure
checkpoints/
├── latest.pth                    # Latest checkpoint
├── best_model.pth               # Best performing model
├── epoch_000010.pth            # Regular checkpoints
├── epoch_000020.pth
└── deployed/
    └── production_model.pth     # Production-ready model
```

### 3. Memory Management
```python
import genesis
import gc

def efficient_checkpoint_save(model, optimizer, file_path):
    """Save checkpoint with memory optimization."""
    # Create checkpoint dict
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }
    
    # Save checkpoint
    genesis.save(checkpoint, file_path)
    
    # Clear checkpoint dict from memory
    del checkpoint
    gc.collect()
    
    print(f"Checkpoint saved to {file_path}")
```

### 4. Cross-Device Compatibility
```python
def save_device_agnostic(model, file_path):
    """Save model that can be loaded on any device."""
    # Move to CPU before saving
    model.cpu()
    genesis.save(model.state_dict(), file_path)
    
def load_to_device(file_path, model, device):
    """Load checkpoint to specific device."""
    # Load checkpoint
    state_dict = genesis.load(file_path)
    
    # Load to model
    model.load_state_dict(state_dict)
    
    # Move to target device
    model.to(device)
```

## Migration and Compatibility

### From PyTorch
```python
import genesis
import torch

def convert_pytorch_checkpoint(pytorch_file, genesis_file):
    """Convert PyTorch checkpoint to Genesis format."""
    # Load PyTorch checkpoint
    torch_checkpoint = torch.load(pytorch_file, map_location='cpu')
    
    # Convert to Genesis format (if needed)
    genesis_checkpoint = {
        'model_state_dict': torch_checkpoint['model_state_dict'],
        'optimizer_state_dict': torch_checkpoint.get('optimizer_state_dict', {}),
        'epoch': torch_checkpoint.get('epoch', 0),
        'converted_from_pytorch': True
    }
    
    # Save in Genesis format
    genesis.save(genesis_checkpoint, genesis_file)
    print(f"Converted {pytorch_file} -> {genesis_file}")
```

The Genesis serialization system provides robust, efficient, and safe model checkpointing capabilities essential for production deep learning workflows.