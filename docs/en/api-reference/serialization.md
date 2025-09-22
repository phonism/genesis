# Model Serialization and Checkpointing

Genesis provides model serialization and checkpointing functionality to save and load model states, optimizer states, and training progress.

## Overview

The serialization system in Genesis provides:
- Model state dictionary saving/loading
- Optimizer state saving/loading
- Checkpoint management with atomic write operations
- Backup file creation for safety

## Core Functions

### save()
```python
def save(state_dict, file_path):
    """
    Save state dictionary to file with atomic write operation.
    
    Args:
        state_dict (dict): Dictionary containing state to save
        file_path (str): Path where to save the file
        
    Features:
        - Creates backup file (.genesis.bak) before overwriting
        - Atomic write with automatic cleanup on success
        - Rollback to backup on failure
        - Clears state_dict from memory after save
        
    Implementation:
        - Uses dill (enhanced pickle) for serialization
        - Supports lambda functions and complex Python objects
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
        
    Implementation:
        - Uses dill for deserialization
        - Compatible with files saved by save()
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
        - "model_state_dict": Model parameters and buffers
        - "optimizer_state_dict": Optimizer state
        
    Implementation:
        - Internally calls save() with structured dictionary
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
        
    Implementation:
        - Loads pickle file and extracts state dictionaries
        - Returns tuple for convenient unpacking
        
    Example:
        >>> model_state, optimizer_state = genesis.load_checkpoint('checkpoint.pth')
        >>> model.load_state_dict(model_state)
        >>> optimizer.load_state_dict(optimizer_state)
    """
```

## Basic Usage

### Saving a Model
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

## Advanced Usage

### Complete Training State
```python
import genesis

# Save complete training state
checkpoint = {
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': loss,
    'best_accuracy': best_accuracy,
    'training_history': training_history
}
genesis.save(checkpoint, 'full_checkpoint.pth')

# Load complete training state
checkpoint = genesis.load('full_checkpoint.pth')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
start_epoch = checkpoint['epoch'] + 1
best_accuracy = checkpoint['best_accuracy']
```

### Best Model Tracking
```python
import genesis
import os

best_loss = float('inf')

for epoch in range(num_epochs):
    # Training
    train_loss = train_one_epoch(model, train_loader, optimizer)
    val_loss = validate(model, val_loader)
    
    # Save regular checkpoint
    genesis.save_checkpoint(
        model.state_dict(),
        optimizer.state_dict(),
        f'checkpoint_epoch_{epoch}.pth'
    )
    
    # Save best model
    if val_loss < best_loss:
        best_loss = val_loss
        genesis.save(model.state_dict(), 'best_model.pth')
        print(f"New best model saved with loss: {val_loss:.4f}")
```

### Error Handling
```python
import genesis
import os

def safe_load_checkpoint(file_path, model, optimizer=None):
    """Safely load checkpoint with error handling."""
    try:
        if not os.path.exists(file_path):
            print(f"Checkpoint {file_path} not found")
            return False
            
        # Load checkpoint
        if file_path.endswith('.pth'):
            # Try loading as checkpoint first
            try:
                model_state, optimizer_state = genesis.load_checkpoint(file_path)
                model.load_state_dict(model_state)
                if optimizer:
                    optimizer.load_state_dict(optimizer_state)
            except:
                # Fall back to loading as state dict
                state_dict = genesis.load(file_path)
                if 'model_state_dict' in state_dict:
                    model.load_state_dict(state_dict['model_state_dict'])
                    if optimizer and 'optimizer_state_dict' in state_dict:
                        optimizer.load_state_dict(state_dict['optimizer_state_dict'])
                else:
                    model.load_state_dict(state_dict)
        
        print("Checkpoint loaded successfully")
        return True
        
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return False

# Usage
success = safe_load_checkpoint('checkpoint.pth', model, optimizer)
```

## Implementation Details

### Atomic Write Operation
The save function implements atomic writes to prevent data corruption:

1. **Backup Creation**: Before overwriting an existing file, creates a `.genesis.bak` backup
2. **Write Operation**: Saves new data to the target file
3. **Memory Cleanup**: Clears the state_dict from memory after successful save
4. **Cleanup/Rollback**: On success, removes backup; on failure, restores from backup

### Serialization Format
- Uses `dill` (enhanced pickle) for serialization
- Supports lambda functions and complex Python objects
- Binary format for efficient storage
- Compatible with standard pickle for basic objects

### Memory Management
The save function includes automatic memory cleanup:
```python
# After successful save, clear the dictionary to free memory
for key in list(state_dict.keys()):
    del state_dict[key]
```

## Best Practices

### 1. Checkpoint Frequency
- Save checkpoints regularly (e.g., every epoch or N steps)
- Keep multiple recent checkpoints
- Save best model separately

### 2. File Organization
```python
# Recommended structure
checkpoints/
├── checkpoint_epoch_10.pth     # Regular checkpoints
├── checkpoint_epoch_20.pth
├── best_model.pth              # Best performing model
└── final_model.pth             # Final trained model
```

### 3. Checkpoint Naming
```python
# Include useful information in filename
filename = f"checkpoint_epoch_{epoch}_loss_{loss:.4f}.pth"
genesis.save_checkpoint(model.state_dict(), optimizer.state_dict(), filename)
```

### 4. Resume Training
```python
# Check for existing checkpoint
checkpoint_path = 'checkpoint_latest.pth'
if os.path.exists(checkpoint_path):
    model_state, optimizer_state = genesis.load_checkpoint(checkpoint_path)
    model.load_state_dict(model_state)
    optimizer.load_state_dict(optimizer_state)
    print("Resumed from checkpoint")
else:
    print("Starting fresh training")
```

## Limitations and Considerations

1. **File Format**: Uses pickle/dill format, not compatible with PyTorch `.pt` files
2. **Memory Usage**: The save function clears the input dictionary from memory
3. **No Compression**: Files are not compressed (consider using external compression if needed)
4. **Single Device**: No special handling for multi-GPU model states

## Migration Notes

### From PyTorch
Genesis uses a similar API to PyTorch but with some differences:
- Uses `dill` instead of standard `pickle` for better lambda support
- Automatic backup file creation for safety
- Memory cleanup after saving

### Loading PyTorch Models
To load PyTorch models in Genesis, you'll need to convert them:
```python
import torch
import genesis

# Load PyTorch model
torch_state = torch.load('pytorch_model.pt')

# Convert and save in Genesis format
# Note: Tensor conversion may be needed depending on backend
genesis.save(torch_state, 'genesis_model.pth')
```

## Advanced Checkpointing

### Robust Checkpoint Loading

Handle loading errors gracefully with validation:

```python
import genesis

def safe_load_checkpoint(file_path, model, optimizer=None):
    """Load checkpoint with comprehensive error handling."""
    try:
        checkpoint = genesis.load(file_path)
        
        # Load model state
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            print("Model state loaded successfully")
        else:
            print("Warning: No model state found in checkpoint")
            return False
            
        # Load optimizer state (optional)
        if optimizer and 'optimizer_state_dict' in checkpoint:
            try:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                print("Optimizer state loaded successfully")
            except Exception as e:
                print(f"Warning: Could not load optimizer state: {e}")
        
        # Return additional information
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
├── best_model.pth               # Best performance model
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
    """Memory-optimized checkpoint saving."""
    # Create checkpoint dictionary
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }
    
    # Save checkpoint
    genesis.save(checkpoint, file_path)
    
    # Clear checkpoint dictionary from memory
    del checkpoint
    gc.collect()
```

### 4. Version Management
```python
class CheckpointManager:
    """Manage checkpoint versions and cleanup."""
    
    def __init__(self, checkpoint_dir, max_checkpoints=5):
        self.checkpoint_dir = checkpoint_dir
        self.max_checkpoints = max_checkpoints
        
    def save_checkpoint(self, model, optimizer, epoch, loss):
        """Save checkpoint with automatic cleanup."""
        file_path = f"{self.checkpoint_dir}/epoch_{epoch:06d}.pth"
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            'timestamp': time.time()
        }
        
        genesis.save(checkpoint, file_path)
        self._cleanup_old_checkpoints()
        
    def _cleanup_old_checkpoints(self):
        """Remove old checkpoints keeping only the most recent."""
        import os
        import glob
        
        checkpoints = glob.glob(f"{self.checkpoint_dir}/epoch_*.pth")
        checkpoints.sort()
        
        while len(checkpoints) > self.max_checkpoints:
            os.remove(checkpoints.pop(0))

# Usage
manager = CheckpointManager('checkpoints/', max_checkpoints=3)
manager.save_checkpoint(model, optimizer, epoch, loss)
```

## See Also

- [Optimizers](optim/optimizers.md) - Optimizer state management
- [Neural Network Modules](nn/modules.md) - Model state_dict methods
- [Training Examples](../../../samples/) - Complete training scripts with checkpointing