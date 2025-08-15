# Mixed Precision Training Guide

Mixed precision training is a technique that uses both 16-bit (half precision) and 32-bit (single precision) floating-point numbers during training to reduce memory usage and accelerate training while maintaining model accuracy. Genesis provides comprehensive support for mixed precision training with automatic mixed precision (AMP) capabilities.

## Overview

### Benefits of Mixed Precision Training

- **Memory Efficiency**: Reduces memory usage by ~50%
- **Speed Improvement**: Faster training on modern GPUs with Tensor Cores
- **Model Accuracy**: Maintains training stability with automatic loss scaling
- **Larger Models**: Enables training of larger models on the same hardware

### Supported Precision Types

Genesis supports multiple precision formats:

- **float32 (FP32)**: Standard single precision (default)
- **float16 (FP16)**: IEEE half precision 
- **bfloat16 (BF16)**: Brain float format with larger dynamic range

## Data Type System

### Understanding Genesis DTypes

```python
import genesis

# Available precision types
print("Available dtypes:")
print(f"FP32: {genesis.float32}")  # Standard precision
print(f"FP16: {genesis.float16}")  # Half precision
print(f"BF16: {genesis.bfloat16}") # Brain float

# Check dtype properties
dtype = genesis.float16
print(f"Name: {dtype.name}")
print(f"Size: {dtype.itemsize} bytes")
print(f"Is floating: {dtype.is_floating_point}")
print(f"NumPy type: {dtype.numpy_dtype}")
```

### Creating Mixed Precision Tensors

```python
import genesis

# Create tensors with different precisions
fp32_tensor = genesis.randn(1000, 1000, dtype=genesis.float32)
fp16_tensor = genesis.randn(1000, 1000, dtype=genesis.float16) 
bf16_tensor = genesis.randn(1000, 1000, dtype=genesis.bfloat16)

print(f"FP32 memory: {fp32_tensor.numel() * 4} bytes")
print(f"FP16 memory: {fp16_tensor.numel() * 2} bytes") 
print(f"BF16 memory: {bf16_tensor.numel() * 2} bytes")

# Type conversion
fp16_from_fp32 = fp32_tensor.half()    # Convert to FP16
fp32_from_fp16 = fp16_tensor.float()   # Convert to FP32
```

## Automatic Mixed Precision (AMP)

### Basic AMP Usage

Genesis provides automatic mixed precision through the `autocast` context and enable flag:

```python
import genesis
import genesis.nn as nn

# Enable automatic mixed precision globally
genesis.enable_autocast = True

# Create model and data
model = nn.Linear(784, 10).cuda()
x = genesis.randn(32, 784, device='cuda')
labels = genesis.randint(0, 10, (32,), device='cuda')

# Forward pass with automatic casting
outputs = model(x)  # Automatically uses mixed precision

# Loss computation (typically done in FP32)
criterion = nn.CrossEntropyLoss()
loss = criterion(outputs, labels)

print(f"Input dtype: {x.dtype}")
print(f"Output dtype: {outputs.dtype}")
print(f"Loss dtype: {loss.dtype}")
```

### Manual AMP Control

For fine-grained control, use the `autocast` context manager:

```python
import genesis

# Disable global autocast
genesis.enable_autocast = False

# Model setup
model = nn.Sequential(
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Linear(256, 10)
).cuda()

x = genesis.randn(32, 784, device='cuda')

# Manual mixed precision control
with genesis.autocast():
    # Operations inside this block use FP16/BF16
    hidden = model[0](x)  # Linear layer in FP16
    activated = model[1](hidden)  # ReLU in FP16
    
# Operations outside use default precision
outputs = model[2](activated)  # This will be FP32

print(f"Hidden dtype: {hidden.dtype}")
print(f"Activated dtype: {activated.dtype}")
print(f"Output dtype: {outputs.dtype}")
```

## Training with Mixed Precision

### Simple Mixed Precision Training Loop

```python
import genesis
import genesis.nn as nn
import genesis.optim as optim

# Model setup
model = nn.Sequential(
    nn.Linear(784, 512),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(512, 256),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(256, 10)
).cuda()

# Optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Loss function
criterion = nn.CrossEntropyLoss()

# Enable mixed precision
genesis.enable_autocast = True

def train_epoch_amp(model, dataloader, optimizer, criterion):
    model.train()
    total_loss = 0.0
    
    for batch_idx, (data, targets) in enumerate(dataloader):
        data = data.cuda()
        targets = targets.cuda()
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass with mixed precision
        outputs = model(data)
        loss = criterion(outputs, targets)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping (important for stability)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # Optimizer step
        optimizer.step()
        
        total_loss += loss.item()
        
        if batch_idx % 100 == 0:
            print(f'Batch {batch_idx}: loss={loss.item():.4f}')
    
    return total_loss / len(dataloader)

# Training
for epoch in range(10):
    avg_loss = train_epoch_amp(model, train_loader, optimizer, criterion)
    print(f'Epoch {epoch}: avg loss = {avg_loss:.4f}')
```

### Advanced Mixed Precision with Loss Scaling

For training stability, especially with FP16, loss scaling is recommended:

```python
class GradScaler:
    """Gradient scaler for mixed precision training."""
    
    def __init__(self, init_scale=2**16, growth_factor=2.0, backoff_factor=0.5, 
                 growth_interval=2000):
        self.scale = init_scale
        self.growth_factor = growth_factor
        self.backoff_factor = backoff_factor
        self.growth_interval = growth_interval
        self._growth_tracker = 0
    
    def scale_loss(self, loss):
        """Scale loss to prevent gradient underflow."""
        return loss * self.scale
    
    def unscale_gradients(self, optimizer):
        """Unscale gradients before optimizer step."""
        for param_group in optimizer.param_groups:
            for param in param_group['params']:
                if param.grad is not None:
                    param.grad.data.div_(self.scale)
    
    def step(self, optimizer):
        """Step optimizer with gradient overflow detection."""
        # Check for gradient overflow
        has_overflow = self._check_overflow(optimizer)
        
        if has_overflow:
            # Skip optimizer step and reduce scale
            self.scale *= self.backoff_factor
            self.scale = max(self.scale, 1.0)
            self._growth_tracker = 0
            return False
        else:
            # Normal optimizer step
            optimizer.step()
            
            # Increase scale periodically
            self._growth_tracker += 1
            if self._growth_tracker >= self.growth_interval:
                self.scale *= self.growth_factor
                self._growth_tracker = 0
            
            return True
    
    def _check_overflow(self, optimizer):
        """Check if any gradients have overflowed."""
        for param_group in optimizer.param_groups:
            for param in param_group['params']:
                if param.grad is not None:
                    if genesis.isnan(param.grad).any() or genesis.isinf(param.grad).any():
                        return True
        return False

# Training with gradient scaling
scaler = GradScaler()

def train_with_scaling(model, dataloader, optimizer, criterion, scaler):
    model.train()
    total_loss = 0.0
    successful_steps = 0
    
    for batch_idx, (data, targets) in enumerate(dataloader):
        data = data.cuda()
        targets = targets.cuda()
        
        optimizer.zero_grad()
        
        # Forward pass with mixed precision
        with genesis.autocast():
            outputs = model(data)
            loss = criterion(outputs, targets)
        
        # Scale loss to prevent gradient underflow
        scaled_loss = scaler.scale_loss(loss)
        scaled_loss.backward()
        
        # Unscale gradients and check for overflow
        scaler.unscale_gradients(optimizer)
        
        # Gradient clipping on unscaled gradients
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # Step optimizer with overflow detection
        if scaler.step(optimizer):
            successful_steps += 1
        
        total_loss += loss.item()
        
        if batch_idx % 100 == 0:
            print(f'Batch {batch_idx}: loss={loss.item():.4f}, scale={scaler.scale:.0f}')
    
    success_rate = successful_steps / len(dataloader)
    print(f'Training success rate: {success_rate:.1%}')
    
    return total_loss / len(dataloader)
```

## Precision-Specific Considerations

### FP16 (Half Precision)

```python
import genesis

# FP16 characteristics
fp16_info = {
    'range': '±65,504',
    'precision': '~3-4 decimal digits',
    'special_values': ['inf', '-inf', 'nan'],
    'benefits': ['Faster on Tensor Cores', '50% memory reduction'],
    'challenges': ['Limited range', 'Gradient underflow']
}

# Best practices for FP16
def create_fp16_model():
    model = nn.Sequential(
        nn.Linear(784, 256),
        nn.LayerNorm(256),  # LayerNorm works well with FP16
        nn.ReLU(),
        nn.Linear(256, 10)
    )
    
    # Initialize with appropriate scale for FP16
    for module in model.modules():
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight, gain=1.0)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    
    return model

# Monitor FP16 training
def check_fp16_health(model):
    """Check model health during FP16 training."""
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            param_norm = param.norm().item()
            
            print(f"{name}:")
            print(f"  Param norm: {param_norm:.2e}")
            print(f"  Grad norm: {grad_norm:.2e}")
            
            # Check for problematic values
            if grad_norm < 1e-7:
                print(f"  WARNING: Very small gradients detected!")
            if grad_norm > 1e4:
                print(f"  WARNING: Very large gradients detected!")
```

### BF16 (Brain Float)

```python
import genesis

# BF16 advantages
bf16_info = {
    'range': 'Same as FP32 (±3.4×10^38)',
    'precision': '~2-3 decimal digits', 
    'benefits': ['Larger range than FP16', 'More stable training'],
    'hardware': ['A100', 'H100', 'TPUs']
}

# BF16 is often more stable than FP16
def train_with_bf16():
    # Create model with BF16
    model = nn.Linear(1000, 100).cuda()
    x = genesis.randn(32, 1000, dtype=genesis.bfloat16, device='cuda')
    
    # BF16 forward pass
    output = model(x)
    print(f"Input: {x.dtype}, Output: {output.dtype}")
    
    # BF16 typically doesn't need loss scaling
    loss = output.sum()
    loss.backward()
    
    return model

# Compare precisions
def compare_precisions():
    sizes = [100, 1000, 10000]
    
    for size in sizes:
        # Create test data
        data_fp32 = genesis.randn(size, size)
        data_fp16 = data_fp32.half()
        data_bf16 = data_fp32.to(genesis.bfloat16)
        
        # Simple computation
        result_fp32 = genesis.matmul(data_fp32, data_fp32)
        result_fp16 = genesis.matmul(data_fp16, data_fp16)
        result_bf16 = genesis.matmul(data_bf16, data_bf16)
        
        # Compare accuracy
        error_fp16 = (result_fp32 - result_fp16.float()).abs().mean()
        error_bf16 = (result_fp32 - result_bf16.float()).abs().mean()
        
        print(f"Size {size}x{size}:")
        print(f"  FP16 error: {error_fp16:.2e}")
        print(f"  BF16 error: {error_bf16:.2e}")
```

## Memory Optimization

### Memory Usage Analysis

```python
import genesis

def analyze_memory_usage():
    """Analyze memory usage of different precision types."""
    
    # Model sizes
    sizes = [(1000, 1000), (2000, 2000), (5000, 5000)]
    
    for h, w in sizes:
        print(f"\nTensor size: {h}x{w}")
        
        # Create tensors
        fp32_tensor = genesis.randn(h, w, dtype=genesis.float32, device='cuda')
        fp16_tensor = genesis.randn(h, w, dtype=genesis.float16, device='cuda')
        bf16_tensor = genesis.randn(h, w, dtype=genesis.bfloat16, device='cuda')
        
        # Memory usage
        fp32_memory = fp32_tensor.numel() * 4  # 4 bytes per float32
        fp16_memory = fp16_tensor.numel() * 2  # 2 bytes per float16
        bf16_memory = bf16_tensor.numel() * 2  # 2 bytes per bfloat16
        
        print(f"  FP32: {fp32_memory / 1e6:.1f} MB")
        print(f"  FP16: {fp16_memory / 1e6:.1f} MB ({fp16_memory/fp32_memory:.1%})")
        print(f"  BF16: {bf16_memory / 1e6:.1f} MB ({bf16_memory/fp32_memory:.1%})")
        
        # Cleanup
        del fp32_tensor, fp16_tensor, bf16_tensor
        genesis.cuda.empty_cache()

analyze_memory_usage()
```

### Gradient Checkpointing with Mixed Precision

```python
import genesis
import genesis.nn as nn

class CheckpointedModule(nn.Module):
    """Module with gradient checkpointing support."""
    
    def __init__(self, layers):
        super().__init__()
        self.layers = nn.ModuleList(layers)
        self.checkpoint = True
    
    def forward(self, x):
        def run_layers(x, layers):
            for layer in layers:
                x = layer(x)
            return x
        
        if self.training and self.checkpoint:
            # Use gradient checkpointing to save memory
            return genesis.utils.checkpoint(run_layers, x, self.layers)
        else:
            return run_layers(x, self.layers)

# Create memory-efficient model
def create_checkpointed_model():
    layers = [
        nn.Linear(1024, 1024),
        nn.ReLU(),
        nn.Linear(1024, 1024),
        nn.ReLU(),
        nn.Linear(1024, 512),
        nn.ReLU(),
        nn.Linear(512, 10)
    ]
    
    return CheckpointedModule(layers)

# Training with checkpointing and mixed precision
def train_memory_efficient():
    model = create_checkpointed_model().cuda()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Enable mixed precision
    genesis.enable_autocast = True
    
    for epoch in range(10):
        for batch in dataloader:
            data, targets = batch
            data = data.cuda()
            targets = targets.cuda()
            
            optimizer.zero_grad()
            
            # Forward pass with checkpointing and mixed precision
            outputs = model(data)
            loss = nn.CrossEntropyLoss()(outputs, targets)
            
            # Backward pass
            loss.backward()
            optimizer.step()
        
        print(f"Epoch {epoch} completed")
```

## Performance Benchmarking

### Mixed Precision Performance Comparison

```python
import genesis
import time

def benchmark_precision_performance():
    """Benchmark different precision formats."""
    
    # Model setup
    sizes = [512, 1024, 2048]
    batch_sizes = [16, 32, 64]
    
    results = {}
    
    for size in sizes:
        for batch_size in batch_sizes:
            print(f"\nBenchmarking: size={size}, batch_size={batch_size}")
            
            # Create models
            model_fp32 = nn.Linear(size, size).cuda()
            model_fp16 = nn.Linear(size, size).cuda().half()
            
            # Create data
            data_fp32 = genesis.randn(batch_size, size, device='cuda')
            data_fp16 = data_fp32.half()
            
            # Benchmark FP32
            torch.cuda.synchronize()
            start_time = time.time()
            
            for _ in range(100):
                output_fp32 = model_fp32(data_fp32)
            
            torch.cuda.synchronize()
            fp32_time = time.time() - start_time
            
            # Benchmark FP16
            torch.cuda.synchronize()
            start_time = time.time()
            
            for _ in range(100):
                output_fp16 = model_fp16(data_fp16)
            
            torch.cuda.synchronize()
            fp16_time = time.time() - start_time
            
            # Results
            speedup = fp32_time / fp16_time
            print(f"  FP32 time: {fp32_time:.3f}s")
            print(f"  FP16 time: {fp16_time:.3f}s") 
            print(f"  Speedup: {speedup:.2f}x")
            
            results[(size, batch_size)] = {
                'fp32_time': fp32_time,
                'fp16_time': fp16_time,
                'speedup': speedup
            }
    
    return results

# Run benchmark
benchmark_results = benchmark_precision_performance()
```

## Best Practices and Troubleshooting

### Best Practices

1. **Start Simple**: Begin with automatic mixed precision before manual control
2. **Monitor Training**: Watch for gradient underflow/overflow
3. **Use Loss Scaling**: Essential for FP16 stability
4. **Gradient Clipping**: Helps prevent gradient explosion
5. **Layer-wise Precision**: Some layers may need FP32 (e.g., batch norm)

### Common Issues and Solutions

```python
# Issue 1: Gradient Underflow
def handle_gradient_underflow():
    """Handle gradient underflow in FP16 training."""
    
    # Solution 1: Use loss scaling
    scaler = GradScaler(init_scale=2**16)
    
    # Solution 2: Skip problematic batches
    def safe_backward(loss, scaler):
        scaled_loss = scaler.scale_loss(loss)
        scaled_loss.backward()
        
        # Check for problems before optimizer step
        has_inf_or_nan = any(
            genesis.isinf(p.grad).any() or genesis.isnan(p.grad).any()
            for p in model.parameters() 
            if p.grad is not None
        )
        
        if has_inf_or_nan:
            print("Skipping step due to inf/nan gradients")
            optimizer.zero_grad()
            return False
        
        return True

# Issue 2: Model Divergence
def prevent_model_divergence():
    """Prevent model divergence in mixed precision."""
    
    # Solution 1: Lower learning rate
    optimizer = optim.Adam(model.parameters(), lr=0.0001)  # Lower LR
    
    # Solution 2: Warmup schedule
    scheduler = optim.get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=1000, num_training_steps=10000
    )
    
    # Solution 3: Monitor loss closely
    def check_loss_stability(loss, loss_history):
        loss_history.append(loss.item())
        
        if len(loss_history) > 100:
            recent_losses = loss_history[-50:]
            if any(l > 10 * min(recent_losses) for l in recent_losses):
                print("WARNING: Loss instability detected!")
                return False
        
        return True

# Issue 3: Accuracy Degradation
def maintain_accuracy():
    """Maintain model accuracy with mixed precision."""
    
    # Solution 1: Use BF16 instead of FP16
    genesis.enable_autocast = True
    default_dtype = genesis.bfloat16
    
    # Solution 2: Keep critical layers in FP32
    class MixedPrecisionModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.features = nn.Sequential(
                nn.Linear(784, 256),  # FP16/BF16
                nn.ReLU(),
                nn.Linear(256, 128),  # FP16/BF16
                nn.ReLU()
            )
            
            # Keep output layer in FP32 for stability
            self.classifier = nn.Linear(128, 10).float()
        
        def forward(self, x):
            with genesis.autocast():
                features = self.features(x)
            
            # Output layer in FP32
            output = self.classifier(features.float())
            return output
```

### Debugging Mixed Precision Training

```python
def debug_mixed_precision():
    """Debug mixed precision training issues."""
    
    # 1. Check tensor dtypes throughout the model
    def print_tensor_info(tensor, name):
        print(f"{name}:")
        print(f"  Shape: {tensor.shape}")
        print(f"  Dtype: {tensor.dtype}")
        print(f"  Device: {tensor.device}")
        print(f"  Requires grad: {tensor.requires_grad}")
        print(f"  Min/Max: {tensor.min():.2e} / {tensor.max():.2e}")
        print()
    
    # 2. Monitor gradient norms
    def check_gradient_norms(model):
        total_norm = 0.0
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                total_norm += grad_norm ** 2
                print(f"{name}: grad_norm = {grad_norm:.2e}")
        
        total_norm = total_norm ** 0.5
        print(f"Total gradient norm: {total_norm:.2e}")
        return total_norm
    
    # 3. Validate numerical stability
    def check_numerical_stability(tensor):
        """Check for numerical issues."""
        has_nan = genesis.isnan(tensor).any()
        has_inf = genesis.isinf(tensor).any()
        
        if has_nan:
            print("WARNING: NaN values detected!")
        if has_inf:
            print("WARNING: Inf values detected!")
        
        return not (has_nan or has_inf)

# Usage in training loop
for epoch in range(num_epochs):
    for batch_idx, (data, targets) in enumerate(dataloader):
        # Forward pass
        outputs = model(data)
        loss = criterion(outputs, targets)
        
        # Debug information
        if batch_idx % 100 == 0:
            print(f"Epoch {epoch}, Batch {batch_idx}:")
            print_tensor_info(data, "Input")
            print_tensor_info(outputs, "Output") 
            print_tensor_info(loss, "Loss")
            
            # Check gradients after backward pass
            loss.backward()
            grad_norm = check_gradient_norms(model)
            
            if grad_norm > 10.0:
                print("WARNING: Large gradient norm detected!")
```

This comprehensive guide covers all aspects of mixed precision training in Genesis, from basic usage to advanced optimization techniques and troubleshooting strategies.