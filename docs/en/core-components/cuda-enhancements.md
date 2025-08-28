# CUDA Enhancements

Genesis has made significant improvements in CUDA support, providing better device management, memory operations, and error handling capabilities.

## Device Management Enhancements

### New CUDA Functions

#### `set_device(device)`
Set the current CUDA device.

```python
import genesis.cuda as cuda

# Set current device to GPU 1
cuda.set_device(1)
print(f"Current device: {cuda.current_device()}")
```

#### `device_count()`
Get the number of available CUDA devices.

```python
device_count = cuda.device_count()
print(f"Available GPUs: {device_count}")
```

#### `get_device_name(device)`
Get the name of the specified device.

```python
device_name = cuda.get_device_name(0)
print(f"Device 0 name: {device_name}")
```

#### `synchronize(device=None)`
Synchronize CUDA operations to ensure all CUDA operations are complete.

```python
# Synchronize current device
cuda.synchronize()

# Synchronize specific device
cuda.synchronize(device=1)
```

## Device Property Enhancements

### New Device Class Properties

```python
device = genesis.device('cuda:0')

# PyTorch-compatible properties
print(f"Device type: {device.type}")      # 'cuda'
print(f"Device index: {device.index}")    # 0
```

These properties provide full compatibility with PyTorch, making migration easier.

## Tensor Validation Functions

### Numerical Validity Checking

Genesis now provides complete tensor numerical validity checking functionality:

#### `isinf(tensor)`
Check for infinite values in tensors.

```python
import genesis

tensor = genesis.tensor([1.0, float('inf'), -float('inf'), 2.0], device='cuda')
inf_mask = genesis.isinf(tensor)
print(inf_mask)  # [False, True, True, False]

# Check if there are any infinite values
has_inf = genesis.isinf(tensor).any()
if has_inf:
    print("Tensor contains infinite values!")
```

#### `isnan(tensor)`
Check for NaN values in tensors.

```python
tensor = genesis.tensor([1.0, float('nan'), 2.0, 3.0], device='cuda')
nan_mask = genesis.isnan(tensor)
print(nan_mask)  # [False, True, False, False]

# Check if there are any NaN values
has_nan = genesis.isnan(tensor).any()
if has_nan:
    print("Tensor contains NaN values!")
```

#### `isfinite(tensor)`
Check for finite values in tensors.

```python
tensor = genesis.tensor([1.0, float('inf'), float('nan'), 2.0], device='cuda')
finite_mask = genesis.isfinite(tensor)
print(finite_mask)  # [True, False, False, True]

# Keep only finite values
finite_tensor = tensor[finite_mask]
```

### Application in Training

```python
def check_model_gradients(model):
    """Check numerical stability of model gradients"""
    for name, param in model.named_parameters():
        if param.grad is not None:
            if genesis.isnan(param.grad).any():
                print(f"Warning: {name} gradients contain NaN!")
                return False
            if genesis.isinf(param.grad).any():
                print(f"Warning: {name} gradients contain infinity!")
                return False
    return True

# Use in training loop
for batch in dataloader:
    optimizer.zero_grad()
    loss = model(batch)
    loss.backward()
    
    if not check_model_gradients(model):
        print("Skipping batch due to gradient anomaly")
        continue
    
    optimizer.step()
```

## Mixed Precision Training Enhancements

### GradScaler Improvements

```python
import genesis.amp as amp

# Create gradient scaler
scaler = amp.GradScaler()

# Training loop
for batch in dataloader:
    with amp.autocast():
        outputs = model(batch['input'])
        loss = criterion(outputs, batch['target'])
    
    # Scale loss and backward pass
    scaler.scale(loss).backward()
    
    # Check gradient validity (now uses genesis native functions)
    scaler.step(optimizer)  # Internally uses genesis.isinf/isnan checks
    scaler.update()
```

## CUDA Memory Management

### Improved Memory Allocation

```python
import genesis.cuda as cuda

# Check CUDA availability
if cuda.is_available():
    print(f"CUDA available, device count: {cuda.device_count()}")
    
    # Get current device info
    current_dev = cuda.current_device()
    device_name = cuda.get_device_name(current_dev)
    print(f"Current device: {current_dev} ({device_name})")
    
    # Set specific device
    cuda.set_device(0)
    
    # Synchronize to ensure operations complete
    cuda.synchronize()
```

## Triton Kernel Optimizations

### Numerical Check Kernels

Genesis implements efficient Triton kernels for numerical checks:

```python
# High-performance numerical checks (internal implementation)
@triton.jit
def isinf_kernel(input_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    """Triton kernel to check if elements are infinite"""
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input values
    x = tl.load(input_ptr + offsets, mask=mask)
    
    # Check for infinity
    finite_max = 3.4028235e+38  # Maximum finite float32 value
    is_pos_inf = x > finite_max
    is_neg_inf = x < -finite_max
    result = is_pos_inf | is_neg_inf
    
    # Store result as boolean
    tl.store(output_ptr + offsets, result.to(tl.int8), mask=mask)
```

## Error Handling Improvements

### CUDA Error Checking

```python
import genesis.cuda as cuda

try:
    # CUDA operations
    cuda.set_device(0)
    cuda.synchronize()
except RuntimeError as e:
    if "CUDA" in str(e):
        print(f"CUDA error: {e}")
        # Handle CUDA error
    else:
        raise
```

### Device Availability Checking

```python
def safe_cuda_operation():
    """Safe CUDA operation example"""
    if not cuda.is_available():
        print("CUDA not available, using CPU")
        return genesis.device('cpu')
    
    try:
        device_count = cuda.device_count()
        if device_count == 0:
            print("No available CUDA devices")
            return genesis.device('cpu')
        
        # Select device
        cuda.set_device(0)
        return genesis.device('cuda:0')
    
    except RuntimeError as e:
        print(f"CUDA initialization failed: {e}")
        return genesis.device('cpu')
```

## Best Practices

### 1. Numerical Stability Checks

```python
def validate_tensor(tensor, name="tensor"):
    """Validate tensor numerical stability"""
    if genesis.isnan(tensor).any():
        raise ValueError(f"{name} contains NaN values")
    if genesis.isinf(tensor).any():
        raise ValueError(f"{name} contains infinite values")
    return True

# Use at critical points
loss = criterion(outputs, targets)
validate_tensor(loss, "loss")

gradients = model.get_gradients()
for name, grad in gradients.items():
    validate_tensor(grad, f"gradient_{name}")
```

### 2. Device Management

```python
class DeviceManager:
    """Device manager"""
    
    def __init__(self):
        self.available_devices = []
        self._detect_devices()
    
    def _detect_devices(self):
        """Detect available devices"""
        if cuda.is_available():
            for i in range(cuda.device_count()):
                device_name = cuda.get_device_name(i)
                self.available_devices.append({
                    'index': i,
                    'name': device_name,
                    'type': 'cuda'
                })
        else:
            self.available_devices.append({
                'index': None,
                'name': 'CPU',
                'type': 'cpu'
            })
    
    def get_best_device(self):
        """Get the best available device"""
        if self.available_devices and self.available_devices[0]['type'] == 'cuda':
            cuda.set_device(0)
            return genesis.device('cuda:0')
        return genesis.device('cpu')

# Usage example
device_manager = DeviceManager()
device = device_manager.get_best_device()
model = model.to(device)
```

### 3. Memory Management

```python
def memory_safe_operation(tensor, operation):
    """Memory-safe operation"""
    try:
        # Ensure on correct device
        if tensor.device.type == 'cuda':
            cuda.set_device(tensor.device.index)
        
        # Execute operation
        result = operation(tensor)
        
        # Synchronize to ensure completion
        if tensor.device.type == 'cuda':
            cuda.synchronize()
        
        return result
    
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print("GPU out of memory, attempting cleanup")
            # Can add memory cleanup logic here
        raise

# Usage example
result = memory_safe_operation(tensor, lambda x: x.matmul(weight))
```

## Performance Monitoring

### CUDA Operation Performance Monitoring

```python
import time

def profile_cuda_operation(operation, tensor, name="operation"):
    """Profile CUDA operation performance"""
    if tensor.device.type == 'cuda':
        cuda.synchronize()  # Ensure previous operations complete
    
    start_time = time.time()
    result = operation(tensor)
    
    if tensor.device.type == 'cuda':
        cuda.synchronize()  # Ensure operation completes
    
    end_time = time.time()
    print(f"{name} took: {(end_time - start_time)*1000:.2f} ms")
    
    return result

# Usage example
result = profile_cuda_operation(
    lambda x: genesis.matmul(x, weight),
    input_tensor,
    "matrix multiplication"
)
```

These CUDA enhancements provide better device management, numerical stability checking, and error handling, making Genesis more stable and user-friendly.