# Device API Reference

The Device module provides a unified interface for managing computation devices (CPU and CUDA) in Genesis.

## Device Class

```python
genesis.device(device_type)
```

### Parameters

- **device_type**: String specifying the device
  - `'cpu'`: CPU device
  - `'cuda'`: Default CUDA device (GPU 0)
  - `'cuda:0'`, `'cuda:1'`, etc.: Specific CUDA device by index

### Properties

- **type**: Device type ('cpu' or 'cuda')
- **index**: Device index (for CUDA devices)

### Methods

```python
device.is_available()    # Check if device is available
```

## Global Functions

### Device Query

```python
genesis.cuda_available()
```
Check if CUDA is available on the system.

**Returns**: `bool` - True if CUDA is available, False otherwise

### Current Device Management

```python
# Get current default device
genesis.device.get_default_device()

# Set default device
genesis.device.set_default_device(device)
```

## Device Context

Genesis automatically manages device context for operations. All tensors and operations are executed on their respective devices.

### Automatic Device Selection

When creating tensors without specifying a device:
- If CUDA is available and enabled, uses CUDA by default
- Otherwise, uses CPU

## Examples

### Basic Device Usage

```python
import genesis

# Create device objects
cpu_device = genesis.device('cpu')
cuda_device = genesis.device('cuda')

# Check availability
if cuda_device.is_available():
    print("CUDA is available")
    device = cuda_device
else:
    print("Using CPU")
    device = cpu_device

# Create tensor on specific device
x = genesis.randn(100, 100, device=device)
```

### Multi-GPU Support

```python
import genesis

# Check number of GPUs
if genesis.cuda_available():
    # Use specific GPU
    device0 = genesis.device('cuda:0')
    device1 = genesis.device('cuda:1')

    # Create tensors on different GPUs
    x = genesis.randn(100, 100, device=device0)
    y = genesis.randn(100, 100, device=device1)

    # Move tensor to different device
    y_on_device0 = y.to(device0)
    result = x + y_on_device0  # Operations require same device
```

### Device-Agnostic Code

```python
import genesis

def get_device():
    """Get the best available device"""
    if genesis.cuda_available():
        return genesis.device('cuda')
    return genesis.device('cpu')

# Write device-agnostic code
device = get_device()

# All tensors created on the selected device
model_weights = genesis.randn(784, 256, device=device)
input_data = genesis.randn(32, 784, device=device)

# Computation happens on the selected device
output = input_data @ model_weights
```

### Device Memory Management

```python
import genesis

if genesis.cuda_available():
    device = genesis.device('cuda')

    # Create large tensor on GPU
    large_tensor = genesis.randn(10000, 10000, device=device)

    # Process data
    result = large_tensor.sum()

    # Move result to CPU for further processing
    result_cpu = result.cpu()
    print(f"Sum: {result_cpu.item()}")
```

## Performance Considerations

1. **Device Transfers**: Moving data between devices (CPU ↔ GPU) is expensive. Minimize transfers.

2. **Batch Operations**: GPUs excel at parallel operations. Use larger batch sizes when possible.

3. **Memory Management**: GPU memory is limited. Monitor usage and free unused tensors.

4. **Device Placement**: Ensure all tensors in an operation are on the same device.

## Environment Variables

Genesis respects the following environment variables:

- `CUDA_VISIBLE_DEVICES`: Limit which GPUs are visible to Genesis
  ```bash
  # Only use GPU 0 and 1
  CUDA_VISIBLE_DEVICES=0,1 python script.py
  ```

- `GENESIS_DEVICE`: Set default device
  ```bash
  # Force CPU usage
  GENESIS_DEVICE=cpu python script.py
  ```

## See Also

- [Tensor API](tensor.md) - Tensor operations and device placement
- [CUDA Backend](../backends/cuda.md) - CUDA-specific implementation details
- [Memory Management](memory.md) - Advanced memory management