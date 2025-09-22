# Random Number Generation API

Genesis provides a PyTorch-compatible random number generation API with thread-safe state management and reproducibility guarantees.

## Functions

### Global Random State Management

#### `genesis.manual_seed(seed)`

Set the global random seed for reproducible results.

**Parameters:**
- `seed` (int): Random seed value

**Example:**
```python
import genesis

# Set seed for reproducibility
genesis.manual_seed(42)
x = genesis.rand(100, 100)  # Reproducible random tensor
```

#### `genesis.seed()`

Set random seed from current time or system entropy.

**Example:**
```python
genesis.seed()  # Random seed from system
```

#### `genesis.initial_seed()`

Get the initial random seed used.

**Returns:**
- int: The initial seed value

#### `genesis.get_rng_state()`

Get the current random number generator state.

**Returns:**
- Tensor: Current RNG state

**Example:**
```python
# Save current state
state = genesis.get_rng_state()
# ... perform random operations ...
genesis.set_rng_state(state)  # Restore state
```

#### `genesis.set_rng_state(new_state)`

Set the random number generator state.

**Parameters:**
- `new_state` (Tensor): RNG state to restore

### Thread-Safe Random Generation

#### `genesis.fork_rng(devices=None, enabled=True)`

Context manager for thread-safe random number generation.

**Parameters:**
- `devices` (list, optional): Devices to fork RNG state for
- `enabled` (bool): Whether to actually fork (default: True)

**Example:**
```python
with genesis.fork_rng():
    genesis.manual_seed(999)
    # Random operations here don't affect global state
    x = genesis.rand(10, 10)
# Global state is restored here
```

## Generator Class

### `genesis.Generator(device='cpu')`

Random number generator for controlled random state.

**Parameters:**
- `device` (str): Device for the generator (default: 'cpu')

**Methods:**

#### `generator.manual_seed(seed)`

Set the generator's random seed.

**Parameters:**
- `seed` (int): Random seed value

**Example:**
```python
gen = genesis.Generator()
gen.manual_seed(12345)

# Use generator for specific operations
x = genesis.rand(100, 100, generator=gen)
```

#### `generator.seed()`

Set random seed from system entropy.

#### `generator.initial_seed()`

Get the generator's initial seed.

#### `generator.get_state()`

Get the generator's current state.

#### `generator.set_state(new_state)`

Set the generator's state.

## Global Generator

#### `genesis.default_generator`

The default global random number generator instance.

**Example:**
```python
# Access default generator
gen = genesis.default_generator
state = gen.get_state()
```

## Usage Examples

### Basic Random Generation

```python
import genesis

# Set seed for reproducibility
genesis.manual_seed(42)

# Generate random tensors
x = genesis.rand(100, 100, device='cuda')
y = genesis.randn(50, 50, device='cpu')
z = genesis.randint(0, 10, (20, 20))
```

### Advanced State Management

```python
import genesis

# Save global state
saved_state = genesis.get_rng_state()

# Perform some random operations
genesis.manual_seed(123)
x = genesis.rand(10, 10)

# Restore previous state
genesis.set_rng_state(saved_state)
y = genesis.rand(10, 10)  # Same as if seed(123) never happened
```

### Thread-Safe Usage

```python
import genesis
import threading

def worker():
    with genesis.fork_rng():
        genesis.manual_seed(42)
        # Each thread has independent random state
        return genesis.rand(100, 100)

# Multiple threads won't interfere with each other
threads = [threading.Thread(target=worker) for _ in range(4)]
```

### Custom Generator Usage

```python
import genesis

# Create custom generator
gen1 = genesis.Generator()
gen1.manual_seed(111)

gen2 = genesis.Generator() 
gen2.manual_seed(222)

# Different generators produce different sequences
x1 = genesis.rand(10, 10, generator=gen1)
x2 = genesis.rand(10, 10, generator=gen2)  # Different from x1
```

## Implementation Notes

- The RNG API is designed for PyTorch compatibility
- Thread safety is guaranteed through proper state isolation
- State management supports both global and per-generator control
- All random functions accept an optional `generator` parameter
- The implementation uses NumPy's random number generator internally
- CUDA device random generation inherits from CPU state management

## See Also

- [Tensor Creation Functions](genesis.md#tensor-creation) - Functions that use random generation
- [Memory Management](memory.md) - For device-specific considerations