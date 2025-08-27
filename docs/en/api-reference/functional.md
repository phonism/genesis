# Functional API Reference

Functional interface for Genesis operations - stateless functions for neural network operations.

## Core Functions

### Activation Functions
- `F.relu(x)` - ReLU activation
- `F.softmax(x, dim=-1)` - Softmax
- `F.gelu(x)` - GELU activation

### Loss Functions
- `F.mse_loss(input, target)` - Mean squared error
- `F.cross_entropy(input, target)` - Cross entropy loss

### Convolution Operations
- `F.conv2d(input, weight, bias)` - 2D convolution
- `F.linear(input, weight, bias)` - Linear transformation

### Tensor Manipulation Functions

#### `sort(input, dim=-1, descending=False, stable=False)`
Sort elements along a dimension.

```python
values, indices = genesis.sort(tensor, dim=1, descending=False)
```

**Parameters:**
- `input`: Input tensor
- `dim`: Dimension along which to sort
- `descending`: If True, sort in descending order
- `stable`: If True, stable sort (preserves order of equal elements)

**Returns:** Tuple of (values, indices) tensors

#### `topk(input, k, dim=-1, largest=True, sorted=True)`
Returns the k largest/smallest elements along a dimension.

```python
values, indices = genesis.topk(tensor, k=3, dim=1, largest=True)
```

**Parameters:**
- `input`: Input tensor
- `k`: Number of top values to return
- `dim`: Dimension along which to find top-k values
- `largest`: If True, return largest values; if False, return smallest
- `sorted`: If True, return values in sorted order

**Returns:** Tuple of (values, indices) tensors

#### `argsort(input, dim=-1, descending=False)`
Returns indices that sort a tensor along a dimension.

```python
indices = genesis.argsort(tensor, dim=1, descending=False)
```

**Parameters:**
- `input`: Input tensor
- `dim`: Dimension along which to sort
- `descending`: If True, sort in descending order

**Returns:** Tensor of indices

#### `gather(input, dim, index)`
Gather values along an axis specified by index.

```python
output = genesis.gather(tensor, dim=1, index=indices)
```

**Parameters:**
- `input`: Input tensor
- `dim`: Dimension along which to gather
- `index`: Index tensor with same number of dimensions as input

**Returns:** Tensor containing gathered values

#### `scatter_add(input, dim, index, src)`
Add values from src to input at positions specified by index.

```python
genesis.scatter_add(tensor, dim=1, index=indices, src=values)
```

**Parameters:**
- `input`: Input tensor (modified in-place)
- `dim`: Dimension along which to scatter
- `index`: Index tensor
- `src`: Source tensor containing values to add

**Returns:** Modified input tensor

#### `bincount(input, weights=None, minlength=0)`
Count occurrences of each value in integer tensor.

```python
counts = genesis.bincount(tensor, minlength=10)
```

**Parameters:**
- `input`: 1D integer tensor
- `weights`: Optional weights tensor
- `minlength`: Minimum length of output

**Returns:** Tensor containing counts

### Utility Functions

#### `allclose(input, other, rtol=1e-05, atol=1e-08, equal_nan=False)`
Test if all elements of input and other are close.

```python
result = genesis.allclose(tensor1, tensor2, rtol=1e-5, atol=1e-8)
```

**Parameters:**
- `input`: First tensor
- `other`: Second tensor
- `rtol`: Relative tolerance
- `atol`: Absolute tolerance
- `equal_nan`: Whether to consider NaN values as equal

**Returns:** Boolean scalar tensor

### Creation Functions

#### `eye(n, m=None, device=None, dtype=genesis.float32)`
Generate identity matrix.

```python
identity = genesis.eye(5)  # 5x5 identity matrix
rect_matrix = genesis.eye(3, 5)  # 3x5 matrix
```

#### `ones_like(tensor, dtype=None, device=None)`
Generate ones tensor with same shape as input.

```python
ones_tensor = genesis.ones_like(input_tensor)
```

#### `from_numpy(array, device=None, dtype=None)`
Create tensor from numpy array.

```python
np_array = np.array([1, 2, 3])
tensor = genesis.from_numpy(np_array)
```

*This reference covers the main functional operations. For complete API details, see the source documentation.*