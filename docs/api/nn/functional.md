# Functional Operations Interface (genesis.nn.functional)

Genesis functional interface provides stateless tensor operation functions that can be called directly on tensors without creating module instances.

## Module Overview

`genesis.nn.functional` (commonly imported as `F`) includes:
- Activation functions (relu, sigmoid, softmax, etc.)
- Loss functions (cross_entropy, mse_loss, etc.) 
- Tensor operations (matmul, transpose, reshape, etc.)
- Normalization functions (layer_norm, batch_norm, etc.)
- Attention mechanisms (scaled_dot_product_attention, etc.)

## Activation Functions

### relu
```python
def relu(x: Tensor, inplace: bool = False) -> Tensor:
    """
    ReLU activation function: f(x) = max(0, x)
    
    Args:
        x: Tensor - Input tensor
        inplace: bool - Whether to perform operation inplace
        
    Returns:
        Tensor - Activated tensor
        
    Example:
        >>> x = genesis.randn(10)
        >>> y = F.relu(x)
        >>> # Negative values become 0, positive unchanged
    """
```

### sigmoid
```python
def sigmoid(x: Tensor) -> Tensor:
    """
    Sigmoid activation function: f(x) = 1 / (1 + exp(-x))
    
    Args:
        x: Tensor - Input tensor
        
    Returns:
        Tensor - Sigmoid-activated tensor (values in [0, 1])
        
    Example:
        >>> x = genesis.randn(5, 5)
        >>> y = F.sigmoid(x)
        >>> # All values will be between 0 and 1
    """
```

### softmax
```python
def softmax(x: Tensor, dim: int = -1) -> Tensor:
    """
    Softmax function: f(x_i) = exp(x_i) / sum(exp(x_j))
    
    Args:
        x: Tensor - Input tensor
        dim: int - Dimension along which to apply softmax
        
    Returns:
        Tensor - Softmax output (sums to 1 along dim)
        
    Example:
        >>> x = genesis.randn(2, 3)
        >>> y = F.softmax(x, dim=1)
        >>> # Each row sums to 1
    """
```

### gelu
```python
def gelu(x: Tensor, approximate: str = 'none') -> Tensor:
    """
    Gaussian Error Linear Unit: f(x) = x * Φ(x)
    
    Args:
        x: Tensor - Input tensor
        approximate: str - Approximation method ('none', 'tanh')
        
    Returns:
        Tensor - GELU-activated tensor
        
    Example:
        >>> x = genesis.randn(100)
        >>> y = F.gelu(x)
        >>> # Smooth activation, similar to ReLU but differentiable at 0
    """
```

## Loss Functions

### cross_entropy
```python
def cross_entropy(input: Tensor, target: Tensor, weight: Tensor = None, 
                  reduction: str = 'mean') -> Tensor:
    """
    Cross entropy loss for classification tasks.
    
    Args:
        input: Tensor - Logits tensor of shape (N, C) 
        target: Tensor - Target class indices of shape (N,)
        weight: Tensor - Manual rescaling weight for each class
        reduction: str - Reduction method ('mean', 'sum', 'none')
        
    Returns:
        Tensor - Cross entropy loss
        
    Example:
        >>> logits = genesis.randn(32, 10)  # 32 samples, 10 classes
        >>> targets = genesis.randint(0, 10, (32,))
        >>> loss = F.cross_entropy(logits, targets)
    """
```

### mse_loss
```python
def mse_loss(input: Tensor, target: Tensor, reduction: str = 'mean') -> Tensor:
    """
    Mean Squared Error loss.
    
    Args:
        input: Tensor - Predicted values
        target: Tensor - Target values
        reduction: str - Reduction method ('mean', 'sum', 'none')
        
    Returns:
        Tensor - MSE loss
        
    Example:
        >>> pred = genesis.randn(100, 1)
        >>> target = genesis.randn(100, 1) 
        >>> loss = F.mse_loss(pred, target)
    """
```

## Tensor Operations

### matmul
```python
def matmul(input: Tensor, other: Tensor) -> Tensor:
    """
    Matrix multiplication of two tensors.
    
    Args:
        input: Tensor - First tensor
        other: Tensor - Second tensor
        
    Returns:
        Tensor - Matrix multiplication result
        
    Example:
        >>> a = genesis.randn(3, 4)
        >>> b = genesis.randn(4, 5) 
        >>> c = F.matmul(a, b)  # Shape: (3, 5)
    """
```

### transpose
```python
def transpose(input: Tensor, dim0: int, dim1: int) -> Tensor:
    """
    Transpose two dimensions of a tensor.
    
    Args:
        input: Tensor - Input tensor
        dim0: int - First dimension to transpose
        dim1: int - Second dimension to transpose
        
    Returns:
        Tensor - Transposed tensor
        
    Example:
        >>> x = genesis.randn(2, 3, 4)
        >>> y = F.transpose(x, 0, 2)  # Shape: (4, 3, 2)
    """
```

## Normalization Functions

### layer_norm
```python
def layer_norm(input: Tensor, normalized_shape: list, weight: Tensor = None,
               bias: Tensor = None, eps: float = 1e-5) -> Tensor:
    """
    Layer normalization.
    
    Args:
        input: Tensor - Input tensor
        normalized_shape: list - Shape over which to normalize
        weight: Tensor - Learnable scale parameter
        bias: Tensor - Learnable shift parameter
        eps: float - Small value to avoid division by zero
        
    Returns:
        Tensor - Layer-normalized tensor
        
    Example:
        >>> x = genesis.randn(32, 128)
        >>> y = F.layer_norm(x, [128])
    """
```

### batch_norm
```python
def batch_norm(input: Tensor, running_mean: Tensor, running_var: Tensor,
               weight: Tensor = None, bias: Tensor = None, training: bool = True,
               momentum: float = 0.1, eps: float = 1e-5) -> Tensor:
    """
    Batch normalization.
    
    Args:
        input: Tensor - Input tensor (N, C, ...)
        running_mean: Tensor - Running mean statistics
        running_var: Tensor - Running variance statistics
        weight: Tensor - Learnable scale parameter
        bias: Tensor - Learnable shift parameter
        training: bool - Training mode flag
        momentum: float - Momentum for updating running statistics
        eps: float - Small value to avoid division by zero
        
    Returns:
        Tensor - Batch-normalized tensor
        
    Example:
        >>> x = genesis.randn(32, 64, 28, 28)
        >>> running_mean = genesis.zeros(64)
        >>> running_var = genesis.ones(64)
        >>> y = F.batch_norm(x, running_mean, running_var)
    """
```

## Attention Mechanisms

### scaled_dot_product_attention
```python
def scaled_dot_product_attention(query: Tensor, key: Tensor, value: Tensor,
                                 attn_mask: Tensor = None, dropout_p: float = 0.0,
                                 is_causal: bool = False) -> Tensor:
    """
    Scaled dot-product attention mechanism.
    
    Args:
        query: Tensor - Query tensor (..., L, E)
        key: Tensor - Key tensor (..., S, E)
        value: Tensor - Value tensor (..., S, Ev)
        attn_mask: Tensor - Attention mask
        dropout_p: float - Dropout probability
        is_causal: bool - Whether to apply causal mask
        
    Returns:
        Tensor - Attention output (..., L, Ev)
        
    Example:
        >>> seq_len, d_model = 10, 64
        >>> q = genesis.randn(1, seq_len, d_model)
        >>> k = genesis.randn(1, seq_len, d_model)
        >>> v = genesis.randn(1, seq_len, d_model)
        >>> out = F.scaled_dot_product_attention(q, k, v)
    """
```

## Utility Functions

### dropout
```python
def dropout(input: Tensor, p: float = 0.5, training: bool = True,
            inplace: bool = False) -> Tensor:
    """
    Applies dropout regularization.
    
    Args:
        input: Tensor - Input tensor
        p: float - Dropout probability
        training: bool - Training mode flag
        inplace: bool - Whether to perform operation inplace
        
    Returns:
        Tensor - Tensor with dropout applied (if training)
        
    Example:
        >>> x = genesis.randn(100, 50)
        >>> y = F.dropout(x, p=0.2, training=True)
    """
```

### linear
```python
def linear(input: Tensor, weight: Tensor, bias: Tensor = None) -> Tensor:
    """
    Linear transformation: y = xW^T + b
    
    Args:
        input: Tensor - Input tensor (..., in_features)
        weight: Tensor - Weight tensor (out_features, in_features)
        bias: Tensor - Bias tensor (out_features,)
        
    Returns:
        Tensor - Linear transformation result (..., out_features)
        
    Example:
        >>> x = genesis.randn(32, 784)
        >>> w = genesis.randn(10, 784)
        >>> b = genesis.randn(10)
        >>> y = F.linear(x, w, b)  # Shape: (32, 10)
    """
```

## Implementation Notes

- All functions support automatic differentiation through Genesis autograd system
- GPU acceleration is automatically applied when tensors are on CUDA devices  
- Memory-efficient implementations using Triton kernels for better performance
- Broadcasting rules follow standard PyTorch conventions
- Inplace operations (where supported) can reduce memory usage but should be used carefully with autograd