# Function System

The Function system in Genesis provides the foundation for automatic differentiation by defining how operations are executed in the forward pass and how gradients are computed in the backward pass.

## 📋 Overview

The Function system is built around the `Function` base class, which encapsulates:
- Forward computation logic
- Backward gradient computation
- Context management for storing intermediate values
- Integration with the automatic differentiation engine

## 🏗️ Architecture

```mermaid
graph TB
    subgraph "Function System"
        A[Function Base Class] --> B[apply() Method]
        A --> C[forward() Method]
        A --> D[backward() Method]
        E[Context] --> F[save_for_backward()]
        E --> G[saved_tensors]
    end

    subgraph "Autograd Integration"
        B --> H[Computation Graph]
        H --> I[Gradient Flow]
        I --> J[Backward Pass]
    end

    subgraph "Built-in Functions"
        K[AddFunction] --> A
        L[MulFunction] --> A
        M[MatMulFunction] --> A
        N[ReluFunction] --> A
    end

    style A fill:#e1f5fe
    style E fill:#f3e5f5
    style H fill:#e8f5e8
```

## 🎯 Core Concepts

### Function Base Class
The `Function` class provides the interface for all operations:

```python
class Function:
    """Base class for all automatic differentiation functions."""

    @staticmethod
    def apply(*args):
        """Apply the function with automatic differentiation support."""
        ctx = Context()

        # Forward pass
        result = cls.forward(ctx, *args)

        # Set up backward pass if any input requires gradients
        if any(tensor.requires_grad for tensor in args if isinstance(tensor, Tensor)):
            result.set_creator(ctx, cls.backward)

        return result

    @staticmethod
    def forward(ctx, *args):
        """Compute forward pass. Must be implemented by subclasses."""
        raise NotImplementedError

    @staticmethod
    def backward(ctx, *grad_outputs):
        """Compute backward pass. Must be implemented by subclasses."""
        raise NotImplementedError
```

### Context Management
The `Context` class manages information needed for backward computation:

```python
class Context:
    """Context for storing information needed during backward pass."""

    def __init__(self):
        self.saved_tensors = []
        self.saved_variables = {}

    def save_for_backward(self, *tensors):
        """Save tensors for use in backward pass."""
        self.saved_tensors.extend(tensors)

    def save_variable(self, name, value):
        """Save a variable for use in backward pass."""
        self.saved_variables[name] = value
```

## 💻 Implementation Examples

### Basic Arithmetic Function
```python
class AddFunction(Function):
    """Addition function with gradient support."""

    @staticmethod
    def forward(ctx, a, b):
        """Forward pass: compute a + b."""
        # No need to save inputs for addition
        return genesis.ops.add(a, b)

    @staticmethod
    def backward(ctx, grad_output):
        """Backward pass: gradients flow unchanged."""
        return grad_output, grad_output

# Usage
add = AddFunction.apply
c = add(a, b)  # Equivalent to a + b with autograd support
```

### Matrix Multiplication Function
```python
class MatMulFunction(Function):
    """Matrix multiplication with gradient support."""

    @staticmethod
    def forward(ctx, a, b):
        """Forward pass: compute a @ b."""
        ctx.save_for_backward(a, b)
        return genesis.ops.matmul(a, b)

    @staticmethod
    def backward(ctx, grad_output):
        """Backward pass: compute gradients using chain rule."""
        a, b = ctx.saved_tensors

        grad_a = genesis.ops.matmul(grad_output, b.transpose(-2, -1))
        grad_b = genesis.ops.matmul(a.transpose(-2, -1), grad_output)

        return grad_a, grad_b

# Usage
matmul = MatMulFunction.apply
c = matmul(a, b)  # Equivalent to a @ b with autograd support
```

### Activation Function with Context
```python
class ReluFunction(Function):
    """ReLU activation with gradient support."""

    @staticmethod
    def forward(ctx, input):
        """Forward pass: compute max(0, input)."""
        output = genesis.ops.maximum(input, 0)
        ctx.save_for_backward(input)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        """Backward pass: gradient is 0 for negative inputs."""
        input, = ctx.saved_tensors
        mask = input > 0
        return grad_output * mask

# Usage
relu = ReluFunction.apply
activated = relu(x)
```

## 🚀 Advanced Features

### In-Place Operations
```python
class AddInplaceFunction(Function):
    """In-place addition function."""

    @staticmethod
    def forward(ctx, a, b):
        """Forward pass: modify a in-place."""
        ctx.save_variable('original_a', a.clone())
        a.add_(b)
        return a

    @staticmethod
    def backward(ctx, grad_output):
        """Backward pass for in-place operation."""
        return grad_output, grad_output
```

### Multi-Output Functions
```python
class SplitFunction(Function):
    """Function that returns multiple outputs."""

    @staticmethod
    def forward(ctx, input, split_sizes):
        """Split input tensor into multiple parts."""
        ctx.save_variable('split_sizes', split_sizes)
        return genesis.ops.split(input, split_sizes)

    @staticmethod
    def backward(ctx, *grad_outputs):
        """Concatenate gradients from multiple outputs."""
        grad_input = genesis.ops.cat(grad_outputs, dim=0)
        return grad_input, None  # None for split_sizes (no gradient)
```

### Custom Context Variables
```python
class ScaleFunction(Function):
    """Scale tensor by a constant factor."""

    @staticmethod
    def forward(ctx, input, scale_factor):
        """Scale input by constant factor."""
        ctx.save_variable('scale_factor', scale_factor)
        return input * scale_factor

    @staticmethod
    def backward(ctx, grad_output):
        """Scale gradient by same factor."""
        scale_factor = ctx.saved_variables['scale_factor']
        return grad_output * scale_factor, None
```

## 🔧 Integration with Operations

### Registering Functions with Dispatcher
```python
# Register function with operation dispatcher
genesis.ops.register_function('add', AddFunction.apply)
genesis.ops.register_function('matmul', MatMulFunction.apply)
genesis.ops.register_function('relu', ReluFunction.apply)

# Now operations automatically use the registered functions
x = genesis.tensor([1, 2, 3], requires_grad=True)
y = genesis.tensor([4, 5, 6], requires_grad=True)
z = x + y  # Automatically uses AddFunction
```

### Custom Operation Definition
```python
def custom_operation(input, param):
    """Define custom operation using Function."""
    return CustomFunction.apply(input, param)

# Register as operation
genesis.ops.register_operation('custom_op', custom_operation)

# Use like any other operation
result = genesis.custom_op(tensor, param)
```

## 📊 Performance Considerations

### Memory Efficiency
```python
class EfficientFunction(Function):
    """Memory-efficient function implementation."""

    @staticmethod
    def forward(ctx, input):
        # Only save what's needed for backward
        ctx.save_for_backward(input.detach())  # Detach to avoid recursive gradients

        # Compute result efficiently
        result = efficient_computation(input)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        # Compute gradient efficiently
        return efficient_gradient_computation(input, grad_output)
```

### Numerical Stability
```python
class StableFunction(Function):
    """Numerically stable function implementation."""

    @staticmethod
    def forward(ctx, input):
        # Use numerically stable computation
        output = stable_computation(input)
        ctx.save_for_backward(input, output)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, output = ctx.saved_tensors
        # Use stable gradient computation
        return stable_gradient(input, output, grad_output)
```

## 🔍 Debugging and Testing

### Function Testing
```python
def test_function_gradients():
    """Test function gradient computation."""
    x = genesis.tensor([1.0, 2.0, 3.0], requires_grad=True)

    # Test forward pass
    y = CustomFunction.apply(x)

    # Test backward pass
    y.backward(genesis.tensor([1.0, 1.0, 1.0]))

    # Check gradients
    assert x.grad is not None
    print(f"Gradient: {x.grad}")

# Numerical gradient checking
def numerical_gradient_check(func, input, eps=1e-5):
    """Check gradients using numerical differentiation."""
    # Implementation of numerical gradient checking
    pass
```

### Debugging Context
```python
class DebugFunction(Function):
    """Function with debugging information."""

    @staticmethod
    def forward(ctx, input):
        print(f"Forward: input shape = {input.shape}")
        ctx.save_for_backward(input)
        result = computation(input)
        print(f"Forward: output shape = {result.shape}")
        return result

    @staticmethod
    def backward(ctx, grad_output):
        print(f"Backward: grad_output shape = {grad_output.shape}")
        input, = ctx.saved_tensors
        grad_input = gradient_computation(input, grad_output)
        print(f"Backward: grad_input shape = {grad_input.shape}")
        return grad_input
```

## 🔗 See Also

- [Tensor System](tensor.md) - Tensor class and autograd integration
- [Core Components Overview](index.md) - Overall system architecture
- [Tensor System](tensor.md) - Core tensor operations and automatic differentiation
- [Custom Operations Guide](../tutorials/custom-ops.md) - Creating custom operations