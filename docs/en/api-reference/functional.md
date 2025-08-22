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

*This page is under construction. Detailed function signatures and examples will be added.*