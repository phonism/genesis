"""
CPU arithmetic operations implementation.
"""

import torch
from genesis.ops.dispatcher import register_cpu


@register_cpu("add")
def add(x, y):
    """Add two tensors element-wise."""
    return x + y


@register_cpu("sub")
def sub(x, y):
    """Subtract y from x element-wise."""
    return x - y


@register_cpu("mul")
def mul(x, y):
    """Multiply two tensors element-wise."""
    return x * y


@register_cpu("div")
def div(x, y):
    """Divide x by y element-wise."""
    return x / y


@register_cpu("truediv")
def truediv(x, y):
    """Divide x by y element-wise."""
    return x.__truediv__(y)


@register_cpu("rtruediv")
def rtruediv(x, y):
    """Right division (y/x)."""
    return x.__rtruediv__(y)


@register_cpu("pow")
def pow(x, scalar):
    """Raise tensor to power."""
    return x ** scalar


@register_cpu("neg")
def neg(x):
    """Negate tensor."""
    return -x


@register_cpu("abs")
def abs(x):
    """Absolute value."""
    return torch.abs(x)


@register_cpu("sign")
def sign(x):
    """Sign of tensor elements."""
    return torch.sign(x)


@register_cpu("sqrt")
def sqrt(x):
    """Square root."""
    return torch.sqrt(x)


@register_cpu("log")
def log(x):
    """Natural logarithm."""
    return torch.log(x)


@register_cpu("exp")
def exp(x):
    """Exponential."""
    return torch.exp(x)


@register_cpu("sin")
def sin(x):
    """Sine."""
    return torch.sin(x)


@register_cpu("cos")
def cos(x):
    """Cosine."""
    return torch.cos(x)


@register_cpu("tanh")
def tanh(x):
    """Hyperbolic tangent."""
    return torch.tanh(x)


@register_cpu("sqrt")
def sqrt(x):
    """Square root."""
    return torch.sqrt(x)


@register_cpu("maximum")
def maximum(x, y):
    """Element-wise maximum."""
    return torch.maximum(x, y)


# Inplace operations
@register_cpu("add_inplace")
def add_inplace(x, y):
    """Add y to x in-place."""
    x += y
    return x


@register_cpu("sub_inplace")
def sub_inplace(x, y):
    """Subtract y from x in-place."""
    x -= y
    return x


@register_cpu("mul_inplace")
def mul_inplace(x, y):
    """Multiply x by y in-place."""
    x *= y
    return x


@register_cpu("div_inplace")
def div_inplace(x, y):
    """Divide x by y in-place."""
    x /= y
    return x