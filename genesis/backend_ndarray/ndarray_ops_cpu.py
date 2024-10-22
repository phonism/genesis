import torch
import operator
from functools import reduce

def prod(x):
    """
    prod
    """
    return reduce(operator.mul, x, 1)

def add(x, y):
    return x + y

def mul(x, y):
    return x * y

def truediv(x, y):
    return x.__truediv__(y)

def pow(x, scalar):
    return x ** scalar

def log(x):
    return torch.log(x)

def exp(x):
    return torch.exp(x)

def sin(x):
    return torch.sin(x)

def cos(x):
    return torch.cos(x)

def sqrt(x):
    return torch.sqrt(x)

def maximum(x, y):
    if isinstance(y, torch.Tensor):
        return torch.maximum(x, y)
    return torch.clamp(x, min=y)

def reduce_sum(x, axis=None, keepdims=False):
    return torch.sum(x, axis=axis, keepdims=keepdims)

def reduce_max(x, axis=None, keepdims=False):
    if axis is None:
        return torch.max(x)
    if isinstance(axis, tuple):
        axis = axis[0]
    return torch.max(x, dim=axis, keepdim=keepdims).values

def reshape(x, new_shape):
    return x.reshape(new_shape)

def permute(x, new_axis):
    return x.permute(new_axis)

def broadcast_to(x, new_shape):
    return x.broadcast_to(new_shape)

def getitem(x, idxs):
    return x.__getitem__(idxs)

def setitem(x, idxs, other):
    return x.__setitem__(idxs, other)

def eq(x, y):
    return x.__eq__(y)

def ge(x, y):
    return x.__ge__(y)

def matmul(a, b, activation=""):
    return a @ b

def from_numpy(data):
    return torch.from_numpy(data)

def from_tensor(data):
    return data

def array(shape):
    return torch.empty(shape, device=torch.device("cpu"), dtype=torch.float32)
