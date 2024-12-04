import torch
import genesis
import operator
from functools import reduce

def add(x, y):
    return x + y

def mul(x, y):
    return x * y

def truediv(x, y):
    return x.__truediv__(y)

def rtruediv(x, y):
    return x.__rtruediv__(y)

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

def view(x, new_shape):
    return x.view(new_shape)

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

def from_numpy(data, device_id=None, dtype=None):
    torch_dtype = None
    if dtype is None or dtype == genesis.float32:
        torch_dtype = torch.float32
    elif dtype == genesis.float16:
        torch_dtype = torch.float16
    elif dtype == genesis.bfloat16:
        torch_dtype = torch.bfloat16
    return torch.from_numpy(data).to(torch_dtype)

def from_tensor(data, device_id=None):
    return data

def array(shape, device_id=None, dtype=None):
    if dtype is None or dtype == genesis.float32:
        arr = torch.empty(shape, device=torch.device("cpu"), dtype=torch.float32)
    elif dtype == genesis.float16:
        arr = torch.empty(shape, device=torch.device("cpu"), dtype=torch.float16)
    elif dtype == genesis.bfloat16:
        arr = torch.empty(shape, device=torch.device("cpu"), dtype=torch.bfloat16)
    return arr
