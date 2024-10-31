import operator
import torch
import os
import math
from functools import reduce
import numpy as np
import genesis

def prod(x):
    return reduce(operator.mul, x, 1)

class Device:
    def __init__(self, name, mod, device_id=None):
        self.name = name
        self.mod = mod
        self.device_id = device_id

    def __eq__(self, other):
        return self.name == other.name

    def __repr__(self):
        return self.name + "()"

    def __getattr__(self, name):
        return getattr(self.mod, name)

    def enabled(self):
        return self.mod is not None

    def randn(self, *shape, dtype="float32"):
        if self.name == "cuda":
            return NDArray(torch.randn(*shape, device=torch.device("cuda:" + str(self.device_id))), device=self)
        else:
            return NDArray(torch.randn(*shape), device=self)

    def rand(self, *shape, dtype="float32"):
        if self.name == "cuda":
            return NDArray(torch.rand(*shape, device=torch.device("cuda:" + str(self.device_id))), device=self)
        else:
            return NDArray(torch.rand(*shape), device=self)

    def one_hot(self, n, i, dtype="float32"):
        return NDArray(torch.nn.functional.one_hot(i.data.data.long(), num_classes=n).float(), device=self)

    def empty(self, shape, dtype="float32"):
        dtype = "float32" if dtype is None else dtype
        return NDArray.make(shape, device=self)

    def full(self, shape, fill_value, dtype="float32"):
        dtype = "float32" if dtype is None else dtype
        arr = self.empty(shape, dtype)
        arr.fill(fill_value)
        return arr

def device(device_name):
    if isinstance(device_name, int):
        return cuda(device_name)
    if device_name == "cuda":
        return cuda(0)
    elif device_name == "cpu":
        return cpu()
    elif device_name.find("cuda") != -1:
        device_id = int(device_name.split("cuda:")[-1])
        return cuda(device_id)

def cpu():
    """Return cuda device"""
    from . import ndarray_ops_cpu
    return Device("cpu", ndarray_ops_cpu)

def cuda(index=0):
    """Return cuda device"""
    try:
        from . import ndarray_ops_gpu
        return Device("cuda", ndarray_ops_gpu, index)
    except:
        return Device("cuda", None)

def default_device():
    return cpu()

def all_devices():
    """return a list of all available devices"""
    return [cuda()]

class NDArray:
    def __init__(self, data, device=None, dtype=None):
        if isinstance(data, np.ndarray):
            device = device if device is not None else default_device()
            self._device = device
            self.data = self._device.from_numpy(data, device_id=self._device.device_id)
        elif isinstance(data, NDArray):
            self._device = device
            self.data = self._device.from_tensor(data.data, device_id=self._device.device_id)
        elif isinstance(data, torch.Tensor):
            self._device = device
            self.data = self._device.from_tensor(data, device_id=self._device.device_id)
        else:
            self._device = device
            self.data = self._device.from_numpy(np.array(data, dtype=np.float32), device_id=self._device.device_id)

    @staticmethod
    def make(shape, device=None):
        array = NDArray.__new__(NDArray)
        array._device = device if device is not None else default_device()
        array.data = array.device.array(shape, device_id=array._device.device_id)
        return array

    def fill(self, value):
        """ Fill (in place) with a constant value. """
        self.data.fill_(value)

    def __repr__(self):
        return "NDArray:" + self.data.__repr__()

    def __str__(self):
        return self.__repr__()

    def numpy(self):
        return self.data.cpu().numpy()

    @property
    def dtype(self):
        return self.data.dtype

    @property
    def shape(self):
        return self.data.shape

    @property
    def device(self):
        return self._device

    @property
    def size(self):
        return prod(self.shape)

    def triu(self, k=0):
        out = NDArray.make(self.shape, device=self.device)
        out.data = torch.triu(self.data, diagonal=k)
        return out

    def split(self, cnt, dim=None):
        result = []
        for ten in torch.split(self.data, cnt, dim=dim):
            out = NDArray.make(self.shape, device=self.device)
            out.data = ten
            result.append(out)
        return result

    @property
    def flat(self):
        out = NDArray.make(self.shape, device=self.device)
        out.data = self.device.reshape(self.data, (self.size,))
        return out

    def __add__(self, y):
        out = NDArray.make(self.shape, device=self.device)
        if isinstance(y, NDArray):
            out.data = self.device.add(self.data, y.data)
        else:
            out.data = self.device.add(self.data, y)
        return out

    def __iadd__(self, y):
        out = NDArray.make(self.shape, device=self.device)
        if isinstance(y, NDArray):
            out.data = self.device.iadd(self.data, y.data)
        else:
            out.data = self.device.iadd(self.data, y)
        return out

    __radd__ = __add__

    def __mul__(self, y):
        out = NDArray.make(self.shape, device=self.device)
        if isinstance(y, NDArray):
            out.data = self.device.mul(self.data, y.data)
        else:
            out.data = self.device.mul(self.data, y)
        return out

    __rmul__ = __mul__

    def __truediv__(self, y):
        out = NDArray.make(self.shape, device=self.device)
        if isinstance(y, NDArray):
            out.data = self.device.truediv(self.data, y.data)
        else:
            out.data = self.device.truediv(self.data, y)
        return out

    def maximum(self, y):
        out = NDArray.make(self.shape, device=self.device)
        if isinstance(y, NDArray):
            out.data = self.device.maximum(self.data, y.data)
        else:
            out.data = self.device.maximum(self.data, y)
        return out
    
    def __neg__(self):
        if self.data.is_contiguous() is False:
            self.data = self.data.contiguous()
        return self * (-1)

    def __sub__(self, other):
        if self.data.is_contiguous() is False:
            self.data = self.data.contiguous()
        if other.data.is_contiguous() is False:
            other.data = other.data.contiguous()
        return self + (-other)

    def __rsub__(self, other):
        if self.data.is_contiguous() is False:
            self.data = self.data.contiguous()
        if isinstance(other, NDArray):
            if other.data.is_contiguous() is False:
                other.data = other.data.contiguous()
        return other + (-self)

    def __pow__(self, scalar):
        out = NDArray.make(self.shape, device=self.device)
        out.data = self.device.pow(self.data, scalar)
        return out

    def log(self):
        out = NDArray.make(self.shape, device=self.device)
        out.data = self.device.log(self.data)
        return out

    def exp(self):
        out = NDArray.make(self.shape, device=self.device)
        out.data = self.device.exp(self.data)
        return out

    def cos(self):
        out = NDArray.make(self.shape, device=self.device)
        out.data = self.device.cos(self.data)
        return out

    def sin(self):
        out = NDArray.make(self.shape, device=self.device)
        out.data = self.device.sin(self.data)
        return out

    def sqrt(self):
        out = NDArray.make(self.shape, device=self.device)
        out.data = self.device.sqrt(self.data)
        return out

    def reshape(self, new_shape):
        out = NDArray.make(new_shape, device=self.device)
        out.data = self.device.reshape(self.data, new_shape)
        return out

    def permute(self, new_axis):
        out = NDArray.make(self.shape, device=self.device)
        out.data = self.device.permute(self.data, new_axis)
        return out

    def broadcast_to(self, new_shape):
        out = NDArray.make(new_shape, device=self.device)
        out.data = self.device.broadcast_to(self.data, new_shape)
        return out

    def __getitem__(self, idxs):
        out = NDArray.make(self.shape, device=self.device)
        out.data = self.device.getitem(self.data, idxs)
        return out

    def __setitem__(self, idxs, other):
        out = NDArray.make(self.shape, device=self.device)
        out.data = self.device.setitem(self.data, idxs, other.data)
        return out

    def __eq__(self, other):
        out = NDArray.make(self.shape, device=self.device)
        if isinstance(other, NDArray):
            out.data = self.device.eq(self.data, other.data)
        else:
            out.data = self.device.eq(self.data, other)
        return out

    def __ge__(self, other):
        out = NDArray.make(self.shape, device=self.device)
        if isinstance(other, NDArray):
            out.data = self.device.ge(self.data, other.data)
        else:
            out.data = self.device.ge(self.data, other)
        return out

    def __ne__(self, other):
        return 1 - (self == other)

    def __gt__(self, other):
        return (self >= other) * (self != other)

    def __lt__(self, other):
        return 1 - (self >= other)

    def __le__(self, other):
        return 1 - (self > other)

    def __matmul__(self, b, activation=""):
        out = NDArray.make(self.shape, device=self.device)
        out.data = self.device.matmul(self.data, b.data, activation=activation)
        return out

    def sum(self, axis=None, keepdims=False):
        out = NDArray.make(self.shape, device=self.device)
        out.data = self.device.reduce_sum(self.data, axis=axis, keepdims=keepdims)
        return out

    def max(self, axis=None, keepdims=False):
        out = NDArray.make(self.shape, device=self.device)
        out.data = self.device.reduce_max(self.data, axis=axis, keepdims=keepdims)
        return out

def array(a, dtype="float32", device=None):
    """ Convenience methods to match numpy a bit more closely."""
    dtype = "float32" if dtype is None else dtype
    #assert dtype == "float32"
    return NDArray(a, device=device, dtype=dtype)


def empty(shape, dtype="float32", device=None):
    device = device if device is not None else default_device()
    return devie.empty(shape, dtype)


def full(shape, fill_value, dtype="float32", device=None):
    device = device if device is not None else default_device()
    return device.full(shape, fill_value, dtype)


def broadcast_to(array, new_shape):
    return array.broadcast_to(new_shape)

def swapaxes(array, x, y):
    new_shape = list(range(len(array.shape)))
    if x < 0:
        x = x + len(new_shape)
    if y < 0:
        y = y + len(new_shape)
    new_shape[x] = y
    new_shape[y] = x
    return array.permute(tuple(new_shape))

def transpose(array, axis):
    x, y = axis
    new_shape = list(range(len(array.shape)))
    if x < 0:
        x = x + len(new_shape)
    if y < 0:
        y = y + len(new_shape)
    new_shape[x] = y
    new_shape[y] = x
    return array.permute(tuple(new_shape))

def norm_axis(a, axis):
    if type(axis) is int:
        axis = (axis,)
    new_axis = []
    for ax in axis:
        if ax < 0:
            ax = ax + len(a.shape)
        new_axis.append(ax)
    return tuple(new_axis)

def sum(a, axis=None, keepdims=False):
    if type(axis) is int:
        axis = (axis, )
    if axis is None:
        return a.sum(axis=axis, keepdims=keepdims)
    axis = norm_axis(a, axis)
    axis = tuple(sorted(list(axis)))
    pre = 0
    for ax in axis:
        if keepdims:
            a = a.sum(axis=ax, keepdims=keepdims)
        else:
            a = a.sum(axis=ax - pre, keepdims=keepdims)
        pre += 1
    return a

def max(a, axis=None, keepdims=False):
    if type(axis) is int:
        axis = (axis, )
    if axis is None:
        return a.max(axis=axis, keepdims=keepdims)
    axis = norm_axis(a, axis)
    axis = tuple(sorted(list(axis)))
    pre = 0
    for ax in axis:
        if keepdims:
            a = a.max(axis=ax, keepdims=keepdims)
        else:
            a = a.max(axis=ax - pre, keepdims=keepdims)
        pre += 1
    return a

def reshape(array, new_shape):
    return array.reshape(new_shape)

def negative(array):
    return -array

def divide(a, b, dtype):
    return a / b

def power(a, b):
    return a ** b

def sin(a):
    return a.sin()

def cos(a):
    return a.cos()

def log(a):
    return a.log()

def exp(a):
    return a.exp()

def matmul(a, b):
    return a.matmul(b)

def maximum(a, b):
    return a.maximum(b)

def diag(a):
    return a.diag()

def triu(a, k=0):
    return a.triu(k=k)

def sqrt(a):
    return a.sqrt()

def split(a, cnt, dim=None):
    return a.split(cnt, dim=dim)
