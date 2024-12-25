"""Operatpr table."""
# Global operator table.
from numbers import Number
from typing import Optional, List
import numpy
from ..autograd import Function, NDArray, Tensor
import genesis
from genesis import init
import math
import torch
from ..backend import array_api, NDArray
#try:
if True:
    # import fused ops
    from .layer_norm import (
            FusedLayerNormFunction, fused_layer_norm,
    )
    from .attention import FusedAttention, fused_attention, scaled_dot_product_attention
    from .triton_ops import dropout, softmax, safe_softmax
#except:
    #print("Triton layers do not imported!")
    #pass

def sum_to_shape(data, shape):
    """Sum the array `data` to match the target `shape`."""
    # Calculate which axes need to be summed
    while len(data.shape) > len(shape):
        data = data.sum(axis=0)
                                
    for i, (dim, target_dim) in enumerate(zip(data.shape, shape)):
        if target_dim == 1:
            data = data.sum(axis=i, keepdims=True) 
    return data

class EWiseAdd(Function):
    @staticmethod
    def forward(ctx, a, b):
        ctx.save_for_backward(a, b)
        requires_grad = a.requires_grad or b.requires_grad
        return Tensor(a.data + b.data, requires_grad=requires_grad, dtype=a.dtype)

    @staticmethod
    def backward(ctx, out_grad: Tensor):
        a, b = ctx.saved_tensors
        grad_a = out_grad.data
        grad_b = out_grad.data
        grad_a = sum_to_shape(grad_a, a.shape)
        grad_b = sum_to_shape(grad_b, b.shape)
        return (Tensor(grad_a, requires_grad=False, dtype=out_grad.dtype), Tensor(grad_b, requires_grad=False, dtype=out_grad.dtype))

def add(a, b):
    return EWiseAdd.apply(a, b)


class AddScalar(Function):
    @staticmethod
    def forward(ctx, a, scalar):
        ctx.save_for_backward(a)
        return Tensor(a.data + scalar, requires_grad=a.requires_grad, dtype=a.dtype)

    @staticmethod
    def backward(ctx, out_grad: Tensor):
        grad = Tensor(out_grad.data, requires_grad=False, dtype=out_grad.dtype)
        return (grad,)

def add_scalar(a, scalar):
    return AddScalar.apply(a, scalar)


class Negate(Function):
    @staticmethod
    def forward(ctx, a):
        ctx.save_for_backward(a)
        return Tensor(array_api.negative(a.data), requires_grad=a.requires_grad, dtype=a.dtype)

    @staticmethod
    def backward(ctx, out_grad):
        a, = ctx.saved_tensors
        grad = Tensor(out_grad.data * (-1), requires_grad=False, dtype=out_grad.dtype)
        return (grad,)


def negate(a):
    return Negate.apply(a)


class EWiseMul(Function):
    @staticmethod
    def forward(ctx, a, b):
        ctx.save_for_backward(a, b)
        requires_grad = a.requires_grad or b.requires_grad
        result = Tensor(a.data * b.data, requires_grad=requires_grad, dtype=a.dtype)
        return result

    @staticmethod
    def backward(ctx, out_grad: Tensor):
        a, b = ctx.saved_tensors
        grad_a = out_grad.data * b.data
        grad_b = out_grad.data * a.data

        grad_a = sum_to_shape(grad_a, a.shape)
        grad_b = sum_to_shape(grad_b, b.shape)
        return (Tensor(grad_a, requires_grad=False, dtype=out_grad.dtype), Tensor(grad_b, requires_grad=False, dtype=out_grad.dtype))


def multiply(a, b):
    return EWiseMul.apply(a, b)


class MulScalar(Function):
    @staticmethod
    def forward(ctx, a, scalar):
        ctx.save_for_backward(a, scalar)
        return Tensor(a.data * scalar, requires_grad=a.requires_grad, dtype=a.dtype)

    @staticmethod
    def backward(ctx, out_grad):
        a, scalar = ctx.saved_tensors
        grad = Tensor(out_grad.data * scalar, requires_grad=False, dtype=out_grad.dtype)
        return (grad, )


def mul_scalar(a, scalar):
    return MulScalar.apply(a, scalar)


class EWiseDiv(Function):
    @staticmethod
    def forward(ctx, a, b):
        ctx.save_for_backward(a, b)
        requires_grad = a.requires_grad or b.requires_grad
        return Tensor(array_api.divide(a.data, b.data, dtype=a.dtype), requires_grad=requires_grad, dtype=a.dtype)

    @staticmethod
    def backward(ctx, out_grad):
        a, b = ctx.saved_tensors
        grad_a = out_grad.data / b.data
        grad_b = out_grad.data * (-1) * a.data / b.data / b.data
        grad_a = sum_to_shape(grad_a, a.shape)
        grad_b = sum_to_shape(grad_b, b.shape)
        return (Tensor(grad_a, requires_grad=False, dtype=out_grad.dtype), 
                Tensor(grad_b, requires_grad=False, dtype=out_grad.dtype))

def divide(a, b):
    return EWiseDiv.apply(a, b)

class DivScalar(Function):
    @staticmethod
    def forward(ctx, a, scalar, reverse=False):
        ctx.save_for_backward(a, scalar)
        ctx.reverse = reverse
        if reverse:
            result_data = array_api.divide(scalar, a.data, dtype=a.dtype)
        else:
            result_data = array_api.divide(a.data, scalar, dtype=a.dtype)
        return Tensor(result_data, requires_grad=a.requires_grad, dtype=a.dtype)

    @staticmethod
    def backward(ctx, out_grad):
        a, scalar = ctx.saved_tensors
        reverse = ctx.reverse
        if reverse:
            grad = Tensor(-scalar * out_grad.data / (a.data ** 2), requires_grad=False, dtype=out_grad.dtype)
        else:
            grad = Tensor(out_grad.data / scalar, requires_grad=False, dtype=out_grad.dtype)
        return (grad, None)

def divide_scalar(a, scalar, reverse=False):
    return DivScalar.apply(a, scalar, reverse=reverse)

class PowScalar(Function):
    @staticmethod
    def forward(ctx, a, scalar, reverse=False):
        ctx.save_for_backward(a, scalar)
        if reverse:
            result_data = array_api.power(scalar, a.data)
        else:
            result_data = array_api.power(a.data, scalar)
        ctx.reverse = reverse
        ctx.result_data = result_data
        return Tensor(result_data, requires_grad=a.requires_grad, dtype=a.dtype)

    @staticmethod
    def backward(ctx, out_grad):
        a, scalar = ctx.saved_tensors
        reverse = ctx.reverse
        if reverse:
            grad = Tensor(out_grad.data * ctx.result_data * math.log(scalar), requires_grad=False, dtype=out_grad.dtype)
        else:
            grad = Tensor(scalar * out_grad.data * pow_scalar(a, scalar - 1).data, requires_grad=False, dtype=out_grad.dtype)
        return (grad, )


def pow_scalar(a, scalar, reverse=False):
    return PowScalar.apply(a, scalar, reverse=reverse)

class Sin(Function):
    @staticmethod
    def forward(ctx, a):
        ctx.save_for_backward(a)
        if genesis.upgrade:
            return Tensor(array_api.sin(a.data).float(), requires_grad=a.requires_grad, dtype=genesis.float32)
        else:
            return Tensor(array_api.sin(a.data), requires_grad=a.requires_grad, dtype=a.dtype)

    @staticmethod
    def backward(ctx, out_grad):
        a, = ctx.saved_tensors
        grad = Tensor(out_grad.data * array_api.cos(a.data), requires_grad=False, dtype=out_grad.dtype)
        return (grad, )

def sin(a):
    if genesis.enable_autocast:
        genesis.upgrade = True
    return Sin.apply(a)

class Cos(Function):
    @staticmethod
    def forward(ctx, a):
        ctx.save_for_backward(a)
        if genesis.upgrade:
            return Tensor(array_api.cos(a.data).float(), requires_grad=a.requires_grad, dtype=genesis.float32)
        else:
            return Tensor(array_api.cos(a.data), requires_grad=a.requires_grad, dtype=a.dtype)

    @staticmethod
    def backward(ctx, out_grad):
        a, = ctx.saved_tensors
        grad = Tensor(out_grad.data * (-1) * array_api.sin(a.data), requires_grad=False, dtype=out_grad.dtype)
        return (grad, )

def cos(a):
    if genesis.enable_autocast:
        genesis.upgrade = True
    return Cos.apply(a)


class Log(Function):
    @staticmethod
    def forward(ctx, a):
        ctx.save_for_backward(a)
        if genesis.upgrade:
            return Tensor(array_api.log(a.data).float(), requires_grad=a.requires_grad, dtype=genesis.float32)
        else:
            return Tensor(array_api.log(a.data), requires_grad=a.requires_grad, dtype=a.dtype)

    @staticmethod
    def backward(ctx, out_grad):
        a, = ctx.saved_tensors
        grad = Tensor(out_grad.data / a.data, requires_grad=False, dtype=out_grad.dtype)
        return (grad, )

def log(a):
    if genesis.enable_autocast:
        genesis.upgrade = True
    return Log.apply(a)


class Exp(Function):
    @staticmethod
    def forward(ctx, a):
        ctx.save_for_backward(a)
        if genesis.upgrade:
            return Tensor(array_api.exp(a.data).float(), requires_grad=a.requires_grad, dtype=genesis.float32)
        else:
            return Tensor(array_api.exp(a.data), requires_grad=a.requires_grad, dtype=a.dtype)

    @staticmethod
    def backward(ctx, out_grad):
        a, = ctx.saved_tensors
        grad = Tensor(out_grad.data * array_api.exp(a.data), requires_grad=False, dtype=out_grad.dtype)
        return (grad, )

def exp(a):
    if genesis.enable_autocast:
        genesis.upgrade = True
    return Exp.apply(a)

class Sqrt(Function):
    @staticmethod
    def forward(ctx, a):
        ctx.save_for_backward(a)
        return Tensor(array_api.sqrt(a.data), requires_grad=a.requires_grad, dtype=a.dtype)

    @staticmethod
    def backward(ctx, out_grad):
        a, = ctx.saved_tensors
        grad = Tensor(out_grad.data / (2 * array_api.sqrt(a.data)), requires_grad=False, dtype=out_grad.dtype)
        return (grad, )

def sqrt(a):
    return Sqrt.apply(a)

class Transpose(Function):
    @staticmethod
    def forward(ctx, a, axis=None):
        ctx.save_for_backward(a, axis)
        if axis is None:
            return Tensor(array_api.swapaxes(a.data, -1, -2), requires_grad=a.requires_grad, dtype=a.dtype)
        return Tensor(array_api.swapaxes(a.data, axis[0], axis[1]), requires_grad=a.requires_grad, dtype=a.dtype)

    @staticmethod
    def backward(ctx, out_grad):
        a, axis = ctx.saved_tensors
        if axis is None:
            grad = Tensor(array_api.swapaxes(out_grad.data, -1, -2), requires_grad=False, dtype=out_grad.dtype)
        else:
            grad = Tensor(array_api.swapaxes(out_grad.data, axis[0], axis[1]), requires_grad=False, dtype=out_grad.dtype)
        return (grad, )


def transpose(a, axis=None):
    return Transpose.apply(a, axis=axis)


class Reshape(Function):
    @staticmethod
    def forward(ctx, a, shape):
        ctx.save_for_backward(a)
        return Tensor(array_api.reshape(a.data, shape), device=a.device, requires_grad=a.requires_grad, dtype=a.dtype)

    @staticmethod
    def backward(ctx, out_grad):
        a, = ctx.saved_tensors
        grad = Tensor(array_api.reshape(out_grad.data, a.shape), device=out_grad.device, requires_grad=False, dtype=out_grad.dtype)
        return (grad, )

def reshape(a, shape):
    return Reshape.apply(a, shape)

class View(Function):
    @staticmethod
    def forward(ctx, a, shape):
        ctx.save_for_backward(a)
        ctx.original_shape = a.shape
        return Tensor(a.data.view(shape), requires_grad=a.requires_grad, device=a.device, dtype=a.dtype)

    @staticmethod
    def backward(ctx, out_grad):
        return (Tensor(out_grad.data.view(ctx.original_shape), device=out_grad.device, requires_grad=False, dtype=out_grad.dtype),)

# TODO for now, view use reshape
def view(a, shape):
    return Reshape.apply(a, shape)

class Flatten(Function):
    @staticmethod
    def forward(ctx, a, start_dim=0, end_dim=None):
        ctx.original_shape = a.shape
        ctx.start_dim = start_dim
        ctx.end_dim = end_dim if end_dim is not None else len(a.shape) - 1
        new_shape = a.shape[:start_dim] + (-1,) + a.shape[ctx.end_dim + 1:] 
        return Tensor(a.data.view(new_shape), device=a.device, requires_grad=a.requires_grad, dtype=a.dtype)
    
    @staticmethod
    def backward(ctx, out_grad):
        return (Tensor(out_grad.data.view(ctx.original_shape), device=out_grad.device, requires_grad=False, dtype=out_grad.dtype),) 

def flatten(a, start_dim=0, end_dim=None):
    return Flatten.apply(a, start_dim, end_dim)

class SetItem(Function):
    @staticmethod
    def forward(ctx, a, index, value):
        ctx.index = index
        ctx.save_for_backward(a) 
        if isinstance(value, Tensor):
            a.data[index] = value.data
        else:
            a.data[index] = value
        return a 
    
    @staticmethod
    def backward(ctx, out_grad):
        a, = ctx.saved_tensors 
        index = ctx.index
        grad = genesis.zeros(*a.shape, dtype=out_grad.dtype, requires_grad=False)
        grad.data[index] = out_grad.data
        return (grad, )
    
def setitem(a, index, value):
    return SetItem.apply(a, index, value)

class GetItem(Function):
    @staticmethod
    def forward(ctx, a, index):
        ctx.save_for_backward(a)
        if isinstance(index, Tensor):
            index = index.data.data
        tensor = Tensor.__new__(Tensor)
        tensor.init([], data=a.data[index], requires_grad=a.requires_grad)
        ctx.index = index
        return tensor 
    
    @staticmethod
    def backward(ctx, out_grad):
        a, = ctx.saved_tensors
        index = ctx.index 
        grad = genesis.zeros(*a.shape, dtype=out_grad.dtype, device=out_grad.device, requires_grad=False)
        grad.data[index] = out_grad.data
        return (grad, )
    
def getitem(a, index):
    return GetItem.apply(a, index)


class BroadcastTo(Function):
    """
    In order to broadcast, the size of the trailing axis for both arrays 
    in an operation must either be the same size or one of them must be one.
    """
    @staticmethod
    def forward(ctx, a, shape):
        ctx.save_for_backward(a, shape)
        return Tensor(array_api.broadcast_to(a.data, shape), device=a.device, requires_grad=a.requires_grad, dtype=a.dtype)

    @staticmethod
    def backward(ctx, out_grad):
        a, shape = ctx.saved_tensors
        input_shape = list(a.shape)
        base_shape = [1] * (len(shape) - len(input_shape)) + input_shape
        axis = []
        for i in range(len(base_shape)):
            if base_shape[i] != shape[i]:
                axis.append(i)
        grad = array_api.sum(out_grad.data, axis=tuple(axis))
        grad = Tensor(array_api.reshape(grad, input_shape), device=out_grad.device, requires_grad=False, dtype=out_grad.dtype)
        return (grad, )


def broadcast_to(a, shape):
    return BroadcastTo.apply(a, shape)


class Summation(Function):
    @staticmethod
    def forward(ctx, a, axis=None, keepdims=False):
        if isinstance(axis, int):
            axis = (axis, )
        ctx.save_for_backward(a)
        ctx.axis = axis
        ctx.keepdims = keepdims
        output = Tensor(array_api.sum(a.data, axis=axis, keepdims=keepdims), device=a.device, requires_grad=a.requires_grad, dtype=a.dtype)
        return output

    @staticmethod
    def backward(ctx, out_grad: Tensor):
        hs, = ctx.saved_tensors
        axis = ctx.axis
        keepdims = ctx.keepdims
        if axis is None:
            axis = hs.shape
        grad_shape = list(out_grad.shape)
        new_axis = []
        for x in axis:
            if x >= 0:
                new_axis.append(x)
            else:
                new_axis.append(x + len(hs.shape))
        if keepdims is False: 
            for x in sorted(new_axis):
                grad_shape.insert(x, 1)

        grad = Tensor(array_api.broadcast_to(
            array_api.reshape(out_grad.data, grad_shape), hs.shape), device=out_grad.device, requires_grad=False, dtype=out_grad.dtype)
        return (grad, )

def summation(a, axis=None, keepdims=False):
    return Summation.apply(a, axis=axis, keepdims=keepdims)

def sum(a, axis=None, keepdims=False):
    return Summation.apply(a, axis=axis, keepdims=keepdims)

class Matmul(Function):
    @staticmethod
    def forward(ctx, a, b):
        ctx.save_for_backward(a, b)
        c = a.data @ b.data
        requires_grad = a.requires_grad or b.requires_grad
        return Tensor(c, device=a.device, requires_grad=requires_grad, dtype=a.dtype)

    @staticmethod
    def backward(ctx, out_grad: Tensor):
        a, b = ctx.saved_tensors
        a_grad = out_grad.data @ array_api.transpose(b.data, (-1, -2))
        b_grad = array_api.transpose(a.data, (-1, -2)) @ out_grad.data

        dim1 = len(a.shape)
        dim2 = len(b.shape)
        dim3 = len(out_grad.shape)

        if dim3 > dim1:
            a_grad = array_api.sum(a_grad, tuple(range(dim3 - dim1)))
        if dim3 > dim2:
            b_grad = array_api.sum(b_grad, tuple(range(dim3 - dim2)))
        a_grad = Tensor(a_grad, device=out_grad.device, requires_grad=False, dtype=out_grad.dtype)
        b_grad = Tensor(b_grad, device=out_grad.device, requires_grad=False, dtype=out_grad.dtype)
        return (a_grad, b_grad)

    def get_total_time(self):
        return self.total_time

def matmul(a, b):
    return Matmul.apply(a, b)


class ReLU(Function):
    @staticmethod
    def forward(ctx, a):
        ctx.save_for_backward(a)
        return Tensor(array_api.maximum(a.data, 0), device=a.device, requires_grad=a.requires_grad, dtype=a.dtype)

    @staticmethod
    def backward(ctx, out_grad):
        a, = ctx.saved_tensors
        input_relu = array_api.maximum(a.data, 0)
        grad = Tensor(out_grad.data * (input_relu > 0), device=out_grad.device, requires_grad=False, dtype=out_grad.dtype)
        return (grad, )

def relu(a):
    return ReLU.apply(a)

class LogSumExp(Function):
    @staticmethod
    def forward(ctx, a, axis=None):
        ctx.save_for_backward(a)
        Z = a.data
        ctx.axis = axis
        ctx.max_value = Z.max(axis, keepdims=True)
        max_z = array_api.broadcast_to(ctx.max_value, Z.shape)
        if genesis.upgrade:
            Z = array_api.exp(Z - max_z).float()
        else:
            Z = array_api.exp(Z - max_z)
        Z = array_api.sum(Z, axis)
        Z = array_api.log(Z)
        if genesis.upgrade:
            return Tensor(Z + array_api.reshape(ctx.max_value, Z.shape), device=a.device, requires_grad=a.requires_grad, dtype=genesis.float32)
        else:
            return Tensor(Z + array_api.reshape(ctx.max_value, Z.shape), device=a.device, requires_grad=a.requires_grad, dtype=a.dtype)

    @staticmethod
    def backward(ctx, out_grad):
        hs, = ctx.saved_tensors
        input_shape = hs.shape
        max_z = array_api.broadcast_to(ctx.max_value, input_shape)
        base_shape = list(input_shape)
        if isinstance(ctx.axis, int): 
            ctx.axis = (ctx.axis,)
        axis = list(range(len(base_shape))) if ctx.axis is None else ctx.axis
        for ax in axis:
            base_shape[ax] = 1
        out_grad = out_grad.data / array_api.sum(array_api.exp(hs.data - max_z), ctx.axis)
        out_grad = array_api.reshape(out_grad, base_shape)
        out_grad = array_api.broadcast_to(out_grad, input_shape)
        out_grad = out_grad * array_api.exp(hs.data - max_z)
        grad = Tensor(out_grad, device=out_grad.device, requires_grad=False, dtype=out_grad.dtype)
        return (grad, )

def logsumexp(a, axis=None):
    if genesis.enable_autocast:
        genesis.upgrade = True
    return LogSumExp.apply(a, axis=axis)

class Max(Function):
    @staticmethod
    def forward(ctx, a, axis=None, keepdims=False):
        ctx.save_for_backward(a)
        if isinstance(axis, int):
            axis = (axis,)
        ctx.axis = axis
        ctx.keepdims = keepdims
        return Tensor(array_api.max(a.data, axis, keepdims=keepdims), device=a.device, requires_grad=a.requires_grad, dtype=a.dtype)

    @staticmethod
    def backward(ctx, out_grad):
        # Your code here
        hs, = ctx.saved_tensors
        if ctx.axis is None:
            axis = hs.shape
        else:
            axis = ctx.axis
        grad_shape = list(out_grad.shape)
        new_axis = []
        for x in axis:
            if x >= 0:
                new_axis.append(x)
            else:
                new_axis.append(x + len(hs.shape))
        if ctx.keepdims is False:
            for x in sorted(new_axis):
                grad_shape.insert(x, 1)
        mask = (hs.data == (array_api.broadcast_to(
            array_api.max(hs.data, axis=ctx.axis, keepdims=True), hs.shape)))
        grad = Tensor(array_api.broadcast_to(
            array_api.reshape(out_grad.data, grad_shape), hs.shape) * mask,
            device=out_grad.device,
            requires_grad=False, dtype=out_grad.dtype)
        return (grad,)

def max(a, axis=None, keepdims=False):
    return Max.apply(a, axis=axis, keepdims=keepdims)

class Stack(Function):
    @staticmethod
    def forward(ctx, tensors, dim):
        ctx.dim = dim
        data = torch.stack([t.data.data for t in tensors], axis=dim)
        requires_grad = False
        for t in tensors:
            if t.requires_grad:
                requires_grad = True
        return Tensor(data, device=tensors[0].device, requires_grad=requires_grad, dtype=tensors[0].dtype)
    
    @staticmethod
    def backward(ctx, out_grad):
        grads = array_api.split(out_grad.data, 1, dim=ctx.dim)
        result = tuple(Tensor(g.data.squeeze(dim=ctx.dim), device=out_grad.device, requires_grad=False, dtype=out_grad.dtype) for g in grads)
        return result

def stack(tensors, dim=0):
    return Stack.apply(tensors, dim=dim)

class Split(Function):
    @staticmethod
    def forward(ctx, x, axis):
        ctx.save_for_backward(x)
        if axis < 0:
            axis = axis + len(x.shape)
        ctx.axis = axis
        results = []
        for res in array_api.split(x.data, 1, dim=ctx.axis):
            results.append(Tensor(res, device=x.device, dtype=x.dtype))
        return tuple(results)

    @staticmethod
    def backward(ctx, out_grad, idx):
        x, = ctx.saved_tensors
        grad = torch.zeros_like(x.data.data)
        slices = [slice(None)] * len(x.shape)
        slices[ctx.axis] = slice(idx, idx + 1)
        grad[tuple(slices)] = out_grad.data.data
        grad = Tensor(grad, device=out_grad.device, requires_grad=False, dtype=out_grad.dtype)
        return (grad,)

def split(a, axis):
    return Split.apply(a, axis=axis)

class Norm(Function):
    @staticmethod
    def forward(ctx, a):
        ctx.save_for_backward(a)
        return Tensor(array_api.negative(a.data), device=a.device, requires_grad=a.requires_grad, dtype=a.dtype)

    @staticmethod
    def backward(ctx, out_grad):
        a, = ctx.saved_tensors
        grad = Tensor(out_grad.data * (-1), device=out_grad.device, requires_grad=False, dtype=out_grad.dtype)
        return (grad,)
