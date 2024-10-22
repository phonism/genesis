"""Operatpr table."""
# Global operator table.
from numbers import Number
from typing import Optional, List
import numpy
from ..autograd import Function, NDArray, Tensor
from genesis import init

import torch
from ..backend_selection import array_api, NDArray
try:
    # import fused ops
    from .layer_norm import (
            FusedLayerNormFunction, fused_layer_norm,
    )
    from .attention import FusedAttention, fused_attention
except:
    pass

class EWiseAdd(Function):
    @staticmethod
    def forward(ctx, a, b):
        ctx.save_for_backward(a, b)
        return Tensor(a.data + b.data)

    @staticmethod
    def backward(ctx, out_grad: Tensor):
        grad_a = Tensor(out_grad.data, requires_grad=False)
        grad_b = Tensor(out_grad.data, requires_grad=False)
        return (grad_a, grad_b)

def add(a, b):
    return EWiseAdd.apply(a, b)


class AddScalar(Function):
    @staticmethod
    def forward(ctx, a, scalar):
        ctx.save_for_backward(a)
        return Tensor(a.data + scalar)

    @staticmethod
    def backward(ctx, out_grad: Tensor):
        grad = Tensor(out_grad.data, requires_grad=False)
        return (grad,)

def add_scalar(a, scalar):
    return AddScalar.apply(a, scalar)


class Negate(Function):
    @staticmethod
    def forward(ctx, a):
        ctx.save_for_backward(a)
        return Tensor(array_api.negative(a.data))

    @staticmethod
    def backward(ctx, out_grad):
        a, = ctx.saved_tensors
        grad = Tensor(out_grad.data * (-1), requires_grad=False)
        return (grad,)


def negate(a):
    return Negate.apply(a)


class EWiseMul(Function):
    @staticmethod
    def forward(ctx, a, b):
        ctx.save_for_backward(a, b)
        result = Tensor(a.data * b.data)
        return result

    @staticmethod
    def backward(ctx, out_grad: Tensor):
        a, b = ctx.saved_tensors
        grad_a = Tensor(out_grad.data * b.data, requires_grad=False)
        grad_b = Tensor(out_grad.data * a.data, requires_grad=False)
        return (grad_a, grad_b)


def multiply(a, b):
    return EWiseMul.apply(a, b)


class MulScalar(Function):
    @staticmethod
    def forward(ctx, a, scalar):
        ctx.save_for_backward(a, scalar)
        return Tensor(a.data * scalar)

    @staticmethod
    def backward(ctx, out_grad):
        a, scalar = ctx.saved_tensors
        grad = Tensor(out_grad.data * scalar, requires_grad=False)
        return (grad, )


def mul_scalar(a, scalar):
    return MulScalar.apply(a, scalar)


class EWiseDiv(Function):
    @staticmethod
    def forward(ctx, a, b):
        ctx.save_for_backward(a, b)
        return Tensor(array_api.divide(a.data, b.data, dtype=a.dtype))

    @staticmethod
    def backward(ctx, out_grad):
        a, b = ctx.saved_tensors
        grad_a = Tensor(out_grad.data / b.data, requires_grad=False)
        grad_b = Tensor(out_grad.data * (-1) * a.data / b.data / b.data, requires_grad=False)
        return (grad_a, grad_b)

def divide(a, b):
    return EWiseDiv.apply(a, b)


class DivScalar(Function):
    @staticmethod
    def forward(ctx, a, scalar):
        ctx.save_for_backward(a, scalar)
        return Tensor(array_api.divide(a.data, scalar, dtype=a.dtype))

    @staticmethod
    def backward(ctx, out_grad):
        a, scalar = ctx.saved_tensors
        grad = Tensor(out_grad.data / scalar, requires_grad=False)
        return (grad,)


def divide_scalar(a, scalar):
    return DivScalar.apply(a, scalar)


class PowScalar(Function):
    @staticmethod
    def forward(ctx, a, scalar):
        ctx.save_for_backward(a, scalar)
        return Tensor(array_api.power(a.data, scalar))

    @staticmethod
    def backward(ctx, out_grad):
        a, scalar = ctx.saved_tensors
        grad = Tensor(scalar * out_grad.data * pow_scalar(a, scalar - 1).data, require_grad=False)
        return (grad, )


def pow_scalar(a, scalar):
    return PowScalar.apply(a, scalar)

class Sin(Function):
    @staticmethod
    def forward(ctx, a):
        ctx.save_for_backward(a)
        return Tensor(array_api.sin(a.data))

    @staticmethod
    def backward(ctx, out_grad):
        a, = ctx.saved_tensors
        grad = Tensor(out_grad.data * array_api.cos(a.data), requires_grad=False)
        return (grad, )

def sin(a):
    return Sin.apply(a)

class Cos(Function):
    @staticmethod
    def forward(ctx, a):
        ctx.save_for_backward(a)
        return Tensor(array_api.cos(a.data))

    @staticmethod
    def backward(ctx, out_grad):
        a, = ctx.saved_tensors
        grad = Tensor(out_grad.data * (-1) * array_api.sin(a.data), requires_grad=False)
        return (grad, )

def cos(a):
    return Cos.apply(a)


class Log(Function):
    @staticmethod
    def forward(ctx, a):
        ctx.save_for_backward(a)
        return Tensor(array_api.log(a.data))

    @staticmethod
    def backward(ctx, out_grad):
        a, = ctx.saved_tensors
        grad = Tensor(out_grad.data / a.data, requires_grad=False)
        return (grad, )

def log(a):
    return Log.apply(a)


class Exp(Function):
    @staticmethod
    def forward(ctx, a):
        ctx.save_for_backward(a)
        return Tensor(array_api.exp(a.data))

    @staticmethod
    def backward(ctx, out_grad):
        a, = ctx.saved_tensors
        grad = Tensor(out_grad.data * array_api.exp(a.data), requires_grad=False)
        return (grad, )

def exp(a):
    return Exp.apply(a)

class Sqrt(Function):
    @staticmethod
    def forward(ctx, a):
        ctx.save_for_backward(a)
        return Tensor(array_api.sqrt(a.data))

    @staticmethod
    def backward(ctx, out_grad):
        a, = ctx.saved_tensors
        grad = Tensor(out_grad.data / (2 * array_api.sqrt(a.data)), requires_grad=False)
        return (grad, )

def sqrt(a):
    return Sqrt.apply(a)

class Transpose(Function):
    @staticmethod
    def forward(ctx, a, axis=None):
        ctx.save_for_backward(a, axis)
        if axis is None:
            return Tensor(array_api.swapaxes(a.data, -1, -2))
        return Tensor(array_api.swapaxes(a.data, axis[0], axis[1]))

    @staticmethod
    def backward(ctx, out_grad):
        a, axis = ctx.saved_tensors
        if axis is None:
            grad = Tensor(array_api.swapaxes(out_grad.data, -1, -2), requires_grad=False)
        else:
            grad = Tensor(array_api.swapaxes(out_grad.data, axis[0], axis[1]), requires_grad=False)
        return (grad, )


def transpose(a, axis=None):
    return Transpose.apply(a, axis=axis)


class Reshape(Function):
    @staticmethod
    def forward(ctx, a, shape):
        ctx.save_for_backward(a)
        return Tensor(array_api.reshape(a.data, shape))

    @staticmethod
    def backward(ctx, out_grad):
        a, = ctx.saved_tensors
        grad = Tensor(array_api.reshape(out_grad.data, a.shape), requires_grad=False)
        return (grad, )

def reshape(a, shape):
    return Reshape.apply(a, shape)


class BroadcastTo(Function):
    """
    In order to broadcast, the size of the trailing axis for both arrays 
    in an operation must either be the same size or one of them must be one.
    """
    @staticmethod
    def forward(ctx, a, shape):
        ctx.save_for_backward(a, shape)
        return Tensor(array_api.broadcast_to(a.data, shape))

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
        grad = Tensor(array_api.reshape(grad, input_shape), requires_grad=False)
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
        output = Tensor(array_api.sum(a.data, axis=axis, keepdims=keepdims))
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
            array_api.reshape(out_grad.data, grad_shape), hs.shape), requires_grad=False)
        #grad = Tensor(hs.data, requires_grad=False)
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
        return Tensor(c, device=a.device)

    @staticmethod
    def backward(ctx, out_grad: Tensor):
        a, b = ctx.saved_tensors
        a_grad = out_grad.data @ array_api.swapaxes(b.data, -1, -2)
        b_grad = array_api.transpose(a.data, (-1, -2)) @ out_grad.data

        dim1 = len(a.shape)
        dim2 = len(b.shape)
        dim3 = len(out_grad.shape)

        # 如果输出的shape比输入高，说明在前面做了broadcast，那么就要把这些broadcast给sum起来
        if dim3 > dim1:
            a_grad = array_api.sum(a_grad, tuple(range(dim3 - dim1)))
        if dim3 > dim2:
            b_grad = array_api.sum(b_grad, tuple(range(dim3 - dim2)))
        a_grad = Tensor(a_grad, requires_grad=False)
        b_grad = Tensor(b_grad, requires_grad=False)
        return (a_grad, b_grad)

    def get_total_time(self):
        return self.total_time

def matmul(a, b):
    return Matmul.apply(a, b)


class ReLU(Function):
    @staticmethod
    def forward(ctx, a):
        ctx.save_for_backward(a)
        return Tensor(array_api.maximum(a.data, 0))

    @staticmethod
    def backward(ctx, out_grad):
        a, = ctx.saved_tensors
        input_relu = array_api.maximum(a.data, 0)
        grad = Tensor(out_grad.data * (input_relu > 0), require_grad=False)
        return (grad, )

def relu(a):
    return ReLU.apply(a)

class LogSumExp(Function):
    @staticmethod
    def forward(ctx, Z, axis=None):
        ctx.save_for_backward(Z)
        Z = Z.data
        ctx.axis = axis
        ctx.max_value = Z.max(axis, keepdims=True)
        max_z = array_api.broadcast_to(ctx.max_value, Z.shape)
        Z = array_api.exp(Z - max_z)
        Z = array_api.sum(Z, axis)
        Z = array_api.log(Z)
        return Tensor(Z + array_api.reshape(ctx.max_value, Z.shape))

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
        grad = Tensor(out_grad, requires_grad=False)
        return (grad, )

def logsumexp(a, axis=None):
    return LogSumExp.apply(a, axis=axis)

class Equal(Function):
    @staticmethod
    def forward(ctx, a, b):
        ctx.save_for_backward(a, b)
        return Tensor(a.data == b.data)

    @staticmethod
    def backward(ctx, out_grad):
        a, b = ctx.saved_tensors
        grad_a = array_api.reduce_sum(out_grad.data, axis=None, keepdims=False)
        grad_b = array_api.reduce_sum(out_grad.data, axis=None, keepdims=False)
        grad_a = Tensor(grad_a, requires_grad=False)
        grad_b = Tensor(grad_b, requires_grad=False)
        return (grad_a, grad_b)

def equal(a, b):
    return Equal.apply(a, b)


class Max(Function):
    @staticmethod
    def forward(ctx, a, axis=None, keepdims=False):
        ctx.save_for_backward(a)
        if isinstance(axis, int):
            axis = (axis,)
        ctx.axis = axis
        ctx.keepdims = keepdims
        return Tensor(array_api.max(a.data, axis, keepdims=keepdims))

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
            requires_grad=False)
        return (grad,)

def max(a, axis=None, keepdims=False):
    return Max.apply(a, axis=axis, keepdims=keepdims)

class Stack(Function):
    @staticmethod
    def forward(ctx, tensors, dim):
        ctx.dim = dim
        data = torch.stack([t.data.data for t in tensors], axis=dim)
        return Tensor(data)
    
    @staticmethod
    def backward(ctx, out_grad):
        grads = array_api.split(out_grad.data, 1, dim=ctx.dim)
        return tuple(Tensor(g.data.squeeze(dim=ctx.dim), requires_grad=False) for g in grads)

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
            results.append(Tensor(res))
        return tuple(results)

    @staticmethod
    def backward(ctx, out_grad, idx):
        x, = ctx.saved_tensors
        grad = torch.zeros_like(x.data.data)
        slices = [slice(None)] * len(x.shape)
        slices[ctx.axis] = slice(idx, idx + 1)
        grad[tuple(slices)] = out_grad.data.data
        grad = Tensor(grad, requires_grad=False)
        return (grad,)

def split(a, axis):
    return Split.apply(a, axis=axis)

