"""Operatpr table."""
# Global operator table.
from functools import reduce as functools_reduce
import operator
from numbers import Number
from typing import Optional, List
from ..autograd import Function, NDArray, Tensor
import genesis
from genesis import init
import math
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
    
    # Collect axes that need to be summed (where target is 1 but current is not)
    axes_to_sum = []
    for i, (dim, target_dim) in enumerate(zip(data.shape, shape)):
        if target_dim == 1 and dim != 1:
            axes_to_sum.append(i)
    
    # Sum over all collected axes at once, keeping dimensions
    if axes_to_sum:
        data = data.sum(axis=tuple(axes_to_sum), keepdims=True)
    
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

class EWiseSub(Function):
    @staticmethod
    def forward(ctx, a, b):
        ctx.save_for_backward(a, b)
        requires_grad = a.requires_grad or b.requires_grad
        return Tensor(a.data - b.data, requires_grad=requires_grad, dtype=a.dtype)

    @staticmethod
    def backward(ctx, out_grad: Tensor):
        a, b = ctx.saved_tensors
        grad_a = out_grad.data
        grad_b = -out_grad.data  # Gradient for subtraction: d/da(a-b) = 1, d/db(a-b) = -1
        grad_a = sum_to_shape(grad_a, a.shape)
        grad_b = sum_to_shape(grad_b, b.shape)
        return (Tensor(grad_a, requires_grad=False, dtype=out_grad.dtype), Tensor(grad_b, requires_grad=False, dtype=out_grad.dtype))

def sub(a, b):
    return EWiseSub.apply(a, b)


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
        return Tensor(a.data / b.data, requires_grad=requires_grad, dtype=a.dtype)

    @staticmethod
    def backward(ctx, out_grad):
        a, b = ctx.saved_tensors
        grad_a = out_grad.data / b.data
        grad_b = out_grad.data * (-1) * a.data / b.data / b.data
        grad_a = sum_to_shape(grad_a, a.shape)
        grad_b = sum_to_shape(grad_b, b.shape)
        return (
            Tensor(grad_a, requires_grad=False, dtype=out_grad.dtype), 
            Tensor(grad_b, requires_grad=False, dtype=out_grad.dtype)
        )

def divide(a, b):
    return EWiseDiv.apply(a, b)

class DivScalar(Function):
    @staticmethod
    def forward(ctx, a, scalar, reverse=False):
        ctx.save_for_backward(a, scalar)
        ctx.reverse = reverse
        if reverse:
            result_data = scalar / a.data
        else:
            result_data = a.data / scalar
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

class Abs(Function):
    @staticmethod
    def forward(ctx, a):
        ctx.save_for_backward(a)
        return Tensor(array_api.abs(a.data), requires_grad=a.requires_grad, dtype=a.dtype)

    @staticmethod
    def backward(ctx, out_grad):
        a, = ctx.saved_tensors
        # Gradient of abs(x) is sign(x), but undefined at x=0
        # We use the convention that sign(0) = 0
        grad = Tensor(array_api.sign(a.data) * out_grad.data, requires_grad=False, dtype=out_grad.dtype)
        return (grad, )

def abs(a):
    return Abs.apply(a)

class Clamp(Function):
    @staticmethod
    def forward(ctx, a, min_val=None, max_val=None):
        ctx.save_for_backward(a, min_val, max_val)
        return Tensor(array_api.clamp(a.data, min_val, max_val), requires_grad=a.requires_grad, dtype=a.dtype)

    @staticmethod
    def backward(ctx, out_grad):
        a, min_val, max_val = ctx.saved_tensors
        # Gradient of clamp: 1 where min_val <= x <= max_val, 0 otherwise
        mask = array_api.ones(a.shape, device=a.device, dtype=a.dtype)
        if min_val is not None:
            mask = mask * array_api.greater_equal(a.data, min_val)
        if max_val is not None:
            mask = mask * array_api.less_equal(a.data, max_val)
        grad = Tensor(mask * out_grad.data, requires_grad=False, dtype=out_grad.dtype)
        return (grad,)

def clamp(a, min_val=None, max_val=None):
    return Clamp.apply(a, min_val, max_val)

def clip(a, min_val=None, max_val=None):
    return Clamp.apply(a, min_val, max_val)

class Where(Function):
    @staticmethod
    def forward(ctx, condition, x, y):
        """Element-wise selection of values from x or y based on condition."""
        ctx.save_for_backward(condition, x, y)
        return Tensor(array_api.where(condition.data, x.data, y.data), requires_grad=(x.requires_grad or y.requires_grad), dtype=x.dtype)

    @staticmethod
    def backward(ctx, out_grad):
        """Backward pass for where operation."""
        condition, x, y = ctx.saved_tensors
        
        # Simple gradient computation - just pass through out_grad conditionally
        x_grad = None
        y_grad = None
        
        if x.requires_grad:
            x_grad = genesis.where(condition, out_grad, genesis.zeros_like(out_grad))
        
        if y.requires_grad:
            y_grad = genesis.where(condition, genesis.zeros_like(out_grad), out_grad)
        
        return (None, x_grad, y_grad)

def where(condition, x, y):
    """Element-wise selection of values from x or y based on condition."""
    return Where.apply(condition, x, y)

class Argmax(Function):
    @staticmethod
    def forward(ctx, a, dim=None, keepdim=False):
        """Find indices of maximum values along dimension."""
        ctx.save_for_backward(a, dim, keepdim)
        result_data = array_api.argmax(a.data, dim=dim, keepdim=keepdim)
        return Tensor(result_data, requires_grad=False, dtype=genesis.int64)

    @staticmethod
    def backward(ctx, out_grad):
        """Argmax is not differentiable - gradient is None."""
        return (None, None, None)

class Argmin(Function):
    @staticmethod
    def forward(ctx, a, dim=None, keepdim=False):
        """Find indices of minimum values along dimension."""
        ctx.save_for_backward(a, dim, keepdim)
        result_data = array_api.argmin(a.data, dim=dim, keepdim=keepdim)
        return Tensor(result_data, requires_grad=False, dtype=genesis.int64)

    @staticmethod
    def backward(ctx, out_grad):
        """Argmin is not differentiable - gradient is None."""
        return (None, None, None)

def argmax(a, dim=None, keepdim=False):
    """Return indices of maximum values along specified dimension."""
    return Argmax.apply(a, dim, keepdim)

def argmin(a, dim=None, keepdim=False):
    """Return indices of minimum values along specified dimension."""
    return Argmin.apply(a, dim, keepdim)

class Permute(Function):
    @staticmethod
    def forward(ctx, a, dims):
        """Permute the dimensions of the input tensor."""
        ctx.save_for_backward(a, dims)
        result_data = array_api.permute(a.data, dims)
        return Tensor(result_data, requires_grad=a.requires_grad, dtype=a.dtype)

    @staticmethod
    def backward(ctx, out_grad):
        """Backward pass for permute - reverse the permutation."""
        a, dims = ctx.saved_tensors
        
        # Create inverse permutation
        inv_dims = [0] * len(dims)
        for i, d in enumerate(dims):
            inv_dims[d] = i
        
        # Apply inverse permutation to gradient
        grad = genesis.permute(out_grad, inv_dims)
        return (grad, None)

def permute(a, dims):
    """Permute the dimensions of the input tensor."""
    return Permute.apply(a, dims)

class Gather(Function):
    @staticmethod
    def forward(ctx, input, dim, index):
        """Gather values along dimension using indices."""
        ctx.save_for_backward(input, dim, index)
        result_data = array_api.gather(input.data, dim, index.data)
        return Tensor(result_data, requires_grad=input.requires_grad, dtype=input.dtype)

    @staticmethod
    def backward(ctx, out_grad):
        """Backward pass for gather - scatter gradient back to original positions."""
        input, dim, index = ctx.saved_tensors
        
        # Input gradient: scatter out_grad back to original positions
        input_grad = None
        if input.requires_grad:
            # Create tensor of zeros with same shape as input but same dtype as out_grad
            input_grad = genesis.zeros(input.shape, dtype=out_grad.dtype, device=input.device, requires_grad=False)
            
            # Use array_api.scatter directly to avoid triggering autograd
            result_data = array_api.scatter(input_grad.data, dim, index.data, out_grad.data)
            input_grad = Tensor(result_data, requires_grad=False, dtype=out_grad.dtype, device=input.device)
        
        # Dim gradient: always None (scalar, not a tensor)
        dim_grad = None
        
        # Index gradient: always zero tensor with same shape as index
        index_grad = None
        if index.requires_grad:
            zeros_data = array_api.zeros_like(index.data)
            index_grad = Tensor(zeros_data, requires_grad=False, dtype=index.dtype, device=index.device)
        
        return (input_grad, index_grad)

class Scatter(Function):
    @staticmethod
    def forward(ctx, input, dim, index, src):
        """Scatter values from src along dimension using indices."""
        ctx.save_for_backward(input, dim, index, src)
        result_data = array_api.scatter(input.data, dim, index.data, src.data)
        return Tensor(result_data, requires_grad=(input.requires_grad or src.requires_grad), dtype=input.dtype)

    @staticmethod
    def backward(ctx, out_grad):
        """Backward pass for scatter."""
        input, dim, index, src = ctx.saved_tensors
        
        # Input gradient: scattered positions get zero, others get out_grad
        input_grad = None
        if input.requires_grad:
            # Create tensor of zeros with same shape as src but same dtype as out_grad  
            zeros_at_indices = genesis.zeros(src.shape, dtype=out_grad.dtype, device=src.device, requires_grad=False)
            
            # Use array_api.scatter directly to avoid triggering autograd
            result_data = array_api.scatter(out_grad.data, dim, index.data, zeros_at_indices.data)
            input_grad = Tensor(result_data, requires_grad=False, dtype=out_grad.dtype, device=input.device)
        
        # Dim gradient: always None (scalar, not a tensor)
        dim_grad = None
        
        # Index gradient: always zero tensor with same shape as index
        index_grad = None
        if index.requires_grad:
            zeros_data = array_api.zeros_like(index.data)
            index_grad = Tensor(zeros_data, requires_grad=False, dtype=index.dtype, device=index.device)
        
        # Source gradient: gather from out_grad using same indices
        src_grad = None
        if src.requires_grad:
            result_data = array_api.gather(out_grad.data, dim, index.data)
            src_grad = Tensor(result_data, requires_grad=False, dtype=src.dtype, device=src.device)
        
        return (input_grad, index_grad, src_grad)

def gather(input, dim, index):
    """Gather values along dimension using indices."""
    return Gather.apply(input, dim, index)

def scatter(input, dim, index, src):
    """Scatter values from src along dimension using indices."""
    return Scatter.apply(input, dim, index, src)

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

class Expand(Function):
    @staticmethod
    def forward(ctx, a, new_shape):
        ctx.a_shape = a.shape
        ctx.new_shape = new_shape
        return Tensor(array_api.expand(a.data, new_shape), requires_grad=a.requires_grad, device=a.device, dtype=a.dtype)

    @staticmethod
    def backward(ctx, out_grad):
        grad_input = out_grad 
        for i, (a_dim, new_dim) in enumerate(zip(ctx.a_shape, ctx.new_shape)):
            if a_dim == 1 and new_dim > 1:
                grad_input = array_api.reduce_sum(grad_input.data, axis=i, keepdims=True) 
        grad_input = Tensor(grad_input.view(ctx.a_shape), device=out_grad.device, requires_grad=False, dtype=out_grad.dtype)
        return grad_input, None

def expand(a, shape):
    return Expand.apply(a, shape)


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

def _is_basic_indexing(index):
    """
    Check if index is basic indexing (view/slice path).
    Basic indexing includes: int, slice, ..., None, Ellipsis
    Returns True for view path, False for gather path.
    """
    if isinstance(index, (int, slice, type(None), type(Ellipsis))):
        return True
    if isinstance(index, tuple):
        for idx in index:
            if isinstance(idx, (Tensor, list)):
                return False
            if isinstance(idx, tuple) and any(isinstance(x, (Tensor, list)) for x in idx):
                return False
        return True
    if isinstance(index, (Tensor, list)):
        return False
    return True

class SetItem(Function):
    @staticmethod
    def forward(ctx, a, index, value):
        ctx.index = index
        ctx.save_for_backward(a)
        
        # Determine indexing type
        ctx.is_basic = _is_basic_indexing(index)
        
        # Convert index to proper format for NDArray layer
        if isinstance(index, Tensor):
            index_data = index.data
        else:
            index_data = index
            
        if isinstance(value, Tensor):
            a.data[index_data] = value.data
        else:
            a.data[index_data] = value
        return a 
    
    @staticmethod
    def backward(ctx, out_grad):
        a, = ctx.saved_tensors 
        index = ctx.index
        
        # For setitem, gradient just passes through for the unchanged parts
        # The indexed parts get gradient from out_grad at those positions
        grad = out_grad.detach()  # Pass through gradient
        return (grad, None, None)
    
def setitem(a, index, value):
    return SetItem.apply(a, index, value)

class GetItemView(Function):
    """View/Slice path for basic indexing - returns a view sharing storage."""
    @staticmethod
    def forward(ctx, a, index):
        ctx.save_for_backward(a)
        ctx.index = index
        
        # Basic indexing returns a view
        result_data = a.data[index]
        
        tensor = Tensor.__new__(Tensor)
        tensor.init([], data=result_data, requires_grad=a.requires_grad)
        return tensor
    
    @staticmethod
    def backward(ctx, out_grad):
        a, = ctx.saved_tensors
        index = ctx.index
        
        # Create zero gradient tensor
        grad = genesis.zeros(*a.shape, dtype=out_grad.dtype, device=out_grad.device, requires_grad=False)
        
        # For view indexing, gradient flows back to original positions
        grad.data[index] = out_grad.data
        return (grad, None)

class GetItemGather(Function):
    """Gather path for advanced indexing - creates a copy."""
    @staticmethod
    def forward(ctx, a, index):
        ctx.save_for_backward(a, index if isinstance(index, Tensor) else None)
        ctx.original_shape = a.shape
        
        if isinstance(index, Tensor):
            # Tensor indexing
            result_data = a.data[index.data]
            ctx.tensor_index = True
        else:
            # Other advanced indexing (list, array, etc.)
            result_data = a.data[index]
            ctx.tensor_index = False
            ctx.index = index
        
        tensor = Tensor.__new__(Tensor)
        tensor.init([], data=result_data, requires_grad=a.requires_grad)
        return tensor
    
    @staticmethod  
    def backward(ctx, out_grad):
        saved = ctx.saved_tensors
        a = saved[0]
        
        # Create gradient tensor  
        grad = genesis.zeros(*ctx.original_shape, dtype=out_grad.dtype, device=out_grad.device, requires_grad=False)
        
        if ctx.tensor_index:
            index = saved[1]
            # Handle mixed indexing case where index is a tuple  
            if isinstance(index.data, tuple) and len(index.data) == 2:
                # Mixed 2D indexing: need to scatter 1D out_grad to 2D grad tensor
                row_idx, col_idx = index.data
                # Use the existing CUDAStorage mixed indexing setitem
                grad.data[index.data] = out_grad.data
            else:
                grad.data[index.data] = out_grad.data
        else:
            # Use saved index for other cases
            if isinstance(ctx.index, list):
                # For list indices, manually accumulate gradients for duplicates
                for i, idx in enumerate(ctx.index):
                    # Extract the i-th row from out_grad and add to grad[idx]
                    grad_row = grad[idx]
                    out_grad_row = out_grad[i]
                    grad[idx] = grad_row + out_grad_row
            else:
                grad.data[ctx.index] = out_grad.data
        
        return (grad, None)

def getitem(a, index):
    """
    Main getitem dispatcher - routes to View or Gather path based on index type.
    """
    if _is_basic_indexing(index):
        return GetItemView.apply(a, index)
    else:
        return GetItemGather.apply(a, index)


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
        
        # For bool tensors, sum should return int64 (like PyTorch)
        result_dtype = genesis.int64 if a.dtype == genesis.bool else a.dtype
        
        # Get the sum result
        sum_result = array_api.sum(a.data, axis=axis, keepdims=keepdims)
        
        # For bool tensors, the GPU ops already converted to int64, so we need to update the dtype
        if a.dtype == genesis.bool and hasattr(sum_result, 'dtype'):
            sum_result._dtype = result_dtype
        
        output = Tensor(sum_result, device=a.device, requires_grad=a.requires_grad, dtype=result_dtype)
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

class Mean(Function):
    @staticmethod
    def forward(ctx, a, axis=None, keepdims=False):
        """
        Forward pass for mean operation using sum + divide approach.
        """
        if isinstance(axis, int):
            axis = (axis, )
        ctx.save_for_backward(a)
        ctx.axis = axis
        ctx.keepdims = keepdims
        
        # Calculate number of elements being reduced for gradient scaling
        if axis is None:
            # Full reduction
            ctx.num_elements = a.numel()
        else:
            # Partial reduction - calculate elements in reduced dimensions
            shape = a.shape
            ndim = len(shape)
            normalized_axis = tuple(ax if ax >= 0 else ax + ndim for ax in axis)
            ctx.num_elements = functools_reduce(operator.mul, [shape[ax] for ax in normalized_axis], 1)
        
        # Use sum + divide approach directly with array_api (like PyTorch)
        sum_data = array_api.sum(a.data, axis=axis, keepdims=keepdims)
        mean_data = sum_data / ctx.num_elements
        output = Tensor(mean_data, device=a.device, requires_grad=a.requires_grad, dtype=a.dtype)
        return output

    @staticmethod
    def backward(ctx, out_grad: Tensor):
        """
        Backward pass for mean operation.
        
        The gradient of mean is out_grad / num_elements broadcasted to input shape.
        """
        hs, = ctx.saved_tensors
        axis = ctx.axis
        keepdims = ctx.keepdims
        num_elements = ctx.num_elements
        
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

        # Scale gradient by 1/num_elements (since mean = sum/num_elements)
        scaled_grad = out_grad.data / num_elements
        grad = Tensor(array_api.broadcast_to(
            array_api.reshape(scaled_grad, grad_shape), hs.shape), 
            device=out_grad.device, requires_grad=False, dtype=out_grad.dtype)
        return (grad, )

def mean(a, axis=None, keepdims=False):
    """
    Compute the arithmetic mean along the specified axis.
    
    Args:
        a: Input tensor
        axis: Axis or axes along which to compute mean. None means reduce all axes.
        keepdims: Whether to keep reduced dimensions as size 1
        
    Returns:
        Tensor containing the mean values
    """
    return Mean.apply(a, axis=axis, keepdims=keepdims)

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
        # Normalize negative dimension
        base_ndim = len(tensors[0].shape)
        if dim < 0:
            dim = base_ndim + 1 + dim  # +1 because we're adding a new dimension
        
        ctx.dim = dim
        ctx.num_tensors = len(tensors)  # Save the number of tensors for backward
        
        # Clean implementation: use basic NDArray operations to implement stack
        device = tensors[0].device
        
        # Check if any tensor requires grad
        requires_grad = any(t.requires_grad for t in tensors)
        
        # Get the output shape: insert len(tensors) at position dim
        base_shape = list(tensors[0].data.shape)
        output_shape = base_shape[:dim] + [len(tensors)] + base_shape[dim:]
        
        # Create output array
        stacked_data = array_api.empty(tuple(output_shape), device=device, dtype=tensors[0].dtype)
        
        # Fill the output array by placing each tensor at the correct position
        for i, t in enumerate(tensors):
            # Create index to place this tensor at position i in dimension dim
            indices = [slice(None)] * len(output_shape)
            indices[dim] = i
            stacked_data[tuple(indices)] = t.data
        
        return Tensor(stacked_data, device=device, requires_grad=requires_grad, dtype=tensors[0].dtype)
    
    @staticmethod
    def backward(ctx, out_grad):
        # Split into ctx.num_tensors parts along ctx.dim dimension
        # Clean abstraction: use NDArray operations only
        result = []
        
        # Extract individual slices from the stacked tensor
        for i in range(ctx.num_tensors):
            # Use NDArray's indexing to extract slice i from dimension ctx.dim
            indices = [slice(None)] * len(out_grad.data.shape)
            indices[ctx.dim] = i
            slice_data = out_grad.data[tuple(indices)]
            
            # Create tensor from the slice
            result.append(Tensor.make_const(slice_data, requires_grad=False))
        
        return tuple(result)

def stack(tensors, dim=0):
    return Stack.apply(tensors, dim=dim)

class Cat(Function):
    @staticmethod 
    def forward(ctx, tensors, dim):
        """
        GPU-native concatenation using array API.
        """
        ctx.dim = dim
        ctx.save_for_backward(*tensors)
        
        # Extract NDArray objects from tensors
        ndarrays = [t.data for t in tensors]
        
        # Use array_api.cat for GPU-native concatenation
        result_ndarray = array_api.cat(ndarrays, dim=dim)
        
        requires_grad = any(t.requires_grad for t in tensors)
        return Tensor(result_ndarray, device=tensors[0].device, 
                     requires_grad=requires_grad, dtype=tensors[0].dtype) 
    
    @staticmethod
    def backward(ctx, out_grad):
        # Get sizes from saved tensors
        sizes = [t.data.shape[ctx.dim] for t in ctx.saved_tensors]
        
        # Use NDArray split method directly
        grad_splits = out_grad.data.split(sizes, dim=ctx.dim)
        
        # Convert split results to Tensors
        result = []
        for grad_ndarray in grad_splits:
            result.append(Tensor.make_const(grad_ndarray, requires_grad=False))
        
        return tuple(result) 

def cat(tensors, dim=0):
    return Cat.apply(tensors, dim=dim)

class Squeeze(Function):
    @staticmethod
    def forward(ctx, tensor, dim):
        ctx.dim = dim
        # Use NDArray squeeze method directly
        squeezed_ndarray = tensor.data.squeeze(dim)
        return Tensor(squeezed_ndarray, device=tensor.device, requires_grad=tensor.requires_grad, dtype=tensor.dtype)
    
    @staticmethod
    def backward(ctx, out_grad):
        # Use NDArray unsqueeze method directly
        unsqueezed_ndarray = out_grad.data.unsqueeze(ctx.dim)
        return (Tensor.make_const(unsqueezed_ndarray, requires_grad=False), ) 

def squeeze(tensor, dim):
    return Squeeze.apply(tensor, dim)

class Unsqueeze(Function):
    @staticmethod
    def forward(ctx, tensor, dim):
        ctx.dim = dim
        # Use NDArray unsqueeze method directly
        unsqueezed_ndarray = tensor.data.unsqueeze(dim)
        return Tensor(unsqueezed_ndarray, device=tensor.device, requires_grad=tensor.requires_grad, dtype=tensor.dtype)
    
    @staticmethod
    def backward(ctx, out_grad):
        # Use NDArray squeeze method directly
        squeezed_ndarray = out_grad.data.squeeze(ctx.dim)
        return (Tensor.make_const(squeezed_ndarray, requires_grad=False), )

def unsqueeze(tensor, dim):
    return Unsqueeze.apply(tensor, dim)

class Split(Function):
    @staticmethod
    def forward(ctx, x, dim):
        ctx.save_for_backward(x)
        if dim < 0:
            dim = dim + len(x.shape)
        ctx.dim = dim
        results = []
        
        # Split along the dim dimension - each split should have size 1 in that dimension
        for i in range(x.shape[dim]):
            # Create slice for index i (using slice to preserve dimension)
            indices = [slice(None)] * len(x.shape)
            indices[dim] = slice(i, i+1)  # Use slice(i, i+1) instead of i to preserve dimension
            slice_tensor = x.data[tuple(indices)]
            results.append(Tensor(slice_tensor, device=x.device, dtype=x.dtype, requires_grad=x.requires_grad))
        return tuple(results)

    @staticmethod
    def backward(ctx, out_grad, idx):
        x, = ctx.saved_tensors
        result = genesis.zeros_like(x, requires_grad=False)
        slices = [slice(None)] * len(x.shape)
        slices[ctx.dim] = slice(idx, idx+1)
        result.data[tuple(slices)] = out_grad.data
        return (result,)

def split(a, dim):
    return Split.apply(a, dim=dim)

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

class Sigmoid(Function):
    """
    Sigmoid activation function: 1 / (1 + exp(-x))
    """
    @staticmethod
    def forward(ctx, a):
        ctx.save_for_backward(a)
        # sigmoid(x) = 1 / (1 + exp(-x))
        sigmoid_out = 1 / (1 + array_api.exp(-a.data))
        return Tensor(sigmoid_out, device=a.device, requires_grad=a.requires_grad, dtype=a.dtype)

    @staticmethod
    def backward(ctx, out_grad):
        a, = ctx.saved_tensors
        # derivative of sigmoid: sigmoid(x) * (1 - sigmoid(x))
        sigmoid_out = 1 / (1 + array_api.exp(-a.data))
        grad = Tensor(out_grad.data * sigmoid_out * (1 - sigmoid_out), device=out_grad.device, requires_grad=False, dtype=out_grad.dtype)
        return (grad,)

def sigmoid(a):
    """Apply sigmoid activation function"""
    return Sigmoid.apply(a)

class Tanh(Function):
    """
    Tanh activation function
    """
    @staticmethod
    def forward(ctx, a):
        ctx.save_for_backward(a)
        # tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
        # More numerically stable: tanh(x) = (exp(2x) - 1) / (exp(2x) + 1)
        exp_2x = array_api.exp(2 * a.data)
        tanh_out = (exp_2x - 1) / (exp_2x + 1)
        return Tensor(tanh_out, device=a.device, requires_grad=a.requires_grad, dtype=a.dtype)

    @staticmethod
    def backward(ctx, out_grad):
        a, = ctx.saved_tensors
        # derivative of tanh: 1 - tanh^2(x)
        exp_2x = array_api.exp(2 * a.data)
        tanh_out = (exp_2x - 1) / (exp_2x + 1)
        grad = Tensor(out_grad.data * (1 - tanh_out * tanh_out), device=out_grad.device, requires_grad=False, dtype=out_grad.dtype)
        return (grad,)

def tanh(a):
    """Apply tanh activation function"""
    return Tanh.apply(a)


class ScatterAddFunction(Function):
    @staticmethod
    def forward(ctx, input, dim, index, src):
        ctx.dim = dim
        ctx.save_for_backward(index, src)
        result_data = array_api.scatter_add(input.data, dim, index.data, src.data)
        # Create tensor with proper requires_grad
        result = Tensor(result_data, device=input.device, requires_grad=input.requires_grad or src.requires_grad)
        return result
    
    @staticmethod  
    def backward(ctx, out_grad):
        dim = ctx.dim
        index, src = ctx.saved_tensors
        
        # Gradient w.r.t. input: just pass through the out_grad
        input_grad = out_grad if out_grad is not None else None
        
        # Gradient w.r.t. src: gather the out_grad at the scattered positions
        src_grad = None
        if src.requires_grad and out_grad is not None:
            # Use array_api.gather to get gradients from the scattered positions
            src_grad_data = array_api.gather(out_grad.data, dim, index.data)
            src_grad = Tensor(src_grad_data, device=src.device, requires_grad=False)
            
        # Return gradients for: input, index, src (dim is not a tensor input)
        # index doesn't need gradients (integer indices)
        return input_grad, None, src_grad


def scatter_add(input, dim, index, src):
    """
    Scatter-add values from src along dimension using indices.
    
    Args:
        input: Input tensor to scatter-add into
        dim: Dimension to scatter along
        index: Tensor with indices
        src: Source tensor with values to add
        
    Returns:
        Tensor with scattered-added values
    """
    return ScatterAddFunction.apply(input, dim, index, src)


def repeat_interleave(input, repeats, dim=None):
    """
    Repeat elements of tensor along specified dimension.
    
    Args:
        input: Input tensor
        repeats: Number of repetitions for each element
        dim: Dimension to repeat along (if None, flatten first)
        
    Returns:
        Tensor with repeated elements
    """
    return Tensor.make_const(array_api.repeat_interleave(input.data, repeats, dim))


def one_hot(indices, num_classes):
    """
    One-hot encoding of indices.
    
    Args:
        indices: Integer tensor with class indices
        num_classes: Number of classes
        
    Returns:
        Tensor with one-hot encoding
    """
    # Use the existing one_hot from init module
    return genesis.init.one_hot(num_classes, indices)


def log_softmax(input, dim=-1):
    """
    Log softmax function for numerical stability.
    
    Args:
        input: Input tensor
        dim: Dimension to apply log_softmax along
        
    Returns:
        Log softmax of input
    """
    # Use log-sum-exp trick for numerical stability
    max_vals = max(input, dim, keepdims=True)
    shifted = input - max_vals
    log_sum_exp = log(summation(exp(shifted), axis=dim, keepdims=True))
    return shifted - log_sum_exp


def maximum(input, other):
    """
    Element-wise maximum of tensors.
    
    Args:
        input: First tensor
        other: Second tensor or scalar
        
    Returns:
        Element-wise maximum
    """
    if isinstance(other, (int, float)):
        # Create a tensor filled with the scalar value
        other = genesis.tensor([other]).broadcast_to(input.shape)
    return Maximum.apply(input, other)


class Maximum(Function):
    @staticmethod
    def forward(ctx, a, b):
        ctx.save_for_backward(a, b)
        result_data = array_api.maximum(a.data, b.data)
        requires_grad = a.requires_grad or b.requires_grad
        return Tensor(result_data, requires_grad=requires_grad, dtype=a.dtype)
    
    @staticmethod
    def backward(ctx, out_grad):
        a, b = ctx.saved_tensors
        # Gradient flows to the larger input
        a_mask = (a.data >= b.data).astype(out_grad.dtype)
        b_mask = (b.data >= a.data).astype(out_grad.dtype)
        
        grad_a = out_grad.data * a_mask
        grad_b = out_grad.data * b_mask
        
        return (
            Tensor(grad_a, requires_grad=False, dtype=out_grad.dtype),
            Tensor(grad_b, requires_grad=False, dtype=out_grad.dtype)
        )


def cross_entropy(input, target, weight=None, ignore_index=-100, reduction='mean'):
    """
    Efficient sparse cross entropy loss function.
    
    Combines log_softmax and sparse NLL loss without one-hot conversion.
    This avoids the O(N*C) memory overhead of one-hot encoding.
    
    Args:
        input: Tensor of shape (N, C) where N is batch size, C is number of classes
        target: Tensor of shape (N,) containing class indices
        weight: Manual rescaling weight for each class
        ignore_index: Index to ignore in loss computation
        reduction: Reduction method ('mean', 'sum', 'none')
        
    Returns:
        Cross entropy loss tensor
    """
    return sparse_cross_entropy(input, target, weight, ignore_index, reduction)

def sparse_cross_entropy(input, target, weight=None, ignore_index=-100, reduction='mean'):
    """
    Efficient sparse cross entropy implementation avoiding one-hot conversion.
    
    This implementation uses sparse indexing to gather log probabilities
    directly from the log_softmax output, similar to PyTorch's CUDA kernel.
    """
    # Flatten input to 2D for easier indexing: (N*..., C)
    original_shape = input.shape
    batch_size = original_shape[0] if len(original_shape) > 1 else 1
    num_classes = original_shape[-1]
    
    # Reshape input to (N, C) where N = batch_size * other_dims
    input_2d = input.view(-1, num_classes)  # Shape: (N, C)
    target_1d = target.view(-1)             # Shape: (N,)
    
    # Compute log softmax for numerical stability
    log_probs = log_softmax(input_2d, dim=-1)  # Shape: (N, C)
    
    # Create mask for ignore_index
    if ignore_index != -100:
        mask = (target_1d != ignore_index)  # Shape: (N,)
        valid_targets = target_1d * mask.long()  # Zero out ignored indices
    else:
        mask = None
        valid_targets = target_1d
    
    # Sparse indexing: gather log probabilities for target classes
    # This is the key optimization - no one-hot conversion needed!
    batch_indices = genesis.arange(input_2d.shape[0], device=input.device)  # [0, 1, 2, ..., N-1]
    
    # Use advanced indexing to gather: log_probs[batch_idx, target_idx]
    selected_log_probs = log_probs[batch_indices, valid_targets]  # Shape: (N,)
    
    # Compute negative log likelihood
    nll = -selected_log_probs  # Shape: (N,)
    
    # Apply ignore_index mask
    if mask is not None:
        nll = nll * mask.float()
    
    # Apply class weights if provided
    if weight is not None:
        class_weights = weight[valid_targets]  # Shape: (N,)
        nll = nll * class_weights
    
    # Apply reduction
    if reduction == 'none':
        # Reshape back to original batch dimensions
        return nll.view(original_shape[:-1])  # Remove last dimension (classes)
    elif reduction == 'sum':
        return summation(nll)
    elif reduction == 'mean':
        if mask is not None:
            # Only average over non-ignored elements
            valid_count = summation(mask.float())
            return summation(nll) / valid_count if valid_count > 0 else genesis.tensor(0.0, device=input.device)
        else:
            return mean(nll)
    else:
        raise ValueError(f"Invalid reduction mode: {reduction}")
