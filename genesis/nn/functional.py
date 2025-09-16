"""Operator table with new dispatcher integration."""
# Global operator table.
from functools import reduce as functools_reduce
import operator
from numbers import Number
from typing import Optional, List
from ..function import Function
from genesis.tensor import Tensor
import genesis
from genesis import init
import math
# Import new dispatcher system
from genesis.ops import OperationDispatcher, DeviceType
try:
    # import fused ops
    from .layer_norm import (
            FusedLayerNormFunction, fused_layer_norm,
    )
    from .attention import FusedAttention, fused_attention, scaled_dot_product_attention
    from .triton_ops import dropout, safe_softmax
    from .triton_ops.softmax import softmax as triton_softmax
except Exception as e:
    print(f"Triton layers do not imported! Error: {e}")
    import traceback
    traceback.print_exc()
    pass

def sum_to_shape(data, shape):
    """Sum the array `data` to match the target `shape`."""
    # Calculate which axes need to be summed
    while len(data.shape) > len(shape):
        data = data.sum(dim=0)
    
    # Collect axes that need to be summed (where target is 1 but current is not)
    axes_to_sum = []
    for i, (dim, target_dim) in enumerate(zip(data.shape, shape)):
        if target_dim == 1 and dim != 1:
            axes_to_sum.append(i)
    
    # Sum over all collected axes at once, keeping dimensions
    if axes_to_sum:
        data = data.sum(dim=tuple(axes_to_sum), keepdim=True)
    
    return data

class EWiseAdd(Function):
    @staticmethod
    def forward(ctx, a, b):
        ctx.save_for_backward(a, b)
        requires_grad = a.requires_grad or b.requires_grad
        # Use new dispatcher system - pass tensors, get tensor back
        result_tensor = OperationDispatcher.dispatch("add", a, b)
        result_tensor.requires_grad = requires_grad
        return result_tensor

    @staticmethod
    def backward(ctx, out_grad: Tensor):
        a, b = ctx.saved_tensors
        
        # Create new tensors for gradients by copying data
        grad_a = out_grad.clone()
        grad_b = out_grad.clone()
        
        # Sum to shape to handle broadcasting
        if grad_a.shape != a.shape:
            grad_a = sum_to_shape(grad_a, a.shape)
        if grad_b.shape != b.shape:
            grad_b = sum_to_shape(grad_b, b.shape)
            
        return (grad_a, grad_b)

def add(a, b):
    if isinstance(b, Tensor):
        return EWiseAdd.apply(a, b)
    else:
        # b is a scalar
        return add_scalar(a, b)

class EWiseSub(Function):
    @staticmethod
    def forward(ctx, a, b):
        ctx.save_for_backward(a, b)
        requires_grad = a.requires_grad or b.requires_grad
        # Use new dispatcher system - pass tensors, get tensor back
        result_tensor = OperationDispatcher.dispatch("sub", a, b)
        result_tensor.requires_grad = requires_grad
        return result_tensor

    @staticmethod
    def backward(ctx, out_grad: Tensor):
        a, b = ctx.saved_tensors
        # Gradients for subtraction: d/da(a-b) = 1, d/db(a-b) = -1
        grad_a = out_grad.clone()
        grad_b = OperationDispatcher.dispatch("neg", out_grad)  # Use dispatcher for negation
        
        # Sum to shape to handle broadcasting
        if grad_a.shape != a.shape:
            grad_a = sum_to_shape(grad_a, a.shape)
        if grad_b.shape != b.shape:
            grad_b = sum_to_shape(grad_b, b.shape)
            
        return (grad_a, grad_b)

def sub(a, b):
    if isinstance(b, Tensor):
        return EWiseSub.apply(a, b)
    else:
        # b is a scalar
        return sub_scalar(a, b)


class AddScalar(Function):
    @staticmethod
    def forward(ctx, a, scalar):
        ctx.save_for_backward(a)
        # Use dispatcher for scalar addition 
        result_tensor = OperationDispatcher.dispatch("add", a, scalar)
        result_tensor.requires_grad = a.requires_grad
        return result_tensor

    @staticmethod
    def backward(ctx, out_grad: Tensor):
        # Gradient for tensor is just out_grad cloned, no gradient for scalar
        grad = out_grad.clone()
        return (grad,)

def add_scalar(a, scalar):
    return AddScalar.apply(a, scalar)


class SubScalar(Function):
    @staticmethod
    def forward(ctx, a, scalar, reverse=False):
        ctx.save_for_backward(a, scalar)
        ctx.reverse = reverse
        if reverse:
            # For reverse subtraction (scalar - tensor), use rsub operation
            result_tensor = OperationDispatcher.dispatch("rsub", a, scalar)
        else:
            # Use dispatcher for tensor - scalar
            result_tensor = OperationDispatcher.dispatch("sub", a, scalar)
        result_tensor.requires_grad = a.requires_grad
        return result_tensor

    @staticmethod
    def backward(ctx, out_grad):
        a, scalar = ctx.saved_tensors
        reverse = ctx.reverse
        if reverse:
            # Gradient of scalar - x is -out_grad
            grad = OperationDispatcher.dispatch("neg", out_grad)
        else:
            # Gradient of x - scalar is out_grad
            grad = out_grad.clone()
        grad.requires_grad = False
        return (grad, None)


def sub_scalar(a, scalar, reverse=False):
    return SubScalar.apply(a, scalar, reverse=reverse)


class Negate(Function):
    @staticmethod
    def forward(ctx, a):
        ctx.save_for_backward(a)
        result_data = OperationDispatcher.dispatch("neg", a)
        result_data.requires_grad = a.requires_grad
        return result_data

    @staticmethod
    def backward(ctx, out_grad):
        a, = ctx.saved_tensors
        # Use neg operation instead of .data * (-1)
        grad = OperationDispatcher.dispatch("mul", out_grad, -1)
        grad.requires_grad = False
        return (grad,)


def negate(a):
    return Negate.apply(a)

def neg(a):
    """Alias for negate function - PyTorch compatibility."""
    return negate(a)


class EWiseMul(Function):
    @staticmethod
    def forward(ctx, a, b):
        ctx.save_for_backward(a, b)
        requires_grad = a.requires_grad or b.requires_grad
        # Use new dispatcher system - pass tensors, get tensor back
        result_tensor = OperationDispatcher.dispatch("mul", a, b)
        result_tensor.requires_grad = requires_grad
        return result_tensor

    @staticmethod
    def backward(ctx, out_grad: Tensor):
        a, b = ctx.saved_tensors
        # Gradients for multiplication: d/da(a*b) = b, d/db(a*b) = a
        # Use dispatcher to avoid creating computation graph in backward
        grad_a = OperationDispatcher.dispatch("mul", out_grad, b)
        grad_b = OperationDispatcher.dispatch("mul", out_grad, a)
        
        # Sum to shape to handle broadcasting
        if grad_a.shape != a.shape:
            grad_a = sum_to_shape(grad_a, a.shape)
        if grad_b.shape != b.shape:
            grad_b = sum_to_shape(grad_b, b.shape)
            
        return (grad_a, grad_b)


def multiply(a, b):
    return EWiseMul.apply(a, b)

def mul(a, b):
    if isinstance(b, Tensor):
        return EWiseMul.apply(a, b)
    else:
        # b is a scalar
        return mul_scalar(a, b)

def div(a, b):
    return EWiseDiv.apply(a, b)


class MulScalar(Function):
    @staticmethod
    def forward(ctx, a, scalar):
        ctx.save_for_backward(a, scalar)
        # Use dispatcher for scalar multiplication 
        result_tensor = OperationDispatcher.dispatch("mul", a, scalar)
        result_tensor.requires_grad = a.requires_grad
        return result_tensor

    @staticmethod
    def backward(ctx, out_grad):
        a, scalar = ctx.saved_tensors
        # Use dispatcher for scalar multiplication in backward pass
        grad = OperationDispatcher.dispatch("mul", out_grad, scalar)
        grad.requires_grad = False
        return (grad, )


def mul_scalar(a, scalar):
    return MulScalar.apply(a, scalar)


class EWiseDiv(Function):
    @staticmethod
    def forward(ctx, a, b):
        ctx.save_for_backward(a, b)
        requires_grad = a.requires_grad or b.requires_grad
        # Use new dispatcher system - pass tensors, get tensor back
        result_tensor = OperationDispatcher.dispatch("div", a, b)
        result_tensor.requires_grad = requires_grad
        return result_tensor

    @staticmethod
    def backward(ctx, out_grad):
        a, b = ctx.saved_tensors
        
        # Compute gradients using dispatcher operations
        # grad_a = out_grad / b
        grad_a = OperationDispatcher.dispatch("div", out_grad, b)
        
        # grad_b = -out_grad * a / (b * b)
        temp1 = OperationDispatcher.dispatch("mul", out_grad, a)
        temp2 = OperationDispatcher.dispatch("mul", b, b)
        temp3 = OperationDispatcher.dispatch("div", temp1, temp2)
        grad_b = OperationDispatcher.dispatch("neg", temp3)
        
        # Sum to shape to handle broadcasting
        if grad_a.shape != a.shape:
            grad_a = sum_to_shape(grad_a, a.shape)
        if grad_b.shape != b.shape:
            grad_b = sum_to_shape(grad_b, b.shape)
            
        return (grad_a, grad_b)

def divide(a, b):
    return EWiseDiv.apply(a, b)

def truediv(a, b):
    """True division - alias for divide."""
    if isinstance(b, Tensor):
        return EWiseDiv.apply(a, b)
    else:
        # b is a scalar
        return divide_scalar(a, b)

def floordiv(a, b):
    """Floor division - divide and cast to integer."""
    # First do regular division
    result = truediv(a, b)
    # Cast to int64 for floor division behavior
    return result.to(genesis.int64)

class DivScalar(Function):
    @staticmethod
    def forward(ctx, a, scalar, reverse=False):
        ctx.save_for_backward(a, scalar)
        ctx.reverse = reverse
        if reverse:
            # For reverse division (scalar / tensor), use rdiv operation
            result_tensor = OperationDispatcher.dispatch("rdiv", a, scalar)
        else:
            # Use dispatcher for tensor / scalar 
            result_tensor = OperationDispatcher.dispatch("div", a, scalar)
        result_tensor.requires_grad = a.requires_grad
        return result_tensor

    @staticmethod
    def backward(ctx, out_grad):
        a, scalar = ctx.saved_tensors
        reverse = ctx.reverse
        if reverse:
            # Gradient of scalar/x is -scalar/x^2 * out_grad
            # First compute x^2
            x_squared = OperationDispatcher.dispatch("mul", a, a)
            # Then compute -scalar/x^2 * out_grad
            neg_scalar_over_x2 = OperationDispatcher.dispatch("rdiv", x_squared, -scalar)
            grad = OperationDispatcher.dispatch("mul", neg_scalar_over_x2, out_grad)
        else:
            # Gradient of x/scalar is out_grad/scalar
            grad = OperationDispatcher.dispatch("div", out_grad, scalar)
        grad.requires_grad = False
        return (grad, None)

def divide_scalar(a, scalar, reverse=False):
    return DivScalar.apply(a, scalar, reverse=reverse)

class PowScalar(Function):
    @staticmethod
    def forward(ctx, a, scalar, reverse=False):
        ctx.save_for_backward(a, scalar)
        ctx.reverse = reverse
        if reverse:
            # For reverse power (scalar ** tensor), use rpower operation
            result_tensor = OperationDispatcher.dispatch("rpower", a, scalar)
        else:
            # Use dispatcher for tensor ** scalar 
            result_tensor = OperationDispatcher.dispatch("pow", a, scalar)
        result_tensor.requires_grad = a.requires_grad
        ctx.result_tensor = result_tensor  # Save for backward
        return result_tensor

    @staticmethod
    def backward(ctx, out_grad):
        a, scalar = ctx.saved_tensors
        reverse = ctx.reverse
        result_tensor = ctx.result_tensor
        if reverse:
            # Gradient of scalar^x is scalar^x * ln(scalar) * out_grad
            ln_scalar = math.log(scalar)
            grad = OperationDispatcher.dispatch("mul", result_tensor, ln_scalar)
            grad = OperationDispatcher.dispatch("mul", grad, out_grad)
        else:
            # Gradient of x^scalar is scalar * x^(scalar-1) * out_grad
            if scalar == 0:
                grad = genesis.zeros_like(a)
            else:
                x_pow_s_minus_1 = OperationDispatcher.dispatch("pow", a, scalar - 1)
                grad = OperationDispatcher.dispatch("mul", x_pow_s_minus_1, scalar)
                grad = OperationDispatcher.dispatch("mul", grad, out_grad)
        grad.requires_grad = False
        return (grad, None)


def pow_scalar(a, scalar, reverse=False):
    return PowScalar.apply(a, scalar, reverse=reverse)


def pow(a, b):
    """Power operation - supports both tensor and scalar exponents."""
    if isinstance(b, Tensor):
        # For tensor-tensor power, we need to implement EWisePow
        # For now, just raise NotImplementedError
        raise NotImplementedError("Tensor-tensor power not implemented yet")
    else:
        # b is a scalar
        return pow_scalar(a, b)

class Sin(Function):
    @staticmethod
    def forward(ctx, a):
        ctx.save_for_backward(a)
        # Use new dispatcher system - pass tensor, get tensor back
        result_tensor = OperationDispatcher.dispatch("sin", a)
        if genesis.upgrade:
            # Convert to float32 for upgrade mode
            result_tensor = result_tensor.to(genesis.float32)
            result_tensor.requires_grad = a.requires_grad
            return result_tensor
        else:
            result_tensor.requires_grad = a.requires_grad
            return result_tensor

    @staticmethod
    def backward(ctx, out_grad):
        a, = ctx.saved_tensors
        # Gradient of sin(x) is cos(x)
        cos_a = OperationDispatcher.dispatch("cos", a)
        grad = OperationDispatcher.dispatch("mul", out_grad, cos_a)
        grad.requires_grad = False
        return (grad, )

def sin(a):
    if genesis.enable_autocast:
        genesis.upgrade = True
    return Sin.apply(a)

class Cos(Function):
    @staticmethod
    def forward(ctx, a):
        ctx.save_for_backward(a)
        # Use new dispatcher system - pass tensor, get tensor back
        result_tensor = OperationDispatcher.dispatch("cos", a)
        if genesis.upgrade:
            # Convert to float32 for upgrade mode
            result_tensor = result_tensor.to(genesis.float32)
            result_tensor.requires_grad = a.requires_grad
            return result_tensor
        else:
            result_tensor.requires_grad = a.requires_grad
            return result_tensor

    @staticmethod
    def backward(ctx, out_grad):
        a, = ctx.saved_tensors
        # Gradient of cos(x) is -sin(x)
        sin_a = OperationDispatcher.dispatch("sin", a)
        neg_sin_a = OperationDispatcher.dispatch("neg", sin_a)
        grad = OperationDispatcher.dispatch("mul", out_grad, neg_sin_a)
        grad.requires_grad = False
        return (grad, )

def cos(a):
    if genesis.enable_autocast:
        genesis.upgrade = True
    return Cos.apply(a)


class Log(Function):
    @staticmethod
    def forward(ctx, a):
        ctx.save_for_backward(a)
        # Use new dispatcher system - pass tensor, get tensor back
        result_tensor = OperationDispatcher.dispatch("log", a)
        if genesis.upgrade:
            # Convert to float32 for upgrade mode
            result_tensor = result_tensor.to(genesis.float32)
            result_tensor.requires_grad = a.requires_grad
            return result_tensor
        else:
            result_tensor.requires_grad = a.requires_grad
            return result_tensor

    @staticmethod
    def backward(ctx, out_grad):
        a, = ctx.saved_tensors
        # Gradient of log(x) is 1/x
        grad = OperationDispatcher.dispatch("div", out_grad, a)
        grad.requires_grad = False
        return (grad, )

def log(a):
    if genesis.enable_autocast:
        genesis.upgrade = True
    return Log.apply(a)


class Exp(Function):
    @staticmethod
    def forward(ctx, a):
        ctx.save_for_backward(a)
        # Use new dispatcher system - pass tensor, get tensor back
        result_tensor = OperationDispatcher.dispatch("exp", a)
        if genesis.upgrade:
            # Convert to float32 for upgrade mode
            result_tensor = result_tensor.to(genesis.float32)
            result_tensor.requires_grad = a.requires_grad
            return result_tensor
        else:
            result_tensor.requires_grad = a.requires_grad
            return result_tensor

    @staticmethod
    def backward(ctx, out_grad):
        a, = ctx.saved_tensors
        # Gradient of exp(x) is exp(x)
        exp_a = OperationDispatcher.dispatch("exp", a)
        grad = OperationDispatcher.dispatch("mul", out_grad, exp_a)
        grad.requires_grad = False
        return (grad, )

def exp(a):
    if genesis.enable_autocast:
        genesis.upgrade = True
    return Exp.apply(a)

class Sqrt(Function):
    @staticmethod
    def forward(ctx, a):
        ctx.save_for_backward(a)
        # Use new dispatcher system - pass tensor, get tensor back
        result_tensor = OperationDispatcher.dispatch("sqrt", a)
        result_tensor.requires_grad = a.requires_grad
        return result_tensor

    @staticmethod
    def backward(ctx, out_grad):
        a, = ctx.saved_tensors
        # Gradient of sqrt(x) is 1/(2*sqrt(x))
        sqrt_a = OperationDispatcher.dispatch("sqrt", a)
        two_sqrt_a = OperationDispatcher.dispatch("mul", sqrt_a, 2)
        grad = OperationDispatcher.dispatch("div", out_grad, two_sqrt_a)
        grad.requires_grad = False
        return (grad, )

def sqrt(a):
    return Sqrt.apply(a)


class Equal(Function):
    @staticmethod
    def forward(ctx, a, b):
        # Use new dispatcher system - pass tensors, get tensor back
        result_tensor = OperationDispatcher.dispatch("eq", a, b)
        result_tensor.requires_grad = False  # Boolean tensors don't need gradients
        return result_tensor

    @staticmethod
    def backward(ctx, out_grad):
        # Equality comparison is not differentiable
        return (None, None)


def eq(a, b):
    """Element-wise equality comparison."""
    return Equal.apply(a, b)

class Abs(Function):
    @staticmethod
    def forward(ctx, a):
        ctx.save_for_backward(a)
        # Use new dispatcher system - pass tensor, get tensor back
        result_tensor = OperationDispatcher.dispatch("abs", a)
        result_tensor.requires_grad = a.requires_grad
        return result_tensor

    @staticmethod
    def backward(ctx, out_grad):
        a, = ctx.saved_tensors
        # Gradient of abs(x) is sign(x), but undefined at x=0
        # We use the convention that sign(0) = 0
        sign_result = OperationDispatcher.dispatch("sign", a)
        grad = OperationDispatcher.dispatch("mul", sign_result, out_grad)
        return (grad, )

def abs(a):
    return Abs.apply(a)

class Clamp(Function):
    @staticmethod
    def forward(ctx, a, min_val=None, max_val=None):
        ctx.save_for_backward(a, min_val, max_val)
        result = OperationDispatcher.dispatch("clamp", a, min_val, max_val)
        result.requires_grad = a.requires_grad
        return result

    @staticmethod
    def backward(ctx, out_grad):
        a, min_val, max_val = ctx.saved_tensors
        # Gradient of clamp: 1 where min_val <= x <= max_val, 0 otherwise
        mask = genesis.ones(a.shape, device=a.device, dtype=a.dtype)
        if min_val is not None:
            mask = mask * (a >= min_val)
        if max_val is not None:
            mask = mask * (a <= max_val)
        grad = mask * out_grad
        grad.requires_grad = False
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
        result = OperationDispatcher.dispatch("where", condition, x, y)
        result.requires_grad = x.requires_grad or y.requires_grad
        return result

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
        result_data = OperationDispatcher.dispatch("argmax", a, dim=dim, keepdim=keepdim)
        result_data.requires_grad = False
        return result_data

    @staticmethod
    def backward(ctx, out_grad):
        """Argmax is not differentiable - gradient is None."""
        return (None, None, None)

class Argmin(Function):
    @staticmethod
    def forward(ctx, a, dim=None, keepdim=False):
        """Find indices of minimum values along dimension."""
        ctx.save_for_backward(a, dim, keepdim)
        result_data = OperationDispatcher.dispatch("argmin", a, dim=dim, keepdim=keepdim)
        result_data.requires_grad = False
        return result_data

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
        result_data = OperationDispatcher.dispatch("permute", a, dims)
        result_data.requires_grad = a.requires_grad
        return result_data

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
        result_data = OperationDispatcher.dispatch("gather", input, dim, index)
        result_data.requires_grad = input.requires_grad
        return result_data

    @staticmethod
    def backward(ctx, out_grad):
        """Backward pass for gather - scatter gradient back to original positions."""
        input, dim, index = ctx.saved_tensors
        
        # Input gradient: scatter out_grad back to original positions
        input_grad = None
        if input.requires_grad:
            # Create tensor of zeros with same shape as input but same dtype as out_grad
            input_grad = genesis.zeros(input.shape, dtype=out_grad.dtype, device=input.device, requires_grad=False)
            
            # Use dispatcher directly to avoid triggering autograd
            result_data = OperationDispatcher.dispatch("scatter", input_grad, dim, index, out_grad)
            result_data.requires_grad = False
            input_grad = result_data
        
        # Dim gradient: always None (scalar, not a tensor)
        dim_grad = None
        
        # Index gradient: always zero tensor with same shape as index
        index_grad = None
        if index.requires_grad:
            zeros_data = OperationDispatcher.dispatch("zeros_like", index)
            zeros_data.requires_grad = False
            index_grad = zeros_data
        
        return (input_grad, index_grad)

class Scatter(Function):
    @staticmethod
    def forward(ctx, input, dim, index, src):
        """Scatter values from src along dimension using indices."""
        ctx.save_for_backward(input, dim, index, src)
        result_data = OperationDispatcher.dispatch("scatter", input, dim, index, src)
        result_data.requires_grad = input.requires_grad or src.requires_grad
        return result_data

    @staticmethod
    def backward(ctx, out_grad):
        """Backward pass for scatter."""
        input, dim, index, src = ctx.saved_tensors
        
        # Input gradient: scattered positions get zero, others get out_grad
        input_grad = None
        if input.requires_grad:
            # Create tensor of zeros with same shape as src but same dtype as out_grad  
            zeros_at_indices = genesis.zeros(src.shape, dtype=out_grad.dtype, device=src.device, requires_grad=False)
            
            # Use dispatcher directly to avoid triggering autograd
            result_data = OperationDispatcher.dispatch("scatter", out_grad, dim, index, zeros_at_indices)
            result_data.requires_grad = False
            input_grad = result_data
        
        # Dim gradient: always None (scalar, not a tensor)
        dim_grad = None
        
        # Index gradient: always zero tensor with same shape as index
        index_grad = None
        if index.requires_grad:
            zeros_data = OperationDispatcher.dispatch("zeros_like", index)
            zeros_data.requires_grad = False
            index_grad = zeros_data
        
        # Source gradient: gather from out_grad using same indices
        src_grad = None
        if src.requires_grad:
            result_data = OperationDispatcher.dispatch("gather", out_grad, dim, index)
            result_data.requires_grad = False
            src_grad = result_data
        
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
        ctx.save_for_backward(a)
        ctx.axis = axis
        if axis is None:
            result = OperationDispatcher.dispatch("transpose", a, -1, -2)
            result.requires_grad = a.requires_grad
            return result
        result = OperationDispatcher.dispatch("transpose", a, axis[0], axis[1])
        result.requires_grad = a.requires_grad
        return result

    @staticmethod
    def backward(ctx, out_grad):
        a, = ctx.saved_tensors
        axis = ctx.axis
        if axis is None:
            grad = OperationDispatcher.dispatch("transpose", out_grad, -1, -2)
        else:
            grad = OperationDispatcher.dispatch("transpose", out_grad, axis[0], axis[1])
        return (grad, )


def transpose(a, axis=None):
    return Transpose.apply(a, axis=axis)


class Reshape(Function):
    @staticmethod
    def forward(ctx, a, shape):
        ctx.save_for_backward(a)
        result = OperationDispatcher.dispatch("reshape", a, shape)
        result.requires_grad = a.requires_grad
        return result

    @staticmethod
    def backward(ctx, out_grad):
        a, = ctx.saved_tensors
        grad = OperationDispatcher.dispatch("reshape", out_grad, a.shape)
        return (grad,)
        return (grad, )

def reshape(a, *shape):
    """Reshape tensor to new shape - PyTorch compatible API.

    Args:
        a: Input tensor
        *shape: New shape as multiple arguments or single tuple/list

    Examples:
        reshape(tensor, 3, 4)      # Multiple arguments
        reshape(tensor, (3, 4))    # Tuple argument
        reshape(tensor, -1)        # Single dimension
    """
    # Handle different argument patterns like PyTorch
    if len(shape) == 1:
        # Single argument - could be tuple, list, or single int
        if isinstance(shape[0], (tuple, list)):
            new_shape = tuple(shape[0])
        else:
            new_shape = (shape[0],)
    else:
        # Multiple arguments
        new_shape = shape

    return Reshape.apply(a, new_shape)

class Expand(Function):
    @staticmethod
    def forward(ctx, a, new_shape):
        ctx.a_shape = a.shape
        ctx.new_shape = new_shape
        result = OperationDispatcher.dispatch("expand", a, new_shape)
        result.requires_grad = a.requires_grad
        return result

    @staticmethod
    def backward(ctx, out_grad):
        grad_input = out_grad 
        for i, (a_dim, new_dim) in enumerate(zip(ctx.a_shape, ctx.new_shape)):
            if a_dim == 1 and new_dim > 1:
                grad_input = OperationDispatcher.dispatch("sum", grad_input, axis=i, keepdims=True) 
        grad_input = grad_input.view(ctx.a_shape)
        grad_input.requires_grad = False
        return (grad_input,)

def expand(a, *shape):
    # Handle both expand(tensor, (2, 3, 4)) and expand(tensor, 2, 3, 4) like PyTorch
    if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
        shape = shape[0]
    elif len(shape) > 1:
        shape = tuple(shape)
    else:
        shape = shape[0] if shape else ()
    return Expand.apply(a, shape)


class View(Function):
    @staticmethod
    def forward(ctx, a, shape):
        ctx.save_for_backward(a)
        ctx.original_shape = a.shape
        result = a.view(shape)
        result.requires_grad = a.requires_grad
        return result

    @staticmethod
    def backward(ctx, out_grad):
        reshaped_grad = OperationDispatcher.dispatch("reshape", out_grad, ctx.original_shape)
        reshaped_grad.requires_grad = False
        return (reshaped_grad,)

# TODO for now, view use reshape
def view(a, *shape):
    # Handle both view(tensor, (shape,)) and view(tensor, dim1, dim2, ...)
    if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
        shape = shape[0]
    return Reshape.apply(a, shape)

class Flatten(Function):
    @staticmethod
    def forward(ctx, a, start_dim=0, end_dim=None):
        ctx.original_shape = a.shape
        ctx.start_dim = start_dim
        ctx.end_dim = end_dim if end_dim is not None else len(a.shape) - 1
        new_shape = a.shape[:start_dim] + (-1,) + a.shape[ctx.end_dim + 1:]
        result = a.view(new_shape)
        result.requires_grad = a.requires_grad
        return result

    @staticmethod
    def backward(ctx, out_grad):
        reshaped_grad = OperationDispatcher.dispatch("reshape", out_grad, ctx.original_shape)
        reshaped_grad.requires_grad = False
        return (reshaped_grad,) 

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
        
        # Use dispatcher for setitem operation
        result = OperationDispatcher.dispatch("setitem", a, index, value)
        return result 
    
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
        
        # Use dispatcher for indexing - like EWiseAdd does
        result_tensor = OperationDispatcher.dispatch("getitem", a, index)
        result_tensor.requires_grad = a.requires_grad
        return result_tensor
    
    @staticmethod
    def backward(ctx, out_grad):
        a, = ctx.saved_tensors
        index = ctx.index
        
        # Create zero gradient tensor
        grad = genesis.zeros(*a.shape, dtype=out_grad.dtype, device=out_grad.device, requires_grad=False)
        
        # For view indexing, gradient flows back to original positions
        grad = OperationDispatcher.dispatch("setitem", grad, index, out_grad)
        return (grad, None)

class GetItemGather(Function):
    """Gather path for advanced indexing - creates a copy."""
    @staticmethod
    def forward(ctx, a, index):
        ctx.save_for_backward(a, index if isinstance(index, Tensor) else None)
        ctx.original_shape = a.shape
        
        if isinstance(index, Tensor):
            # Tensor indexing - use OperationDispatcher
            result_tensor = OperationDispatcher.dispatch("getitem", a, index)
            ctx.tensor_index = True
        else:
            # Other advanced indexing (list, array, etc.) - use OperationDispatcher
            result_tensor = OperationDispatcher.dispatch("getitem", a, index)
            ctx.tensor_index = False
            ctx.index = index
        
        result_tensor.requires_grad = a.requires_grad
        return result_tensor
    
    @staticmethod  
    def backward(ctx, out_grad):
        saved = ctx.saved_tensors
        a = saved[0]
        
        # Create gradient tensor  
        grad = genesis.zeros(*ctx.original_shape, dtype=out_grad.dtype, device=out_grad.device, requires_grad=False)
        
        if ctx.tensor_index:
            index = saved[1]
            # Use OperationDispatcher for setitem
            grad = OperationDispatcher.dispatch("setitem", grad, index, out_grad)
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
                # Use OperationDispatcher for other index types
                grad = OperationDispatcher.dispatch("setitem", grad, ctx.index, out_grad)
        
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
        result = OperationDispatcher.dispatch("broadcast_to", a, shape)
        result.requires_grad = a.requires_grad
        return result

    @staticmethod
    def backward(ctx, out_grad):
        a, shape = ctx.saved_tensors
        input_shape = list(a.shape)
        base_shape = [1] * (len(shape) - len(input_shape)) + input_shape
        axis = []
        for i in range(len(base_shape)):
            if base_shape[i] != shape[i]:
                axis.append(i)
        grad = OperationDispatcher.dispatch("sum", out_grad, axis=tuple(axis))
        grad = OperationDispatcher.dispatch("reshape", grad, input_shape)
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
        
        # Get the sum result using dispatcher
        sum_result = OperationDispatcher.dispatch("sum", a, axis=axis, keepdims=keepdims)
        
        # For bool tensors, the GPU ops already converted to int64, so we need to update the dtype
        if a.dtype == genesis.bool and hasattr(sum_result, 'dtype'):
            sum_result._dtype = result_dtype
        
        sum_result.requires_grad = a.requires_grad
        output = sum_result
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

        # Reshape gradient to handle keepdims=False case
        reshaped_grad = OperationDispatcher.dispatch("reshape", out_grad, new_shape=grad_shape)
        # Broadcast to original input shape
        grad = OperationDispatcher.dispatch("broadcast_to", reshaped_grad, new_shape=hs.shape)
        grad.requires_grad = False
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
        
        # Use sum + divide approach directly with dispatcher (like PyTorch)
        sum_result = OperationDispatcher.dispatch("sum", a, axis=axis, keepdims=keepdims)
        output = OperationDispatcher.dispatch("truediv", sum_result, ctx.num_elements)
        output.requires_grad = a.requires_grad
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
        scaled_grad = OperationDispatcher.dispatch("truediv", out_grad, num_elements)
        reshaped_grad = OperationDispatcher.dispatch("reshape", scaled_grad, grad_shape)
        grad = OperationDispatcher.dispatch("broadcast_to", reshaped_grad, hs.shape)
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
        requires_grad = a.requires_grad or b.requires_grad
        # Use new dispatcher system - pass tensors, get tensor back
        result_tensor = OperationDispatcher.dispatch("matmul", a, b)
        result_tensor.requires_grad = requires_grad
        return result_tensor

    @staticmethod
    def backward(ctx, out_grad: Tensor):
        a, b = ctx.saved_tensors
        # Use dispatcher for transpose and matmul operations
        b_transposed = OperationDispatcher.dispatch("transpose", b, -1, -2)
        a_grad = OperationDispatcher.dispatch("matmul", out_grad, b_transposed)
        
        a_transposed = OperationDispatcher.dispatch("transpose", a, -1, -2)
        b_grad = OperationDispatcher.dispatch("matmul", a_transposed, out_grad)

        dim1 = len(a.shape)
        dim2 = len(b.shape)
        dim3 = len(out_grad.shape)

        if dim3 > dim1:
            a_grad = OperationDispatcher.dispatch("sum", a_grad, axis=tuple(range(dim3 - dim1)))
        if dim3 > dim2:
            b_grad = OperationDispatcher.dispatch("sum", b_grad, axis=tuple(range(dim3 - dim2)))
        
        return (a_grad, b_grad)

    def get_total_time(self):
        return self.total_time

def matmul(a, b):
    return Matmul.apply(a, b)


class ReLU(Function):
    @staticmethod
    def forward(ctx, a):
        ctx.save_for_backward(a)
        # Use new dispatcher system - pass tensor, get tensor back
        result_tensor = OperationDispatcher.dispatch("maximum", a, 0)
        result_tensor.requires_grad = a.requires_grad
        return result_tensor

    @staticmethod
    def backward(ctx, out_grad):
        a, = ctx.saved_tensors
        # Gradient is out_grad where input > 0, 0 elsewhere
        relu_mask = OperationDispatcher.dispatch("gt", a, 0)
        grad = OperationDispatcher.dispatch("mul", out_grad, relu_mask)
        grad.requires_grad = False
        return (grad, )

def relu(a):
    return ReLU.apply(a)

class LogSumExp(Function):
    @staticmethod
    def forward(ctx, a, axis=None):
        ctx.save_for_backward(a)
        ctx.axis = axis
        max_value = OperationDispatcher.dispatch("max", a, axis=axis, keepdims=True)
        ctx.max_value = max_value
        max_z = OperationDispatcher.dispatch("broadcast_to", max_value, a.shape)
        Z_minus_max = OperationDispatcher.dispatch("sub", a, max_z)
        Z_exp = OperationDispatcher.dispatch("exp", Z_minus_max)
        Z_sum = OperationDispatcher.dispatch("sum", Z_exp, axis=axis)
        Z_log = OperationDispatcher.dispatch("log", Z_sum)
        reshaped_max = OperationDispatcher.dispatch("reshape", max_value, Z_log.shape)
        result = OperationDispatcher.dispatch("add", Z_log, reshaped_max)
        result.requires_grad = a.requires_grad
        return result

    @staticmethod
    def backward(ctx, out_grad):
        hs, = ctx.saved_tensors
        input_shape = hs.shape
        max_z = OperationDispatcher.dispatch("broadcast_to", ctx.max_value, input_shape)
        base_shape = list(input_shape)
        if isinstance(ctx.axis, int): 
            ctx.axis = (ctx.axis,)
        axis = list(range(len(base_shape))) if ctx.axis is None else ctx.axis
        for ax in axis:
            base_shape[ax] = 1
        
        # Compute exp(hs - max_z) and sum
        hs_minus_max = OperationDispatcher.dispatch("sub", hs, max_z)
        exp_hs = OperationDispatcher.dispatch("exp", hs_minus_max)
        sum_exp = OperationDispatcher.dispatch("sum", exp_hs, axis=ctx.axis)
        
        # Scale out_grad
        scaled_grad = OperationDispatcher.dispatch("truediv", out_grad, sum_exp)
        reshaped_grad = OperationDispatcher.dispatch("reshape", scaled_grad, base_shape)
        broadcasted_grad = OperationDispatcher.dispatch("broadcast_to", reshaped_grad, input_shape)
        grad = OperationDispatcher.dispatch("mul", broadcasted_grad, exp_hs)
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
        result = OperationDispatcher.dispatch("max", a, axis=axis, keepdims=keepdims)
        result.requires_grad = a.requires_grad
        return result

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
        max_vals = OperationDispatcher.dispatch("max", hs, axis=ctx.axis, keepdims=True)
        broadcasted_max = OperationDispatcher.dispatch("broadcast_to", max_vals, hs.shape)
        mask = OperationDispatcher.dispatch("eq", hs, broadcasted_max)
        
        reshaped_grad = OperationDispatcher.dispatch("reshape", out_grad, grad_shape)
        broadcasted_grad = OperationDispatcher.dispatch("broadcast_to", reshaped_grad, hs.shape)
        grad = OperationDispatcher.dispatch("mul", broadcasted_grad, mask)
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
        base_shape = list(tensors[0].shape)
        output_shape = base_shape[:dim] + [len(tensors)] + base_shape[dim:]
        
        # Expand each tensor along the stack dimension and then concatenate
        expanded_tensors = []
        for t in tensors:
            # Add a new dimension at position dim with size 1
            new_shape = list(t.shape)
            new_shape.insert(dim, 1)
            # Use reshape to add the dimension (this should work since it's just a view)
            expanded = t.view(new_shape)
            expanded_tensors.append(expanded)
        
        # Now concatenate along the dimension we just created
        result = cat(expanded_tensors, dim)
        result.requires_grad = requires_grad
        return result
    
    @staticmethod
    def backward(ctx, out_grad):
        # Split into ctx.num_tensors parts along ctx.dim dimension
        # Use dispatcher for indexing
        result = []
        
        # Extract individual slices from the stacked tensor
        for i in range(ctx.num_tensors):
            # Use dispatcher getitem to extract slice i from dimension ctx.dim
            indices = [slice(None)] * len(out_grad.shape)
            indices[ctx.dim] = i
            slice_tensor = OperationDispatcher.dispatch("getitem", out_grad, tuple(indices))
            slice_tensor.requires_grad = False
            result.append(slice_tensor)
        
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
        
        # Use dispatcher for concatenation
        result = OperationDispatcher.dispatch("cat", tensors, dim=dim)
        
        requires_grad = any(t.requires_grad for t in tensors)
        result.requires_grad = requires_grad
        return result 
    
    @staticmethod
    def backward(ctx, out_grad):
        # Get sizes from saved tensors
        sizes = [t.shape[ctx.dim] for t in ctx.saved_tensors]
        
        # Use dispatcher for split operation (returns tuple)
        grad_splits = OperationDispatcher.dispatch_tuple("split", out_grad, sizes, dim=ctx.dim)
        
        # grad_splits should be a tuple of tensors with requires_grad=False
        result = []
        for grad_tensor in grad_splits:
            grad_tensor.requires_grad = False
            result.append(grad_tensor)
        
        return tuple(result) 

def cat(tensors, dim=0):
    return Cat.apply(tensors, dim=dim)

class Squeeze(Function):
    @staticmethod
    def forward(ctx, tensor, dim):
        ctx.dim = dim
        # Use dispatcher for squeeze operation
        result = OperationDispatcher.dispatch("squeeze", tensor, dim)
        result.requires_grad = tensor.requires_grad
        return result
    
    @staticmethod
    def backward(ctx, out_grad):
        # Use dispatcher for unsqueeze operation
        unsqueezed = OperationDispatcher.dispatch("unsqueeze", out_grad, ctx.dim)
        unsqueezed.requires_grad = False
        return (unsqueezed,) 

def squeeze(tensor, dim):
    return Squeeze.apply(tensor, dim)

class Unsqueeze(Function):
    @staticmethod
    def forward(ctx, tensor, dim):
        ctx.dim = dim
        # Use dispatcher for unsqueeze operation
        result = OperationDispatcher.dispatch("unsqueeze", tensor, dim)
        result.requires_grad = tensor.requires_grad
        return result
    
    @staticmethod
    def backward(ctx, out_grad):
        # Use dispatcher for squeeze operation
        squeezed = OperationDispatcher.dispatch("squeeze", out_grad, ctx.dim)
        squeezed.requires_grad = False
        return (squeezed,)

def unsqueeze(tensor, dim):
    return Unsqueeze.apply(tensor, dim)

class Split(Function):
    @staticmethod
    def forward(ctx, x, dim):
        ctx.save_for_backward(x)
        if dim < 0:
            dim = dim + len(x.shape)
        ctx.dim = dim
        
        # Use registered split operation via dispatch_tuple
        # Split into individual elements along dimension (size 1 splits)
        split_sizes = [1] * x.shape[dim]  # Each split has size 1
        results = OperationDispatcher.dispatch_tuple("split", x, split_sizes, dim=dim)
        
        # Set requires_grad for all result tensors
        for result in results:
            result.requires_grad = x.requires_grad
        
        return results

    @staticmethod
    def backward(ctx, out_grad, idx):
        x, = ctx.saved_tensors
        result = genesis.zeros_like(x, requires_grad=False)
        slices = [slice(None)] * len(x.shape)
        slices[ctx.dim] = slice(idx, idx+1)
        
        # Use dispatcher setitem operation (equivalent to result.data[slices] = out_grad.data)
        result = OperationDispatcher.dispatch("setitem", result, tuple(slices), out_grad)
        result.requires_grad = False
        return (result,)

def split(a, dim):
    return Split.apply(a, dim=dim)

class Norm(Function):
    @staticmethod
    def forward(ctx, a):
        ctx.save_for_backward(a)
        result_data = OperationDispatcher.dispatch("neg", a)
        result_data.requires_grad = a.requires_grad
        return result_data

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
        neg_a = OperationDispatcher.dispatch("neg", a)
        exp_neg_a = OperationDispatcher.dispatch("exp", neg_a)
        one_plus_exp = OperationDispatcher.dispatch("add", 1, exp_neg_a)
        sigmoid_out = OperationDispatcher.dispatch("truediv", 1, one_plus_exp)
        sigmoid_out.requires_grad = a.requires_grad
        return sigmoid_out

    @staticmethod
    def backward(ctx, out_grad):
        a, = ctx.saved_tensors
        # derivative of sigmoid: sigmoid(x) * (1 - sigmoid(x))
        neg_a = OperationDispatcher.dispatch("neg", a)
        exp_neg_a = OperationDispatcher.dispatch("exp", neg_a)
        one_plus_exp = OperationDispatcher.dispatch("add", 1, exp_neg_a)
        sigmoid_out = OperationDispatcher.dispatch("truediv", 1, one_plus_exp)
        one_minus_sigmoid = OperationDispatcher.dispatch("sub", 1, sigmoid_out)
        grad_temp = OperationDispatcher.dispatch("mul", sigmoid_out, one_minus_sigmoid)
        grad = OperationDispatcher.dispatch("mul", out_grad, grad_temp)
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
        two_a = OperationDispatcher.dispatch("mul", 2, a)
        exp_2x = OperationDispatcher.dispatch("exp", two_a)
        exp_2x_minus_1 = OperationDispatcher.dispatch("sub", exp_2x, 1)
        exp_2x_plus_1 = OperationDispatcher.dispatch("add", exp_2x, 1)
        tanh_out = OperationDispatcher.dispatch("truediv", exp_2x_minus_1, exp_2x_plus_1)
        tanh_out.requires_grad = a.requires_grad
        return tanh_out

    @staticmethod
    def backward(ctx, out_grad):
        a, = ctx.saved_tensors
        # derivative of tanh: 1 - tanh^2(x)
        two_a = OperationDispatcher.dispatch("mul", 2, a)
        exp_2x = OperationDispatcher.dispatch("exp", two_a)
        exp_2x_minus_1 = OperationDispatcher.dispatch("sub", exp_2x, 1)
        exp_2x_plus_1 = OperationDispatcher.dispatch("add", exp_2x, 1)
        tanh_out = OperationDispatcher.dispatch("truediv", exp_2x_minus_1, exp_2x_plus_1)
        tanh_squared = OperationDispatcher.dispatch("mul", tanh_out, tanh_out)
        one_minus_tanh_sq = OperationDispatcher.dispatch("sub", 1, tanh_squared)
        grad = OperationDispatcher.dispatch("mul", out_grad, one_minus_tanh_sq)
        return (grad,)

def tanh(a):
    """Apply tanh activation function"""
    return Tanh.apply(a)


class ScatterAddFunction(Function):
    @staticmethod
    def forward(ctx, input, dim, index, src):
        ctx.dim = dim
        ctx.save_for_backward(index, src)
        result = OperationDispatcher.dispatch("scatter_add", input, dim, index, src)
        # Set proper requires_grad
        result.requires_grad = input.requires_grad or src.requires_grad
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
            # Use dispatcher to gather gradients from the scattered positions
            src_grad = OperationDispatcher.dispatch("gather", out_grad, dim, index)
            
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
    result = OperationDispatcher.dispatch("repeat_interleave", input, repeats, dim)
    return result


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


def softmax(input, dim=-1):
    """
    Softmax function with proper CPU fallback.

    Args:
        input: Input tensor
        dim: Dimension to apply softmax along

    Returns:
        Softmax of input
    """
    return triton_softmax(input, dim)


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
        result_data = OperationDispatcher.dispatch("maximum", a, b)
        result_data.requires_grad = a.requires_grad or b.requires_grad
        return result_data
    
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
