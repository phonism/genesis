"""
API binding layer for Genesis Tensor methods.

This module binds functional operations as Tensor methods using lambda functions,
similar to the existing pattern but with the new dispatcher system.
"""


def bind_tensor_methods():
    """
    Bind functional operations as Tensor methods.
    This function should be called during package initialization.
    """
    from genesis.tensor import Tensor
    from genesis import functional as F
    
    # Arithmetic operations - direct binding
    Tensor.__add__ = lambda self, other: F.add(self, other)
    Tensor.__radd__ = lambda self, other: F.add(self, other)  # Addition is commutative
    Tensor.__sub__ = lambda self, other: F.sub(self, other)
    Tensor.__rsub__ = lambda self, other: F.sub_scalar(self, other, reverse=True)  # scalar - tensor
    Tensor.__mul__ = lambda self, other: F.mul(self, other)
    Tensor.__rmul__ = lambda self, other: F.mul(self, other)  # Multiplication is commutative
    Tensor.__truediv__ = lambda self, other: F.truediv(self, other)
    Tensor.__rtruediv__ = lambda self, other: F.divide_scalar(self, other, reverse=True)  # scalar / tensor
    Tensor.__floordiv__ = lambda self, other: F.floordiv(self, other)  # floor division
    Tensor.__pow__ = lambda self, other: F.pow(self, other)
    Tensor.__rpow__ = lambda self, other: F.pow_scalar(self, other, reverse=True)  # scalar ** tensor
    Tensor.__neg__ = lambda self: F.neg(self)
    Tensor.__abs__ = lambda self: F.abs(self)

    # In-place operations
    Tensor.__iadd__ = lambda self, other: F.add_inplace(self, other)
    # Tensor.__isub__ = lambda self, other: F.sub_inplace(self, other)  # TODO: implement
    # Tensor.__imul__ = lambda self, other: F.mul_inplace(self, other)  # TODO: implement

    # Method versions
    Tensor.add = lambda self, other: F.add(self, other)
    Tensor.sub = lambda self, other: F.sub(self, other)
    Tensor.mul = lambda self, other: F.mul(self, other)
    Tensor.div = lambda self, other: F.truediv(self, other)
    Tensor.pow = lambda self, other: F.pow(self, other)
    Tensor.neg = lambda self: F.neg(self)
    Tensor.abs = lambda self: F.abs(self)
    
    # Math functions
    Tensor.exp = lambda self: F.exp(self)
    Tensor.log = lambda self: F.log(self)
    Tensor.sqrt = lambda self: F.sqrt(self)
    Tensor.rsqrt = lambda self: F.pow(self, -0.5)  # 1/sqrt(x) = x^(-0.5)
    Tensor.sin = lambda self: F.sin(self)
    Tensor.cos = lambda self: F.cos(self)
    Tensor.tanh = lambda self: F.tanh(self)
    
    # Matrix operations
    Tensor.matmul = lambda self, other: F.matmul(self, other)
    Tensor.mm = lambda self, other: F.matmul(self, other)
    Tensor.__matmul__ = lambda self, other: F.matmul(self, other)
    
    # View operations - direct binding
    Tensor.view = lambda self, *shape: F.view(self, *shape)
    Tensor.reshape = lambda self, *shape: F.reshape(self, *shape)
    
    # Permutation operations - direct binding
    Tensor.permute = lambda self, *dims: F.permute(self, dims if len(dims) > 1 else dims[0])
    Tensor.transpose = lambda self, dim0, dim1: F.transpose(self, axis=(dim0, dim1))
    
    # T property for 2D transpose
    def T_getter(self):
        """Transpose property for 2D tensors."""
        if len(self.shape) != 2:
            raise ValueError("T property only valid for 2D tensors")
        return F.transpose(self, axis=(0, 1))

    Tensor.T = property(T_getter)

    # t() method for 2D transpose (PyTorch compatible)
    Tensor.t = lambda self: F.t(self)
    
    # Shape operations - direct binding
    Tensor.squeeze = lambda self, dim=None: F.squeeze(self, dim)
    Tensor.unsqueeze = lambda self, dim: F.unsqueeze(self, dim)
    Tensor.expand = lambda self, *shape: F.expand(self, *shape)
    
    # Element-wise operations
    Tensor.clamp = lambda self, min=None, max=None: F.clamp(self, min_val=min, max_val=max)
    Tensor.clip = lambda self, min=None, max=None: F.clip(self, min_val=min, max_val=max)
    
    # Reduction operations
    Tensor.sum = lambda self, dim=None, keepdim=False: F.sum(self, dim, keepdim)
    Tensor.mean = lambda self, dim=None, keepdim=False: F.mean(self, dim, keepdim)
    Tensor.max = lambda self, dim=None, keepdim=False: F.max(self, dim, keepdim)
    Tensor.min = lambda self, dim=None, keepdim=False: F.min(self, dim, keepdim)
    Tensor.argmax = lambda self, dim=None, keepdim=False: F.argmax(self, dim, keepdim)
    Tensor.argmin = lambda self, dim=None, keepdim=False: F.argmin(self, dim, keepdim)
    Tensor.argsort = lambda self, dim=-1, descending=False: F.argsort(self, dim, descending)
    
    # Comparison operations
    Tensor.eq = lambda self, other: F.eq(self, other)
    Tensor.ne = lambda self, other: F.ne(self, other)
    Tensor.lt = lambda self, other: F.lt(self, other)
    Tensor.le = lambda self, other: F.le(self, other)
    Tensor.gt = lambda self, other: F.gt(self, other)
    Tensor.ge = lambda self, other: F.ge(self, other)
    
    # Comparison operators
    Tensor.__eq__ = lambda self, other: F.eq(self, other)
    Tensor.__ne__ = lambda self, other: F.ne(self, other)
    Tensor.__lt__ = lambda self, other: F.lt(self, other)
    Tensor.__le__ = lambda self, other: F.le(self, other)

    # Data type conversion methods
    Tensor.long = lambda self: self.to_dtype("int64")
    Tensor.int = lambda self: self.to_dtype("int32")
    Tensor.float = lambda self: self.to_dtype("float32")
    Tensor.double = lambda self: self.to_dtype("float64")
    Tensor.half = lambda self: self.to_dtype("float16")
    Tensor.__gt__ = lambda self, other: F.gt(self, other)
    Tensor.__ge__ = lambda self, other: F.ge(self, other)
    


def bind_nn_functional_methods():
    """
    Bind nn.functional operations as Tensor methods.
    This function should be called during package initialization.
    """
    from genesis.tensor import Tensor
    from genesis.nn import functional as NF
    
    # Neural network activation functions
    Tensor.relu = lambda self: NF.relu(self)
    Tensor.sigmoid = lambda self: NF.sigmoid(self)
    Tensor.softmax = lambda self, dim=-1: NF.softmax(self, dim)
    Tensor.log_softmax = lambda self, dim=-1: NF.log_softmax(self, dim)
    
    # Tensor manipulation functions
    Tensor.repeat_interleave = lambda self, repeats, dim=None: NF.repeat_interleave(self, repeats, dim)
    Tensor.scatter_add = lambda self, dim, index, src: NF.scatter_add(self, dim, index, src)
    
    # Loss functions (these typically need targets, so method versions might be less useful)
    # Tensor.cross_entropy = lambda self, target: NF.cross_entropy(self, target)
    # Tensor.mse_loss = lambda self, target: NF.mse_loss(self, target)
    
    # Dropout (though this is typically used in training mode)
    Tensor.dropout = lambda self, p=0.5, training=True: NF.dropout(self, p, training)

    # Indexing operations
    Tensor.__getitem__ = lambda self, key: NF.getitem(self, key)
    Tensor.__setitem__ = lambda self, key, value: NF.setitem(self, key, value)

    # Length method for PyTorch compatibility
    Tensor.__len__ = lambda self: self.shape[0] if self.shape else 0
    
    # Gather and scatter operations
    Tensor.gather = lambda self, dim, index: NF.gather(self, dim, index)
    Tensor.scatter = lambda self, dim, index, src: NF.scatter(self, dim, index, src)
