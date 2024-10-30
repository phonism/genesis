import genesis
from typing import List, Optional, NamedTuple, Tuple, Union

import numpy
from genesis import init
from .backend import Device, array_api, NDArray, default_device
import operator
from functools import reduce

TENSOR_COUNTER = 0

class Context:
    def __init__(self):
        self.saved_tensors = []

    def save_for_backward(self, *tensors):
        self.saved_tensors.extend(tensors)

    @property
    def saved_tensors(self):
        return self._saved_tensors

    @saved_tensors.setter
    def saved_tensors(self, tensors):
        self._saved_tensors = tensors

class Function:
    """
    operator definitions
    """
    def __init__(self):
        self.inputs = []
        self.ctx = Context()

    @staticmethod
    def forward(ctx, *args, **kwarge):

        """
        Calculate forward pass of operator.

        Args:
            input (NDArray): A list of input arrays to the function

        Returns:
            Array: Array output of the operation
        """
        raise NotImplementedError()

    @staticmethod
    def backward(ctx, *args) -> Union["Tensor", Tuple["Tensor"]]:
        """
        Compute partial adjoint for each input value for a given output adjoint.

        Args:
            out_grad (Tensor): The adjoint with respect to the output value. 
            node (Tensor): The value node of forward evaluation.

        Returns:
            Tensor or Tuple[Tensor]: A list containing partial gradient adjoints to be propagated to each of the input node.
        """
        raise NotImplementedError()

    @classmethod
    def apply(cls, *args, **kwarge):
        instance = cls()  # Create a new instance for each call
        result = cls.forward(instance.ctx, *args, **kwarge)
        instance.is_tuple_result = isinstance(result, tuple)

        if instance.is_tuple_result:
            for idx, res in enumerate(result):
                if isinstance(res, Tensor) and res.requires_grad:
                    res.set_creator(instance, idx)
        elif isinstance(result, Tensor) and result.requires_grad:
            result.set_creator(instance)

        instance.inputs = []
        for t in args:
            if isinstance(t, Tensor):
                instance.inputs.append(t)
            if isinstance(t, list) and all(isinstance(item, Tensor) for item in t):
                for tt in t:
                    instance.inputs.append(tt)
        return result


class Tensor:
    """
    basic type
    """
    grad: "Tensor"
    op: Optional[Function]
    inputs: List["Tensor"]
    data: NDArray
    requires_grad: bool

    def __init__(self, array, *, device: Optional[Device] = None, dtype=None, requires_grad=True, **kwargs):
        if isinstance(array, Tensor):
            if device is None:
                device = array.device
            if dtype is None:
                dtype = array.dtype
            if device == array.device and dtype == array.dtype:
                data = array.data
            else:
                data = Tensor._array_from_numpy(array.numpy(), device=device, dtype=dtype)
        elif isinstance(array, NDArray):
            data = Tensor._array_from_numpy(array, device=array.device, dtype=dtype)
        else:
            device = device if device else default_device()
            data = Tensor._array_from_numpy(array, device=device, dtype=dtype)
        self.creator = None
        self.grad = None

        self.init([], data=data, requires_grad=requires_grad)

    def init(self, inputs: List["Tensor"], *, data: List[object] = None, requires_grad: Optional[bool] = None):
        """
        Initialize a new Tensor object with the given operation and input tensors.

        Args:
            op (Optional[Op]): The operation producing this tensor, if any. It can be None if the tensor is created directly without an operation.
            inputs (List["Tensor"]): A list of input Tensor objects that this tensor depends on.
            data (List[object], optional): Pre-computed data or intermediates that can be reused. None by default.
            requires_grad (Optional[bool], optional): Whether this tensor requires the computation of gradients. If None, it is inferred from the input tensors.
        """
        global TENSOR_COUNTER
        TENSOR_COUNTER += 1

        if requires_grad is None:
            # check the inputs op requires grad
            requires_grad = any(x.requires_grad for x in inputs)
        self.inputs = inputs
        self.data = data
        self.requires_grad = requires_grad
        self.creator = None
        self.hooks = []

    def register_hook(self, hook):
        self.hooks.append(hook)

    def apply_hooks(self, grad):
        for hook in self.hooks:
            hook(grad)

    def __del__(self):
        global TENSOR_COUNTER
        TENSOR_COUNTER -= 1

    def detach(self):
        """
        generate a const Tensor
        """
        return Tensor.make_const(self.data)

    @staticmethod
    def _array_from_numpy(numpy_array, device, dtype):
        return array_api.array(numpy_array, device=device, dtype=dtype)

    @staticmethod
    def make_const(data, requires_grad=False):
        """
        make const
        """
        tensor = Tensor.__new__(Tensor)
        if isinstance(data, Tensor):
            tensor.init([], data=data.data, requires_grad=requires_grad)
        else:
            tensor.init([], data=data, requires_grad=requires_grad)
        return tensor

    def is_leaf(self) -> bool:
        """
        check current value is the leaf node in the computation graph
        """
        return self.creator is None

    def set_creator(self, creator, idx=-1):
        self.creator = creator
        self.idx = idx

    def copy_(self, value):
        assert isinstance(value, Tensor)
        assert value.dtype == self.dtype, "%s %s" % (value.dtype, self.dtype)
        self.data = value.data.clone()

    def backward(self, out_grad=None):
        out_grad = out_grad if out_grad else init.ones(*self.shape, dtype=self.dtype, device=self.device)
        self.grad = out_grad
        self.apply_hooks(self.grad)
        node_to_output_grads_list: Dict[Tensor, List[Tensor]] = {}
        node_to_output_grads_list[self] = [out_grad]

        topo_order = topo_sort(self)
        for node in reversed(topo_order):
            node.grad = reduce(operator.add, node_to_output_grads_list[node])
            node.apply_hooks(node.grad)
            if node.creator is not None:
                for nd in node.creator.inputs:
                    if nd not in node_to_output_grads_list:
                        node_to_output_grads_list[nd] = []
                if node.creator.is_tuple_result is False:
                    backward_grad = node.creator.backward(node.creator.ctx, node.grad)
                else:
                    backward_grad = node.creator.backward(node.creator.ctx, node.grad, node.idx)
                for i, nd in enumerate(node.creator.inputs):
                    node_to_output_grads_list[nd].append(backward_grad[i])

    @property
    def shape(self):
        return self.data.shape

    @property
    def dtype(self):
        return self.data.dtype

    @property
    def device(self):
        data = self.data
        return data.device

    def to(self, device):
        tensor = Tensor(self, device=genesis.device(device))
        return tensor

    def set_device(self, device_name):
        self.data = array_api.array(self.data, device=genesis.device(device_name))

    def __repr__(self):
        return "Id:" + str(id(self)) + " Tensor(" + str(self.data) + ")"

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        tensor = Tensor.__new__(Tensor)
        tensor.init([], data=self.data[index], requires_grad=self.requires_grad)
        return tensor

    def __str__(self):
        return str(self.data)

    def numpy(self):
        data = self.data
        if array_api is numpy:
            return data
        return data.numpy()

    def __add__(self, other):
        if isinstance(other, Tensor):
            return genesis.nn.functional.add(self, other)
        else:
            return genesis.nn.functional.add_scalar(self, other)

    def __sub__(self, other):
        if isinstance(other, Tensor):
            return genesis.nn.functional.add(self, genesis.nn.functional.negate(other))
        else:
            return genesis.nn.functional.add_scalar(self, -other)

    def __mul__(self, other):
        if isinstance(other, Tensor):
            return genesis.nn.functional.multiply(self, other)
        else:
            return genesis.nn.functional.mul_scalar(self, other)

    def __truediv__(self, other):
        if isinstance(other, Tensor):
            return genesis.nn.functional.divide(self, other)
        else:
            return genesis.nn.functional.divide_scalar(self, other)

    def __pow__(self, other):
        if isinstance(other, Tensor):
            raise TypeError("pow value must be a scalar")
        else:
            return genesis.nn.functional.pow_scalar(self, other)

    def __neg__(self):
        return genesis.nn.functional.negate(self)

    def equal(self, other):
        if isinstance(other, Tensor):
            return genesis.nn.functional.equal(self, other)
        else:
            return genesis.nn.functional.equal(self, other)

    def sin(self):
        return genesis.nn.functional.sin(self)

    def cos(self):
        return genesis.nn.functional.cos(self)

    def log(self):
        return genesis.nn.functional.log(self)

    def exp(self):
        return genesis.nn.functional.exp(self)

    def transpose(self, axis=None):
        return genesis.nn.functional.transpose(self, axis)

    def reshape(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = shape[0]
        return genesis.nn.functional.reshape(self, shape)

    def summation(self, axis=None, keepdims=False):
        return genesis.nn.functional.summation(self, axis=axis, keepdims=keepdims)

    def sum(self, axis=None, keepdims=False):
        return genesis.nn.functional.summation(self, axis=axis, keepdims=keepdims)

    def broadcast_to(self, shape):
        return genesis.nn.functional.broadcast_to(self, shape)

    def __matmul__(self, other):
        return genesis.nn.functional.matmul(self, other)

    def matmul(self, other):
        return genesis.nn.functional.matmul(self, other)

    def sqrt(self):
        return genesis.nn.functional.sqrt(self)

    __radd__ = __add__
    __rsub__ = __sub__
    __rmul__ = __mul__
    __rmatmul__ = __matmul__

def topo_sort(node):
    visited = set()
    topo_order = []

    def dfs(n):
        if n in visited:
            return
        visited.add(n)
        if n.creator is not None:
            for input_node in n.creator.inputs:
                if isinstance(input_node, Tensor):
                    dfs(input_node)
        topo_order.append(n) 
    dfs(node)
    return topo_order
