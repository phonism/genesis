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

def _cast(value, dtype):
    if isinstance(value, Tensor):
        if value.is_floating_point():
            if dtype == genesis.float16:
                return value.half()
            else:
                return value.float()
        else:
            return value
    elif isinstance(value, dict):
        return {_cast(k, dtype): _cast(v, dtype) for k, v in value.items()}
    elif isinstance(value, list) or isinstance(value, tuple):
        return type(value)(_cast(v, dtype) for v in value)
    else:
        return value

def check_dtype(value, dtype): 
    if isinstance(value, Tensor):
        return value.dtype == dtype
    elif isinstance(value, dict):
        return any(check_dtype(k, dtype) or check_dtype(v, dtype) for k, v in value.items())
    elif isinstance(value, list) or isinstance(value, tuple):
        return any(check_dtype(v, dtype) for v in value)
    else:
        return False

class Function:
    """
    operator definitions
    """
    def __init__(self):
        self.inputs = []
        self.ctx = Context()

    @staticmethod
    def forward(ctx, *args, **kwargs):

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
    def apply(cls, *args, **kwargs):
        instance = cls()  # Create a new instance for each call

        if genesis.enable_autocast and genesis.upgrade is False:
            result = cls.forward(
                    instance.ctx, *_cast(args, genesis.float16), **_cast(kwargs, genesis.float16))
        else:
            has_float32 = check_dtype(args, genesis.float32) or check_dtype(kwargs, genesis.float32)
            has_float16 = check_dtype(args, genesis.float16) or check_dtype(kwargs, genesis.float16)
            if has_float32 and has_float16:
                result = cls.forward(instance.ctx, *_cast(args, genesis.float32), **_cast(kwargs, genesis.float32))
            else:
                result = cls.forward(instance.ctx, *args, **kwargs)
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
        self.grad = None
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
        self.apply_hooks(self.grad)
        node_to_output_grads_list: Dict[Tensor, List[Tensor]] = {}
        node_to_output_grads_list[self] = [out_grad]

        topo_order = topo_sort(self)
        for node in reversed(topo_order):
            if node.requires_grad is False:
                continue
            if node.grad is None:
                node.grad = reduce(operator.add, node_to_output_grads_list[node])
            else:
                node.grad += reduce(operator.add, node_to_output_grads_list[node])
            node.apply_hooks(node.grad)
            if node.creator is not None:
                for nd in node.creator.inputs:
                    if nd not in node_to_output_grads_list:
                        node_to_output_grads_list[nd] = []
                if check_dtype(node.creator.ctx.saved_tensors, genesis.float16):
                    grad = node.grad.half()
                else:
                    grad = node.grad
                if node.creator.is_tuple_result is False:
                    backward_grad = node.creator.backward(node.creator.ctx, grad)
                else:
                    backward_grad = node.creator.backward(node.creator.ctx, grad, node.idx)
                for i, nd in enumerate(node.creator.inputs):
                    if nd.requires_grad is False:
                        continue
                    node_to_output_grads_list[nd].append(backward_grad[i].float())

    @property
    def shape(self):
        return self.data.shape

    @property
    def ndim(self):
        return len(self.shape)

    @property
    def size(self):
        return self.data.size

    def size(self, dim=None):
        if dim is not None:
            return self.data.size(dim)
        return self.data.shape

    @property
    def dtype(self):
        return self.data.dtype

    @property
    def device(self):
        data = self.data
        return data.device

    def float(self):
        tensor = Tensor.__new__(Tensor)
        tensor.init([], data=self.data.float(), requires_grad=self.requires_grad)
        return tensor

    def half(self):
        tensor = Tensor.__new__(Tensor)
        tensor.init([], data=self.data.half(), requires_grad=self.requires_grad)
        return tensor

    def to(self, device):
        tensor = Tensor(self, device=genesis.device(device))
        return tensor

    def contiguous(self):
        self.data = self.data.contiguous()
        return self

    def set_device(self, device_name):
        self.data = array_api.array(self.data, device=genesis.device(device_name))

    def __repr__(self):
        return "Id:" + str(id(self)) + " Tensor(" + str(self.data) + ")"

    def __hash__(self):
        return id(self)

    def __bool__(self):
        return bool(self.data)

    def __len__(self):
        return self.data.size

    def __getitem__(self, index):
        return genesis.nn.functional.getitem(self, index)

    def __setitem__(self, index, value):
        return genesis.nn.functional.setitem(self, index, value)

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

    def __rtruediv__(self, other):
        if isinstance(other, Tensor):
            return genesis.nn.functional.divide(other, self)
        else:
            return genesis.nn.functional.divide_scalar(self, other, reverse=True)

    def __pow__(self, other):
        if isinstance(other, Tensor):
            raise TypeError("pow value must be a scalar")
        else:
            return genesis.nn.functional.pow_scalar(self, other)

    def __rpow__(self, other):
        if isinstance(other, Tensor):
            raise TypeError("pow value must be a scalar")
        else:
            return genesis.nn.functional.pow_scalar(self, other, reverse=True)

    def __neg__(self):
        return genesis.nn.functional.negate(self)

    def equal(self, other):
        if isinstance(other, Tensor):
            return Tensor(self.data == other.data, device=self.device, requires_grad=False)
        else:
            return Tensor(self.data == other, device=self.device, requires_grad=False)

    def __eq__(self, other):
        return self.equal(other)

    def __ne__(self, other):
        if isinstance(other, Tensor):
            return Tensor(self.data != other.data, device=self.device, requires_grad=False)
        else:
            return Tensor(self.data != other, device=self.device, requires_grad=False)

    def __lt__(self, other):
        if isinstance(other, Tensor):
            return Tensor(self.data < other.data, device=self.device, requires_grad=False)
        else:
            return Tensor(self.data < other, device=self.device, requires_grad=False)

    def __gt__(self, other):
        if isinstance(other, Tensor):
            return Tensor(self.data > other.data, device=self.device, requires_grad=False)
        else:
            return Tensor(self.data > other, device=self.device, requires_grad=False)

    def sin(self):
        return genesis.nn.functional.sin(self)

    def cos(self):
        return genesis.nn.functional.cos(self)

    def log(self):
        return genesis.nn.functional.log(self)

    def exp(self):
        return genesis.nn.functional.exp(self)


    def transpose(self, *axis):
        if not axis:
            axis = None
        elif len(axis) == 1 and isinstance(axis[0], (tuple, list)):
            axis = axis[0]
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

    def view(self, *new_shape):
        if len(new_shape) == 1 and isinstance(new_shape[0], (tuple, list)):
            new_shape = new_shape[0]
        return genesis.nn.functional.view(self, new_shape)

    def flatten(self, start_dim=0, end_dim=None):
        return genesis.nn.functional.flatten(self, start_dim=start_dim, end_dim=end_dim)

    def split(self, dim=-1):
        return genesis.nn.functional.split(self, dim)


    __radd__ = __add__
    __rsub__ = __sub__
    __rmul__ = __mul__
    __rmatmul__ = __matmul__

    def is_floating_point(self):
        return self.data.is_floating_point()

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
