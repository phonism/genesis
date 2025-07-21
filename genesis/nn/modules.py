"""
The module
"""

from typing import (
    List, Callable, Any, Optional, Dict, Iterator, Tuple
)
import genesis
from ..autograd import Tensor
import genesis.nn.functional as F
from genesis import init
import numpy as np

class Parameter(Tensor):
    """
    A special kind of tensor that represents parameters.
    """


def _unpack_params(value: object) -> List[Tensor]:
    """
    Unpack parameters from a value.
    """
    if isinstance(value, Parameter):
        return [value]
    elif isinstance(value, Module):
        return value.parameters()
    elif isinstance(value, dict):
        params = []
        for k, v in value.items():
            params += _unpack_params(v)
        return params
    elif isinstance(value, (list, tuple)):
        params = []
        for v in value:
            params += _unpack_params(v)
        return params
    else:
        return []


def _unpack_vars(value: object) -> List[Tensor]:
    """
    Unpack variables from a value.
    """
    if isinstance(value, Tensor):
        return [value]
    elif isinstance(value, Module):
        return value.vars()
    elif isinstance(value, dict):
        var_list = []
        for k, v in value.items():
            var_list += _unpack_vars(v)
        return var_list
    elif isinstance(value, (list, tuple)):
        var_list = []
        for v in value:
            var_list += _unpack_vars(v)
        return var_list
    else:
        return []


def _child_modules(value: object) -> List["Module"]:
    """
    Unpack child modules from a value.
    """
    if isinstance(value, Module):
        modules = [value]
        modules.extend(_child_modules(value.__dict__))
        return modules
    if isinstance(value, dict):
        modules = []
        for k, v in value.items():
            modules += _child_modules(v)
        return modules
    elif isinstance(value, (list, tuple)):
        modules = []
        for v in value:
            modules += _child_modules(v)
        return modules
    return []


class Module:
    """
    Base class for all neural network modules.
    """
    def __init__(self):
        self.training = True

    def register_buffer(self, name: str, buffer: Tensor, persistent: bool = True):
        """
        Register a buffer to the module.
        """
        self.__dict__[name] = buffer

    def parameters(self) -> List[Tensor]:
        """
        Return the list of parameters in the module.
        """
        return _unpack_params(self.__dict__)
    
    def num_parameters(self) -> int:
        """
        Return the number of parameters in the module.
        """
        num_parameters = 0
        for p in self.parameters():
            cur = 1
            for x in p.shape:
                cur *= x
            num_parameters += cur
        return num_parameters

    def vars(self) -> List[Tensor]:
        """
        Return the list of variables in the module.
        """
        return _unpack_vars(self.__dict__)

    def _children(self) -> List["Module"]:
        """
        Return the list of child modules in the module.
        """
        return _child_modules(self.__dict__)

    def state_dict(self, prefix="") -> Dict[str, Tensor]:
        """
        Return the state dictionary of the module.
        """
        state_dict = {}
        for name, param in self.__dict__.items():
            if isinstance(param, genesis.Tensor):
                # TODO: we need to dump genesis.Tensor
                state_dict[prefix + name] = param.data.data
            elif isinstance(param, Module):
                state_dict.update(param.state_dict(prefix + name + "."))
            elif isinstance(param, (list, tuple)):
                for idx, v in enumerate(param):
                    if isinstance(v, Module):
                        state_dict.update(v.state_dict(prefix + str(idx) + "."))
        return state_dict

    def load_state_dict(self, state_dict, strict=True) -> None:
        """
        Load the state dictionary of the module.
        """
        missing_keys = []
        unexpected_keys = list(state_dict.keys())

        def load(module, prefix=""):
            """
            Load the state dictionary of the module.
            """
            for name, param in module.__dict__.items():
                full_name = prefix + name

                if isinstance(param, genesis.Tensor):
                    if full_name in state_dict:
                        param.data.data.copy_(state_dict[full_name])
                        unexpected_keys.remove(full_name)
                    elif strict:
                        missing_keys.append(full_name)
                elif isinstance(param, Module):
                    load(param, full_name + ".")
                elif isinstance(param, (list, tuple)):
                    for idx, sub_param in enumerate(param):
                        if isinstance(sub_param, Module):
                            load(sub_param, prefix + str(idx) + ".")
        load(self)
        if strict:
            if len(missing_keys) > 0:
                raise KeyError(f"Missing keys in state_dict: {missing_keys}")
            if len(unexpected_keys) > 0:
                raise KeyError(f"Unexpected keys in state_dict: {unexpected_keys}")

    def train(self) -> None:
        """
        Set the module in training mode.
        """
        self.training = True
        for m in self._children():
            m.training = True

    def eval(self) -> None:
        """
        Set the module in evaluation mode.
        """
        self.training = False
        for m in self._children():
            m.training = False

    def to(self, device: str) -> None:
        """
        Move the module to the specified device.
        """
        self.cuda(device)

    def cuda(self, device_name: str = "cuda") -> None:
        """
        Move the module to the specified cuda device.
        """
        for idx in range(len(self.parameters())):
            self.parameters()[idx].set_device(device_name)
        for idx in range(len(self.vars())):
            self.vars()[idx].set_device(device_name)
        for idx in range(len(self._children())):
            self._children()[idx].cuda(device_name)

    def forward(self, *args, **kwargs) -> Tensor:
        """
        Forward pass of the module.
        """
        raise NotImplementedError("forward method not implemented.")

    def __call__(self, *args, **kwargs) -> Tensor:
        """
        Call the module.
        """
        return self.forward(*args, **kwargs)


class Sequential(Module):
    """
    Sequential container for modules.
    """
    def __init__(self, *modules) -> None:
        super().__init__()
        self.modules = modules

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the module.
        """
        for module in self.modules:
            x = module(x)
        return x


class ModuleList(Module):
    def __init__(self, modules: Optional[List[Module]] = None) -> None:
        super().__init__()
        self._modules = []
        if modules is not None:
            self.extend(list(modules))
    
    def append(self, module: Module) -> None:
        """
        Append a module to the module list.
        """
        if not isinstance(module, Module):
            raise ValueError("All elements must be instances of nn.Module")
        self._modules.append(module)

    def extend(self, modules: List[Module]) -> None:
        """
        Extend the module list with a list of modules.
        """
        if not all(isinstance(module, Module) for module in modules):
            raise ValueError("All elements must be instances of nn.Module")
        for module in modules:
            self.append(module) 

    def __getitem__(self, idx: int) -> Module:
        """
        Get a module by index.
        """
        return self._modules[idx] 

    def __len__(self) -> int:
        """
        Get the number of modules in the module list.
        """
        return len(self._modules) 

    def __iter__(self) -> Iterator[Module]:
        """
        Iterate over the modules in the module list.
        """
        return iter(self._modules) 


class Linear(Module):
    """
    Linear layer.
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device: Optional[str] = None,
        dtype: Optional[str] = "float32"
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(
            init.randn(self.out_features, self.in_features, std=0.02),
            device=device, dtype=dtype)

        self.bias = None
        if bias:
            self.bias = Parameter(init.zeros(self.out_features), device=device, dtype=dtype)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the linear layer.
        """
        x = x @ self.weight.transpose(0, 1)
        if self.bias:
            x = x + self.bias
        return x


class Flatten(Module):
    """
    Flatten layer.
    """
    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the flatten layer.
        """
        return x.reshape(x.shape[0], -1)


class ReLU(Module):
    """
    ReLU activation function.
    """
    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the ReLU activation function.
        """
        x = genesis.relu(x)
        return x


class Dropout(Module):
    """
    Dropout layer.
    """
    def __init__(self, p: float = 0.5):
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the dropout layer.
        """
        if self.training and self.p > 0.0:
            mask = init.randb(*x.shape, p=(1 - self.p), dtype=x.dtype, device=x.device)
            x = x * mask / (1 - self.p)
        return x


class Residual(Module):
    """
    Residual connection.
    """
    def __init__(self, fn: Module):
        super().__init__()
        self.fn = fn

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the residual connection.
        """
        return self.fn(x) + x


class BatchNorm1d(Module):
    """
    Batch normalization layer.
    """
    def __init__(
        self,
        dim: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        device: Optional[str] = None,
        dtype: Optional[str] = "float32"
    ) -> None:
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.momentum = momentum

        self.weight = Parameter(init.ones(dim, device=device, dtype=dtype))
        self.bias = Parameter(init.zeros(dim, device=device, dtype=dtype))
        self.running_mean = init.zeros(dim, device=device, dtype=dtype)
        self.running_var = init.ones(dim, device=device, dtype=dtype)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the batch normalization layer.
        """
        if self.training:
            batch = x.shape[0]
            mean = F.summation(x, axis=0) / batch
            self.running_mean = (self.momentum * mean.detach() + (1 - self.momentum) * self.running_mean).detach()
            var = F.summation((x - mean) ** 2, axis=0) / batch
            self.running_var = (self.momentum * var.detach() + (1 - self.momentum) * self.running_var).detach()
        else:
            mean = self.running_mean
            var = self.running_var
        x = (x - mean) / (var + self.eps) ** 0.5
        x = self.weight * x + self.bias
        return x


class LayerNorm(Module):
    """
    Layer normalization layer.
    """
    def __init__(
        self,
        dim: int,
        eps: float = 1e-5,
        device: Optional[str] = None,
        dtype: Optional[str] = "float32"
    ) -> None:
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = Parameter(init.ones(dim, device=device, dtype=dtype))
        self.bias = Parameter(init.zeros(dim, device=device, dtype=dtype))

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the layer normalization layer.
        """
        if x.shape[-1] != self.dim:
            raise RuntimeError("Input dims should be %d" % self.dim)
        mean = F.summation(x, axis=-1, keepdims=True) / x.shape[-1]
        var = F.summation((x - mean) ** 2, axis=-1, keepdims=True) / self.dim
        output = (x - mean) / F.sqrt(var + self.eps)
        output = self.weight * output + self.bias
        return output


class FusedLayerNorm(Module):
    """
    Fused layer normalization layer.
    """
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.dim = dim
        self.weight = Parameter(init.ones(dim))
        self.bias = Parameter(init.zeros(dim))

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the fused layer normalization layer.
        """
        return F.fused_layer_norm(x, self.weight, self.bias, self.eps)


class RMSNorm(Module):
    """
    RMS normalization layer.
    """
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.dim = dim
        self.weight = Parameter(init.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the RMS normalization layer.
        """
        x_square = x ** 2
        x_mean = F.summation(x_square, axis=-1, keepdims=True) / x_square.shape[-1]
        rms = x / F.sqrt(x_mean + self.eps)
        return rms * self.weight


class SoftmaxLoss(Module):
    """
    Softmax loss.
    """
    def forward(self, logits: Tensor, y: Tensor) -> Tensor:
        """
        Forward pass of the softmax loss.
        """
        num, classes = logits.shape
        mask = (y != -1)
        valid_logits = logits[mask] 
        valid_y = y[mask]

        y_one_hot = init.one_hot(classes, valid_y, dtype=logits.dtype, device=logits.device)
        logsum = F.logsumexp(valid_logits, axis=(1,))
        logits_y = F.summation(valid_logits * y_one_hot, axis=(1,))
        loss = logsum - logits_y
        return F.summation(loss) / valid_logits.shape[0]


class Softmax(Module):
    """
    Softmax activation function.
    """
    def __init__(self, dim=-1):
        super().__init__()
        self.dim = dim

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the softmax activation function.
        """
        if x.device == genesis.cpu() or genesis.use_triton is False:
            x_exp = F.exp(x - F.max(x, self.dim, keepdims=True))
            x = x_exp / F.summation(x_exp, axis=self.dim, keepdims=True)
            return x
        else:
            return F.softmax(x, dim=self.dim)


class Embedding(Module):
    """
    Embedding layer.
    """
    def __init__(
        self, 
        num_embeddings, 
        embedding_dim
    ):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = Parameter(init.randn(num_embeddings, embedding_dim, std=0.02))

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the embedding layer.
        """
        x_one_hot = init.one_hot(self.num_embeddings, x.data.flat, device=x.device)
        res = x_one_hot @ self.weight
        return res.reshape((*x.shape, self.embedding_dim))

    
class RotaryEmbedding(Module):
    """
    Rotary embedding layer.
    """
    def __init__(
        self, 
        dim, 
        max_position_embeddings: int = 2048, 
        base: int = 10000
    ):
        super().__init__()
        self.inv_freq = genesis.Tensor(1.0 / (base ** (np.arange(0, dim, 2).astype(np.float32) / dim)))
        self.max_seq_len_cached = max_position_embeddings
        t = genesis.Tensor(np.arange(self.max_seq_len_cached, dtype="float32"))
        t = t.reshape(t.shape[0], 1)
        self.inv_freq = self.inv_freq.reshape(1, self.inv_freq.shape[0])
        freqs = t @ self.inv_freq
        emb = F.stack((freqs, freqs), dim=-1).transpose().reshape(freqs.shape[0], freqs.shape[1] * 2)
        self.cos_cached = emb.cos().reshape((1, 1) + (emb.shape))
        self.sin_cached = emb.sin().reshape((1, 1) + (emb.shape))

    def forward(self, x: Tensor, seq_len: Optional[int] = None) -> Tuple[Tensor, Tensor]:
        """
        Forward pass of the rotary embedding layer.
        """
        return (
            self.cos_cached[:, :, :seq_len, :],
            self.sin_cached[:, :, :seq_len, :],
        )


class SiLU(Module):
    """
    SiLU activation function.
    """
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the SiLU activation function.
        """
        return x / (F.exp(-x) + 1)


class FeedFowardSwiGLU(Module):
    """ 
    SwiGLU: https://arxiv.org/pdf/2002.05202.pdf
    """
    def __init__(
        self, 
        dim: int, 
        hidden_dim: int
    ):
        super().__init__()
        self.gate = Linear(dim, hidden_dim, bias=False)
        self.down = Linear(hidden_dim, dim, bias=False)
        self.up = Linear(dim, hidden_dim, bias=False)
        self.act = SiLU()
        self.dropout = Dropout(0.1)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the feed forward SwiGLU layer.
        """
        out = self.down(self.act(self.gate(x)) * self.up(x))
        return self.dropout(out)


class MultiheadAttention(Module):
    """
    Multihead attention layer.
    """
    def __init__(
        self, 
        dim: int = 64, 
        heads: int = 1, 
        device: Optional[str] = None, 
        dtype: Optional[str] = "float32"
    ):
        self.dim = dim
        self.heads = heads
        self.w_qkv = Parameter(
            init.kaiming_uniform(self.dim, self.dim * 3),
            device=device, dtype=dtype)
        self.w_out = Parameter(
            init.kaiming_uniform(self.dim, self.dim),
            device=device, dtype=dtype)
        self.softmax = Softmax()

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the multihead attention layer.
        """
        q, k, v = F.split((x @ self.w_qkv).reshape(x.shape[0], x.shape[1], 3, self.dim), axis=2)
        q, k, v = [a.reshape(x.shape[0], x.shape[1], self.heads, self.dim // self.heads).transpose((1, 2)) for a in [q, k, v]]
        mask = genesis.triu((-float("inf") * init.ones(x.shape[1], x.shape[1], device=x.device)), k=1, device=x.device)
        atten = self.softmax(q @ F.transpose(k) / np.sqrt(self.dim // self.heads) + mask)
        return (atten @ v).transpose((1, 2)).reshape(x.shape[0], x.shape[1], self.dim) @ self.w_out, atten


class FusedMultiheadAttention(Module):
    """
    Fused multihead attention layer.
    """
    def __init__(
        self, 
        dim: int = 64, 
        heads: int = 1, 
        device: Optional[str] = None, 
        dtype: Optional[str] = "float32"
    ):
        self.dim = dim
        self.heads = heads
        self.w_qkv = Parameter(
            init.kaiming_uniform(self.dim, self.dim * 3),
            device=device, dtype=dtype)
        self.w_out = Parameter(
            init.kaiming_uniform(self.dim, self.dim),
            device=device, dtype=dtype)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the fused multihead attention layer.
        """
        q, k, v = F.split((x @ self.w_qkv).reshape(x.shape[0], x.shape[1], 3, self.dim), axis=2)
        q, k, v = [a.reshape(x.shape[0], x.shape[1], self.heads, self.dim // self.heads).transpose((1, 2)) for a in [q, k, v]]
        return F.fused_attention(q, k, v).transpose((1, 2)).reshape(x.shape[0], x.shape[1], self.dim) @ self.w_out, None
