"""
The module
"""

from typing import List, Callable, Any
import genesis
from ..autograd import Tensor
import genesis.nn.functional as F
from genesis import init
import numpy as np

class Parameter(Tensor):
    """A special kind of tensor that represents parameters."""

def _unpack_params(value: object) -> List[Tensor]:
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
    def __init__(self):
        self.training = True

    def parameters(self) -> List[Tensor]:
        """Return the list of parameters in the module."""
        return _unpack_params(self.__dict__)
    
    def num_parameters(self):
        num_parameters = 0
        for p in self.parameters():
            cur = 1
            for x in p.shape:
                cur *= x
            num_parameters += cur
        return num_parameters

    def vars(self) -> List[Tensor]:
        """Return the list of parameters in the module."""
        return _unpack_vars(self.__dict__)

    def _children(self) -> List["Module"]:
        return _child_modules(self.__dict__)

    def state_dict(self, prefix=""):
        state_dict = {}
        for name, param in self.__dict__.items():
            if isinstance(param, genesis.Tensor):
                state_dict[prefix + name] = param
            elif isinstance(param, Module):
                state_dict.update(param.state_dict(prefix + name + "."))
            elif isinstance(param, (list, tuple)):
                for idx, v in enumerate(param):
                    state_dict.update(v.state_dict(prefix + name + "." + str(idx) + "."))
        return state_dict

    def load_state_dict(self, state_dict, strict=True):
        missing_keys = []
        unexpected_keys = list(state_dict.keys())

        def load(module, prefix=""):
            for name, param in module.__dict__.items():
                full_name = prefix + name

                # 如果是一个 Tensor，则直接从 state_dict 中加载
                if isinstance(param, genesis.Tensor):
                    if full_name in state_dict:
                        param.copy_(state_dict[full_name])
                        unexpected_keys.remove(full_name)
                    elif strict:
                        missing_keys.append(full_name)
                elif isinstance(param, Module):
                    load(param, full_name + ".")
                elif isinstance(param, (list, tuple)):
                    for idx, sub_param in enumerate(param):
                        if isinstance(sub_param, Module):
                            load(sub_param, full_name + "." + str(idx) + ".")

        load(self)

        if strict:
            if len(missing_keys) > 0:
                raise KeyError(f"Missing keys in state_dict: {missing_keys}")
            if len(unexpected_keys) > 0:
                raise KeyError(f"Unexpected keys in state_dict: {unexpected_keys}")


    def train(self):
        self.training = True
        for m in self._children():
            m.traning = True

    def eval(self):
        self.training = False
        for m in self._children():
            m.training = False

    def cuda(self):
        for idx in range(len(self.parameters())):
            self.parameters()[idx].set_device()
        for idx in range(len(self.vars())):
            self.vars()[idx].set_device()
        for idx in range(len(self._children())):
            self._children()[idx].cuda()

    def forward(self, *args, **kwargs):
        raise NotImplementedError("forward method not implemented.")

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class Sequential(Module):
    def __init__(self, *modules):
        super().__init__()
        self.modules = modules

    def forward(self, x: Tensor) -> Tensor:
        for module in self.modules:
            x = module(x)
        return x


class Linear(Module):
    def __init__(self, in_features, out_features, bias=True, device=None, dtype="float32"):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(
                init.kaiming_uniform(self.in_features, self.out_features),
                device=device, dtype=dtype)
        self.bias = None
        if bias:
            self.bias = Parameter(
                    F.transpose(init.kaiming_uniform(self.out_features, 1)),
                    device=device, dtype=dtype)

    def forward(self, x: Tensor) -> Tensor:
        x = F.matmul(x, self.weight)
        if self.bias:
            x = x + F.broadcast_to(F.reshape(self.bias, (1,) * (len(x.shape) - 1) + (self.out_features,)), x.shape)
        return x


class Flatten(Module):
    def forward(self, x) -> Tensor:
        return F.reshape(x, (x.shape[0], -1))


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        x = F.relu(x)
        return x


class Dropout(Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        if self.training and self.p > 0.0:
            mask = init.randb(*x.shape, p=(1 - self.p), dtype=x.dtype, device=x.device)
            x = x * mask / (1 - self.p)
        return x

class Residual(Module):
    def __init__(self, fn: Module):
        super().__init__()
        self.fn = fn

    def forward(self, x: Tensor) -> Tensor:
        return self.fn(x) + x

class BatchNorm1d(Module):
    def __init__(self, dim: int, eps=1e-5, momentum=0.1, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.momentum = momentum

        self.weight = Parameter(init.ones(dim, device=device, dtype=dtype))
        self.bias = Parameter(init.zeros(dim, device=device, dtype=dtype))
        self.running_mean = init.zeros(dim, device=device, dtype=dtype)
        self.running_var = init.ones(dim, device=device, dtype=dtype)

    def forward(self, x: Tensor) -> Tensor:
        if self.training:
            batch = x.shape[0]
            mean = F.summation(x, axis=0) / batch
            self.running_mean = (self.momentum * mean.detach() + (1 - self.momentum) * self.running_mean).detach()
            mean = F.broadcast_to(F.reshape(mean, (1, self.dim)), x.shape)
            var = F.summation((x - mean) ** 2, axis=0) / batch
            self.running_var = (self.momentum * var.detach() + (1 - self.momentum) * self.running_var).detach()
            var = F.broadcast_to(F.reshape(var, (1, self.dim)), x.shape)
        else:
            mean = self.running_mean.reshape((1, self.dim)).broadcast_to(x.shape)
            var = self.running_var.reshape((1, self.dim)).broadcast_to(x.shape)
        x = (x - mean) / (var + self.eps) ** 0.5
        w = F.broadcast_to(F.reshape(self.weight, (1, self.dim)), x.shape)
        b = F.broadcast_to(F.reshape(self.bias, (1, self.dim)), x.shape)
        x = w * x + b
        return x

class LayerNorm(Module):
    def __init__(self, dim, eps=1e-5, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = Parameter(init.ones(dim, device=device, dtype=dtype))
        self.bias = Parameter(init.zeros(dim, device=device, dtype=dtype))

    def forward(self, x):
        if x.shape[-1] != self.dim:
            raise RuntimeError("Input dims should be %d" % self.dim)
        mean = F.summation(x, axis=-1) / x.shape[-1]
        mean = F.broadcast_to(F.reshape(mean, mean.shape + (1,)), x.shape)
        var = F.summation((x - mean) ** 2, axis=-1) / self.dim
        var = F.broadcast_to(F.reshape(var, var.shape + (1,)), x.shape)
        weight = F.broadcast_to(F.reshape(self.weight, (1, ) * (len(x.shape) - 1) + (self.dim,)), x.shape)
        bias = F.broadcast_to(F.reshape(self.bias, (1, ) * (len(x.shape) - 1) + (self.dim,)), x.shape)
        output = (x - mean) / F.sqrt(var + self.eps)
        output = weight * output + bias
        return output

class FusedLayerNorm(Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.dim = dim
        self.weight = Parameter(init.ones(dim))
        self.bias = Parameter(init.zeros(dim))

    def forward(self, x):
        return F.fused_layer_norm(x, self.weight, self.bias, self.eps)

class RMSNorm(Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.dim = dim
        self.weight = Parameter(init.ones(dim))

    def forward(self, x):
        x_square = x ** 2
        x_mean = F.summation(x_square, axis=-1) / x_square.shape[-1]
        x_mean = F.broadcast_to(F.reshape(x_mean, x_mean.shape + (1,)), x.shape)
        rms = x / F.sqrt(x_mean + self.eps)
        weight = F.broadcast_to(F.reshape(self.weight, (1, ) * (len(x.shape) - 1) + (self.dim,)), x.shape)
        return rms * weight

class SoftmaxLoss(Module):
    def forward(self, logits: Tensor, y: Tensor) -> Tensor:
        num, classes = logits.shape
        y_one_hot = init.one_hot(classes, y, dtype=logits.dtype, device=logits.device)
        logsum = F.logsumexp(logits, axis=(1,))
        logits_y = F.summation(logits * y_one_hot, axis=(1,))
        loss = logsum - logits_y
        return F.summation(loss) / logits.shape[0]

class Softmax(Module):
    def __init__(self, dim=-1):
        super().__init__()
        self.dim = dim

    def forward(self, x: Tensor) -> Tensor:
        x_exp = F.exp(x - F.broadcast_to(F.max(x, self.dim, keepdims=True), x.shape))
        x = x_exp / F.broadcast_to(F.summation(x_exp, axis=self.dim, keepdims=True), x.shape)
        return x

class Embedding(Module):
    def __init__(self, num_embeddings, embedding_dim):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = Parameter(init.randn(num_embeddings, embedding_dim))

    def forward(self, x):
        x_one_hot = init.one_hot(self.num_embeddings, x.data.flat, device=x.device)
        res = x_one_hot @ self.weight
        return res.reshape((*x.shape, self.embedding_dim))
    
class RotaryEmbedding(Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000):
        super().__init__()
        self.inv_freq = genesis.Tensor(1.0 / (base ** (np.arange(0, dim, 2).astype(np.float32) / dim)))

        self.max_seq_len_cached = max_position_embeddings
        t = genesis.Tensor(np.arange(self.max_seq_len_cached, dtype="float32"))
        t = F.reshape(t, (t.shape[0], 1))
        self.inv_freq = F.reshape(self.inv_freq, (1, self.inv_freq.shape[0]))
        freqs = t @ self.inv_freq
        emb = F.reshape(F.stack((freqs, freqs), dim=-1).transpose(), (freqs.shape[0], freqs.shape[1] * 2))
        self.cos_cached = F.reshape(emb.cos(), (1, 1) + (emb.shape))
        self.sin_cached = F.reshape(emb.sin(), (1, 1) + (emb.shape))

    def forward(self, x, seq_len=None):
        return (
            self.cos_cached[:, :, :seq_len, :],
            self.sin_cached[:, :, :seq_len, :],
        )

class SiLU(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x / (F.exp(-x) + 1)

class FeedFowardSwiGLU(Module):
    """ 
    SwiGLU: https://arxiv.org/pdf/2002.05202.pdf
    """
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.gate = Linear(dim, hidden_dim, bias=False)
        self.down = Linear(hidden_dim, dim, bias=False)
        self.up = Linear(dim, hidden_dim, bias=False)
        self.act = SiLU()
        self.dropout = Dropout(0.1)

    def forward(self, x):
        out = self.down(self.act(self.gate(x)) * self.up(x))
        return self.dropout(out)


class MultiheadAttention(Module):
    def __init__(self, dim=64, heads=1, device=None, dtype="float32"):
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
        q, k, v = F.split(F.reshape(x @ self.w_qkv, (x.shape[0], x.shape[1], 3, self.dim)), axis=2)
        q, k, v = [F.reshape(a, (x.shape[0], x.shape[1], self.heads, self.dim // self.heads)).transpose((1, 2)) for a in [q, k, v]]
        mask = genesis.triu((-float("inf") * init.ones(x.shape[1], x.shape[1], device=x.device)), k=1, device=x.device)
        mask = F.broadcast_to(F.reshape(mask, (1, 1,) + mask.shape), (k.shape[0], k.shape[1],) + mask.shape)
        atten = self.softmax(q @ F.transpose(k) / np.sqrt(self.dim // self.heads) + mask)
        return F.reshape((atten @ v).transpose((1, 2)), (x.shape[0], x.shape[1], self.dim)) @ self.w_out, atten

class FusedMultiheadAttention(Module):
    def __init__(self, dim=64, heads=1, device=None, dtype="float32"):
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
        q, k, v = F.split(F.reshape(x @ self.w_qkv, (x.shape[0], x.shape[1], 3, self.dim)), axis=2)
        q, k, v = [F.reshape(a, (x.shape[0], x.shape[1], self.heads, self.dim // self.heads)).transpose((1, 2)) for a in [q, k, v]]
        return F.reshape(F.fused_attention(q, k, v).transpose((1, 2)), (x.shape[0], x.shape[1], self.dim)) @ self.w_out, None
