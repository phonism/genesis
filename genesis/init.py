"""
init
"""

import genesis
import math

def rand(*shape, low=0.0, high=1.0, device=None, dtype=genesis.float32, requires_grad=False):
    """ Generate random numbers uniform between low and high """
    device = genesis.device('cpu') if device is None else device
    array = device.rand(*shape) * (high - low) + low
    return genesis.Tensor(array, device=device, dtype=dtype, requires_grad=requires_grad)

def randn(*shape, mean=0.0, std=1.0, device=None, dtype=genesis.float32, requires_grad=False):
    """ Generate random normal with specified mean and std deviation """
    device = genesis.device('cpu') if device is None else device
    array = device.randn(*shape) * std + mean
    return genesis.Tensor(array, device=device, dtype=dtype, requires_grad=requires_grad)

def constant(*shape, c=1.0, device=None, dtype=genesis.float32, requires_grad=False):
    """ Generate constant Tensor """
    device = genesis.device('cpu') if device is None else device
    array = device.full(shape, c, dtype=dtype)  # Remove duplicate multiplication
    return genesis.Tensor(array, device=device, dtype=dtype, requires_grad=requires_grad)

def ones(*shape, device=None, dtype=genesis.float32, requires_grad=False):
    """ Generate all-ones Tensor """
    return constant(*shape, c=1.0, device=device, dtype=dtype, requires_grad=requires_grad)

def zeros(*shape, device=None, dtype=genesis.float32, requires_grad=False):
    """ Generate all-zeros Tensor """
    return constant(*shape, c=0.0, device=device, dtype=dtype, requires_grad=requires_grad)

def full(shape, fill_value, device=None, dtype=genesis.float32, requires_grad=False):
    """ Generate Tensor filled with specified value """
    if isinstance(shape, (int,)):
        shape = (shape,)
    elif not isinstance(shape, tuple):
        shape = tuple(shape)
    return constant(*shape, c=fill_value, device=device, dtype=dtype, requires_grad=requires_grad)

def empty(*shape, device=None, dtype=genesis.float32, requires_grad=False):
    """ Generate empty Tensor (uninitialized data) """
    device = genesis.device('cpu') if device is None else device
    array = device.empty(shape, dtype=dtype)
    return genesis.Tensor(array, device=device, dtype=dtype, requires_grad=requires_grad)

def empty_like(tensor, dtype=None, device=None, requires_grad=None):
    """ Generate empty Tensor with same shape as input tensor """
    if dtype is None:
        dtype = tensor.dtype
    if device is None:
        device = tensor.device
    if requires_grad is None:
        requires_grad = tensor.requires_grad  # Inherit requires_grad from input tensor
    
    # Use efficient empty instead of zeros
    array = device.empty(tensor.shape, dtype=dtype)
    return genesis.Tensor(array, device=device, dtype=dtype, requires_grad=requires_grad)

def zeros_like(tensor, dtype=None, device=None, requires_grad=None):
    """ Generate zeros Tensor with same shape as input tensor """
    if dtype is None:
        dtype = tensor.dtype
    if device is None:
        device = tensor.device
    if requires_grad is None:
        requires_grad = tensor.requires_grad  # Inherit requires_grad from input tensor
    
    return zeros(*tensor.shape, device=device, dtype=dtype, requires_grad=requires_grad)

def randb(*shape, p=0.5, device=None, dtype="bool", requires_grad=False):
    """ Generate binary random Tensor """
    device = genesis.device('cpu') if device is None else device
    array = device.rand(*shape) <= p
    return genesis.Tensor(array, device=device, dtype=dtype, requires_grad=requires_grad)

def one_hot(n, i, device=None, dtype=genesis.float32, requires_grad=False):
    """ Generate one-hot encoding Tensor """
    device = genesis.device('cpu') if device is None else device
    # i should be an NDArray, pass it directly to device.one_hot
    return genesis.Tensor(device.one_hot(n, i, dtype=genesis.float32), device=device, requires_grad=requires_grad)

def xavier_uniform(fan_in, fan_out, gain=1.0, **kwargs):
    a = gain * math.sqrt(6 / (fan_in + fan_out))
    return rand(fan_in, fan_out, low=-a, high=a, **kwargs)

def xavier_normal(fan_in, fan_out, gain=1.0, **kwargs):
    std = gain * math.sqrt(2 / (fan_in + fan_out))
    return randn(fan_in, fan_out, mean=0, std=std, **kwargs)

def kaiming_uniform(fan_in, fan_out, nonlinearity="relu", **kwargs):
    assert nonlinearity == "relu", "Only relu supported currently"
    gain = math.sqrt(2)
    bound = gain * math.sqrt(3 / fan_in)
    return rand(fan_in, fan_out, low=-bound, high=bound, **kwargs)

def kaiming_normal(fan_in, fan_out, nonlinearity="relu", **kwargs):
    assert nonlinearity == "relu", "Only relu supported currently"
    gain = math.sqrt(2)
    std = gain / math.sqrt(fan_in)
    return randn(fan_in, fan_out, mean=0, std=std, **kwargs)
