"""Neural network module base classes and parameter management.

This module provides the foundational Module class and Parameter class that serve
as building blocks for all neural network layers in Genesis.
"""

from typing import (
    List, Callable, Any, Optional, Dict, Iterator, Tuple
)
import genesis
from genesis.tensor import Tensor
import numpy as np


class Parameter(Tensor):
    """A tensor subclass that represents trainable parameters.

    Parameters are automatically tracked for gradient computation and
    optimization during training.
    """

    def __new__(cls, data, requires_grad=True):
        """Create a Parameter from a Tensor using proper initialization."""
        # Always convert to tensor first if needed
        if not isinstance(data, Tensor):
            data = genesis.tensor(data, requires_grad=requires_grad)

        # Use Tensor.__new__ to create the instance properly
        instance = Tensor.__new__(cls)
        # Store the tensor data for __init__ to use
        instance._init_data = data
        instance._init_requires_grad = requires_grad
        return instance

    def __init__(self, data, requires_grad=True):
        """Initialize the Parameter from a Tensor."""
        # Use the stored data from __new__
        if hasattr(self, '_init_data'):
            data = self._init_data
            requires_grad = self._init_requires_grad
            delattr(self, '_init_data')
            delattr(self, '_init_requires_grad')

        # Now data is guaranteed to be a Tensor
        if isinstance(data, Tensor):
            # Initialize using parent class with tensor's attributes
            super().__init__(data.storage, data.shape, data.stride, data.offset)
            self.requires_grad = requires_grad
            if self.requires_grad:
                self.grad = None
        else:
            # This should never happen now
            raise ValueError("Parameter expects a Tensor")

    def to(self, device):
        """Move parameter to device and return a Parameter (not Tensor)."""
        # Use parent Tensor.to() to get moved tensor
        moved_tensor = super().to(device)
        # Create new Parameter from moved tensor
        return Parameter(moved_tensor, requires_grad=self.requires_grad)


def _unpack_params(value: object) -> List[Tensor]:
    """Recursively extract parameters from nested structures."""
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
    """Recursively extract all tensors from nested structures."""
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
    """Recursively extract child modules from nested structures."""
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
    """Base class for all neural network modules.
    
    Provides parameter management, training mode control, and device placement
    functionality for building complex neural network architectures.
    """
    
    def __init__(self):
        self.training = True

    def register_buffer(self, name: str, buffer: Tensor, persistent: bool = True):
        """Register a buffer (non-parameter tensor) to the module.

        Buffers are tensors that should be saved/loaded with the model but do not require gradients.

        Args:
            name: Name of the buffer
            buffer: Tensor to register as buffer
            persistent: Whether buffer persists during serialization
        """
        # Buffers should not require gradients by design
        if buffer is not None and hasattr(buffer, 'requires_grad'):
            buffer.requires_grad = False
        self.__dict__[name] = buffer

    def parameters(self) -> List[Tensor]:
        """Return iterator over module parameters."""
        return _unpack_params(self.__dict__)
    
    def num_parameters(self) -> int:
        """
        Return the number of parameters in the module.

        Note: This method correctly handles weight tying by deduplicating
        parameters that share the same underlying tensor.
        """
        seen = set()
        num_parameters = 0
        for p in self.parameters():
            # Use id() to check if we've already counted this parameter
            param_id = id(p.data if hasattr(p, 'data') else p)
            if param_id not in seen:
                seen.add(param_id)
                cur = 1
                for x in p.shape:
                    cur *= x
                num_parameters += cur
        return num_parameters

    def named_parameters(self, prefix: str = "", recurse: bool = True) -> Iterator[Tuple[str, Tensor]]:
        """
        Return an iterator over module parameters with their names.
        
        Args:
            prefix: prefix to prepend to all parameter names
            recurse: whether to include parameters of submodules
            
        Yields:
            (string, Parameter): Tuple containing name and parameter
        """
        for name, param in self.__dict__.items():
            if isinstance(param, Parameter):
                yield prefix + name, param
            elif recurse and isinstance(param, Module):
                for sub_name, sub_param in param.named_parameters(prefix + name + ".", recurse):
                    yield sub_name, sub_param
            elif recurse and isinstance(param, (list, tuple)):
                for idx, v in enumerate(param):
                    if isinstance(v, Module):
                        for sub_name, sub_param in v.named_parameters(prefix + name + "." + str(idx) + ".", recurse):
                            yield sub_name, sub_param

    def buffers(self) -> Iterator[Tensor]:
        """Return iterator over module buffers."""
        for name, value in self.__dict__.items():
            if isinstance(value, Tensor) and not isinstance(value, Parameter):
                yield value
            elif isinstance(value, Module):
                for buffer in value.buffers():
                    yield buffer
                    
    def named_buffers(self, prefix: str = "", recurse: bool = True) -> Iterator[Tuple[str, Tensor]]:
        """
        Return an iterator over module buffers with their names.
        
        Args:
            prefix: prefix to prepend to all buffer names
            
        Yields:
            (string, Tensor): Tuple containing name and buffer
        """
        for name, value in self.__dict__.items():
            if isinstance(value, Tensor) and not isinstance(value, Parameter):
                yield prefix + name, value
            elif recurse and isinstance(value, Module):
                for sub_name, sub_buffer in value.named_buffers(prefix + name + ".", recurse):
                    yield sub_name, sub_buffer

    def vars(self) -> List[Tensor]:
        """
        Return the list of variables in the module.
        """
        return _unpack_vars(self.__dict__)

    def modules(self) -> Iterator["Module"]:
        """Return iterator over all modules in this module."""
        yield self
        for name, value in self.__dict__.items():
            if isinstance(value, Module):
                for module in value.modules():
                    yield module
                    
    def named_modules(self, memo=None, prefix="") -> Iterator[Tuple[str, "Module"]]:
        """
        Return an iterator over all modules with their names.
        
        Args:
            memo: a memo to store the set of modules already added to the result
            prefix: a prefix that will be added to the name of the module
            
        Yields:
            (string, Module): Tuple containing name and module
        """
        if memo is None:
            memo = set()
        if self not in memo:
            memo.add(self)
            yield prefix, self
            for name, module in self.__dict__.items():
                if isinstance(module, Module):
                    submodule_prefix = prefix + ('.' if prefix else '') + name
                    for m in module.named_modules(memo, submodule_prefix):
                        yield m

    def _children(self) -> List["Module"]:
        """
        Return the list of child modules in the module.
        """
        return _child_modules(self.__dict__)

    def state_dict(self, destination=None, prefix="") -> Dict[str, Tensor]:
        """
        Return the state dictionary of the module.
        """
        if destination is None:
            destination = {}
        state_dict = destination
        for name, param in self.__dict__.items():
            if isinstance(param, genesis.Tensor):
                # Use tensor directly, not internal data attributes
                state_dict[prefix + name] = param
            elif isinstance(param, Module):
                param.state_dict(state_dict, prefix + name + ".")
            elif isinstance(param, (list, tuple)):
                for idx, v in enumerate(param):
                    if isinstance(v, Module):
                        v.state_dict(state_dict, prefix + str(idx) + ".")
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
                        # Preserve Parameter type when loading state dict
                        source_tensor = state_dict[full_name]
                        if isinstance(param, Parameter):
                            # If original was a Parameter, create a new Parameter with the loaded data
                            new_param = Parameter(source_tensor, requires_grad=param.requires_grad)
                            module.__dict__[name] = new_param
                        else:
                            # If original was just a Tensor, replace directly
                            module.__dict__[name] = source_tensor
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

    def train(self, mode: bool = True) -> "Module":
        """
        Set the module in training mode.
        
        Args:
            mode: If True, sets to training mode. If False, sets to evaluation mode.
            
        Returns:
            Self for method chaining
        """
        self.training = mode
        for m in self._children():
            m.train(mode)
        return self

    def eval(self) -> "Module":
        """
        Set the module in evaluation mode.
        
        Returns:
            Self for method chaining
        """
        return self.train(False)

    def to(self, device) -> "Module":
        """
        Move the module to the specified device.
        
        Args:
            device: Device object or device string
            
        Returns:
            Self for method chaining
        """
        # Move parameters and buffers in this module (not recursively)
        for name, value in self.__dict__.items():
            if isinstance(value, Parameter):
                self.__dict__[name] = value.to(device)
            elif isinstance(value, Tensor) and not isinstance(value, Parameter):
                self.__dict__[name] = value.to(device)
        
        # Recursively move child modules
        for child in self._children():
            child.to(device)
        
        return self

    def cuda(self, device_name: str = "cuda") -> None:
        """
        Move the module to the specified cuda device.
        """
        # Move all parameters and buffers to CUDA
        # We need to replace the parameter objects, not modify them in-place
        for name, param in self.named_parameters():
            # Find the parent module and attribute name
            *parent_names, attr_name = name.split('.')
            parent_module = self
            for parent_name in parent_names:
                # Handle ModuleList case where we need to use indexing instead of getattr
                if hasattr(parent_module, '_modules') and parent_name.isdigit():
                    # This is a ModuleList with numeric index access
                    parent_module = parent_module[int(parent_name)]
                else:
                    parent_module = getattr(parent_module, parent_name)

            # Move parameter to device and replace
            moved_param = param.to(device_name)
            if isinstance(param, Parameter):
                moved_param = Parameter(moved_param, requires_grad=param.requires_grad)
            setattr(parent_module, attr_name, moved_param)

        # Also move buffers registered with register_buffer
        for name, value in self.__dict__.items():
            if isinstance(value, Tensor) and not isinstance(value, Parameter):
                # This is likely a buffer - move it to CUDA
                # Use detach to break the computation graph for buffers
                if hasattr(value, 'detach'):
                    self.__dict__[name] = value.detach().to(device_name)
                else:
                    self.__dict__[name] = value.to(device_name)

        for child_module in self._children():
            child_module.cuda(device_name)
        return self

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