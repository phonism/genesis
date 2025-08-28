"""Container modules for organizing neural network components."""

from typing import List, Optional, Iterator, Tuple
from ...autograd import Tensor
from .module import Module


class Sequential(Module):
    """
    Sequential container for modules.
    """
    def __init__(self, *modules) -> None:
        super().__init__()
        # Handle both Sequential(mod1, mod2) and Sequential([mod1, mod2]) patterns
        if len(modules) == 1 and isinstance(modules[0], (list, tuple)):
            self._modules_list = modules[0]
        else:
            self._modules_list = modules
        # Register modules as attributes for proper parameter discovery
        for i, module in enumerate(self._modules_list):
            setattr(self, f'_{i}', module)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the module.
        """
        for module in self._modules_list:
            x = module(x)
        return x


class ModuleList(Module):
    """A list container for neural network modules.
    
    Holds modules in a list and allows for easy iteration and access.
    Parameters of child modules are automatically registered.
    
    Args:
        modules: Optional list of modules to initialize with
    """
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
    
    def named_parameters(self, prefix: str = "", recurse: bool = True) -> Iterator[Tuple[str, Tensor]]:
        """
        Return an iterator over module parameters with their names.
        For ModuleList, use numeric indices directly without '_modules' prefix.
        """
        if recurse:
            for idx, module in enumerate(self._modules):
                for name, param in module.named_parameters(prefix + str(idx) + ".", recurse):
                    yield name, param