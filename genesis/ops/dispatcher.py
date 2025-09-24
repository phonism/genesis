"""
PyTorch-style dispatcher for routing operations to device-specific implementations.
"""

from typing import Callable, Dict, Tuple, Any, Optional
from genesis.device import DeviceType
from genesis.storage import Storage
from genesis.tensor import Tensor


class OperationDispatcher:
    """Central dispatcher for routing operations to device implementations."""
    
    _registry: Dict[Tuple[str, DeviceType], Callable] = {}
    
    @classmethod
    def register(cls, op_name: str, device_type: DeviceType, impl_func: Callable):
        """Register a device-specific implementation.
        
        Args:
            op_name: Name of the operation
            device_type: Target device type
            impl_func: Implementation function
        """
        key = (op_name, device_type)
        cls._registry[key] = impl_func
    
    @classmethod
    def dispatch(cls, op_name: str, *tensors, **kwargs) -> Any:
        """Dispatch operation to device implementation.
        
        Args:
            op_name: Name of the operation
            *tensors: Input tensor(s) - device inferred from first tensor
            **kwargs: Operation keyword arguments
            
        Returns:
            Result of the operation (storage object)
            
        Raises:
            NotImplementedError: If no implementation found
        """
        if not tensors:
            raise ValueError("At least one tensor is required for dispatch")
        
        # Get device - handle case where first arg is a list of tensors
        if isinstance(tensors[0], (list, tuple)):
            if not tensors[0]:
                raise ValueError("Empty tensor list in dispatch")
            device = tensors[0][0].device
        else:
            device = tensors[0].device
        key = (op_name, device.type)
        if key not in cls._registry:
            raise NotImplementedError(
                f"Operation '{op_name}' not implemented for device '{device.type}'"
            )

        # Extract storage backends for tensor arguments, keep scalars as-is
        mixed_args = []
        for arg in tensors:
            if isinstance(arg, Tensor):
                # It's a tensor
                mixed_args.append(arg.storage._backend)
            elif isinstance(arg, (list, tuple)):
                # Check if it's a list/tuple of tensors or other data
                if arg and isinstance(arg[0], Tensor):
                    # It's a list/tuple of tensors
                    tensor_backends = [t.storage._backend for t in arg]
                    mixed_args.append(tensor_backends)
                else:
                    # It's a list/tuple of scalars or other data
                    mixed_args.append(arg)
            else:
                # It's a scalar
                mixed_args.append(arg)
        
        # Call kernel with mixed storage objects and scalars
        impl_func = cls._registry[key]
        result_storage = impl_func(*mixed_args, **kwargs)
        
        # Wrap result storage back into tensor (PyTorch pattern)
        storage_wrapper = Storage(result_storage, device)
        # Use result storage shape, not input shape (for reductions etc)
        # Ensure shape is a tuple, not torch.Size
        result_shape = tuple(result_storage.shape)
        return Tensor(storage_wrapper, result_shape)

    @classmethod
    def dispatch_inplace(cls, op_name: str, target_tensor, *args, **kwargs):
        """
        Dispatch in-place operation that modifies target_tensor directly.

        Args:
            op_name: Name of the operation (should end with '_inplace')
            target_tensor: Tensor to modify in-place
            *args: Additional arguments
            **kwargs: Operation keyword arguments

        Returns:
            target_tensor (same object, modified in-place)
        """
        # Ensure operation name is for in-place variant
        if not op_name.endswith('_inplace'):
            op_name = f"{op_name}_inplace"

        # Get device from target tensor
        device = target_tensor.device
        key = (op_name, device.type)

        if key not in cls._registry:
            raise NotImplementedError(f"Operation {op_name} not implemented for {device.type}")

        # Prepare arguments: target tensor backend + other arguments
        mixed_args = [target_tensor.storage._backend]

        for arg in args:
            if hasattr(arg, 'storage'):
                # It's a tensor
                mixed_args.append(arg.storage._backend)
            else:
                # It's a scalar or other data
                mixed_args.append(arg)

        # Call in-place operation
        impl_func = cls._registry[key]
        result_storage = impl_func(*mixed_args, **kwargs)

        # For in-place operations, result_storage should be the same as target storage
        # and we return the original tensor object
        if result_storage is not target_tensor.storage._backend:
            raise RuntimeError(f"In-place operation {op_name} returned different storage object")

        return target_tensor

    @classmethod
    def dispatch_tuple(cls, op_name: str, *tensors, **kwargs):
        """Dispatch operation that returns multiple tensors (like split, topk).
        
        Args:
            op_name: Name of the operation
            *tensors: Input tensor(s) - device inferred from first tensor
            **kwargs: Operation keyword arguments
            
        Returns:
            Tuple of result tensors
            
        Raises:
            NotImplementedError: If no implementation found
        """
        if not tensors:
            raise ValueError("At least one tensor is required for dispatch")
        
        # Get device - handle case where first arg is a list of tensors
        if isinstance(tensors[0], (list, tuple)):
            if not tensors[0]:
                raise ValueError("Empty tensor list in dispatch")
            device = tensors[0][0].device
        else:
            device = tensors[0].device
        key = (op_name, device.type)
        if key not in cls._registry:
            raise NotImplementedError(
                f"Operation '{op_name}' not implemented for device '{device.type}'"
            )
        
        # Extract storage backends for tensor arguments, keep scalars as-is
        mixed_args = []
        for arg in tensors:
            if isinstance(arg, Tensor):
                # It's a tensor
                mixed_args.append(arg.storage._backend)
            elif isinstance(arg, (list, tuple)):
                # Check if it's a list/tuple of tensors or other data
                if arg and isinstance(arg[0], Tensor):
                    # It's a list/tuple of tensors
                    tensor_backends = [t.storage._backend for t in arg]
                    mixed_args.append(tensor_backends)
                else:
                    # It's a list/tuple of scalars or other data
                    mixed_args.append(arg)
            else:
                # It's a scalar
                mixed_args.append(arg)
        
        # Call kernel with mixed storage objects and scalars
        impl_func = cls._registry[key]
        result_storages = impl_func(*mixed_args, **kwargs)
        
        # Handle tuple/list of storage results
        result_tensors = []
        for storage in result_storages:
            storage_wrapper = Storage(storage, device)
            # Ensure shape is a tuple, not torch.Size
            result_shape = tuple(storage.shape)
            result_tensors.append(Tensor(storage_wrapper, result_shape))
        return tuple(result_tensors)

    @classmethod
    def dispatch_creation(cls, op_name: str, device, *args, **kwargs):
        """Dispatch creation operation that doesn't take tensor inputs.
        
        Args:
            op_name: Name of the operation
            device: Target device for creation
            *args: Creation arguments
            **kwargs: Creation keyword arguments
            
        Returns:
            Result tensor
            
        Raises:
            NotImplementedError: If no implementation found
        """
        key = (op_name, device.type)
        if key not in cls._registry:
            raise NotImplementedError(
                f"Operation '{op_name}' not implemented for device '{device.type}'"
            )
        
        # Call creation function directly
        impl_func = cls._registry[key]
        result_storage = impl_func(*args, **kwargs)
        
        # Wrap result storage back into tensor
        storage_wrapper = Storage(result_storage, device)
        result_shape = result_storage.shape
        return Tensor(storage_wrapper, result_shape)

    @classmethod
    def has_implementation(cls, op_name: str, device_type: DeviceType) -> bool:
        """Check if an implementation exists.
        
        Args:
            op_name: Name of the operation
            device_type: Target device type
            
        Returns:
            True if implementation exists
        """
        return (op_name, device_type) in cls._registry


# Convenience decorators
def register_cpu(op_name: str):
    """Decorator to register CPU implementation."""
    def decorator(func: Callable):
        OperationDispatcher.register(op_name, DeviceType.CPU, func)
        return func
    return decorator


def register_cuda(op_name: str):
    """Decorator to register CUDA implementation."""
    def decorator(func: Callable):
        OperationDispatcher.register(op_name, DeviceType.CUDA, func)
        return func
    return decorator
