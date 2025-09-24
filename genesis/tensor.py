"""
Tensor implementation - lightweight metadata + storage reference
"""
import math
import numpy as np
from typing import List, Optional, Tuple, Union
from functools import reduce
import operator
from genesis.dtypes import DType, default_dtype, float32, get_dtype
from genesis.device import device as make_device, cpu, Device
from genesis import init
from genesis.storage import Storage

class Tensor:
    """
    Tensor: lightweight metadata + storage reference
    """
    
    def __init__(self, 
                 storage,
                 shape: Tuple[int, ...],
                 stride: Optional[Tuple[int, ...]] = None,
                 offset: int = 0):
        """
        Initialize tensor from storage - PyTorch internal style
        
        Args:
            storage: Storage object holding the data
            shape: Tensor shape (sizes)
            stride: Memory stride pattern
            offset: Storage offset
        """
        self.storage = storage
        self.shape = shape
        self.offset = offset
        
        # Compute default stride (row-major)
        if stride is None:
            stride = self._compute_default_stride(shape)
        self._stride = stride
        
        # View tracking
        self.base: Optional['Tensor'] = None  # Base tensor if this is a view
        self._version: int = 0  # Version counter for in-place operations
        
        # Autograd related
        self.requires_grad = False
        self.grad: Optional['Tensor'] = None
        self.grad_fn: Optional['Function'] = None
        self.is_leaf = True
        self._hooks: List = []
        
        # Computation graph tracking
        self.creator: Optional['Function'] = None
        self.inputs: List['Tensor'] = []
        
        # Cache
        self._is_contiguous: Optional[bool] = None
        self._numel: Optional[int] = None
    
    @staticmethod
    def _compute_default_stride(shape: Tuple[int, ...]) -> Tuple[int, ...]:
        """Compute default stride (row-major)"""
        if not shape:
            return ()
        
        stride = [1]
        for dim in reversed(shape[1:]):
            stride.append(stride[-1] * dim)
        
        return tuple(reversed(stride))
    
    def numel(self) -> int:
        """Return total number of elements"""
        if self._numel is None:
            self._numel = math.prod(self.shape) if self.shape else 1
        return self._numel
    
    @property
    def is_view(self) -> bool:
        """Check if tensor is a view of another tensor"""
        return self.base is not None
    
    @property
    def storage_offset(self) -> int:
        """Return storage offset (alias for offset)"""
        return self.offset
    
    @property
    def dtype(self):
        """Return tensor data type"""
        return self.storage.dtype
    
    @property
    def device(self):
        """Return tensor device"""
        return self.storage.device
    
    @property
    def ndim(self):
        """Return number of dimensions"""
        return len(self.shape)

    def element_size(self) -> int:
        """Return bytes per element"""
        return self.storage.element_size()

    def is_contiguous(self) -> bool:
        """Check if tensor has contiguous memory layout (stride pattern and zero offset)."""
        if self._is_contiguous is None:
            # Check both tensor-level and storage-level contiguity
            expected_stride = self._compute_default_stride(self.shape)
            tensor_contiguous = (self._stride == expected_stride) and (self.offset == 0)

            # Also check storage-level contiguity for broadcast tensors
            # Use _backend since Storage wrapper doesn't have is_contiguous method
            storage_contiguous = True
            if hasattr(self.storage, '_backend') and hasattr(self.storage._backend, 'is_contiguous'):
                storage_contiguous = self.storage._backend.is_contiguous()

            # Both must be true for tensor to be truly contiguous
            self._is_contiguous = tensor_contiguous and storage_contiguous
        return self._is_contiguous
    
    def contiguous(self) -> 'Tensor':
        """Return contiguous storage tensor"""
        if self.is_contiguous():
            return self
        
        # Use storage-level contiguous method
        new_storage = self.storage.contiguous(self.shape, self._stride, self.offset)
        return Tensor(new_storage, self.shape, stride=None, offset=0)
    
    def _generate_indices(self):
        """Generate all possible indices"""
        if not self.shape:
            yield ()
            return
            
        def generate_recursive(dims):
            if not dims:
                yield ()
                return
            
            for i in range(dims[0]):
                for rest in generate_recursive(dims[1:]):
                    yield (i,) + rest
        
        yield from generate_recursive(self.shape)
    
    def _get_item_by_indices(self, indices: Tuple[int, ...]):
        """Get element by multi-dimensional indices"""
        flat_index = self.offset
        for idx, stride in zip(indices, self._stride):
            flat_index += idx * stride
        return self.storage[flat_index]
    
    def backward(self, out_grad=None):
        """Compute gradients using reverse-mode automatic differentiation.
        
        Args:
            out_grad: Optional output gradient tensor. Defaults to ones tensor.
            
        This method implements topological sorting and backward pass computation
        for the computational graph rooted at this tensor.
        """
        out_grad = out_grad if out_grad else init.ones(*self.shape, dtype=self.dtype, device=self.device)
        if hasattr(self, 'apply_hooks'):
            self.apply_hooks(self.grad)
        node_to_output_grads_list = {}
        node_to_output_grads_list[self] = [out_grad]

        # Topological sort to determine computation order
        def topo_sort(node):
            """Local topological sort implementation."""
            visited = set()
            topo_order = []

            def dfs(n):
                if n in visited:
                    return
                visited.add(n)
                if hasattr(n, 'creator') and n.creator is not None:
                    for input_node in n.creator.inputs:
                        if isinstance(input_node, Tensor):
                            dfs(input_node)
                topo_order.append(n)
            
            dfs(node)
            return topo_order
            
        if hasattr(self, 'creator') and self.creator is not None:
            topo_order = topo_sort(self)
        else:
            topo_order = [self]
        
        # Reverse pass through computation graph
        for node in reversed(topo_order):
            if node.requires_grad is False:
                continue
            
            # Accumulate gradients for current node - optimized for memory efficiency
            grad_list = node_to_output_grads_list[node]
            if node.grad is None:
                # Use first gradient as base, accumulate rest in-place
                if len(grad_list) == 1:
                    node.grad = grad_list[0]
                else:
                    node.grad = grad_list[0].clone()  # Clone to avoid modifying original
                    for grad in grad_list[1:]:
                        node.grad += grad  # In-place accumulation

                # Ensure gradient is contiguous only if needed
                if hasattr(node.grad, 'data') and hasattr(node.grad.data, 'data'):
                    cuda_tensor = node.grad.data.data
                    if hasattr(cuda_tensor, 'is_contiguous') and not cuda_tensor.is_contiguous():
                        node.grad.data.data = cuda_tensor.contiguous()
            else:
                # Accumulate all gradients in-place
                for grad in grad_list:
                    node.grad += grad
            if hasattr(node, 'apply_hooks'):
                node.apply_hooks(node.grad)
            
            # Propagate gradients to input nodes
            if hasattr(node, 'creator') and node.creator is not None:
                for nd in node.creator.inputs:
                    if nd not in node_to_output_grads_list:
                        node_to_output_grads_list[nd] = []
                        
                # Use current node's gradient for backward computation
                grad = node.grad
                
                # Compute backward gradients
                if hasattr(node.creator, 'is_tuple_result') and node.creator.is_tuple_result:
                    backward_grad = node.creator.backward(node.creator.ctx, grad, node.idx)
                else:
                    backward_grad = node.creator.backward(node.creator.ctx, grad)
                
                # Distribute gradients to input tensors
                for i, nd in enumerate(node.creator.inputs):
                    if nd.requires_grad is False:
                        continue
                    node_to_output_grads_list[nd].append(backward_grad[i])
    
    def to(self, *args, **kwargs):
        """Move tensor to device and/or change dtype - compatible with torch.Tensor.to"""
        # Handle different argument patterns like PyTorch
        device = None
        dtype = None

        # Parse positional arguments
        for arg in args:
            if isinstance(arg, DType):  # DType object
                dtype = arg
            elif isinstance(arg, Device):  # Device object
                device = arg
            elif isinstance(arg, str):  # device string or dtype string
                if 'cuda' in arg or arg == 'cpu':
                    device = arg
                else:
                    dtype = arg
            else:
                device = arg

        # Parse keyword arguments
        if 'device' in kwargs:
            device = kwargs['device']
        if 'dtype' in kwargs:
            dtype = kwargs['dtype']

        result = self

        # Handle dtype conversion
        if dtype is not None:
            result = result.to_dtype(dtype)

        # Handle device conversion
        if device is not None:
            target_device = make_device(device)
            if result.storage.device != target_device:
                new_storage = result.storage.to(target_device)
                result = Tensor(new_storage, result.shape, result._stride, result.offset)
                result.requires_grad = self.requires_grad

        return result
    
    def data_ptr(self):
        """Get pointer to tensor data - compatible with torch.Tensor.data_ptr()"""
        return self.storage.data_ptr()
    
    def cpu(self) -> 'Tensor':
        """Move tensor to CPU - compatible with torch.Tensor.cpu"""
        return self.to('cpu')
    
    def cuda(self, device=None) -> 'Tensor':
        """Move tensor to GPU - compatible with torch.Tensor.cuda"""
        if device is None:
            return self.to('cuda')
        else:
            return self.to(f'cuda:{device}')

    def to_dtype(self, dtype) -> 'Tensor':
        """Convert tensor to different dtype - compatible with PyTorch"""
        from genesis.dtypes import get_dtype
        target_dtype = get_dtype(dtype)

        # Convert storage to new dtype
        new_storage = self.storage.to_dtype(target_dtype)

        # Create new tensor with converted storage
        new_tensor = Tensor(new_storage, self.shape, self._stride, self.offset)
        new_tensor.requires_grad = self.requires_grad
        return new_tensor

    def numpy(self):
        """
        Return tensor as numpy array - compatible with torch.Tensor.numpy()
        
        For CUDA tensors, directly gets data from CUDA storage.
        Ensures the result is contiguous.
        
        Returns:
            numpy.ndarray: The tensor data as a numpy array
        """
        # Ensure tensor is contiguous for proper memory layout
        tensor_contig = self.contiguous()
        
        # For CUDA tensors, get numpy data directly from CUDA storage
        if self.device.is_cuda():
            numpy_data = tensor_contig.storage._backend.to_numpy()
            # Handle scalar case where shape is ()
            if tensor_contig.shape == ():
                return numpy_data.reshape(())
            else:
                return numpy_data.reshape(*tensor_contig.shape)
        else:
            # For CPU tensors, use storage's numpy method
            numpy_data = tensor_contig.storage._backend.numpy()
            # Handle scalar case where shape is ()
            if tensor_contig.shape == ():
                return numpy_data.reshape(())
            else:
                return numpy_data.reshape(*tensor_contig.shape)
    
    def requires_grad_(self, requires_grad: bool = True):
        """
        Set requires_grad flag (in-place operation), PyTorch-compatible method.
        
        Args:
            requires_grad: Whether to require gradients
            
        Returns:
            Self tensor for chaining
        """
        self.requires_grad = requires_grad
        return self
    
    def detach(self) -> 'Tensor':
        """
        Return a new tensor detached from computation graph.
        
        Returns:
            New tensor with requires_grad=False
        """
        result = Tensor(self.storage, self.shape, self._stride, self.offset)
        result.requires_grad = False
        return result
    
    def clone(self) -> 'Tensor':
        """
        Create a deep copy of the tensor with independent data storage.

        Returns:
            New tensor with same data but independent storage
        """
        # Create new storage with copied data
        cloned_storage = self.storage.clone()
        result = Tensor(cloned_storage, self.shape, self._stride, self.offset)
        result.requires_grad = self.requires_grad
        return result

    def copy_(self, src: 'Tensor') -> 'Tensor':
        """
        Copy data from source tensor into this tensor in-place.
        Similar to PyTorch's copy_ method.

        Args:
            src: Source tensor to copy from

        Returns:
            self (modified in-place)
        """
        # Import here to avoid circular dependency
        from genesis.ops.dispatcher import OperationDispatcher

        # Ensure shapes are compatible
        if self.shape != src.shape:
            raise RuntimeError(f"Shape mismatch: cannot copy tensor of shape {src.shape} into tensor of shape {self.shape}")

        # Use dispatcher to handle the copy operation
        # This will route to the appropriate backend (CPU/CUDA)
        OperationDispatcher.dispatch("copy", self, src)

        # Return self for chaining
        return self

    def apply_hooks(self, grad):
        """Apply hooks to gradient"""
        for hook in self._hooks:
            hook(grad)
    
    def register_hook(self, hook):
        """Register backward hook"""
        self._hooks.append(hook)
    
    def set_creator(self, function, idx=None):
        """Set creator function for autograd"""
        self.creator = function
        self.idx = idx if idx is not None else None
        self.is_leaf = False
    
    def all(self, dim=None, keepdim=False):
        """
        Test whether all tensor elements evaluate to True.
        
        Args:
            dim: Dimension or dimensions along which to check. None means check all dimensions.
            keepdim: Whether to keep reduced dimensions as size 1
            
        Returns:
            Boolean tensor indicating if all elements are True
        """
        import genesis
        # Convert to boolean tensor (non-zero is True) 
        bool_tensor = self != 0
        
        if dim is None:
            # Reduce all dimensions - return scalar tensor like PyTorch
            sum_result = bool_tensor.sum()
            # Convert to Python values to compare, then create result tensor
            total_elements = self.numel()
            result_val = sum_result.item() == total_elements
            # Create scalar bool tensor like PyTorch
            result = genesis.tensor(result_val, dtype=genesis.bool, device=self.device)
            return result
        else:
            # Reduce along specific dimension
            total_along_dim = bool_tensor.sum(dim=dim, keepdim=keepdim)
            if isinstance(dim, int):
                expected_count = self.shape[dim]
            elif isinstance(dim, (tuple, list)):
                expected_count = 1
                for d in dim:
                    expected_count *= self.shape[d]
            else:
                expected_count = self.shape[dim]
            return total_along_dim == expected_count
    
    def float(self):
        """Convert tensor to float32 dtype."""
        if self.dtype == float32:
            return self
        
        # Use storage dtype conversion
        new_storage = self.storage.to_dtype(float32)
        result = Tensor(new_storage, self.shape, self._stride, self.offset)
        result.requires_grad = self.requires_grad
        return result
    
    def any(self, dim=None, keepdim=False):
        """
        Test whether any tensor elements evaluate to True.
        
        Args:
            dim: Dimension or dimensions along which to check. None means check all dimensions.
            keepdim: Whether to keep reduced dimensions as size 1
            
        Returns:
            Boolean tensor indicating if any elements are True
        """
        import genesis
        # Convert to boolean tensor (non-zero is True) 
        bool_tensor = self != 0
        
        if dim is None:
            # Reduce all dimensions - return scalar boolean
            total_sum = bool_tensor.sum()
            # Any element is true if sum > 0
            result = total_sum > 0
            return result
        else:
            # Reduce along specific dimension
            total_along_dim = bool_tensor.sum(dim=dim, keepdim=keepdim)
            return total_along_dim > 0
    
    def fill_(self, value):
        """
        Fill tensor with specified value (in-place operation).

        Args:
            value: Value to fill tensor with

        Returns:
            Self tensor for chaining
        """
        # Use storage-level fill operation
        self.storage._backend.fill_(value)
        return self

    def item(self):
        """
        Return the value of a tensor with a single element as a Python number.
        
        Returns:
            Python scalar value
        """
        if self.numel() != 1:
            raise ValueError("only one element tensors can be converted to Python scalars")
        # Get the scalar value from storage backend
        return self.storage._backend.item()
    
    @property
    def __cuda_array_interface__(self):
        """
        CUDA Array Interface for Triton compatibility.

        Returns:
            Dict with CUDA array interface specification
        """
        if not self.device.is_cuda():
            raise RuntimeError("__cuda_array_interface__ only available for CUDA tensors")

        # Delegate to storage backend's implementation
        # The storage backend should handle any offset internally
        return self.storage._backend.__cuda_array_interface__

    @property
    def stride(self):
        """Return stride tuple or callable for PyTorch compatibility"""
        class StrideAccessor:
            def __init__(self, stride_tuple):
                self._stride_tuple = stride_tuple

            def __call__(self, dim=None):
                if dim is None:
                    return self._stride_tuple
                else:
                    return self._stride_tuple[dim]

            def __getitem__(self, dim):
                return self._stride_tuple[dim]

            def __iter__(self):
                return iter(self._stride_tuple)

            def __len__(self):
                return len(self._stride_tuple)

            def __eq__(self, other):
                return self._stride_tuple == other

            def __repr__(self):
                return repr(self._stride_tuple)

        # Always get stride from backend to ensure consistency
        # This eliminates stride synchronization issues in view operations
        backend_strides = getattr(self.storage._backend, 'strides', self._stride)
        return StrideAccessor(backend_strides)

    def __repr__(self):
        # Use actual stride from backend for accurate representation
        actual_stride = getattr(self.storage._backend, 'strides', self._stride)
        return f"Tensor(shape={self.shape}, dtype={self.storage.dtype}, device={self.device}, stride={actual_stride}, offset={self.offset})"

def tensor(data, dtype: Optional[DType] = None, device = None, requires_grad: bool = False) -> Tensor:
    """
    Create tensor from data - compatible with torch.tensor
    
    Args:
        data: Data (list, scalar, or array-like)
        dtype: Data type
        device: Device placement ('cpu', 'cuda', Device object)
        requires_grad: Whether to track gradients
    """
    t = make_tensor(data, dtype, device)
    t.requires_grad = requires_grad
    return t

def make_tensor(data, dtype = None, device=None, shape: Optional[Tuple[int, ...]] = None) -> Tensor:
    """Create tensor from data - PyTorch internal style make_tensor"""
    # Handle dtype - support both string and DType object
    dtype = get_dtype(dtype)

    if device is None:
        device = cpu()
    else:
        # Handle both string and Device object inputs
        if isinstance(device, str):
            device = make_device(device)
        # If already a Device object, use it directly
    
    # Data standardization - convert to numpy array (PyTorch pattern)
    if isinstance(data, (list, tuple)):
        # Convert list/tuple to numpy array
        standardized_data = np.array(data, dtype=dtype.numpy_dtype)
        tensor_shape = standardized_data.shape
    elif isinstance(data, np.ndarray):
        # Already numpy array, just ensure dtype
        standardized_data = data.astype(dtype.numpy_dtype) if data.dtype != dtype.numpy_dtype else data
        tensor_shape = data.shape
    elif np.isscalar(data):
        # Scalar to numpy array
        standardized_data = np.array(data, dtype=dtype.numpy_dtype)
        tensor_shape = ()
    else:
        raise ValueError(f"Unsupported data type for tensor creation: {type(data)}")
    
    # Use provided shape or infer from data
    if shape is not None:
        tensor_shape = shape
    
    # Create storage with standardized numpy data
    storage = Storage.make_storage(standardized_data, dtype, device)
    return Tensor(storage, tensor_shape)

def ones_like(tensor: Tensor) -> Tensor:
    """Create tensor with same shape filled with ones"""
    ones_data = [1.0] * tensor.numel()
    return make_tensor(ones_data, tensor.storage.dtype, tensor.storage.device, tensor.shape)
