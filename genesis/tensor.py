"""
Tensor implementation - lightweight metadata + storage reference
"""
import math
import numpy as np
from typing import List, Optional, Tuple, Union
from functools import reduce
import operator
from genesis.dtypes import DType, default_dtype, float32, float16, float64, bfloat16, get_dtype
from genesis.device import device as make_device, cpu, Device
from genesis import init
from genesis.storage import Storage

# Import to avoid function-level imports
import genesis

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
        Initialize tensor from storage with efficient memory layout.

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
    def data(self):
        """Return the underlying tensor data without gradient tracking.

        Returns a new Tensor that shares storage with the original but
        doesn't track gradients. This is useful for in-place operations
        that should not be tracked by autograd.

        Returns:
            Tensor: A new tensor view with requires_grad=False
        """
        # Create a new Tensor that shares the same storage
        new_tensor = Tensor(self.storage, self.shape, self._stride, self.offset)
        new_tensor.requires_grad = False
        new_tensor.grad = None
        new_tensor.grad_fn = None
        new_tensor.is_leaf = True
        new_tensor.base = self.base if self.base is not None else self
        return new_tensor
    
    @property
    def ndim(self):
        """Return number of dimensions"""
        return len(self.shape)

    def element_size(self) -> int:
        """Return bytes per element"""
        return self.storage.element_size()

    def is_floating_point(self) -> bool:
        """Check if tensor is floating point type.

        Returns:
            bool: True if dtype is float16, float32, float64, or bfloat16
        """
        return self.dtype in (float16, float32, float64, bfloat16)

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
        result = Tensor(new_storage, self.shape, stride=None, offset=0)
        # Inherit requires_grad to ensure gradient tracking continues
        result.requires_grad = self.requires_grad
        return result
    
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
    
    def _topo_sort_creators(self, root_creator):
        """Perform topological sort on creator graph (not tensor graph).

        Args:
            root_creator: Root creator to start topological sort from

        Returns:
            List of creators in topological order
        """
        visited = set()
        topo_order = []

        def dfs(creator):
            if creator is None or id(creator) in visited:
                return
            visited.add(id(creator))

            # Traverse next_functions instead of inputs
            for next_creator, _ in creator.next_functions:
                dfs(next_creator)

            topo_order.append(creator)

        dfs(root_creator)
        return topo_order

    def _accumulate_gradients(self, node, grad_list):
        """Accumulate gradients for a node efficiently.

        Args:
            node: Node to accumulate gradients for
            grad_list: List of gradients to accumulate
        """
        if node.grad is None:
            # Use first gradient as base
            if len(grad_list) == 1:
                node.grad = grad_list[0]
            else:
                # Sum all gradients - result will be contiguous
                node.grad = grad_list[0]
                for grad in grad_list[1:]:
                    node.grad += grad

        else:
            # Accumulate all gradients
            for grad in grad_list:
                node.grad += grad

    def _propagate_gradients_to_creators(self, creator, out_grad, output_idx, creator_to_grads):
        """Propagate gradients to next creators using next_functions.

        Args:
            creator: Current creator to propagate gradients from
            out_grad: Output gradient for this creator
            output_idx: Index of the output (for multi-output functions)
            creator_to_grads: Dictionary tracking gradient lists for each (creator, output_idx)
        """
        from genesis.function import AccumulateGrad

        # If this is AccumulateGrad, accumulate to leaf tensor
        if isinstance(creator, AccumulateGrad):
            creator.apply_grad(out_grad)
            return

        # Compute backward gradients
        if hasattr(creator, 'is_tuple_result') and creator.is_tuple_result:
            # For multi-output functions, backward needs idx
            backward_grad = creator.backward(creator.ctx, out_grad, output_idx)
        else:
            backward_grad = creator.backward(creator.ctx, out_grad)

        # 🔥 CRITICAL: Release saved_tensors immediately after backward to save memory
        # This mimics PyTorch's behavior where saved_tensors are freed as soon as they're used
        # For tuple results, only clear after all outputs have done backward (reference counting)
        if hasattr(creator, 'ctx') and hasattr(creator.ctx, '_saved_tensors'):
            if hasattr(creator.ctx, '_backward_count'):
                # Tuple result: decrement counter and only clear when all done
                creator.ctx._backward_count -= 1
                if creator.ctx._backward_count <= 0:
                    creator.ctx._saved_tensors = []  # Clear saved tensors to free memory
            else:
                # Single result: clear immediately
                creator.ctx._saved_tensors = []  # Clear saved tensors to free memory

        # Ensure backward_grad is a tuple
        if not isinstance(backward_grad, tuple):
            backward_grad = (backward_grad,)

        # Propagate gradients to next_functions
        if len(backward_grad) != len(creator.next_functions):
            # Some functions may return fewer gradients (for non-tensor inputs)
            # Pad with None
            backward_grad = list(backward_grad)
            while len(backward_grad) < len(creator.next_functions):
                backward_grad.append(None)

        for (next_creator, next_output_idx), grad in zip(creator.next_functions, backward_grad):
            if grad is None:  # Some inputs don't need gradients
                continue

            # Key is (creator_id, output_idx) to handle multi-output functions
            next_key = (id(next_creator), next_output_idx)
            if next_key not in creator_to_grads:
                creator_to_grads[next_key] = []
            creator_to_grads[next_key].append(grad)

    def backward(self, out_grad=None):
        """Compute gradients using reverse-mode automatic differentiation.

        Args:
            out_grad: Optional output gradient tensor. Defaults to ones tensor.

        This method implements creator graph traversal (not tensor graph) for
        backward pass computation, allowing intermediate tensors to be garbage
        collected during forward pass.
        """
        if not self.requires_grad:
            return

        # Keep autocast state during backward (gradients computed in same dtype as forward)
        # PyTorch keeps autocast enabled in backward for better performance
        saved_autocast = genesis.enable_autocast

        # Initialize output gradient
        if out_grad is None:
            if self.shape == ():
                out_grad = init.ones(*self.shape, dtype=self.dtype, device=self.device)
            else:
                raise RuntimeError("grad can be implicitly created only for scalar outputs")

        # If this is a leaf tensor, just set gradient
        if self.creator is None:
            self.grad = out_grad
            if hasattr(self, 'apply_hooks'):
                self.apply_hooks(self.grad)
            genesis.enable_autocast = saved_autocast
            return

        # === New backward algorithm using creator graph ===

        # 1. Topological sort of creator graph
        topo_order = self._topo_sort_creators(self.creator)

        # 2. Initialize gradient dictionary ((creator_id, output_idx) -> list of grads)
        creator_to_grads = {}
        output_idx = self.idx if hasattr(self, 'idx') else 0
        creator_key = (id(self.creator), output_idx)
        creator_to_grads[creator_key] = [out_grad]

        # 3. Reverse pass through creator graph
        num_creators_processed = 0

        for creator in reversed(topo_order):
            # Check all possible output indices for this creator
            # For single-output functions, only idx=0 exists
            # For multi-output functions, multiple indices may have gradients
            creator_id = id(creator)

            # Find all output indices that have gradients for this creator
            output_indices_with_grads = [idx for (cid, idx) in creator_to_grads.keys() if cid == creator_id]

            if not output_indices_with_grads:
                continue

            num_creators_processed += 1

            # Process each output index separately
            for out_idx in output_indices_with_grads:
                creator_key = (creator_id, out_idx)

                # Accumulate all gradients for this (creator, output_idx)
                grad_list = creator_to_grads[creator_key]
                if len(grad_list) == 1:
                    accumulated_grad = grad_list[0]
                else:
                    # Sum all gradients
                    accumulated_grad = grad_list[0]
                    for g in grad_list[1:]:
                        accumulated_grad = accumulated_grad + g

                # 🔥 CRITICAL: Delete processed gradients immediately to free memory
                # This mimics PyTorch's behavior where gradients are freed as soon as they're used
                del creator_to_grads[creator_key]

                # Propagate gradients to next_functions
                self._propagate_gradients_to_creators(creator, accumulated_grad, out_idx, creator_to_grads)

        # Restore autocast state after backward pass
        genesis.enable_autocast = saved_autocast

    def to(self, *args, **kwargs):
        """Move tensor to device and/or change dtype with flexible arguments."""
        # Handle different argument patterns flexibly
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
        """Get pointer to underlying tensor data."""
        return self.storage.data_ptr()
    
    def cpu(self) -> 'Tensor':
        """Move tensor to CPU device."""
        return self.to('cpu')
    
    def cuda(self, device=None) -> 'Tensor':
        """Move tensor to CUDA device."""
        if device is None:
            return self.to('cuda')
        else:
            return self.to(f'cuda:{device}')

    def to_dtype(self, dtype) -> 'Tensor':
        """Convert tensor to different data type."""
        target_dtype = get_dtype(dtype)

        # Convert storage to new dtype
        new_storage = self.storage.to_dtype(target_dtype)

        # Create new tensor with converted storage
        new_tensor = Tensor(new_storage, self.shape, self._stride, self.offset)
        new_tensor.requires_grad = self.requires_grad
        return new_tensor

    def numpy(self):
        """
        Return tensor as numpy array.

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
        Set requires_grad flag (in-place operation) for gradient tracking.
        
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
        # Convert to boolean tensor (non-zero is True) 
        bool_tensor = self != 0
        
        if dim is None:
            # Reduce all dimensions - return scalar tensor
            sum_result = bool_tensor.sum()
            # Convert to Python values to compare, then create result tensor
            total_elements = self.numel()
            result_val = sum_result.item() == total_elements
            # Create scalar bool tensor for boolean operations
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

    def __float__(self):
        """Convert scalar tensor to Python float (PyTorch compatibility).

        Returns:
            float: Python float value

        Raises:
            TypeError: If tensor has more than one element
        """
        if self.numel() != 1:
            raise TypeError(f"only one element tensors can be converted to Python scalars, got {self.numel()} elements")
        return float(self.item())

    def __int__(self):
        """Convert scalar tensor to Python int (PyTorch compatibility).

        Returns:
            int: Python int value

        Raises:
            TypeError: If tensor has more than one element
        """
        if self.numel() != 1:
            raise TypeError(f"only one element tensors can be converted to Python scalars, got {self.numel()} elements")
        return int(self.item())

    def __bool__(self):
        """Convert scalar tensor to Python bool (PyTorch compatibility).

        Returns:
            bool: Python bool value

        Raises:
            TypeError: If tensor has more than one element
        """
        if self.numel() != 1:
            raise TypeError(f"only one element tensors can be converted to Python scalars, got {self.numel()} elements")
        return bool(self.item())

def tensor(data, dtype: Optional[DType] = None, device = None, requires_grad: bool = False) -> Tensor:
    """
    Create tensor from data with optional dtype and device specification.

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
