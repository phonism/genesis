"""
Distributed Data Parallel implementation for Genesis.

Provides DDP (DistributedDataParallel) wrapper for models to enable
multi-GPU and multi-node training using NCCL communication.
"""

import logging
from typing import List, Optional
import genesis
from .comm import all_reduce, ReduceOp, _get_backend, broadcast
from .process_group import is_initialized, get_world_size, get_rank

logger = logging.getLogger(__name__)


class DistributedDataParallel(genesis.nn.Module):
    """Genesis Distributed Data Parallel wrapper.
    
    Wraps a model to enable distributed training across multiple GPUs/nodes.
    Gradients are automatically synchronized across all processes after backward pass.
    
    Args:
        model: The model to wrap for distributed training
        device_ids: List of GPU device IDs to use (default: current device)
        output_device: Device for model outputs (default: device_ids[0])
        broadcast_buffers: Whether to broadcast model buffers (default: True)
        find_unused_parameters: Whether to find unused parameters (default: False)
        gradient_as_bucket_view: Use gradient bucket view for memory efficiency (default: False)
    """
    
    def __init__(
        self,
        model: genesis.nn.Module,
        device_ids: Optional[List[int]] = None,
        output_device: Optional[int] = None,
        broadcast_buffers: bool = True,
        find_unused_parameters: bool = False,
        gradient_as_bucket_view: bool = False
    ):
        super().__init__()
        
        if not is_initialized():
            raise RuntimeError(
                "DDP requires distributed process group to be initialized. "
                "Call genesis.distributed.init_process_group() first."
            )
            
        self.model = model
        self.world_size = get_world_size()
        self.rank = get_rank()
        self.broadcast_buffers = broadcast_buffers
        self.find_unused_parameters = find_unused_parameters
        self.gradient_as_bucket_view = gradient_as_bucket_view
        
        # Set up devices
        if device_ids is None:
            # Use current CUDA device
            current_device = genesis.cuda.current_device()
            device_ids = [current_device]
            
        self.device_ids = device_ids
        self.output_device = output_device if output_device is not None else device_ids[0]

        # Model should already be on the correct device (moved by user before wrapping with DDP)
        # This is the standard DDP behavior

        # Register gradient hooks for automatic synchronization
        self._register_gradient_hooks()

        # Broadcast initial parameters from rank 0 to ensure consistency
        self._broadcast_parameters()
        
        if broadcast_buffers:
            self._broadcast_buffers()

        logger.info(f"DDP: Initialized model on rank {self.rank}/{self.world_size}")
        
    def forward(self, *inputs, **kwargs):
        """Forward pass through the wrapped model."""
        # Move inputs to the correct device if needed
        if inputs:
            inputs = self._move_to_device(inputs, self.device_ids[0])
        if kwargs:
            kwargs = {k: self._move_to_device(v, self.device_ids[0]) for k, v in kwargs.items()}
            
        # Forward pass through model
        outputs = self.model(*inputs, **kwargs)
        
        # Move outputs to output device if different
        if self.output_device != self.device_ids[0]:
            outputs = self._move_to_device(outputs, self.output_device)
            
        return outputs
        
    def _register_gradient_hooks(self):
        """Register gradient hooks for automatic all-reduce."""
        for param in self.model.parameters():
            if param.requires_grad:
                param.register_hook(self._make_gradient_hook(param))
                
    def _make_gradient_hook(self, param):
        """Create gradient hook for a parameter (standard API)."""
        def gradient_hook(grad):
            if grad is not None:
                # All-reduce sums gradients across all processes (in-place)
                all_reduce(grad, ReduceOp.SUM)
                # Average by dividing by world size (in-place to avoid creating new tensor)
                grad.data = grad.data / self.world_size
            return grad
        return gradient_hook
        
    def _broadcast_parameters(self):
        """Broadcast parameters from rank 0 to all other ranks (standard API)."""
        for param in self.model.parameters():
            broadcast(param, src=0)
            
    def _broadcast_buffers(self):
        """Broadcast model buffers from rank 0 to all other ranks.""" 
        for buffer in self.model.buffers():
            broadcast(buffer, src=0)
            
    def _move_to_device(self, obj, device_id):
        """Move tensor or nested structure to specified device."""
        if isinstance(obj, genesis.Tensor):
            return obj.to(genesis.device(f'cuda:{device_id}'))
        elif isinstance(obj, (list, tuple)):
            return type(obj)(self._move_to_device(item, device_id) for item in obj)
        elif isinstance(obj, dict):
            return {k: self._move_to_device(v, device_id) for k, v in obj.items()}
        else:
            # Non-tensor objects returned as-is
            return obj
            
    def state_dict(self, destination=None, prefix=""):
        """Get model state dict."""
        return self.model.state_dict(destination, prefix)
        
    def load_state_dict(self, state_dict, strict=True):
        """Load model state dict."""
        return self.model.load_state_dict(state_dict, strict)
        
    def parameters(self):
        """Get model parameters.""" 
        return self.model.parameters()
        
    def named_parameters(self, prefix="", recurse=True):
        """Get named model parameters."""
        return self.model.named_parameters(prefix, recurse)
        
    def buffers(self):
        """Get model buffers."""
        return self.model.buffers()
        
    def named_buffers(self, prefix="", recurse=True):
        """Get named model buffers."""
        return self.model.named_buffers(prefix, recurse)
        
    def modules(self):
        """Get model modules."""
        return self.model.modules()
        
    def named_modules(self, memo=None, prefix=""):
        """Get named model modules."""
        return self.model.named_modules(memo, prefix)
        
    def train(self, mode=True):
        """Set training mode."""
        self.model.train(mode)
        return self
        
    def eval(self):
        """Set evaluation mode."""
        self.model.eval()
        return self
        
    def to(self, device):
        """Move model to device."""
        self.model.to(device)
        return self
        
    def cuda(self, device=None):
        """Move model to CUDA device."""
        self.model.cuda(device)
        return self
        
    def cpu(self):
        """Move model to CPU."""
        self.model.cpu()
        return self
        
    def zero_grad(self):
        """Zero model gradients."""
        self.model.zero_grad()
        
    def __getattr__(self, name):
        """Delegate attribute access to wrapped model."""
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model, name)
            
    def __repr__(self):
        return f"DistributedDataParallel(\n  (module): {self.model}\n)"