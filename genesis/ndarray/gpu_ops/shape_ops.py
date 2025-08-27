"""
Shape manipulation operations for GPU backend.
"""


def reshape(x, new_shape):
    """
    Reshape tensor to new shape.
    """
    return x.reshape(new_shape)


def view(x, new_shape):
    """
    Return a view of tensor with new shape.
    """
    if x.is_contiguous() is False:
        x = x.contiguous()
    return x.view(new_shape)


def expand(x, new_shape):
    """
    Expand tensor to new shape by broadcasting.
    """
    return x.expand(new_shape)


def permute(x, new_axis):
    """
    Permute tensor dimensions.
    """
    return x.permute(new_axis)


def broadcast_to(x, new_shape):
    """
    Broadcast tensor to new shape.
    """
    return x.broadcast_to(new_shape)


import triton
import triton.language as tl
from ..cuda_storage import CUDAStorage


@triton.jit
def repeat_interleave_kernel(
    input_ptr, output_ptr,
    input_stride, output_stride,
    original_size, repeat_count,
    input_base, output_base,
    total_elements: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    """
    Repeat-interleave kernel: each element is repeated consecutively.
    """
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_elements
    
    # Calculate which original element and which repetition
    original_idx = offsets // repeat_count
    
    # Load from input
    input_offset = input_base + original_idx * input_stride
    values = tl.load(input_ptr + input_offset, mask=mask, other=0.0)
    
    # Store to output
    output_offset = output_base + offsets * output_stride
    tl.store(output_ptr + output_offset, values, mask=mask)


def repeat_interleave(x, repeats, dim=None):
    """
    Repeat elements of tensor along specified dimension.
    
    Args:
        x: Input CUDAStorage tensor
        repeats: Number of repetitions for each element (integer)
        dim: Dimension to repeat along (if None, flatten first)
        
    Returns:
        CUDAStorage: Tensor with repeated elements
    """
    if dim is None:
        # Flatten and repeat
        x_flat = x.view(-1)
        return repeat_interleave(x_flat, repeats, dim=0)
    
    if dim < 0:
        dim = len(x.shape) + dim
    
    # Calculate output shape
    output_shape = list(x.shape)
    output_shape[dim] *= repeats
    
    # Create output tensor
    output = CUDAStorage(output_shape, dtype=x.dtype)
    
    if not x.is_contiguous():
        x = x.contiguous()
    
    # Calculate strides
    input_stride = x.strides[dim] if dim < len(x.strides) else 1
    output_stride = output.strides[dim] if dim < len(output.strides) else 1
    
    # Process each slice along the repeat dimension
    original_size = x.shape[dim]
    
    # Calculate how many slices we need to process
    prefix_size = 1
    for i in range(dim):
        prefix_size *= x.shape[i]
    
    suffix_size = 1  
    for i in range(dim + 1, len(x.shape)):
        suffix_size *= x.shape[i]
    
    # For each combination of prefix and suffix indices
    for prefix_idx in range(prefix_size):
        for suffix_idx in range(suffix_size):
            # Calculate input and output base offsets
            input_base = prefix_idx * x.strides[0] if dim > 0 else 0
            if dim > 1:
                for i in range(1, dim):
                    input_base += (prefix_idx // (prefix_size // x.shape[i])) % x.shape[i] * x.strides[i]
            
            if suffix_size > 1:
                suffix_offset = 0
                remaining = suffix_idx
                for i in range(dim + 1, len(x.shape)):
                    size = x.shape[i]
                    suffix_offset += (remaining % size) * x.strides[i]
                    remaining //= size
                input_base += suffix_offset
            
            # Similar calculation for output base
            output_base = prefix_idx * output.strides[0] if dim > 0 else 0
            if dim > 1:
                for i in range(1, dim):
                    output_base += (prefix_idx // (prefix_size // output.shape[i])) % output.shape[i] * output.strides[i]
            
            if suffix_size > 1:
                suffix_offset = 0
                remaining = suffix_idx
                for i in range(dim + 1, len(output.shape)):
                    size = output.shape[i]
                    suffix_offset += (remaining % size) * output.strides[i]
                    remaining //= size
                output_base += suffix_offset
            
            # Launch kernel for this slice
            total_elements = original_size * repeats
            grid = lambda meta: (triton.cdiv(total_elements, meta["BLOCK_SIZE"]),)
            
            repeat_interleave_kernel[grid](
                x, output,
                input_stride, output_stride,
                original_size, repeats, 
                input_base, output_base,
                total_elements=total_elements,
                BLOCK_SIZE=256
            )
    
    return output