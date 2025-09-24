"""
Comparison operations for GPU backend.
"""
import triton
import triton.language as tl
from genesis.backends.cuda import CUDAStorage
from genesis.ops.dispatcher import register_cuda


# =============================================================================
# TRITON KERNELS
# =============================================================================

@triton.jit
def compare_kernel(
    x_ptr, y_ptr, output_ptr, n_elements,
    op_type: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Element-wise comparison kernel.
    """
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    
    if op_type == 0:  # eq
        result = x == y
    elif op_type == 1:  # lt
        result = x < y
    elif op_type == 2:  # le  
        result = x <= y
    elif op_type == 3:  # gt
        result = x > y
    elif op_type == 4:  # ge
        result = x >= y
    else:  # ne
        result = x != y
    
    tl.store(output_ptr + offsets, result, mask=mask)


@triton.jit
def compare_scalar_kernel(
    x_ptr, output_ptr, n_elements, scalar_val,
    op_type: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Element-wise comparison with scalar kernel.
    """
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    x = tl.load(x_ptr + offsets, mask=mask)
    
    if op_type == 0:  # eq
        result = x == scalar_val
    elif op_type == 1:  # lt
        result = x < scalar_val
    elif op_type == 2:  # le
        result = x <= scalar_val
    elif op_type == 3:  # gt
        result = x > scalar_val
    elif op_type == 4:  # ge
        result = x >= scalar_val
    else:  # ne
        result = x != scalar_val
    
    tl.store(output_ptr + offsets, result, mask=mask)


# =============================================================================
# GPU OPERATIONS
# =============================================================================


@register_cuda("eq")
def eq(x, y):
    """
    Element-wise equality comparison.
    """
    if isinstance(y, CUDAStorage):
        # Tensor comparison
        if x.shape != y.shape:
            raise ValueError(f"Shape mismatch: {x.shape} vs {y.shape}")
        
        output = CUDAStorage(x.shape, dtype="bool")
        
        if not x.is_contiguous():
            x = x.contiguous()
        if not y.is_contiguous():
            y = y.contiguous()
        
        n_elements = output.size
        grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]), )
        compare_kernel[grid](x, y, output, n_elements, 0, BLOCK_SIZE=1024)  # 0 = eq
    else:
        # Scalar comparison
        output = CUDAStorage(x.shape, dtype="bool")
        if not x.is_contiguous():
            x = x.contiguous()
        
        n_elements = output.size
        grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]), )
        compare_scalar_kernel[grid](x, output, n_elements, float(y), 0, BLOCK_SIZE=1024)  # 0 = eq
    
    return output


@register_cuda("ge")
def ge(x, y):
    """
    Element-wise greater-than-or-equal comparison.
    """
    if isinstance(y, CUDAStorage):
        # Tensor comparison
        if x.shape != y.shape:
            raise ValueError(f"Shape mismatch: {x.shape} vs {y.shape}")
        
        output = CUDAStorage(x.shape, dtype="bool")
        
        if not x.is_contiguous():
            x = x.contiguous()
        if not y.is_contiguous():
            y = y.contiguous()
        
        n_elements = output.size
        grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]), )
        compare_kernel[grid](x, y, output, n_elements, 4, BLOCK_SIZE=1024)  # 4 = ge
    else:
        # Scalar comparison
        output = CUDAStorage(x.shape, dtype="bool")
        if not x.is_contiguous():
            x = x.contiguous()
        
        n_elements = output.size
        grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]), )
        compare_scalar_kernel[grid](x, output, n_elements, float(y), 4, BLOCK_SIZE=1024)  # 4 = ge
    
    return output


@register_cuda("gt")
def gt(x, y):
    """
    Element-wise greater-than comparison.
    """
    if isinstance(y, CUDAStorage):
        # Tensor comparison
        if x.shape != y.shape:
            raise ValueError(f"Shape mismatch: {x.shape} vs {y.shape}")
        
        output = CUDAStorage(x.shape, dtype="bool")
        
        if not x.is_contiguous():
            x = x.contiguous()
        if not y.is_contiguous():
            y = y.contiguous()
        
        n_elements = output.size
        grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]), )
        compare_kernel[grid](x, y, output, n_elements, 3, BLOCK_SIZE=1024)  # 3 = gt
    else:
        # Scalar comparison
        output = CUDAStorage(x.shape, dtype="bool")
        if not x.is_contiguous():
            x = x.contiguous()
        
        n_elements = output.size
        grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]), )
        compare_scalar_kernel[grid](x, output, n_elements, float(y), 3, BLOCK_SIZE=1024)  # 3 = gt
    
    return output


@register_cuda("le")
def le(x, y):
    """
    Element-wise less-than-or-equal comparison.
    """
    if isinstance(y, CUDAStorage):
        # Tensor comparison
        if x.shape != y.shape:
            raise ValueError(f"Shape mismatch: {x.shape} vs {y.shape}")
        
        output = CUDAStorage(x.shape, dtype="bool")
        
        if not x.is_contiguous():
            x = x.contiguous()
        if not y.is_contiguous():
            y = y.contiguous()
        
        n_elements = output.size
        grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]), )
        compare_kernel[grid](x, y, output, n_elements, 2, BLOCK_SIZE=1024)  # 2 = le
    else:
        # Scalar comparison
        output = CUDAStorage(x.shape, dtype="bool")
        if not x.is_contiguous():
            x = x.contiguous()
        
        n_elements = output.size
        grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]), )
        compare_scalar_kernel[grid](x, output, n_elements, float(y), 2, BLOCK_SIZE=1024)  # 2 = le
    
    return output


@register_cuda("lt")
def lt(x, y):
    """
    Element-wise less-than comparison.
    """
    if isinstance(y, CUDAStorage):
        # Tensor comparison
        if x.shape != y.shape:
            raise ValueError(f"Shape mismatch: {x.shape} vs {y.shape}")
        
        output = CUDAStorage(x.shape, dtype="bool")
        
        if not x.is_contiguous():
            x = x.contiguous()
        if not y.is_contiguous():
            y = y.contiguous()
        
        n_elements = output.size
        grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]), )
        compare_kernel[grid](x, y, output, n_elements, 1, BLOCK_SIZE=1024)  # 1 = lt
    else:
        # Scalar comparison
        output = CUDAStorage(x.shape, dtype="bool")
        if not x.is_contiguous():
            x = x.contiguous()
        
        n_elements = output.size
        grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]), )
        compare_scalar_kernel[grid](x, output, n_elements, float(y), 1, BLOCK_SIZE=1024)  # 1 = lt
    
    return output


@register_cuda("ne")
def ne(x, y):
    """
    Element-wise not-equal comparison.
    """
    if isinstance(y, CUDAStorage):
        # Tensor comparison
        if x.shape != y.shape:
            raise ValueError(f"Shape mismatch: {x.shape} vs {y.shape}")
        
        output = CUDAStorage(x.shape, dtype="bool")
        
        if not x.is_contiguous():
            x = x.contiguous()
        if not y.is_contiguous():
            y = y.contiguous()
        
        n_elements = output.size
        grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]), )
        compare_kernel[grid](x, y, output, n_elements, 5, BLOCK_SIZE=1024)  # 5 = ne
    else:
        # Scalar comparison
        output = CUDAStorage(x.shape, dtype="bool")
        if not x.is_contiguous():
            x = x.contiguous()
        
        n_elements = output.size
        grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]), )
        compare_scalar_kernel[grid](x, output, n_elements, float(y), 5, BLOCK_SIZE=1024)  # 5 = ne
    
    return output