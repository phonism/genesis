"""
PyTorch-aligned dtype system for Genesis
Provides dtype objects similar to torch.dtype
"""

import numpy as np


class DType:
    """Genesis data type, similar to torch.dtype"""
    
    def __init__(self, name, itemsize, numpy_dtype, triton_name=None, is_floating_point=None):
        self.name = name
        self.itemsize = itemsize
        self.numpy_dtype = numpy_dtype
        self.triton_name = triton_name or name
        # Auto-detect if floating point based on numpy dtype
        if is_floating_point is None:
            self.is_floating_point = np.issubdtype(numpy_dtype, np.floating)
        else:
            self.is_floating_point = is_floating_point
        
    def __str__(self):
        return f"genesis.{self.name}"
        
    def __repr__(self):
        return f"genesis.{self.name}"
        
    def __eq__(self, other):
        if isinstance(other, DType):
            return self.name == other.name
        elif isinstance(other, str):
            return self.name == other  # Backward compatibility for string comparison
        return False
        
    def __hash__(self):
        return hash(self.name)


# Predefined dtype objects - aligned with PyTorch
float32 = DType("float32", 4, np.float32)
float16 = DType("float16", 2, np.float16) 
float64 = DType("float64", 8, np.float64)
int32 = DType("int32", 4, np.int32)
int64 = DType("int64", 8, np.int64)
int16 = DType("int16", 2, np.int16)
int8 = DType("int8", 1, np.int8)
uint8 = DType("uint8", 1, np.uint8)
bool = DType("bool", 1, np.bool_, is_floating_point=False)

# bfloat16 special handling - Triton supports but numpy doesn't natively
bfloat16 = DType("bfloat16", 2, np.float32, "bfloat16", is_floating_point=True)

# Mapping from dtype names to objects
_name_to_dtype = {
    "float32": float32,
    "float16": float16,
    "float64": float64,
    "bfloat16": bfloat16,
    "int32": int32,
    "int64": int64,
    "int16": int16,
    "int8": int8,
    "uint8": uint8,
    "bool": bool,
}

# Mapping from numpy dtype to Genesis dtype
_numpy_to_dtype = {
    np.float32: float32,
    np.float16: float16,
    np.float64: float64,
    np.int32: int32,
    np.int64: int64,
    np.int16: int16,
    np.int8: int8,
    np.uint8: uint8,
    np.bool_: bool,
}


def get_dtype(obj):
    """
    Convert various dtype representations to Genesis DType object
    
    Args:
        obj: DType, str, numpy.dtype, or None
        
    Returns:
        DType object
    """
    if obj is None:
        return float32  # Default type
    elif isinstance(obj, DType):
        return obj
    elif isinstance(obj, str):
        dtype = _name_to_dtype.get(obj)
        if dtype is None:
            raise ValueError(f"Unsupported dtype string: {obj}")
        return dtype
    elif isinstance(obj, np.dtype):
        dtype = _numpy_to_dtype.get(obj.type)
        if dtype is None:
            raise ValueError(f"Unsupported numpy dtype: {obj}")
        return dtype
    elif isinstance(obj, type) and issubclass(obj, np.generic):
        # Handle np.float32 type input
        dtype = _numpy_to_dtype.get(obj)
        if dtype is None:
            raise ValueError(f"Unsupported numpy type: {obj}")
        return dtype
    elif str(type(obj)) == "<class 'torch.dtype'>":
        # Handle PyTorch dtype objects (torch.float32, etc.)
        torch_name = str(obj).split('.')[-1]  # Extract 'float32' from 'torch.float32'
        dtype = _name_to_dtype.get(torch_name)
        if dtype is not None:
            return dtype
        # Fallback to string representation
        return get_dtype(torch_name)
    else:
        raise ValueError(f"Cannot convert {type(obj)} to Genesis DType: {obj}")


def is_floating_point(dtype):
    """Check if dtype is floating point"""
    dtype = get_dtype(dtype)
    return dtype.is_floating_point


def is_integer(dtype):
    """Check if dtype is integer"""
    dtype = get_dtype(dtype)
    return not dtype.is_floating_point and dtype != bool


def infer_dtype_from_data(array):
    """Infer Genesis dtype from input data with PyTorch-like behavior.
    
    Args:
        array: Input data (scalar, list, numpy array, Tensor, etc.)
        
    Returns:
        DType: Inferred Genesis dtype
    """
    # Handle Tensor and NDArray objects (avoid circular imports)
    if hasattr(array, 'dtype') and hasattr(array, '__class__'):
        if array.__class__.__name__ == 'Tensor':
            return array.dtype
        elif hasattr(array, 'device'):
            return get_dtype(array.dtype)
    
    if isinstance(array, type(1)) or isinstance(array, type(1.0)) or isinstance(array, type(True)):
        # Handle scalar values (follow PyTorch defaults)
        if isinstance(array, type(True)):
            return globals()['bool']  # Genesis bool dtype
        elif isinstance(array, type(1)):
            return int64  # PyTorch default for Python int
        else:  # float
            return float32  # PyTorch default for Python float
    else:
        # Handle arrays and lists
        import numpy as np
        if not hasattr(array, 'dtype'):
            array = np.array(array)
        
        # Map numpy dtypes to Genesis dtypes
        dtype_map = {
            np.bool_: globals()['bool'],
            np.int8: int8,
            np.int16: int16, 
            np.int32: int32,
            np.int64: int64,
            np.uint8: uint8,
            np.float16: float16,
            np.float32: float32,
            np.float64: float32,  # Convert float64 to float32 by default like PyTorch
        }
        return dtype_map.get(array.dtype.type, float32)


# Common dtype list
all_dtypes = [float32, float16, float64, bfloat16, int32, int64, int16, int8, uint8, bool]
floating_dtypes = [dt for dt in all_dtypes if dt.is_floating_point]
integer_dtypes = [dt for dt in all_dtypes if is_integer(dt)]