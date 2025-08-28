"""
Native NCCL ctypes bindings for Genesis distributed training.

This module provides direct ctypes bindings to NVIDIA NCCL library,
avoiding any third-party dependencies like CuPy or PyNCCL.
"""

import ctypes
import ctypes.util
import os
import platform
from dataclasses import dataclass
from typing import List, Optional, Union
import genesis
from .comm import ReduceOp


# =============================================================================
# NCCL Basic Types and Constants
# =============================================================================

# NCCL result codes
ncclResult_t = ctypes.c_int
ncclSuccess = 0
ncclUnhandledCudaError = 1
ncclSystemError = 2
ncclInternalError = 3
ncclInvalidArgument = 4
ncclInvalidUsage = 5

# NCCL data types  
ncclDataType_t = ctypes.c_int
ncclInt8 = 0
ncclChar = 0
ncclUint8 = 1
ncclInt32 = 2
ncclInt = 2
ncclUint32 = 3
ncclInt64 = 4
ncclUint64 = 5
ncclFloat16 = 6
ncclHalf = 6
ncclFloat32 = 7
ncclFloat = 7
ncclFloat64 = 8
ncclDouble = 8
ncclBfloat16 = 9

# NCCL reduction operations
ncclRedOp_t = ctypes.c_int
ncclSum = 0
ncclProd = 1
ncclMax = 2
ncclMin = 3

# NCCL communicator and unique ID types
ncclComm_t = ctypes.c_void_p
cudaStream_t = ctypes.c_void_p

class ncclUniqueId(ctypes.Structure):
    """NCCL unique identifier structure."""
    _fields_ = [("internal", ctypes.c_byte * 128)]


# =============================================================================
# Function Signature Definitions (vLLM-style dataclass approach)
# =============================================================================

@dataclass
class NCCLFunction:
    """Defines an NCCL function with its signature."""
    name: str
    restype: type
    argtypes: List[type]


# Define all NCCL functions we need
NCCL_FUNCTIONS = [
    # Initialization and cleanup
    NCCLFunction("ncclGetVersion", ncclResult_t, [ctypes.POINTER(ctypes.c_int)]),
    NCCLFunction("ncclGetUniqueId", ncclResult_t, [ctypes.POINTER(ncclUniqueId)]),
    NCCLFunction("ncclCommInitRank", ncclResult_t, 
                [ctypes.POINTER(ncclComm_t), ctypes.c_int, ncclUniqueId, ctypes.c_int]),
    NCCLFunction("ncclCommDestroy", ncclResult_t, [ncclComm_t]),
    NCCLFunction("ncclCommGetAsyncError", ncclResult_t, [ncclComm_t, ctypes.POINTER(ncclResult_t)]),
    NCCLFunction("ncclCommCount", ncclResult_t, [ncclComm_t, ctypes.POINTER(ctypes.c_int)]),
    NCCLFunction("ncclCommUserRank", ncclResult_t, [ncclComm_t, ctypes.POINTER(ctypes.c_int)]),
    
    # Collective operations
    NCCLFunction("ncclAllReduce", ncclResult_t,
                [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, 
                 ncclDataType_t, ncclRedOp_t, ncclComm_t, cudaStream_t]),
    NCCLFunction("ncclBroadcast", ncclResult_t,
                [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t,
                 ncclDataType_t, ctypes.c_int, ncclComm_t, cudaStream_t]),
    NCCLFunction("ncclReduce", ncclResult_t,
                [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t,
                 ncclDataType_t, ncclRedOp_t, ctypes.c_int, ncclComm_t, cudaStream_t]),
    NCCLFunction("ncclAllGather", ncclResult_t,
                [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t,
                 ncclDataType_t, ncclComm_t, cudaStream_t]),
    NCCLFunction("ncclReduceScatter", ncclResult_t,
                [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t,
                 ncclDataType_t, ncclRedOp_t, ncclComm_t, cudaStream_t]),
    
    # Point-to-point communication
    NCCLFunction("ncclSend", ncclResult_t,
                [ctypes.c_void_p, ctypes.c_size_t, ncclDataType_t,
                 ctypes.c_int, ncclComm_t, cudaStream_t]),
    NCCLFunction("ncclRecv", ncclResult_t,
                [ctypes.c_void_p, ctypes.c_size_t, ncclDataType_t,
                 ctypes.c_int, ncclComm_t, cudaStream_t]),
    
    # Group operations
    NCCLFunction("ncclGroupStart", ncclResult_t, []),
    NCCLFunction("ncclGroupEnd", ncclResult_t, []),
    
    # Error handling
    NCCLFunction("ncclGetErrorString", ctypes.c_char_p, [ncclResult_t]),
]


# =============================================================================
# NCCL Library Loader
# =============================================================================

class NCCLLibrary:
    """Manages loading and accessing NCCL library functions."""
    
    def __init__(self, library_path: Optional[str] = None):
        self.library = None
        self.functions = {}
        self._load_library(library_path)
        self._setup_functions()
        
    def _load_library(self, library_path: Optional[str]):
        """Load NCCL shared library."""
        if library_path is None:
            # Try to find NCCL library automatically
            possible_names = [
                "libnccl.so.2",    # Common NCCL 2.x
                "libnccl.so",      # Generic
                "libnccl.dylib",   # macOS (though NCCL doesn't support macOS)
            ]
            
            # Check environment variable first
            env_path = os.environ.get('GENESIS_NCCL_PATH')
            if env_path:
                possible_names.insert(0, env_path)
                
            # Try to find library
            for name in possible_names:
                try:
                    if os.path.exists(name):
                        library_path = name
                        break
                    else:
                        # Try system search
                        found = ctypes.util.find_library(name.replace('lib', '').replace('.so', ''))
                        if found:
                            library_path = found
                            break
                except:
                    continue
                    
            if library_path is None:
                raise RuntimeError("Could not find NCCL library. Set GENESIS_NCCL_PATH environment variable.")
                
        try:
            self.library = ctypes.CDLL(library_path)
            print(f"Loaded NCCL library: {library_path}")
        except OSError as e:
            raise RuntimeError(f"Failed to load NCCL library {library_path}: {e}")
            
    def _setup_functions(self):
        """Setup all NCCL function signatures."""
        for func_def in NCCL_FUNCTIONS:
            try:
                func = getattr(self.library, func_def.name)
                func.restype = func_def.restype
                func.argtypes = func_def.argtypes
                self.functions[func_def.name] = func
            except AttributeError:
                print(f"Warning: Function {func_def.name} not found in NCCL library")
                
    def __getattr__(self, name: str):
        """Allow direct access to NCCL functions."""
        if name in self.functions:
            return self.functions[name]
        raise AttributeError(f"NCCL function '{name}' not available")


# =============================================================================
# Genesis Type Conversion Utilities
# =============================================================================

def genesis_dtype_to_nccl(dtype) -> int:
    """Convert Genesis dtype to NCCL data type."""
    mapping = {
        genesis.float32: ncclFloat32,
        genesis.float16: ncclFloat16, 
        genesis.bfloat16: ncclBfloat16,
        genesis.float64: ncclFloat64,
        genesis.int8: ncclInt8,
        genesis.int32: ncclInt32,
        genesis.int64: ncclInt64,
        genesis.uint8: ncclUint8,
    }
    
    nccl_type = mapping.get(dtype)
    if nccl_type is None:
        raise ValueError(f"Unsupported dtype for NCCL: {dtype}")
    return nccl_type


def genesis_reduce_op_to_nccl(op) -> int:
    """Convert Genesis reduce operation to NCCL op."""
    mapping = {
        ReduceOp.SUM: ncclSum,
        ReduceOp.PRODUCT: ncclProd,
        ReduceOp.MAX: ncclMax,
        ReduceOp.MIN: ncclMin,
    }
    
    nccl_op = mapping.get(op)
    if nccl_op is None:
        raise ValueError(f"Unsupported reduce operation for NCCL: {op}")
    return nccl_op


def get_tensor_ptr(tensor: genesis.Tensor) -> ctypes.c_void_p:
    """Get ctypes pointer from Genesis tensor."""
    if hasattr(tensor.data, 'ptr'):
        # GPU tensor
        return ctypes.c_void_p(tensor.data.ptr)
    else:
        # CPU tensor - this shouldn't happen in NCCL context
        raise ValueError("NCCL requires GPU tensors")


def check_nccl_result(result: int, operation: str = "NCCL operation"):
    """Check NCCL result code and raise exception if error."""
    if result != ncclSuccess:
        # Try to get error string if available
        try:
            _nccl_lib = get_nccl_library()
            error_str = _nccl_lib.ncclGetErrorString(result)
            error_msg = error_str.decode('utf-8') if error_str else f"Error code {result}"
        except:
            error_msg = f"Error code {result}"
            
        raise RuntimeError(f"{operation} failed: {error_msg}")


# =============================================================================
# Global NCCL Library Instance
# =============================================================================

_nccl_lib: Optional[NCCLLibrary] = None


def get_nccl_library() -> NCCLLibrary:
    """Get the global NCCL library instance."""
    global _nccl_lib
    if _nccl_lib is None:
        _nccl_lib = NCCLLibrary()
    return _nccl_lib


def is_nccl_available() -> bool:
    """Check if NCCL library is available."""
    try:
        get_nccl_library()
        return True
    except:
        return False


# =============================================================================
# High-level NCCL Operations
# =============================================================================

class NCCLCommunicator:
    """High-level wrapper for NCCL communicator."""
    
    def __init__(self):
        self.comm = ncclComm_t()
        self.world_size = 0
        self.rank = 0
        self.initialized = False
        
    def init_rank(self, world_size: int, unique_id: ncclUniqueId, rank: int):
        """Initialize communicator with rank."""
        nccl_lib = get_nccl_library()
        
        result = nccl_lib.ncclCommInitRank(
            ctypes.byref(self.comm),
            world_size,
            unique_id,
            rank
        )
        
        check_nccl_result(result, "ncclCommInitRank")
        
        self.world_size = world_size
        self.rank = rank
        self.initialized = True
        
    def destroy(self):
        """Destroy communicator."""
        if self.initialized:
            nccl_lib = get_nccl_library()
            result = nccl_lib.ncclCommDestroy(self.comm)
            check_nccl_result(result, "ncclCommDestroy")
            self.initialized = False
            
    def all_reduce(self, tensor: genesis.Tensor, op: int, stream: int = 0):
        """Perform all-reduce operation."""
        if not self.initialized:
            raise RuntimeError("Communicator not initialized")
            
        nccl_lib = get_nccl_library()
        
        ptr = get_tensor_ptr(tensor)
        dtype = genesis_dtype_to_nccl(tensor.dtype)
        count = tensor.data.size
        
        result = nccl_lib.ncclAllReduce(
            ptr,                           # sendbuff
            ptr,                           # recvbuff (in-place)
            count,                         # count
            dtype,                         # datatype
            op,                            # op
            self.comm,                     # comm
            ctypes.c_void_p(stream)        # stream
        )
        
        check_nccl_result(result, "ncclAllReduce")


def generate_unique_id() -> ncclUniqueId:
    """Generate NCCL unique ID."""
    nccl_lib = get_nccl_library()
    unique_id = ncclUniqueId()
    
    result = nccl_lib.ncclGetUniqueId(ctypes.byref(unique_id))
    check_nccl_result(result, "ncclGetUniqueId")
    
    return unique_id


# =============================================================================
# Debugging and Utility Functions
# =============================================================================

def get_nccl_version() -> str:
    """Get NCCL version."""
    nccl_lib = get_nccl_library()
    version = ctypes.c_int()
    
    result = nccl_lib.ncclGetVersion(ctypes.byref(version))
    check_nccl_result(result, "ncclGetVersion")
    
    # NCCL version is encoded as MAJOR * 10000 + MINOR * 100 + PATCH
    major = version.value // 10000
    minor = (version.value % 10000) // 100
    patch = version.value % 100
    
    return f"{major}.{minor}.{patch}"


if __name__ == "__main__":
    # Test basic functionality
    try:
        print(f"NCCL version: {get_nccl_version()}")
        print("NCCL library loaded successfully!")
        
        # Test unique ID generation
        uid = generate_unique_id()
        print(f"Generated unique ID: {uid.internal[:16]}...")  # Show first 16 bytes
        
    except Exception as e:
        print(f"NCCL test failed: {e}")