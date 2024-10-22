"""functional table."""
# Global operator table.
from .autograd import Tensor

from .backend_selection import array_api, NDArray

def triu(a: Tensor, k: int, device=None):
    return Tensor.make_const(array_api.triu(a.data, k))
