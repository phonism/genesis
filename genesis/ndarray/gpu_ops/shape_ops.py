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