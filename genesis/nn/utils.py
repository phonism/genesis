import genesis

def norm(data, p):
    powered_elements = data ** p
    sum_of_powers = genesis.sum(powered_elements)
    norm_value = sum_of_powers ** (1 / p)
    return norm_value


def clip_grad_norm_(parameters, max_norm, norm_type=2):
    if isinstance(parameters, genesis.Tensor):
        parameters = [parameters]
    else:
        parameters = list(parameters)

    total_norm = 0.0
    norm_type = float(norm_type)

    for param in parameters:
        if param.grad is not None:
            param_norm = norm(param.grad.data)
            total_norm += param_norm ** norm_type

    total_norm = total_norm ** (1. / norm_type)
    clip_coef = max_norm / (total_norm + 1e-6)

    if clip_coef < 1:
        for param in parameters:
            if param.grad is not None:
                param.grad.data.mul *= clip_coef
    return total_norm
