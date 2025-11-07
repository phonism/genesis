"""AdamW optimizer with fused and unfused implementations."""

import genesis
import triton
import triton.language as tl
from genesis.optim.optimizer import Optimizer


@triton.jit
def fused_adamw_kernel(
    # Pointers
    param_ptr,
    grad_ptr,
    m_ptr,
    v_ptr,
    # Hyperparameters
    lr,
    beta1,
    beta2,
    eps,
    weight_decay,
    step,
    # Meta parameters
    size,
    BLOCK_SIZE: tl.constexpr,
):
    """Fused AdamW kernel that performs all operations in one pass."""
    # Get block offset
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    # Mask for valid elements
    mask = offset < size

    # Load data
    param = tl.load(param_ptr + offset, mask=mask)
    grad = tl.load(grad_ptr + offset, mask=mask)
    m = tl.load(m_ptr + offset, mask=mask)
    v = tl.load(v_ptr + offset, mask=mask)

    # Apply weight decay
    if weight_decay > 0:
        param = param * (1.0 - lr * weight_decay)

    # Update biased first moment
    m = beta1 * m + (1.0 - beta1) * grad

    # Update biased second moment
    v = beta2 * v + (1.0 - beta2) * grad * grad

    # Bias correction using exp(log) formula: beta^step = exp(step * log(beta))
    bias_correction1 = 1.0 - tl.exp(step * tl.log(beta1))
    bias_correction2 = 1.0 - tl.exp(step * tl.log(beta2))

    m_hat = m / bias_correction1
    v_hat = v / bias_correction2

    # Update parameters
    param = param - lr * m_hat / (tl.sqrt(v_hat) + eps)

    # Store results
    tl.store(param_ptr + offset, param, mask=mask)
    tl.store(m_ptr + offset, m, mask=mask)
    tl.store(v_ptr + offset, v, mask=mask)


class AdamW(Optimizer):
    """AdamW optimizer with fused and unfused implementations.

    Args:
        params: Parameters to optimize
        lr: Learning rate (default: 0.001)
        beta1: First moment decay rate (default: 0.9)
        beta2: Second moment decay rate (default: 0.999)
        eps: Small constant for numerical stability (default: 1e-8)
        weight_decay: Weight decay coefficient (default: 0.01)
        fused: Use fused Triton kernel for better performance (default: True)

    Performance:
        - Fused: Single kernel per parameter (~12x fewer launches)
        - Unfused: Multiple kernels per parameter
    """
    def __init__(self, params, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.01, fused=True):
        super().__init__(params)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0
        self.m = {}
        self.v = {}
        self.fused = fused

    def step(self):
        """Perform optimization step."""
        if self.fused:
            self._step_fused()
        else:
            self._step_unfused()

    def _step_fused(self):
        """Fused AdamW update using single Triton kernel per parameter."""
        self.t += 1
        for theta_id, theta in enumerate(self.params):
            # Skip parameters without gradients
            if theta.grad is None:
                continue

            # Initialize momentum buffers if needed
            if theta_id not in self.m:
                self.m[theta_id] = genesis.zeros_like(theta)
            if theta_id not in self.v:
                self.v[theta_id] = genesis.zeros_like(theta)

            # Get gradient and state tensors
            grad = theta.grad
            m = self.m[theta_id]
            v = self.v[theta_id]

            # Ensure contiguity for optimal memory access
            theta_contiguous = theta.contiguous()
            grad_contiguous = grad.contiguous()
            m_contiguous = m.contiguous()
            v_contiguous = v.contiguous()

            size = theta.numel()

            # Launch fused kernel - pass Genesis tensors directly
            BLOCK_SIZE = 1024
            grid = lambda meta: (triton.cdiv(size, meta["BLOCK_SIZE"]),)

            fused_adamw_kernel[grid](
                theta_contiguous,
                grad_contiguous,
                m_contiguous,
                v_contiguous,
                self.lr,
                self.beta1,
                self.beta2,
                self.eps,
                self.weight_decay,
                float(self.t),
                size,
                BLOCK_SIZE=BLOCK_SIZE,
            )

    def _step_unfused(self):
        """Original unfused AdamW update (for comparison)."""
        self.t += 1
        for theta_id, theta in enumerate(self.params):
            # Skip parameters without gradients
            if theta.grad is None:
                continue

            grad = theta.grad.detach()
            if theta_id not in self.m:
                m_cur = (1 - self.beta1) * grad
            else:
                m_cur = self.m[theta_id] * self.beta1 + (1 - self.beta1) * grad

            if theta_id not in self.v:
                v_cur = (1 - self.beta2) * (grad ** 2)
            else:
                v_cur = self.v[theta_id] * self.beta2 + (1 - self.beta2) * (grad ** 2)

            self.m[theta_id] = m_cur.detach()
            self.v[theta_id] = v_cur.detach()
            m_next_hat = m_cur / (1 - self.beta1 ** self.t)
            v_next_hat = v_cur / (1 - self.beta2 ** self.t)
            # Use no_grad context to avoid affecting autograd computation graph
            with genesis.no_grad():
                # Direct tensor subtraction - modern approach, no .data needed
                theta -= self.lr * (m_next_hat / ((v_next_hat ** 0.5) + self.eps)
                        + self.weight_decay * theta)
