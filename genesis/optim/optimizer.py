"""Optimization algorithms for gradient-based training.

This module implements common optimization algorithms including SGD, Adam, and AdamW
with support for momentum, weight decay, and adaptive learning rates.
"""

import genesis
import numpy as np

class Optimizer:
    """Base class for all optimizers.
    
    Provides common functionality for parameter updates, gradient zeroing,
    and state management across different optimization algorithms.
    """
    
    def __init__(self, params):
        """Initialize optimizer with parameters to optimize.
        
        Args:
            params: Iterable of parameters to optimize
        """
        self.params = params

    def step(self):
        """Perform a single optimization step (parameter update)."""
        raise NotImplementedError()

    def zero_grad(self):
        """Zero gradients of all optimized parameters."""
        for p in self.params:
            p.grad = None

    def reset_grad(self):
        """Reset gradients of all optimized parameters (alias for zero_grad)."""
        for p in self.params:
            p.grad = None

    def state_dict(self):
        """Return optimizer state as a dictionary."""
        state_dict = {}
        for name in self.__dict__:
            value = self.__dict__[name]
            if name != "params":
                state_dict[name] = value
        return state_dict

    def load_state_dict(self, state_dict):
        """Load optimizer state from dictionary."""
        for name, value in self.__dict__.items():
            if name in state_dict:
                value = state_dict[name]


class SGD(Optimizer):
    """Stochastic Gradient Descent optimizer with momentum and weight decay.
    
    Implements the classical SGD algorithm with optional momentum for improved
    convergence and weight decay for regularization.
    """
    
    def __init__(self, params, lr=0.01, momentum=0.0, weight_decay=0.0):
        """Initialize SGD optimizer.
        
        Args:
            params: Parameters to optimize
            lr: Learning rate
            momentum: Momentum factor (0 disables momentum)
            weight_decay: Weight decay (L2 penalty) factor
        """
        super().__init__(params)
        self.lr = lr
        self.momentum = momentum
        self.u = {}  # Momentum buffer
        self.weight_decay = weight_decay
    
    def step(self):
        """Perform SGD parameter update with momentum."""
        for idx, p in enumerate(self.params):
            grad = p.grad.detach() + self.weight_decay * p.detach()
            if idx not in self.u.keys():
                self.u[idx] = 0
            # Update momentum buffer and apply update
            self.u[idx] = (self.momentum * self.u[idx] + (1 - self.momentum) * grad).detach()
            # Update parameter in-place using subtract operation
            updated_param = p - self.lr * self.u[idx]
            self.params[idx] = updated_param


class Adam(Optimizer):
    """Adam optimizer with adaptive learning rates.
    
    Implements the Adam algorithm which computes adaptive learning rates
    for each parameter using estimates of first and second moments of gradients.
    """
    
    def __init__(self, params, lr=0.01, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.0):
        """Initialize Adam optimizer.
        
        Args:
            params: Parameters to optimize
            lr: Learning rate
            beta1: Coefficient for first moment estimate
            beta2: Coefficient for second moment estimate  
            eps: Small constant for numerical stability
            weight_decay: Weight decay (L2 penalty) factor
        """
        super().__init__(params)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0       # Time step counter
        self.m = {}      # First moment estimates
        self.v = {}      # Second moment estimates

    def step(self):
        """Perform Adam parameter update with bias correction."""
        self.t += 1
        for theta_id, theta in enumerate(self.params):
            grad = theta.grad.detach() + self.weight_decay * theta.detach()

            # Initialize or update first moment estimate
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
            # Update parameter using tensor operations, not .data attribute
            update = self.lr * m_next_hat / ((v_next_hat ** 0.5) + self.eps)
            updated_param = theta - update
            self.params[theta_id] = updated_param


class AdamW(Optimizer):
    """
    adamw
    """
    def __init__(self, params, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.01):
        super().__init__(params)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0
        self.m = {}
        self.v = {}

    def step(self):
        self.t += 1
        for theta_id, theta in enumerate(self.params):
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
            theta.data -= self.lr * (m_next_hat.data / ((v_next_hat.data ** 0.5) + self.eps) 
                    + self.weight_decay * theta.data)
