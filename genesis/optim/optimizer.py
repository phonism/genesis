"""Optimization module"""
import genesis
import numpy as np

class Optimizer:
    """
    optimizer
    """
    def __init__(self, params):
        self.params = params

    def step(self):
        raise NotImplementedError()

    def zero_grad(self):
        for p in self.params:
            p.grad = None

    def reset_grad(self):
        for p in self.params:
            p.grad = None

    def state_dict(self):
        state_dict = {}
        for name in self.__dict__:
            value = self.__dict__[name]
            if name != "params":
                state_dict[name] = value
        return state_dict

    def load_state_dict(self, state_dict):
        for name, value in self.__dict__.items():
            if name in state_dict:
                value = state_dict[name]


class SGD(Optimizer):
    """
    SGD
    """
    def __init__(self, params, lr=0.01, momentum=0.0, weight_decay=0.0):
        super().__init__(params)
        self.lr = lr
        self.momentum = momentum
        self.u = {}
        self.weight_decay = weight_decay
    
    def step(self):
        for idx, p in enumerate(self.params):
            grad = p.grad.detach() + self.weight_decay * p.detach()
            if idx not in self.u.keys():
                self.u[idx] = 0
            self.u[idx] = (self.momentum * self.u[idx] + (1 - self.momentum) * grad).detach()
            self.params[idx].data = p.data - self.lr * self.u[idx]


class Adam(Optimizer):
    """
    adam
    """
    def __init__(self, params, lr=0.01, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.0):
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
            grad = theta.grad.detach() + self.weight_decay * theta.detach()

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
            theta.data -= self.lr * m_next_hat.data / ((v_next_hat.data ** 0.5) + self.eps)


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
