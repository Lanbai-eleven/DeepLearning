"""Optimization module"""
import needle as ndl
import numpy as np


class Optimizer:
    def __init__(self, params):
        self.params = params

    def step(self):
        raise NotImplementedError()

    def reset_grad(self):
        for p in self.params:
            p.grad = None


class SGD(Optimizer):
    def __init__(self, params, lr=0.01, momentum=0.0, weight_decay=0.0):
        super().__init__(params)
        self.lr = lr
        self.momentum = momentum
        self.u = {}
        self.weight_decay = weight_decay

    def step(self):
        for param in self.params:
            if param.grad is None:
                continue
            
            u = self.momentum * self.u.get(param, 0) + (1-self.momentum) * (param.grad.data + self.weight_decay*param.data)
            # print("param.grad.dtype", param.grad.dtype)
            # print("param.dtype", param.dtype)
            # print("u:", u.dtype)
            u = ndl.Tensor(u, dtype=param.dtype)
            self.u[param] = u
            param.data = param.data - self.lr * u

class Adam(Optimizer):
    def __init__(
        self,
        params,
        lr=0.01,
        beta1=0.9,
        beta2=0.999,
        eps=1e-8,
        weight_decay=0.0,
    ):
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
        for param in self.params:
            if param.grad is None:
                continue

            m = self.m.get(param, 0)
            v = self.v.get(param, 0)
            grad = param.grad.data + self.weight_decay * param.data
            grad = ndl.Tensor(grad, dtype='float32')

            m = self.beta1 * m + (1 - self.beta1) * grad.data
            v = self.beta2 * v + (1 - self.beta2) * grad.data**2

            m_hat = m / (1 - self.beta1 ** self.t)
            v_hat = v / (1 - self.beta2 ** self.t)

            param.data -= self.lr * m_hat / ((v_hat**0.5) + self.eps)
            self.m[param] = m
            self.v[param] = v
