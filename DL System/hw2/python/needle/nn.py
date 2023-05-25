"""The module.
"""
from typing import List, Callable, Any
from needle.autograd import Tensor
from needle import ops
import needle.init as init
import numpy as np


class Parameter(Tensor):
    """A special kind of tensor that represents parameters."""


def _unpack_params(value: object) -> List[Tensor]:
    if isinstance(value, Parameter):
        return [value]
    elif isinstance(value, Module):
        return value.parameters()
    elif isinstance(value, dict):
        params = []
        for k, v in value.items():
            params += _unpack_params(v)
        return params
    elif isinstance(value, (list, tuple)):
        params = []
        for v in value:
            params += _unpack_params(v)
        return params
    else:
        return []


def _child_modules(value: object) -> List["Module"]:
    if isinstance(value, Module):
        modules = [value]
        modules.extend(_child_modules(value.__dict__))
        return modules
    if isinstance(value, dict):
        modules = []
        for k, v in value.items():
            modules += _child_modules(v)
        return modules
    elif isinstance(value, (list, tuple)):
        modules = []
        for v in value:
            modules += _child_modules(v)
        return modules
    else:
        return []




class Module:
    def __init__(self):
        self.training = True

    def parameters(self) -> List[Tensor]:
        """Return the list of parameters in the module."""
        return _unpack_params(self.__dict__)

    def _children(self) -> List["Module"]:
        return _child_modules(self.__dict__)

    def eval(self):
        self.training = False
        for m in self._children():
            m.training = False

    def train(self):
        self.training = True
        for m in self._children():
            m.training = True

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class Identity(Module):
    def forward(self, x):
        return x


class Linear(Module):
    def __init__(self, in_features, out_features, bias=True, device=None, dtype="float32"):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.use_bias = bias
        self.weight = Parameter(init.kaiming_uniform(in_features, out_features),\
                        device=device, dtype=dtype, requires_grad=True)
        if self.use_bias:
            self.bias = Parameter(init.kaiming_uniform(out_features, 1).reshape((1,-1)),\
                        device=device, dtype=dtype, requires_grad=True)

    def forward(self, X: Tensor) -> Tensor:
        Y = X.matmul(self.weight)
        if self.use_bias:
            Y += self.bias.broadcast_to(Y.shape)
        return Y


class Flatten(Module):
    def forward(self, X):
        return X.reshape((X.shape[0], -1))


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        return ops.relu(x)


class Sequential(Module):
    def __init__(self, *modules):
        super().__init__()
        self.modules = modules

    def forward(self, x: Tensor) -> Tensor:
        for module in self.modules:
            x = module(x)
        
        return x

class SoftmaxLoss(Module):
    def forward(self, logits: Tensor, y: Tensor):
        softmax = ops.LogSumExp(axes=(1,))(logits)

        batch_size = logits.shape[0]
        num_classes = logits.shape[-1]
        one_hot = init.one_hot(num_classes, y)

        z = ops.summation(logits * one_hot, axes=(1, ))
        loss = softmax - z
        avg_loss = ops.summation(loss)/batch_size
        return avg_loss

class BatchNorm1d(Module):
    def __init__(self, dim, eps=1e-5, momentum=0.1, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.momentum = momentum
        self.weight = Parameter(init.ones(self.dim))
        self.bias = Parameter(init.zeros(self.dim))
        self.running_mean = init.zeros(self.dim)
        self.running_var = init.ones(self.dim)


    def forward(self, x: Tensor) -> Tensor:
        batch_size = x.shape[0]
        if self.training:
            x_sum = ops.summation(x, axes=(0, ))
            x_mean = ops.divide_scalar(x_sum, batch_size)
            x_var = ops.divide_scalar(ops.summation(ops.power_scalar((x - x_mean.reshape((1,-1)).broadcast_to(x.shape)), 2), axes=(0, )), batch_size)
            x_std = ops.power_scalar(ops.add_scalar(x_var.reshape((1,-1)), self.eps), 0.5)

            x_norm = ops.divide((x - x_mean.reshape((1,-1)).broadcast_to(x.shape)),(x_std.broadcast_to(x.shape)))
            x_norm = ops.multiply(x_norm, self.weight.broadcast_to(x.shape))
            x_norm = ops.add(x_norm, self.bias.broadcast_to(x.shape))

            self.running_mean = ((1 - self.momentum) * self.running_mean + self.momentum * x_mean)
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * x_var
        else:
            x_norm = ops.divide((x - self.running_mean.broadcast_to(x.shape)),(ops.power_scalar(ops.add_scalar(self.running_var.broadcast_to(x.shape), self.eps), 0.5).broadcast_to(x.shape)))
            x_norm = ops.multiply(x_norm, self.weight.broadcast_to(x.shape))
            x_norm = ops.add(x_norm, self.bias.broadcast_to(x.shape))
        
        return x_norm

class LayerNorm1d(Module):
    def __init__(self, dim, eps=1e-5, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = Parameter(init.ones(self.dim), device=device, dtype=dtype)
        self.bias = Parameter(init.zeros(self.dim), device=device, dtype=dtype)

    def forward(self, x: Tensor) -> Tensor:
        batch_size = x.shape[0]
        feature_size = x.shape[-1]

        x_sum = ops.summation(x, axes=(1, ))
        x_mean = ops.divide_scalar(x_sum, feature_size).reshape((-1,1))
        x_var = ops.divide_scalar(ops.summation(ops.power_scalar((x - x_mean.broadcast_to(x.shape)), 2), axes=(1, )), feature_size).reshape((-1,1))
        x_std = ops.power_scalar(ops.add_scalar(x_var, self.eps), 0.5)

        x_norm = ops.divide((x - x_mean.broadcast_to(x.shape)),(x_std.broadcast_to(x.shape)))
        x_norm = ops.multiply(x_norm, self.weight.broadcast_to(x.shape))
        x_norm = ops.add(x_norm, self.bias.broadcast_to(x.shape))
        return x_norm

class Dropout(Module):
    def __init__(self, p = 0.5):
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        if self.training:
            prob = init.randb(*x.shape, p=1 - self.p)
            res = ops.multiply(x, prob) / (1 - self.p)
        else:
            res = x
        
        return res


class Residual(Module):
    def __init__(self, fn: Module):
        super().__init__()
        self.fn = fn

    def forward(self, x: Tensor) -> Tensor:
        return self.fn(x) + x



