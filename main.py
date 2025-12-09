import numpy as np


class Tensor:
    def __init__(self, data, _children=(), _op=""):
        self.data = data
        self.grad = np.zeros_like(self.data)
        self._prev = set(_children)
        self._op = _op
        self._backward = lambda: None

    def __repr__(self):
        return f"Tensor(data={self.data}, grad={self.grad}, op={self._op})"
    
    def __add__(self, other):#

        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data + other.data, (self, other), _op="+")

        def _backward():
            self.grad += out.grad
            other.grad = out.grad

        out._backward = _backward
        return out
    
    def __radd__(self, other):
        return self + other
    
    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data + other.data, (self, other), _op="+")

        def _backward():
            self.grad += out.grad * other.data
            other.grad += out.grad * self.data

        out._backward = _backward
        return out
    
    def matmul(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data @ other.data, (self, other), _op="+")

        def _backward():
            self.grad += out.grad @ other.data.T
            other.grad += out.grad.T @ self.data

        out._backward = _backward
        return out

    def backward(self):
        topo = []
        visited = set()

        def build(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build(child)
                topo.append(v)

        build(self)

        self.grad = np.ones_like(self.data)

        for v in reversed(topo):
            v._backward()

