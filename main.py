import numpy as np


class Tensor:
    def __init__(self, data, _children=(), _op=""):
        self.data = np.asarray(data, dtype=float)
        self.grad = np.zeros_like(self.data, dtype=float)
        self._prev = set(_children)
        self._op = _op
        self._backward = lambda: None

    def __repr__(self):
        return f"Tensor(data={self.data}, grad={self.grad}, op={self._op})"

    def _check_shapes_no_broadcast(self, other):
        a_shape = self.data.shape
        b_shape = other.data.shape
        if a_shape == b_shape:
            return
        if a_shape == () or b_shape == ():
            return  # allow scalar with tensor
        raise ValueError(f"Broadcasting not supported: {a_shape} vs {b_shape}")

    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        self._check_shapes_no_broadcast(other)

        out = Tensor(self.data + other.data, (self, other), _op="+")

        def _backward():
            if self.data.shape == ():
                self.grad += out.grad.sum()
            else:
                self.grad += out.grad

            if other.data.shape == ():
                other.grad += out.grad.sum()
            else:
                other.grad += out.grad

        out._backward = _backward
        return out

    def __radd__(self, other):
        return self + other
    
    def __neg__(self):
        out = Tensor(-self.data, (self,), _op="neg")

        def _backward():
            self.grad += -out.grad

        out._backward = _backward
        return out
    
    def __sub__(self, other):
        return self + (-other)

    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        self._check_shapes_no_broadcast(other)

        out = Tensor(self.data * other.data, (self, other), _op="*")

        def _backward():
            dself = out.grad * other.data
            dother = out.grad * self.data

            if self.data.shape == ():
                self.grad += dself.sum()
            else:
                self.grad += dself

            if other.data.shape == ():
                other.grad += dother.sum()
            else:
                other.grad += dother

        out._backward = _backward
        return out

    def __rmul__(self, other):
        return self * other

    def matmul(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)

        if self.data.ndim != 2 or other.data.ndim != 2:
            raise ValueError("matmul only supports 2D matrices in this minimal version")

        out = Tensor(self.data @ other.data, (self, other), _op="@")

        def _backward():
            self.grad += out.grad @ other.data.T
            other.grad += self.data.T @ out.grad

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

        self.grad = np.ones_like(self.data, dtype=float)

        for v in reversed(topo):
            v._backward()
