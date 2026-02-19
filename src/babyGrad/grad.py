
import numpy as np

def _reduce_to_shape(grad, shape):
    grad = np.asarray(grad, dtype=float)
    if grad.shape == shape:
        return grad
    if shape == ():
        return np.array(grad.sum(), dtype=float)
    while grad.ndim > len(shape):
        grad = grad.sum(axis=0)

    return grad.reshape(shape)

class Tensor:
    def __init__(self, data, _children=(), _op=""):
        self.data = np.asarray(data, dtype=float)   # no extra reshape here
        self.grad = np.zeros_like(self.data, dtype=float)
        self._prev = set(_children)
        self._op = _op
        self._backward = lambda: None

    def __repr__(self):
        return f"Tensor(data={self.data}, grad={self.grad}, op={self._op})"


    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)

        out = Tensor(self.data + other.data, (self, other), _op="+")

        def _backward():

            self.grad += _reduce_to_shape(out.grad, self.data.shape) # out here is 
            other.grad += _reduce_to_shape(out.grad, other.data.shape)

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

        out = Tensor(self.data * other.data, (self, other), _op="*")

        def _backward():
            dself = out.grad * other.data # this is d self/ d out
            dother = out.grad * self.data

            self.grad += _reduce_to_shape(dself, self.data.shape)

            other.grad += dother

        out._backward = _backward
        return out
    
    def inverse(self):
        out = Tensor(1 / self.data, (self,), _op="inv")

        def _backward():
            dself = -1 / (self.data * self.data) * out.grad
            self.grad += _reduce_to_shape(dself, self.data.shape)

        out._backward = _backward
        return out
    
    def __truediv__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        return self * other.inverse()
    
    def __rtruediv__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        return other * self.inverse()


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
    
    def exp(self):
        out = Tensor(np.exp(self.data), (self,), _op="exp")

        def _backward():
            # when we differentiate exp, we get exp itself, so we can use out.data here
            self.grad += out.data * out.grad

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


if __name__ == "__main__":
    a = Tensor(1)
    b = Tensor(2)
    c = a + b

    d = c * 2

    print("Gradients before backward: ", a.grad, b.grad, c.grad)

    d.backward()

    print("Gradients after backward: ", a.grad, b.grad, c.grad)