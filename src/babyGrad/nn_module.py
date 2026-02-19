from . import grad
import numpy as np

class neural_network:
    def __init__(self, layer_sizes=[], activations=[]):
        self.layer_sizes = layer_sizes
        self.input_size = layer_sizes[0]
        self.output_size = layer_sizes[-1]

        self.weights = []
        self.biases = []
        self.activations = activations

        for i in range(len(layer_sizes) - 1):
            self.weights.append(grad.Tensor(np.random.rand(layer_sizes[i], layer_sizes[i+1])))
            self.biases.append(grad.Tensor(np.random.rand(layer_sizes[i+1])))
            # intuitively weights should be (input_size, output_size) and bias should be (output_size,) for each layer so the matrix multiplication and addition works out correctly 


    def sigmoid(self, x):
        out = 1 / (1 + (-x).exp())
        return out   

    def forward(self, x):
        out = grad.Tensor(x)
        if not isinstance(x, grad.Tensor):
            x = np.asarray(x, dtype=float)
            if x.ndim == 1:
                x = x.reshape(1, -1)   # (2,) -> (1, 2)
            out = grad.Tensor(x)
        else:
            out = x

        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            out = out.matmul(w)+b
            if self.activations[i] == "sigmoid":
                out = self.sigmoid(out)
            else:
                # assume sigmoid for now, we can add more activations later
                out = self.sigmoid(out)
        return out
    
    def backward(self, loss):
        loss.backward()

    def parameters(self):
        return self.weights + self.biases
    
    def zero_grad(self):
        for param in self.parameters():
            param.grad = np.zeros_like(param.data)

    def optimize(self, lr):
        for param in self.parameters():
            param.data -= lr * param.grad