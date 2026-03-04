# A basic autograd implementation

Contains support for core arithmetic operations. This will be extended overtime and used for my ML projects. 

The code is centered around a Tensor class build on numpy and contains all the operations.

Supported core operations:
* Addition (`+`)
* Subtraction (`-`)
* Negation (`-x`)
* Multiplication (`*`)
* Division (`/`)
* Matrix multiplication (`matmul` / `@`) for 2D tensors
* Elementwise `exp()`

This will be extended over time.

---

## babyGrad.grad

### 1) Installation
Install the package with pip!
```bash
  pip install git+https://github.com/Rafipilot/babyGrad
```

---

### 2) Creating Tensors

Import the `Tensor` class and create tensors from Python numbers, lists, or NumPy arrays.

```python
import numpy as np
from babyGrad.grad import Tensor

a = Tensor(3.0)                      # scalar
b = Tensor([1, 2, 3])                # vector
c = Tensor(np.array([[1, 2], [3, 4]]))  # matrix
```

Each `Tensor` has:

* `data`: the numeric value (stored as a NumPy array)
* `grad`: gradient w.r.t. some scalar loss (same shape as `data`)
* internal graph fields used for backprop

---

### 3) Basic operations

#### Addition, subtraction, multiplication

```python
from babyGrad.grad import Tensor

x = Tensor(2.0)
y = Tensor(5.0)

z = x * y + 3
z.backward()

print("z:", z.data)     # 13.0
print("dz/dx:", x.grad) # 5.0
print("dz/dy:", y.grad) # 2.0
```
---

### 4) Matrix multiplication

Use `matmul()` (or call it via the method directly). This minimal version only supports **2D @ 2D**.

```python
import numpy as np
from babyGrad.grad import Tensor

A = Tensor(np.array([[1.0, 2.0],
                     [3.0, 4.0]]))

B = Tensor(np.array([[5.0, 6.0],
                     [7.0, 8.0]]))

C = A.matmul(B)
C.backward()  # sets C.grad to ones, then backprops

print("C:\n", C.data)
print("dC/dA:\n", A.grad)
print("dC/dB:\n", B.grad)
```

---

### 5) Calling `backward()`

`backward()`:

* builds a topological ordering of the computation graph
* sets the gradient of the final tensor to `1` (dx/dx is 1!)
* propagates gradients backward through stored `_backward()` functions

For meaningful training, you normally call `backward()` on a **scalar loss**.
Right now, this implementation allows calling `backward()` on non-scalar tensors too (it seeds the output gradient with `ones_like`), which can be useful for quick testing.

---

### 6) Current limitations (by design)

* No broadcasting (except scalar with tensor)
* `matmul` is limited to 2D matrices only

---

## babyGrad.nn_module
The package also includes a nn.module scripts which builds support for creating artifical neural networks (ANNs) easily ontop of the babygrad framework.

### 1) Initialize neural_network()
```python
from babyGrad.nn_module import neural_network
nn = neural_network(layer_sizes=[2, 4, 1], activations=["sigmoid", "sigmoid"]) # Example values
```
The neural network constructor accepts `layer_size`s (which build the network), and `activations` which describe which activation should be used at the i+1 layer.

### 2) forward()
```python
pred = nn.forward(xi)
```
The neural_network.forward accepts a Tensor and applies matrix multiplication through the network and returns the output of the size of the output layer

### 3) zero_grad(), backwards() and optimize()
```python
        neural_net.zero_grad()
        neural_net.backward(loss)
        neural_net.optimize(lr)
```
These are pretty self-explanatory 
  - zero grad sets the stored gradient of all the tensors to 0.
  - backward(loss) just invokes babyGrad.Tensor.backward()
  - optimize(lr) optimizes all of the parameters in the neural network by adjusting them by `-lr*grad`

### 4) Putting it all together: A tiny neural network!
```python
import numpy as np
from babyGrad.grad import Tensor
from babyGrad.nn_module import neural_network

# AND gate
x = Tensor([[1, 1],
            [0, 0],
            [1, 0],
            [0, 1]])
y = Tensor([[1],
            [0],
            [0],
            [0]])

nn = neural_network(layer_sizes=[2, 2, 1],
                    activations=["sigmoid", "sigmoid"])

lr = 0.1
epochs = 2000

for epoch in range(epochs):
    for xi, yi in zip(x.data, y.data):
        xi_batch = xi.reshape(1, -1)
        yi_batch = yi.reshape(1, -1)

        pred = nn.forward(xi_batch)
        target = Tensor(yi_batch)

        loss = (pred - target) * (pred - target)

        nn.zero_grad()
        nn.backward(loss)
        nn.optimize(lr)

# Check predictions
for xi, yi in zip(x.data, y.data):
    pred = nn.forward(xi.reshape(1, -1))
    print("Input:", xi, "Pred:", pred.data, "Target:", yi)
```

## Future work
1) Build out the nn_module.py to support my types of neural networks (convolutional, etc) and features (dropout. different optimizers, etc)
2) Integrate GPU offloading

Notes: This is still very early stage so expect things to break!



