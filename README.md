# A basic autograd implementation

Contains support for addition, subtraction, multiplication and matrix multiplication. This will be extended overtime and used for my ML projects. 

The code is centered around a Tensor class build on numpy and contains all the operations.

Currently supports:

* Addition (`+`)
* Subtraction (`-`)
* Negation (`-x`)
* Multiplication (`*`)
* Matrix multiplication (`matmul` / `@`)

This will be extended over time and used in my ML projects.

---

## Documentation

### 1) Installation

For now, keep the full code in a local file (for example: `main.py`) and make sure you have NumPy installed:

```bash
pip install numpy
```

---

### 2) Creating Tensors

Import the `Tensor` class and create tensors from Python numbers, lists, or NumPy arrays.

```python
import numpy as np
from main import Tensor

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
from tensor import Tensor

x = Tensor(2.0)
y = Tensor(5.0)

z = x * y + 3
z.backward()

print("z:", z.data)     # 13.0
print("dz/dx:", x.grad) # 5.0
print("dz/dy:", y.grad) # 2.0
```

Notes:

* This minimal version **does not support broadcasting** (except scalar with tensor).
* Shapes must match for elementwise ops, or one side must be a scalar.

---

### 4) Matrix multiplication

Use `matmul()` (or call it via the method directly). This minimal version only supports **2D @ 2D**.

```python
import numpy as np
from tensor import Tensor

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
* sets the gradient of the final tensor to `1` (or ones with the same shape)
* propagates gradients backward through stored `_backward()` functions

For meaningful training, you normally call `backward()` on a **scalar loss**.
Right now, this implementation allows calling `backward()` on non-scalar tensors too (it seeds the output gradient with `ones_like`), which can be useful for quick testing.

---

### 6) Current limitations (by design)

* No broadcasting (except scalar with tensor)
* No division, power, exp/log, relu/sigmoid, sum/mean, indexing, etc. yet
* `matmul` is limited to 2D matrices only
* No gradient reset helper yet (you’ll manually zero `grad` when needed)

---

Notes: the minimalNN.py file shows how the code can be used to create a basic artifical neural network. 

If you want, I can also add a tiny `examples.py` and a `zero_grad()` helper method while keeping everything minimal.
