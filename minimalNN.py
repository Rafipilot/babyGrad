import main
import numpy as np

train_data_x = main.Tensor([[1, 1], [0, 0], [1, 0], [0, 1]])
train_data_y = main.Tensor([[1], [0], [0], [0]])

weights = main.Tensor(np.random.rand(2, 1))
bias = main.Tensor(float(np.random.rand()))  # scalar

lr = 0.1

def sigmoid(x):
    out = 1 / (1 + (-x).exp())
    return out

for epoch in range(200):
    for x, y in zip(train_data_x.data, train_data_y.data):
        x = main.Tensor([x])   # (1,2)
        y = main.Tensor([y])   # (1,1)

        pred = x.matmul(weights) + bias     # (1,1) + scalar
        pred = sigmoid(pred)                # (1,1)
        diff = (pred - y)                     # (1,1)
        loss = diff * diff

        weights.grad = np.zeros_like(weights.data)
        bias.grad = np.zeros_like(bias.data)

        loss.backward()

        weights.data = weights.data - lr * weights.grad
        bias.data = bias.data - lr * bias.grad

test_data = main.Tensor([[1, 1], [0, 0], [1, 0], [0, 1]])
preds = test_data.matmul(weights) + bias      # works because bias is scalar
preds = sigmoid(preds)
print("Predictions:")
print(preds.data)
print("bias:", bias.data)
