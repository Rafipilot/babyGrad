import main
import numpy as np


train_data_x = main.Tensor([[1, 1], [0, 0], [1, 0], [0, 1]])

train_data_y = main.Tensor([[1], [0], [0], [0]])

weights = main.Tensor(np.random.rand(2, 1))

for epoch in range(20):
    for x, y in zip(train_data_x.data, train_data_y.data):
        x = main.Tensor([x])
        y = main.Tensor([y])

        pred = x.matmul(weights)
        loss = (pred - y) * (pred - y)
        print(f"Epoch {epoch}, Loss: {loss.data}")

        loss.backward()
        weights.data = weights.data - 0.1 * weights.grad

test_data = main.Tensor([[1, 1], [0, 0], [1, 0], [0, 1]])
predictions = []
for x in test_data.data:
    x = main.Tensor([x])
    prediction = x.matmul(weights)
    predictions.append(prediction)
print("Predictions:")
print(predictions)