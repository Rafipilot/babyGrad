
import babyGrad.grad as grad
from babyGrad.nn_module import neural_network

train_data_x = grad.Tensor([[1, 1], [0, 0], [1, 0], [0, 1]])
train_data_y = grad.Tensor([[1], [0], [0], [0]])

neural_net = neural_network(layer_sizes=[2, 4, 1], activations=["sigmoid", "sigmoid"])

lr = 0.2
for epoch in range(2000):
    for xi, yi in zip(train_data_x.data, train_data_y.data):
        pred = neural_net.forward(xi)
        loss = (pred - yi) * (pred - yi)
        neural_net.zero_grad()
        neural_net.backward(loss)
        neural_net.optimize(lr)

for xi, yi in zip(train_data_x.data, train_data_y.data):
    pred = neural_net.forward(xi)
    print(f"Input: {xi}, Predicted: {pred.data}, Actual: {yi}")


