import numpy as np

# Sigmoid activation and its derivative
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

# Inputs and outputs
X = np.array([[0.5, 0.2]])
y = np.array([[1]])


# Initialize weights and biases
W1 = np.random.rand(2, 2)
b1 = np.zeros((1, 2))
W2 = np.random.rand(2, 1)
b2 = np.zeros((1, 1))

# Forward pass
z1 = np.dot(X, W1) + b1
a1 = sigmoid(z1)
z2 = np.dot(a1, W2) + b2
a2 = sigmoid(z2)  # Predicted output

# Loss (MSE)
loss = np.mean((y - a2) ** 2)

# Backward pass (Backpropagation)
d_loss_a2 = -(y - a2)
d_a2_z2 = (z2)
d_z2_W2 = a1

d_loss_W2 = np.dot(d_z2_W2.T, d_loss_a2 * d_a2_z2)
d_loss_b2 = d_loss_a2 * d_a2_z2

d_a1 = np.dot(d_loss_a2 * d_a2_z2, W2.T)
d_z1 = d_a1 * sigmoid_derivative(z1)
d_loss_W1 = np.dot(X.T, d_z1)
d_loss_b1 = d_z1

# Gradient Descent update
learning_rate = 0.1
W2 -= learning_rate * d_loss_W2
b2 -= learning_rate * d_loss_b2
W1 -= learning_rate * d_loss_W1
b1 -= learning_rate * d_loss_b1

print("Updated weights and biases:")
print("W1:", W1)
print("b1:", b1)
print("W2:", W2)
print("b2:", b2)
