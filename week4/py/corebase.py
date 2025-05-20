import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
# -----------------------------
# Feedforward Neural Network
# -----------------------------
class FeedforwardANN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, activation='relu'):
        super(FeedforwardANN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

        activations = {
            'relu': F.relu,
            'sigmoid': torch.sigmoid,
            'tanh': torch.tanh
        }
        if activation not in activations:
            raise ValueError(f"Unsupported activation function: {activation}")
        self.activation_fn = activations[activation]

    def forward(self, x):
        x = self.activation_fn(self.fc1(x))
        x = self.fc2(x)
        return x


# -----------------------------
# Generate Dummy Data (Binary classification)  OR
#  Training data-----------------------------
def generate_data(samples=1000):
    X = torch.randn(samples, 4)
    y = (X.sum(dim=1) > 0).long()
    return X, y


# -----------------------------
# Training the Model
# -----------------------------
def train(activation='relu', epochs=100, lr=0.01):
    print(f"Training ANN with activation: {activation}")
    input_dim = 4
    hidden_dim = 10
    output_dim = 2

    model = FeedforwardANN(input_dim, hidden_dim, output_dim, activation)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)

    X, y = generate_data()

    for epoch in range(epochs):
        outputs = model(X)
        loss = criterion(outputs, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            preds = torch.argmax(outputs, dim=1)
            acc = (preds == y).float().mean()
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}, Accuracy: {acc.item()*100:.2f}%")

    torch.save(model.state_dict(), f"ann_{activation}.pth")
    print(f"Model saved as ann_{activation}.pth")


# -----------------------------
# Run Training
# -----------------------------
if __name__ == "__main__":
    for act in ['relu', 'sigmoid', 'tanh']:
        train(activation=act)

# https://www.geeksforgeeks.org/pytorch-learn-with-examples/
