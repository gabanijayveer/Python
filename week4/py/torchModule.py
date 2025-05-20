import torch
import torch.nn as nn
import torch.optim as optim
# Define the neural network architecture
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(2, 4)  # First layer: 2 input features to 4 hidden units
        self.fc2 = nn.Linear(4, 1)   # Second layer: 4 hidden units to 1 output
    def forward(self, x):
        x = torch.relu(self.fc1(x))  # Apply ReLU activation function
        x = self.fc2(x)               # Output layer (no activation function for regression)
        return x
# Training data for XOR problem
X_train = torch.tensor([[0.0, 0.0], 
                        [0.0, 1.0], 
                        [1.0, 0.0], 
                        [1.0, 1.0]]) 
y_train = torch.tensor([[0.0], 
                        [1.0], 
                        [1.0], 
                        [0.0]])  # Expected outputs
# Instantiate the model, define loss function and optimizer
model = SimpleNN()  
criterion = nn.MSELoss()  # Mean Squared Error Loss for regression
optimizer = optim.SGD(model.parameters(), lr=0.1)  # Stochastic Gradient Descent
# Training loop
num_epochs = 1000
for epoch in range(num_epochs):
    # Forward pass
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    # Backward pass and optimization
    optimizer.zero_grad()  # Clear previous gradients
    loss.backward()        # Compute gradients
    optimizer.step()       # Update weights
    # Print loss every 100 epochs
    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')
# Test the model
with torch.no_grad():
    test_output = model(X_train)
    print("Predicted outputs:")
    print(test_output)
