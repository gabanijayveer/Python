import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__() 
        self.pool = nn.MaxPool2d(2, 2)
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3)    
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3)   # Input: 3 channels (RGB), 16 filters
        self.fc1 = nn.Linear(32 * 6 * 6, 64)
        self.fc2 = nn.Linear(64, 10)  # Assume 10 classes

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # Conv1 -> ReLU -> Pool
        x = self.pool(F.relu(self.conv2(x)))  # Conv2 -> ReLU -> Pool
        x = x.view(-1, 32 * 6 * 6)            # Flatten
        x = F.relu(self.fc1(x))               # Fully Connected
        x = self.fc2(x)                       # Output

        return x
