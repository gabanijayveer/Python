# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim

# from torchvision import datasets, transforms
# from torch.utils.data import DataLoader

# # Data transformations
# transform = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize((0.1307,), (0.3081,))
# ])

# # Datasets and DataLoaders
# train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
# test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
# test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# # Define CNN model
# class CNN(nn.Module):
#     def __init__(self):
#         super(CNN, self).__init__()
#         self.conv1 = nn.Conv2d(1, 32, kernel_size=3)     # Output: 32x26x26
#         self.conv2 = nn.Conv2d(32, 64, kernel_size=3)    # Output: 64x24x24
#         self.pool = nn.MaxPool2d(2, 2)                   # Reduces size by 2
#         self.fc1 = nn.Linear(64 * 5 * 5, 128)
#         self.fc2 = nn.Linear(128, 10)

#     def forward(self, x):
#         x = F.relu(self.conv1(x))        # 32x26x26
#         x = self.pool(x)                 # 32x13x13
#         x = F.relu(self.conv2(x))        # 64x11x11
#         x = self.pool(x)                 # 64x5x5
#         x = torch.flatten(x, 1)          # Flatten
#         x = F.relu(self.fc1(x))          # FC1
#         x = self.fc2(x)                  # FC2
#         return x

# # Initialize model, loss function, and optimizer
# model = CNN()
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(), lr=0.001)

# # Training loop
# num_epochs = 5
# for epoch in range(num_epochs):
#     model.train()
#     running_loss = 0.0
#     for images, labels in train_loader:
#         optimizer.zero_grad()
#         outputs = model(images)
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()
#         running_loss += loss.item()

#     print(f"Epoch {epoch+1}, Loss: {running_loss / len(train_loader):.4f}")

# # Evaluation
# model.eval()
# correct = 0
# total = 0
# with torch.no_grad():
#     for images, labels in test_loader:
#         outputs = model(images)
#         _, predicted = torch.max(outputs.data, 1)
#         total += labels.size(0)
#         correct += (predicted == labels).sum().item()

# print(f'Test Accuracy: {100 * correct / total:.2f}%')
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv("bank.csv", sep=';')

# Encode categorical columns
label_encoders = {}
for column in df.columns:
    if df[column].dtype == 'object':
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column])
        label_encoders[column] = le

# Split into features and target
X = df.drop("y", axis=1).values.astype('float32')  # 16 features
y = df["y"].values.astype('int64')                 # binary target (0 or 1)

# Normalize features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define custom dataset
class BankDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X).unsqueeze(1)  # shape: (N, 1, 16)
        self.y = torch.tensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# Create DataLoaders
train_dataset = BankDataset(X_train, y_train)
test_dataset = BankDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Define 1D CNN model for tabular data
class TabularCNN(nn.Module):
    def __init__(self):
        super(TabularCNN, self).__init__()
        self.conv1 = nn.Conv1d(1, 32, kernel_size=3)  # (B, 1, 16) -> (B, 32, 14)
        self.pool = nn.MaxPool1d(2)                   # -> (B, 32, 7)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3) # -> (B, 64, 5)
        self.fc1 = nn.Linear(64 * 2, 128)             # Flatten: 64*2 = 128
        self.fc2 = nn.Linear(128, 2)                  # Binary classification (2 outputs)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Initialize model, loss, optimizer
model = TabularCNN()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 5
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for features, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(features)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {running_loss / len(train_loader):.4f}")

# Evaluation
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for features, labels in test_loader:
        outputs = model(features)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f"Test Accuracy: {accuracy:.2f}%")
