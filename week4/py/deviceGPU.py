# import torch

# # Select device (GPU if available, else CPU)
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print("Using device:", device)

# # Create a tensor and move it to the selected device
# tensor_a = torch.rand(2, 3).to(device)
# print("Tensor A on device:")
# print(tensor_a)
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

tensor_size = (10000, 10000)  
a = torch.randn(tensor_size, device=device)  
b = torch.randn(tensor_size, device=device)  

c = a + b  

print("Result shape (moved to CPU for printing):", c.cpu().shape)

print("Current GPU memory usage:")
print(f"Allocated: {torch.cuda.memory_allocated(device) / (1024 ** 2):.2f} MB")
print(f"Cached: {torch.cuda.memory_reserved(device) / (1024 ** 2):.2f} MB")
