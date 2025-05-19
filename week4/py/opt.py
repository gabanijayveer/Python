import torch

x = torch.tensor([2.0], requires_grad=True)
y = x**2 + 3*x + 1
y.backward()  # Computes dy/dx
print(x.grad)  # Gradient of y w.r.t x: dy/dx = 2x + 3

tensor = torch.arange(0, 12).view(3, 4)
print(tensor[:, 1])    # Slice column
print(tensor[1, :])    # Slice row
