import torch

# Creating a 1D tensor
x = torch.tensor([1.0, 2.0, 3.0])
print('1D Tensor: \n', x)

# Creating a 2D tensor
y = torch.zeros((3, 3))
print('2D Tensor: \n', y)

# Creating a 3D tensor
z = torch.rand((3, 3, 3))
print('3D Tensor: \n', z)

# Creating a tensor with specific values
w = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print('Tensor with specific values: \n', w)

# Creating a tensor with random values
v = torch.rand((3, 3))

print('Tensor with random values: \n', v)

# Creating a tensor with ones
u = torch.ones((3, 3))

print('Tensor with ones: \n', u)


a = torch.tensor([1,2,3,4])
b = torch.tensor([5,6,7,8])

#two ways
#1.
print("add")
print(a + b)
#2.
print(torch.add(a, b))
print("sub")
print(torch.sub(a, b))
print("*")
print(torch.mul(a, b))
print("/")
print(torch.div(a, b))
print("power")
print(torch.pow(a, b))
print("sqrt")
print(torch.sqrt(a))
print("add sqrt")
print(torch.sqrt(a+b))

a = torch.tensor([1, 2]) 
b = torch.tensor([3, 4]) 

#matrix multi

print('Matrix Multiplication of a & b: \n', torch.matmul(a.view(2, 1), b.view(1, 2)))

