import torch

batch_size = 10
features = 25
x = torch.rand((batch_size, features))

print(x[0].shape) # shape of the example

print(x[:, 0].shape) # 1st feature of all examples

print(x[2, 0:10]) # 3rd example, 1st 10 features

# fancy indexing
x = torch.arange(10)
indices = [2, 5, 8]
print(x[indices])

x = torch.rand((3, 5))
print(x)
rows = torch.tensor([1, 0])
cols = torch.tensor([4, 0])
print(x[rows, cols]) # prints x[1][4], x[0][0]

# more advanced indexing
x = torch.arange(10)
print(x[(x < 2 ) | (x > 8)])
print(x[x.remainder(2) == 0])

# other useful operations
print(torch.where(x > 5, x, x*2))
# if x > 5 : x
# else : x*2

print(torch.tensor([0,0,1,2,2,3,4]).unique())

print(x.ndimension()) # 1d

y = torch.randn((2,3,5))
print(y.ndimension()) # 3d => 2 X 3 X 5

print(x.numel()) # no of elements in x = 10
print(y.numel()) # no of elements in y = 30 ( 2 X 3 X 5 )