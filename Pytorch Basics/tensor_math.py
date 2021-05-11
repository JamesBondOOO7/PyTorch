import torch

x = torch.tensor([1, 2, 3])
y = torch.tensor([9, 8, 7])

# Addition
z1 = torch.empty(3) # Method 1
torch.add(x, y, out=z1)
z2 = torch.add(x, y) # Method 2
z = x + y # Method 3
print("Addition", z)

# Subtraction
z = x - y
print("Subtraction", z)

# Division
z = torch.true_divide(x, y)
# What happens is it does "Element wise division" if they are of equal shape
print("Division", z) # [1/9, 2/8, 3/7]

# inplace operations
# function with an underscore => implace operation
# func_()

t = torch.zeros(3)
t.add_(x) # Method 1
t += x # Method 2

# Note: t = t + x is not implace

# Exponentiation
# Element wise exponentiation
z = x.pow(2) # Method 1
z = x**2 # Method 2
print("Exponentiation", z)

# Simple Comparision
z = x > 0 # All True
z = x < 0 # All False

# Matrix Multiplication
x1 = torch.rand((2, 5))
x2 = torch.rand((5, 3))
x3 = torch.mm(x1, x2) # Method 1
x4 = x1.mm(x2) # Method 2

# Matrix Exponentiation
matrix_exp = torch.rand(5, 5)
matrix_exp.matrix_power(3) # A X A X A = A3
# Only for Square Matrices
print("Matrix Exponentiation",matrix_exp)

mt = torch.tensor([[1, 2],[3, 4]], dtype=torch.float)
print(mt.matrix_power(2))

# Element wise multiplication
z = x * y
print("Element Wise Multiplication", z)

# dot product : sum(Element wise mult.)
z = torch.dot(x, y)
print("Dot product",z)

# Batch Matrix Multiplication
batch = 32
n = 10
m = 20
p = 30

tensor1 = torch.rand((batch, n, m))
tensor2 = torch.rand((batch, m, p))
out_bmm = torch.bmm(tensor1, tensor2) # shape : ( batch, n, p )
print(out_bmm.shape)


# Example of Broadcasting
x1 = torch.rand((5, 5))
x2 = torch.rand((1, 5))

z = x1 - x2 # each vector of the matrix x1 is subtracted by the vector x2
print(z)

z = x1 ** x2 # each vector of the matrix x1 is exponentiated by the elements of vector x2

# Other useful operations

sum_x = torch.sum(x, dim=0) # dim on which to sum over
print(torch.sum(mt, dim=0))
print(torch.sum(mt, dim=1))

values, indicies = torch.max(x1, dim=0) # max of elements in dim 0, Method 1
print(values, indicies)
print(x1.max(dim=0)) # Method 2

values, indicies = torch.min(x1, dim=0) # min of elements in dim 0, Method 1
print(values, indicies)
print(x1.min(dim=0)) # Method 2

abs_x = torch.abs(x1) # Method 1
print(x1.abs()) # Method 2

z = torch.argmax(x, dim=0) # only returns the index, same as indicies, Method 1
print(z)
print(x.argmax(dim=0)) # Method 2

mean_x = torch.mean(x1.float(), dim=0) # Method 1
print(mean_x)
print(x1.mean(dim=0)) # Method 2

# element wise compare
z = torch.eq(x, y) # Method 1
print(z)
print(x.eq(y)) # Method 2

# sorting
sorted_y, indices = torch.sort(y, dim=0, descending=False) # Method 1
print(sorted_y, indices)
print(y.sort(dim=0, descending=False)) # Method 2

z = torch.clamp(x, min=0, max=5) # Method 1
# i.e, elements < 0 ==> 0
# and elements > 5 ==> 5
# if we don't set max, then it will not be affected for max
print(z)
print(x.clamp(min=0, max=5)) # Method 2

x = torch.tensor([1, 0, 1, 1, 1], dtype=torch.bool)
# Method 1
z1 = torch.any(x) # at least 1 value as true
z2 = torch.all(x) # all values need to be true
print(z1, z2)

# Method 2
print(x.any(), x.all())