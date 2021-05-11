import torch

x = torch.arange(9)

# (9,) => (3,3)
x_3X3 = x.view(3, 3) # Method 1, works with contiguous memory space
print(x_3X3)
x_3X3 = x.reshape(3, 3) # Method 2

y = x_3X3.t()
print(y.contiguous().view(9)) # as y will no longer be in a contiguous space
# we need to explicitly make it contiguous and then view() it !!

print(y.reshape(9)) # reshape works fine

x1 = torch.rand((2, 5))
x2 = torch.rand((2, 5))
print(torch.cat((x1, x2), dim=0).shape) # concatenate along the 1st dim
print(torch.cat((x1, x2), dim=1).shape) # concatenate along the 2nd dim

z = x1.view(-1) # flatten
print(z.shape)

batch = 64
x = torch.rand((batch, 2, 5))
z = x.view(batch, -1)
print(z.shape)

# switch axis
z = x.permute(0, 2, 1) # 0:0, 1:2nd , 2nd:1st
# transpose is a special case of permute()
print(x.shape, z.shape)

x = torch.arange(10) # 10,
print(x.unsqueeze(0).shape) # 10, => 1,10
print(x.unsqueeze(1).shape) # 10, => 10, 1

x = torch.arange(10).unsqueeze(0).unsqueeze(1) # 1 X 1 X 10
print(x.shape)
z = x.squeeze(1)
print(z.shape)