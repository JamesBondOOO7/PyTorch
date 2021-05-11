import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

my_tensor = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32,
                         device=device, requires_grad=True)
# dtype : int, float, etc
# device : cuda/cpu(by default)
# requires_grad : for auto grad

print(my_tensor)

# Attributes of tensor
print(my_tensor.dtype)
print(my_tensor.device)
print(my_tensor.shape)
print(my_tensor.requires_grad)

# Other common initialization methods
x = torch.empty(size= (3, 3)) # 3 X 3 tensor,
# contains the data currently in that memory location ( just random )
print(x)

x = torch.zeros((3, 3)) # A ZERO tensor
print(x)

x = torch.rand((3, 3)) # values from a uniform ditribution, b/w 0 and 1
print(x)

x = torch.ones((3, 3)) # 1s
print(x)

x = torch.eye(5, 5) # Identity
print(x)

x = torch.arange(start=0, end=5, step=1)
print(x)

x = torch.linspace(start=0.1, end=1, steps=10) # 0.1000, 0.2000, 0.3000,... steps
print(x)

x = torch.empty(size=(1, 5)).normal_(mean=0, std=1) # normally distributed values
print(x)

x = torch.empty(size=(1, 5)).uniform_(0, 1) # same like rand
print(x)

x = torch.diag(torch.ones(3)) # Diagonal matrix
print(x)

# Converting tensors to different types
tensor = torch.arange(4) # int64 by default

print(tensor.bool()) # to boolean
print(tensor.short()) # to int16
print(tensor.long()) # to int64
print(tensor.half()) # to float16
print(tensor.float()) # to float32
print(tensor.double()) # to float64

# Array to Tensor conversion and vice versa

import numpy as np
np_array = np.zeros((5, 5))

tensor = torch.from_numpy(np_array) # numpy to tensor
np_array_back = tensor.numpy() # tensor to numpy