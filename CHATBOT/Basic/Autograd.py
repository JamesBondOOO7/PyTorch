import torch

# if requires_grad = True => the tensor object keeps track of how it was created
x = torch.tensor([1., 2., 3.], requires_grad=True)
y = torch.tensor([4., 5., 6.], requires_grad=True)
# As x and y have their requires_grad = True => We can compute gradients w.r.t them

z = x + y
print(z)

# z knows that it was created as a result of x and y
print(z.requires_grad)
print(z.grad_fn)

# if we go further on this
s = z.sum()
print(s)
print(s.requires_grad)
print(s.grad_fn)

# Now, if we backpropagate on s, we can find the gradients of s w.r.t x
s.backward()
print(x.grad)

# By default, Tensors have requires_grad = False
x = torch.randn(2, 3)
y = torch.rand(2, 3)
z = x + y
print(x.requires_grad, y.requires_grad, z.requires_grad)

# to set requires_grad
x.requires_grad_()  # inplace operation
y.requires_grad_()

z = x + y
# z has the computational history to compute gradients
print(x.requires_grad, y.requires_grad, z.requires_grad)

# Detach
# z.detach() returns a tensor that shares the same storage as ``z``
# but with computational history forgotten
new_z = z.detach()
print(new_z.requires_grad)

# We can also stop autograd from tracking history on Tensors
with torch.no_grad():
    print((x + 10).requires_grad)