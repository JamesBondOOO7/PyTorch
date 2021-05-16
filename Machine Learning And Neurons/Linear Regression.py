import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt  # for plotting

# Generate 20 data points
N = 20

# random data on the x-axis in (-5, 5)
X = np.random.random(N)*10 - 5

# a line plus some noise
Y = 0.5 * X - 1 + np.random.randn(N)

# plot the data
plt.scatter(X, Y)
plt.show()

# Create a Linear Regression Model
model = nn.Linear(1, 1)

# Loss and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

# Changing the shape of the data
X = X.reshape(N, 1)
Y = Y.reshape(N, 1)

# PyTorch uses float32 by default
# Numpy uses float64 by default
inputs = torch.from_numpy(X.astype(np.float32))
targets = torch.from_numpy(Y.astype(np.float32))

# print(type(inputs)) " torch.Tensor "

# Train the model
n_epochs = 30
losses = []

for it in range(n_epochs):

    # zero the parameter gradients
    optimizer.zero_grad()

    # Forward pass
    outputs = model(inputs)
    loss = criterion(outputs, targets)

    # keep the loss for plotting
    losses.append(loss.item())

    # Backward and optimize
    loss.backward()
    optimizer.step()

    print(f"Epoch {it + 1}/{n_epochs}. Loss: {loss.item():.4f}")

################################################

# __IMPORTANT__

# zero_grad clears old gradients from the last step (otherwise you’d just accumulate
# the gradients from all loss.backward() calls).

# loss.backward() computes the derivative of the loss w.r.t. the parameters
# (or anything requiring gradients) using backpropagation.

# opt.step() causes the optimizer to take a step based on the gradients of the parameters.

################################################

# Plot thr loss per iteration
plt.plot(losses)
plt.show()

# plot the graph
predicted = model(inputs).detach().numpy()  # Returns a new Tensor, detached from the current graph.
plt.scatter(X, Y, label="Original Data")
plt.plot(X, predicted, label="Fitted line")
plt.legend()
plt.show()

# Another way to get predictions
with torch.no_grad():
    out = model(inputs).numpy()

print(out.shape)

# print weight and bias
w = model.weight.data.numpy()
b = model.bias.data.numpy()
print(w, b)
