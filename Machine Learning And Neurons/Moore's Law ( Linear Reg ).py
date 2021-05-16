import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# get and load in the data
data = pd.read_csv("https://raw.githubusercontent.com/lazyprogrammer/machine_learning_examples/master/tf2.0/moore.csv",
                   header=None).values

X = data[:, 0].reshape(-1, 1)  # make it a 2-D array
Y = data[:, 1].reshape(-1, 1)

# plot the data
plt.scatter(X, Y)
plt.show()

# Moore's Law is => C = A*r^t
# => logC = logA + t*logr
# => y = b + ax ==> a Linear Regression problem

# Making it a Linear model
Y = np.log(Y)
plt.scatter(X, Y)
plt.show()

# Scaling and center both the x and y axes
mx = X.mean()
sx = X.std()
my = Y.mean()
sy = Y.std()

X = (X - mx) / sx
Y = (Y - my) / sy
# Now everything is centered and in a small range

# Now cast to float32
X = X.astype(np.float32)
Y = Y.astype(np.float32)

# Create the linear regression model
model = nn.Linear(1, 1)

# Loss and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.7)

inputs = torch.from_numpy(X)
targets = torch.from_numpy(Y)

# Train the model
n_epochs = 100
losses = []

for it in range(n_epochs):

    # zero the parameter gradients
    optimizer.zero_grad()

    # Forward pass
    outputs = model(inputs)
    loss = criterion(outputs, targets)

    # Record the loss
    losses.append(loss.item())

    # Backward and optimize
    loss.backward()
    optimizer.step()

    print(f"Epoch {it + 1}/{n_epochs}. Loss: {loss.item():.4f}")

# plot the losses
plt.plot(losses)
plt.show()

# plot the graph
predicted = model(inputs).detach().numpy()  # Returns a new Tensor, detached from the current graph.
plt.scatter(X, Y, label="Original Data")
plt.plot(X, predicted, label="Fitted line", color="orange")
plt.legend()
plt.show()

# print weight and bias
w = model.weight.data.numpy()
b = model.bias.data.numpy()
print(w, b)

# y = b + ax
# y and x were originally transformed
# therefore, substituting them, we will get an eqn in which
# a = w * sy / sx
a = w[0, 0] * sy / sx
print(a)
