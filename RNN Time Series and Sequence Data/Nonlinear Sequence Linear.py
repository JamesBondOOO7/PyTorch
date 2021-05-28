import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# make the original data
# this is a time series of the form  x(t) = sin(wt^2)
series = np.sin((0.1*np.arange(400))**2)

# plot it
plt.plot(series)
plt.show()

# build the dataset
T = 10
D = 1
X = []
Y = []

for t in range(len(series) - T):
    x = series[t:t+T]
    X.append(x)
    y = series[t+T]
    Y.append(y)

X = np.array(X).reshape(-1, T)  # make it N X T
Y = np.array(Y).reshape(-1, 1)
N = len(X)
print(f"X.shape : {X.shape}, Y.shape : {Y.shape}")

# try autoregressive linear model
model = nn.Linear(T, 1)

# Loss and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

# Make inputs and targets
X_train = torch.from_numpy(X[:-N//2].astype(np.float32))
y_train = torch.from_numpy(Y[:-N//2].astype(np.float32))
X_test = torch.from_numpy(X[-N//2:].astype(np.float32))
y_test = torch.from_numpy(Y[-N//2:].astype(np.float32))

# Training
def full_gd(model, criterion, optimizer, X_train, y_train, X_test, y_test, epochs=200):

    train_losses = np.zeros(epochs)
    test_losses = np.zeros(epochs)

    for it in range(epochs):

        # Zero grad the optimizer
        optimizer.zero_grad()

        # Forward pass
        outputs = model(X_train)
        loss = criterion(outputs, y_train)

        # Backward and optimize
        loss.backward()
        optimizer.step()
        
        # Save losses
        train_losses[it] = loss.item()
        
        # Test loss
        test_outputs = model(X_test)
        test_loss = criterion(test_outputs, y_test)
        test_losses[it] = test_loss.item()
        
        if (it + 1) % 5 == 0:
            print(f"Epoch {it + 1}/{epochs}, Train Loss : {loss.item():.4f}, Test Loss : {test_loss.item():.4f}")
            
    return train_losses, test_losses


train_losses, test_losses = full_gd(model, criterion, optimizer, X_train, y_train, X_test, y_test)

# plot the train loss and test loss per iteration
plt.plot(train_losses, label='train_loss')
plt.plot(test_losses, label='test loss')
plt.legend()
plt.show()

# One step forecast using TRUE TARGETS ( wrong way )

validation_target = Y[-N//2:]
with torch.no_grad():
    validation_predictions = model(X_test).numpy()

plt.plot(validation_target, label='forecast target')
plt.plot(validation_predictions, label='forecast prediction')
plt.legend()
plt.show()

# Multi step forecast
validation_target = Y[-N//2:]
validation_predictions = []

# last train input
last_x = torch.from_numpy(X[-N//2].astype(np.float32))

while len(validation_predictions) < len(validation_target):
    input_ = last_x.reshape(1, -1)
    p = model(input_)
    # [0,0] # 1x1 array -> scalar

    # update the predictions list
    validation_predictions.append(p[0, 0].item())

    # make the new input
    last_x = torch.cat((last_x[1:], p[0]))

plt.plot(validation_target, label='forecast target')
plt.plot(validation_predictions, label='forecast prediction')
plt.legend()
plt.show()
