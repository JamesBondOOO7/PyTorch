import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# Make the original dataset + some random noise
N = 1000
series = np.sin(0.1*np.arange(N)) + np.random.randn(N)*0.1

# plot it
plt.plot(series)
plt.show()

# build the dataset

T = 10
X = []
Y = []

for t in range(len(series) - T):
    x = series[t:t+T]
    X.append(x)
    y = series[t+T]
    Y.append(y)

X = np.array(X).reshape(-1, T)
Y = np.array(Y).reshape(-1, 1)
N = len(X)
print(f"X.shape : {X.shape}, Y.shape : {Y.shape}")

# Autoregressive model
model = nn.Linear(T, 1)

# Loss and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

# make the inputs and target
# NOTE : We don't shuffle the data. Why?
# because we have to predict the future data
X_train = torch.from_numpy(X[:-N//2].astype(np.float32))
y_train = torch.from_numpy(Y[:-N//2].astype(np.float32))
X_test = torch.from_numpy(X[-N//2:].astype(np.float32))
y_test = torch.from_numpy(Y[-N//2:].astype(np.float32))

print(f"X_train.shape : {X_train.shape}, y_train.shape : {y_train.shape}")

# Training

def full_gd(model, criterion, optimizer, X_train, y_train, X_test, y_test, epochs=200):

    train_losses = np.zeros(epochs)
    test_losses = np.zeros(epochs)

    for it in range(epochs):

        # zero he parameter gradients
        optimizer.zero_grad()

        # forward pass
        outputs = model(X_train)
        loss = criterion(outputs, y_train)

        # backward and optimize
        loss.backward()
        optimizer.step()

        # Save losses
        train_losses[it] = loss.item()

        # Test loss
        test_output = model(X_test)
        test_loss = criterion(test_output, y_test)
        test_losses[it] = test_loss.item()

        if (it + 1) % 5 == 0:
            print(f"Epoch {it+1}/{epochs}, Train Loss: {loss.item():.4f}, Test Loss: {loss.item():.4f}")

    return train_losses, test_losses


train_losses, test_losses = full_gd(model, criterion, optimizer, X_train, y_train, X_test, y_test)

# plot the train loss and test loss per iteration
plt.plot(train_losses, label='train loss')
plt.plot(test_losses, label='test loss')
plt.legend()
plt.show()

# forecast using true targets - "THE WRONG WAY"

validation_target = Y[-N//2:]
validation_predictions = []

i = 0
while len(validation_predictions) < len(validation_target):

    # X_test => (N, T)
    # X_test[i] => (T,)
    # We will reshape it to (1, T)
    input_ = X_test[i].view(1, -1)

    # model output => (N, K) ; K = number of output nodes
    # for this, N=1 and K=1
    # output_shape => (1, 1)
    p = model(input_)[0, 0].item()
    i += 1

    # update the predictions list
    validation_predictions.append(p)

plt.plot(validation_target, label='forecast target')
plt.plot(validation_predictions, label='forecast prediction')
plt.legend()
plt.show()

# forecast future values ( use only self-predictions for making future predictions )
# "THE RIGHT WAY", ( even though we may get poor generalization than the above )
validation_target = Y[-N//2:]
validation_predictions = []

# last train input
# 1-D array of length T
last_x = torch.from_numpy(X[-N//2].astype(np.float32))

while len(validation_predictions) < len(validation_target):

    input_ = last_x.view(1, -1)

    p = model(input_)

    # 1X1 array --> scalar
    validation_predictions.append(p[0, 0].item())

    # update the last_x list
    last_x = torch.cat((last_x[1:], p[0]))
    # Notice, we are appending the previous prediction
    # and not using the provided test case ( true target )

plt.plot(validation_target, label='forecast target')
plt.plot(validation_predictions, label='forecast prediction')
plt.legend()
plt.show()
