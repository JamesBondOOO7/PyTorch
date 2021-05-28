import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('https://raw.githubusercontent.com/lazyprogrammer/machine_learning_examples/master/tf2.0/sbux.csv')
print(df.head())

# RETURNS
df["PrevClose"] = df['close'].shift(1)

# we are concatenating the 1st column's "close" value to the second
# the 2nd column's "close" value to the third
# and so on
# note, this value for the first column will be NaN

print(df.head())

# the return is
# (x[t] - x[t-1]) / x[t-1]
df["Return"] = (df['close'] - df['PrevClose']) / df['PrevClose']
df.head()

plt.plot(df['Return'])
plt.show()

# Hist plot
df['Return'].hist()
plt.show()

series = df["Return"].values[1:].reshape(-1, 1)  # 1st value will be null

# Normalize the data
scaler = StandardScaler()
scaler.fit(series[:len(series) // 2])
series = scaler.transform(series).flatten()

# build the dataset
T = 20
D = 1
X = []
Y = []

for t in range(len(series) - T):
    x = series[t:t+T]
    X.append(x)
    y = series[t+T]
    Y.append(y)

X = np.array(X).reshape(-1, T, 1)
Y = np.array(Y).reshape(-1, 1)
N = len(X)
print(f"X.shape: {X.shape}, Y.shape: {Y.shape}")

# RNN model
class RNN(nn.Module):
    def __init__(self, n_inputs, n_hidden, n_rnnlayers, n_outputs):
        super(RNN, self).__init__()
        self.D = n_inputs
        self.M = n_hidden
        self.K = n_outputs
        self.L = n_rnnlayers

        self.rnn = nn.LSTM(
            input_size=self.D,
            hidden_size=self.M,
            num_layers=self.L,
            batch_first=True
        )
        self.fc = nn.Linear(self.M, self.K)

    def forward(self, X):

        # init the hidden states
        h0 = torch.zeros(self.L, X.size(0), self.M).to(device)
        c0 = torch.zeros(self.L, X.size(0), self.M).to(device)

        # get rnn outputs
        output, hidden = self.rnn(X, (h0, c0))

        # We only want the h(t) at the final time step
        output = self.fc(output[:, -1, :])
        return output

model = RNN(n_inputs=1, n_hidden=5, n_rnnlayers=1, n_outputs=1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

model.to(device)

# Loss and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

# Training
def full_gd(model, criterion, optimizer, X_train, y_train, X_test, y_test, epochs=200):

    train_losses = torch.zeros(epochs)
    test_losses = torch.zeros(epochs)

    for it in range(epochs):

        # zero the parameter gradient
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
        test_outputs = model(X_test)
        test_loss = criterion(test_outputs, y_test)
        test_losses[it] = test_loss.item()

        if (it + 1) % 5 == 0:
            print(f"Epoch {it + 1}/{epochs}, Train loss: {loss.item():.4f}, Test loss: {test_loss.item():.4f}")

    return train_losses, test_losses


# Make inputs and targets
X_train = torch.from_numpy(X[:-N//2].astype(np.float32))
y_train = torch.from_numpy(Y[:-N//2].astype(np.float32))
X_test = torch.from_numpy(X[-N//2:].astype(np.float32))
y_test = torch.from_numpy(Y[-N//2:].astype(np.float32))

# move data to GPU
X_train, y_train = X_train.to(device), y_train.to(device)
X_test, y_test = X_test.to(device), y_test.to(device)

train_losses, test_losses = full_gd(model,criterion,optimizer,X_train,y_train,X_test,y_test)

# Plot the train loss and test loss per iteration
plt.plot(train_losses, label='train loss')
plt.plot(test_losses, label='test loss')
plt.legend()
plt.show()

# One-step forecast using true targets
validation_target = Y
validation_predictions = []

# index of first validation input
i = 0

X_on_device = torch.from_numpy(X.astype(np.float32)).to(device)

while len(validation_predictions) < len(validation_target):
    input_ = X_on_device[i].reshape(1, T, 1)
    p = model(input_)[0, 0].item()  # 1x1 array -> scalar
    i += 1

    # update the predictions list
    validation_predictions.append(p)

plt.plot(validation_target, label='forecast target')
plt.plot(validation_predictions, label='forecast prediction')
plt.legend()
plt.show()

# Multi-step forecast
validation_target = Y[-N // 2:]
validation_predictions = []

# last train input
last_x = X_test[0].view(T)

while len(validation_predictions) < len(validation_target):
    input_ = last_x.reshape(1, T, 1)
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
