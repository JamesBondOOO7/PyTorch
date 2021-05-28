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

# Now turn the full data into numpy arrays

input_data = df[['open', 'high', 'low', 'close', 'volume']].values
targets = df['Return'].values

# Now make the actual data which will go into the neural network
T = 10  # the number of time steps to look at to make a prediction for the next day
D = input_data.shape[1]
N = len(input_data) - T

# Normalize the inputs
Ntrain = len(input_data) * 2 // 3
scaler = StandardScaler()
scaler.fit(input_data[:Ntrain + T - 1])
input_data = scaler.transform(input_data)

# Setup X_train abd Y_train
X_train = np.zeros((Ntrain, T, D))
Y_train = np.zeros((Ntrain, 1))

for t in range(Ntrain):
    X_train[t, :, :] = input_data[t:t+T]
    Y_train[t] = (targets[t+T] > 0)

# Setup X_test and Y_test
X_test = np.zeros((N - Ntrain, T, D))
Y_test = np.zeros((N - Ntrain, 1))

for u in range(N - Ntrain):
    # u counts form 0...(N - Ntrain)
    # t counts from Ntrain...N
    t = u + Ntrain
    X_test[u, :, :] = input_data[t:t+T]
    Y_test[u] = (targets[t+T] > 0)

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

model = RNN(n_inputs=5, n_hidden=50, n_rnnlayers=2, n_outputs=1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

model.to(device)

# Loss and optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Make inputs and targets
X_train = torch.from_numpy(X_train.astype(np.float32))
y_train = torch.from_numpy(Y_train.astype(np.float32))
X_test = torch.from_numpy(X_test.astype(np.float32))
y_test = torch.from_numpy(Y_test.astype(np.float32))

# move data to GPU
X_train, y_train = X_train.to(device), y_train.to(device)
X_test, y_test = X_test.to(device), y_test.to(device)

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

train_losses, test_losses = full_gd(model,criterion,optimizer,X_train,y_train,X_test,y_test, epochs=300)

# Plot the train loss and test loss per iteration
plt.plot(train_losses, label='train loss')
plt.plot(test_losses, label='test loss')
plt.legend()
plt.show()

# Get accuracy
with torch.no_grad():
    p_train = model(X_train)
    p_train = (p_train.cpu().numpy() > 0)
    train_acc = np.mean(y_train.cpu().numpy() == p_train)

    p_test = model(X_test)
    p_test = (p_test.cpu().numpy() == p_test)
    test_acc = np.mean(y_test.cpu().numpy() == p_test)

print(f"Train acc: {train_acc}, Test acc: {test_acc}")
