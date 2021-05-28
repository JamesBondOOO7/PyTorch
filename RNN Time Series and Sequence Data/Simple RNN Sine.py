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

X = np.array(X).reshape(-1, T, 1)
Y = np.array(Y).reshape(-1, 1)
N = len(X)
print(f"X.shape : {X.shape}, Y.shape : {Y.shape}")

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# Define a simple RNN
class SimpleRNN(nn.Module):

    def __init__(self, n_inputs, n_hidden, n_rnnlayers, n_outputs):

        super(SimpleRNN, self).__init__()
        self.D = n_inputs
        self.M = n_hidden
        self.K = n_outputs
        self.L = n_rnnlayers

        # NOTE : batch_first = True
        # applies that the conversion that our data will be of shape :
        # ( num_samples, sequence_length, num_features )
        # rather than
        # ( sequence_length, num_samples, num_features )

        self.rnn = nn.RNN(
            input_size=self.D,
            hidden_size=self.M,
            num_layers=self.L,
            nonlinearity='relu',
            batch_first=True
        )
        self.fc = nn.Linear(self.M, self.K)

    def forward(self, X):
        # init hidden states
        # h0.shape --> (number of stacked RNN layers X Batch size X Hidden layers )
        h0 = torch.zeros(self.L, X.size(0), self.M).to(device)

        # get RNN unit output
        # output is of the shape --> ( Batch size X Time Steps X Hidden State )
        # hidden is of the shape --> ( number of stacked RNN layers X Batch size X Hidden layers )

        output, hidden = self.rnn(X, h0)
        # output : hidden states for the final layer for each time step, the RNN output
        # hidden : hidden states over all hidden layers, but only for the final time stamp

        # we only want h(t) for the final step
        # output --> output classes
        # ( Batch size X Time Steps X Hidden State ) --> ( Batch size X number of output nodes (classes) )
        output = self.fc(output[:, -1, :])

        return output

# Init the model
model = SimpleRNN(n_inputs=1, n_hidden=15, n_rnnlayers=1, n_outputs=1)
model.to(device)

# Loss and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Make the inputs and targets
X_train = torch.from_numpy(X[:-N//2].astype(np.float32))
y_train = torch.from_numpy(Y[:-N//2].astype(np.float32))
X_test = torch.from_numpy(X[-N//2:].astype(np.float32))
y_test = torch.from_numpy(Y[-N//2:].astype(np.float32))

# move the data to GPU
X_train, y_train = X_train.to(device), y_train.to(device)
X_test, y_test = X_test.to(device), y_test.to(device)

# Training
def full_gd(model, criterion, optimizer, X_train, y_train, X_test, y_test, epochs=1000):

    train_losses = np.zeros(epochs)
    test_losses = np.zeros(epochs)

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

        if (it + 1 ) % 5 == 0:
            print(f"Epoch {it + 1}/{epochs}, Train Loss: {loss.item():.4f}, Test Loss: {test_loss.item():.4f}")

    return train_losses, test_losses

train_losses, test_losses = full_gd(model, criterion, optimizer, X_train, y_train, X_test, y_test)

# plot the train loss and test loss per iteration
plt.plot(train_losses, label='train loss')
plt.plot(test_losses, label='test_loss')
plt.legend()
plt.show()


# " WRONG FORECASTING " using true targets

validation_target = Y[-N//2:]
validation_predictions = []

i = 0
while len(validation_predictions) < len(validation_target):

    # making the input_ of the shape --> ( Batch size = 1 X Time steps X Dimension of data = 1 )
    input_ = X_test[i].reshape(1, T, 1)

    # model output's shape --> (1,1)
    p = model(input_)[0, 0].item()
    i += 1

    # update the predictions
    validation_predictions.append(p)

plt.plot(validation_target, label='forecast target')
plt.plot(validation_predictions, label='forecast prediction')
plt.legend()
plt.show()


# Forecast future values ( use only self-predictions for making future predictions )
# " THE RIGHT WAY "

validation_target = Y[-N//2:]
validation_predictions = []

# last train input
last_x = X_test[0].view(T)

while len(validation_predictions) < len(validation_target):

    input_ = last_x.reshape(1, T, 1)

    # model output's shape --> (1,1)
    p = model(input_)

    # update the predictions list
    validation_predictions.append(p[0, 0].item())

    # make the new input
    last_x = torch.cat((last_x[1:], p[0]))

plt.plot(validation_target, label='forecast target')
plt.plot(validation_predictions, label='forecast prediction')
plt.legend()
plt.show()

# Note : Autoregressive model performs better than RNN !!
# Reason ?
# because this data is quite simple and can be represented
# in the form of linear relations

# but RNN model is quite flexible and so it doesn't fits well
# it is over-parameterized
# So, sometimes it may be able to capture the pattern like periodicity
# in this case, and sometimes not
