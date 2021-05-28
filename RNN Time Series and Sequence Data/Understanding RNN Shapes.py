import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# Imp Terms
# N = number of samples
# T = sequence length
# D = number of input features
# M = number of hidden units
# K = number of output units

# Make some data
N = 1
T = 10
D = 3
M = 5
K = 2
X = np.random.randn(N, T, D)

# Make a RNN
class SimpleRNN(nn.Module):

    def __init__(self, n_inputs, n_hidden, n_outputs):
        super(SimpleRNN, self).__init__()

        self.D = n_inputs
        self.M = n_hidden
        self.K = n_outputs
        self.rnn = nn.RNN(
            input_size=self.D,
            hidden_size=self.M,
            nonlinearity='tanh',
            batch_first=True
        )
        self.fc = nn.Linear(self.M, self.K)

    def forward(self, X):

        # init the hidden states
        h0 = torch.zeros(1, X.size(0), self.M)

        # get the RNN output
        output, hidden = self.rnn(X, h0)

        output = self.fc(output)
        return output


# Instantiate the model
model = SimpleRNN(n_inputs=D, n_hidden=M, n_outputs=K)

# Get the output
inputs = torch.from_numpy(X.astype(np.float32))
output = model(inputs)
print(output.shape)

# Save for later
Yhat_torch = output.detach().numpy()
print(f"Yhat_torch : {Yhat_torch.shape}")

# Parameters of the RNN model
W_xh, W_hh, b_xh, b_hh = [param.data.numpy() for param in model.rnn.parameters()]

print(f"W_xh : {W_xh.shape}")
print(f"W_hh : {W_hh.shape}")
print(f"b_xh : {b_xh.shape}")
print(f"b_hh : {b_hh.shape}")

# Now get the Weights of the Fully Connected layer
Wo, bo = [param.data.numpy() for param in model.fc.parameters()]

print(f"Wo : {Wo.shape}")
print(f"bo : {bo.shape}")


# Let's see if we can replicate the output
h_last = np.zeros(M)  # init the hidden layer
x = X[0]  # the one and only sample
Yhats = np.zeros((T, K))  # where we store outputs

for t in range(T):
    h = np.tanh(x[t].dot(W_xh.T) + b_xh + h_last.dot(W_hh.T) + b_hh)
    y = h.dot(Wo.T) + bo  # we only care about this value on the last iteration
    Yhats[t] = y

    # assign h to h_last
    h_last = h

print(f"Yhat : {np.array(Yhats).shape}")  # should be same as the Yhat_torch

# Lets check whether Yhats == Yhats_torch
print(np.allclose(Yhats, Yhat_torch))
