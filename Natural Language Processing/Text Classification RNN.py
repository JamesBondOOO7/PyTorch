import torch
import torch.nn as nn
import torchtext.legacy.data as ttd
from torchtext.vocab import GloVe
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

df = pd.read_csv("./spam.csv", encoding='ISO-8859-1')
print(df.head())

# drop unnecessary columns
df = df.drop(["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"], axis=1)
print(df.head())

# rename the columns
df.columns = ["labels", "data"]
print(df.head())

# labels is categorical --> converting it to integer labels
df["b_labels"] = df["labels"].map({'ham': 0, 'spam': 1})
df2 = df[["data", "b_labels"]]
print(df2.head())

# Saving the processed file
df2.to_csv('spam2.csv', index=False)

# Field object
TEXT = ttd.Field(
    sequential=True,
    batch_first=True,
    lower=False,
    # tokenize='spacy',
    pad_first=True
)

LABEL = ttd.Field(sequential=False,
                  use_vocab=False,
                  is_target=True)

# Note: if you don't specify use_vocab=False, then PyTorch will
# complain later when you try to iterate over the dataset that
# the attribute `vocab` doesn't exist.

# Note 2: if you don't specify is_target=True, then PyTorch will
# assume it's part of the input, so when you iterate over the
# dataset it will be like:
# for (inputs, targets), _ in iterator:
# where the 2nd element (_) should have been the target.

dataset = ttd.TabularDataset(
    path="spam2.csv",
    format='csv',
    skip_header=True,
    fields=[('data', TEXT), ('label', LABEL)]
)

train_dataset, test_dataset = dataset.split()  # 70% trainig

# Building the vocab
TEXT.build_vocab(train_dataset,)
vocab = TEXT.vocab
print(f"Vocab size : {len(vocab)}")
print(vocab.stoi)
print(vocab.itos)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

train_iter, test_iter = ttd.Iterator.splits(
    (train_dataset, test_dataset), sort_key=lambda x: len(x.data),
    batch_sizes=(32, 256), device=device
)

for inputs, targets in train_iter:
    print("inputs:", inputs, "shape:", inputs.shape)
    print("targets:", targets, "shape:", targets.shape)
    break

for inputs, targets in test_iter:
    print("inputs:", inputs, "shape:", inputs.shape)
    print("targets:", targets, "shape:", targets.shape)
    break


# Define the model
class RNN(nn.Module):

    def __init__(self, n_vocab, embed_dim, n_hidden, n_rnnlayers, n_outputs):
        super(RNN, self).__init__()

        self.V = n_vocab
        self.D = embed_dim
        self.M = n_hidden
        self.K = n_outputs
        self.L = n_rnnlayers

        # Embedding Layer : V --> D
        self.embed = nn.Embedding(self.V, self.D)

        # RNN layer
        self.rnn = nn.LSTM(
            input_size=self.D,
            hidden_size=self.M,
            num_layers=self.L,
            batch_first=True
        )

        # Fully connected layer
        self.fc = nn.Linear(self.M, self.K)

    def forward(self, X):

        # init the hidden states
        h0 = torch.zeros(self.L, X.size(0), self.M).to(device)
        c0 = torch.zeros(self.L, X.size(0), self.M).to(device)

        # embedding layer
        # turns word indexes into word vectors
        # (X) : N X T --> (out) : N X T X D
        out = self.embed(X)

        # get rnn output
        # (out) : N X T X M, (hidden) : L X N X M
        out, hidden = self.rnn(out, (h0, c0))
        # output : hidden states for the final layer for each time step, the RNN output
        # hidden : hidden states over all hidden layers, but only for the final time stamp

        # max pool about 1st time dimension, i.e the time stamp dimension
        # (out) : N X T X M --> (out) : N X M
        out, _ = torch.max(out, 1)

        # we only want h(T) at the final time step
        # (out) : N X M --> (out) : N X K
        out = self.fc(out)

        return out

model = RNN(
    n_vocab=len(vocab),
    embed_dim=20,
    n_hidden=15,
    n_rnnlayers=1,
    n_outputs=1
)
model.to(device)

# Loss and optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters())

# Training loop function
def batch_gd(model, criterion, optimizer, train_iter, test_iter, epochs):

    train_losses = np.zeros(epochs)
    test_losses = np.zeros(epochs)

    for it in range(epochs):

        t0 = datetime.now()
        train_loss = []

        for inputs, targets in train_iter:

            targets = targets.view(-1, 1).float()

            # move data to GPU
            inputs, targets = inputs.to(device), targets.to(device)

            # zero grad the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # Backward and optimize
            loss.backward()
            optimizer.step()

            train_loss.append(loss.item())

        # Get train loss and test loss
        train_loss = np.mean(train_loss)

        model.eval()
        test_loss = []

        with torch.no_grad():
            for inputs, targets in test_iter:

                targets = targets.view(-1, 1).float()

                # move data to GPU
                # inputs, targets = inputs.to(device), targets.to(device)

                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                test_loss.append(loss.item())

            test_loss = np.mean(test_loss)
        model.train()

        # Save losses
        train_losses[it] = train_loss
        test_losses[it] = test_loss

        dt = datetime.now() - t0
        print(f"Epoch {it + 1}/{epochs}, Train loss: {train_loss:.4f}")
        print(f"Test loss: {test_loss:.4f}, Duration: {dt}")

    return train_losses, test_losses


train_losses, test_losses = batch_gd(model, criterion, optimizer, train_iter, test_iter, 15)

# Plot the train loss and test loss per iteration
plt.plot(train_losses, label='train loss')
plt.plot(test_losses, label='test loss')
plt.legend()
plt.show()

# Accuarcy

n_correct = 0
n_total = 0

for inputs, targets in train_iter:
    targets = targets.view(-1, 1).float()

    # move data to GPU
    inputs, targets = inputs.to(device), targets.to(device)

    # Forward pass
    outputs = model(inputs)

    predictions = (outputs > 0)

    # update counts
    n_correct += ( predictions == targets ).sum().item()
    n_total += targets.shape[0]

train_acc = n_correct/n_total

n_correct = 0
n_total = 0

for inputs, targets in test_iter:
    targets = targets.view(-1, 1).float()

    # move data to GPU
    inputs, targets = inputs.to(device), targets.to(device)

    # Forward pass
    outputs = model(inputs)

    predictions = (outputs > 0)

    # update counts
    n_correct += (predictions == targets).sum().item()
    n_total += targets.shape[0]

test_acc = n_correct / n_total
print(f"Train acc: {train_acc:.4f}, Test acc: {test_acc:.4f}")
