import torch
import torch.nn as nn
import torch.nn.functional as F
import torchtext.legacy.data as ttd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# same preprocessing as using RNN
# directly use spam2.csv

# Field Objects
TEXT = ttd.Field(
    sequential=True,
    batch_first=True,
    lower=False,
    pad_first=True
)

LABEL = ttd.Field(
    sequential=False,
    use_vocab=False,
    is_target=True
)

dataset = ttd.TabularDataset(
    path="spam2.csv",
    format='csv',
    skip_header=True,
    fields=[('data', TEXT), ('label', LABEL)]
)

train_dataset, test_dataset = dataset.split()
TEXT.build_vocab(train_dataset,)
vocab = TEXT.vocab

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

train_iter, test_iter = ttd.Iterator.splits(
    (train_dataset, test_dataset), sort_key=lambda x: len(x.data),
    batch_sizes=(32, 256), device=device
)

# Define the model
class CNN(nn.Module):

    def __init__(self, n_vocab, embed_dim, n_outputs):
        super(CNN, self).__init__()
        self.V = n_vocab
        self.D = embed_dim
        self.K = n_outputs

        # T words --> T X D dimension
        # where each word is D dimensional
        self.embed = nn.Embedding(self.V, self.D)

        # conv layers
        self.conv1 = nn.Conv1d(in_channels=self.D, out_channels=32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool1d(kernel_size=2)
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)

        self.fc = nn.Linear(128, self.K)

    def forward(self, X):

        # embedding layer
        # turns word indexes into word vectors
        # X : N X T --> out : N X T X D
        out = self.embed(X)

        # NOTE: Output of embedding layer is always : ( N X T X D )
        # conv1d expects : ( N X D X T )

        # conv layers
        out = out.permute(0, 2, 1)
        out = self.conv1(out)
        out = F.relu(out)
        out = self.pool1(out)
        out = self.conv2(out)
        out = F.relu(out)
        out = self.pool2(out)
        out = self.conv3(out)
        out = F.relu(out)

        # change the order back to N X T' X M'
        out = out.permute(0, 2, 1)

        # max pool
        out, _ = torch.max(out, 1)

        # final dense layer
        out = self.fc(out)

        return out

model = CNN(
    n_vocab=len(vocab),
    embed_dim=20,
    n_outputs=1
)
model.to(device)

# Loss and Criterion
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
