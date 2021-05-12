# Pipeline :
# Imports
# Create a CNN
# Set device
# Hyperparameters
# Load data
# Init network
# Loss and Optimizer
# Train Network
# Check accuracy on training and test for model performance

# Step 1 : Imports
import torch
import torch.nn as nn # Neural Network modules, loss functions, Activation functions
import torch.optim as optim # Optimizers like SGD, ADAM, etc
import torch.nn.functional as F # Activation functions like tanh, relu, etc
from torch.utils.data import DataLoader # for mini batches
import torchvision.datasets as datasets # for datasets
import torchvision.transforms as transforms # for transformations on dataset

# Step 2 : Create a CNN
class CNN(nn.Module):
    def __init__(self, in_channels=1, num_classes=10):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)) # same conv
        self.pool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)) # same conv
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.fc1 = nn.Linear(32*4*4, 128)
        self.dropout = nn.Dropout(p=0.3)
        self.fc2 = nn.Linear(128, 32)
        self.fc3 = nn.Linear(32, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        x = x.reshape(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x


# Step 3 : Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Step 4 : Hyperparameters
in_channels = 1
num_classes = 10
learning_rate = 0.001
batch_size = 64
num_epoch = 10

# Step 5 : Load Data
# download the dataset and convert it to a tensor
train_dataset = datasets.MNIST(root='dataset/', train=True, transform=transforms.ToTensor(), download=True)

# creating a data loader for training using batches
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = datasets.MNIST(root='dataset/', train=False, transform=transforms.ToTensor(), download=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

# Step 6 : Init Network
model = CNN().to(device)

# Step 7 : Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Step 8 : Train Network

for epoch in range(num_epoch):
    for batch_idx, (data, targets) in enumerate(train_loader):
        # Get data to cuda if possible
        data = data.to(device=device)
        targets = targets.to(device=device)

        # forward
        scores = model(data)
        loss = criterion(scores, targets)

        # backward
        optimizer.zero_grad() # making grad to 0
        # so that it doesn't use the gradient from the previous batch
        loss.backward()

        # gradient descent or adam step
        optimizer.step()

# Step 9 : check accuracy on training and test to see how good our model is
def check_accuracy(loader, model):

    if loader.dataset.train:
        print("Checking Accuracy on training data")
    else:
        print("Checking Accuracy on test data")

    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device)
            y = y.to(device=device)

            scores = model(x) # 64 X 10
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)

        print(f"Got {num_correct} / {num_samples} with accuracy {float(num_correct)/float(num_samples) * 100:.2f}")

    model.train()

check_accuracy(train_loader, model)
check_accuracy(test_loader, model)

########################################################
# OUTPUT :

# Checking Accuracy on training data
# Got 58764 / 60000 with accuracy 97.94
# Checking Accuracy on test data
# Got 9707 / 10000 with accuracy 97.07

########################################################