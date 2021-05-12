# Pipeline :
# Imports
# Create Fully Connected Networks
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

# Step 2 : Create Fully Connected Networks
class NN(nn.Module):

    def __init__(self, input_size, num_classes):
        super(NN, self).__init__() # calls the init of parent class
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# test the code till here :
# model = NN(784, 10) # MNIST dataset, image.shape => (28, 28) --> flatten -> 784
# x = torch.randn(64, 784) # batch_size = 64
# print(model(x).shape)

# Step 3 : Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Step 4 : Hyperparameters
input_size = 784
num_classes = 10
learning_rate = 0.001
batch_size = 64
num_epoch = 3

# Step 5 : Load Data
# download the dataset and convert it to a tensor
train_dataset = datasets.MNIST(root='dataset/', train=True, transform=transforms.ToTensor(), download=True)

# creating a data loader for training using batches
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = datasets.MNIST(root='dataset/', train=False, transform=transforms.ToTensor(), download=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

# Step 6 : Init Network
model = NN(input_size=input_size, num_classes=num_classes).to(device)

# Step 7 : Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Step 8 : Train Network

for epoch in range(num_epoch):
    for batch_idx, (data, targets) in enumerate(train_loader):
        # Get data to cuda if possible
        data = data.to(device=device)
        targets = targets.to(device=device)

        data = data.reshape(data.shape[0], -1) # Flatten the images

        # forward
        scores = model(data)
        loss = criterion(scores, targets)

        # backward
        optimizer.zero_grad() # making grad to 0
        # so that it doesn't use the gradient from the previous batch
        loss.backward()

        # gradient descent or adam step
        optimizer.step()

# check accuracy on training and test to see how good our model is

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
            x = x.reshape(x.shape[0], -1)

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
# Got 58467 / 60000 with accuracy 97.45
# Checking Accuracy on test data
# Got 9681 / 10000 with accuracy 96.81

########################################################