import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.metrics import confusion_matrix
import itertools
from torchsummary import summary  # for model summary

# AUGMENTATIONS
transformer_train = torchvision.transforms.Compose([
  # torchvision.transforms.ColorJitter(
  # brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
  transforms.RandomCrop(32, padding=4),
  torchvision.transforms.RandomHorizontalFlip(p=0.5),
  # torchvision.transforms.RandomRotation(degrees=15),
  torchvision.transforms.RandomAffine(0, translate=(0.1, 0.1)),
  # torchvision.transforms.RandomPerspective(),
  transforms.ToTensor(),
])

# load the training data
train_dataset = torchvision.datasets.CIFAR10(
    root='/dataset',
    train=True,
    transform=transformer_train,
    download=True
)

print(train_dataset.data.shape)

# load the test data
test_dataset = torchvision.datasets.CIFAR10(
    root='/dataset',
    train=False,
    transform=transforms.ToTensor(),
    download=True
)

# we are working with color images now
print(train_dataset.data.shape)

# number of classes
K = len(set(train_dataset.targets))
print(f"Number of classes : {K}")

# Data loader
batch_size = 128
train_loader = torch.utils.data.DataLoader(
    dataset=train_dataset,
    batch_size=batch_size,
    shuffle=True)

test_loader = torch.utils.data.DataLoader(
    dataset=test_dataset,
    batch_size=batch_size,
    shuffle=False)

# NOTE: the data transformer mapped the data to (0, 1)
# and also moved the color channel before height/width

# Define the model
class CNN(nn.Module):

    def __init__(self, K):
        super(CNN, self).__init__()

        # define the conv layers
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),  # same padding
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2)
        )  # Instead of using striding convolutions, we applied Pooling

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2)
        )

        # define the linear layers
        self.fc1 = nn.Linear(128 * 4 * 4, 1024)
        self.fc2 = nn.Linear(1024, K)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        x = x.view(x.size(0), -1)

        x = F.dropout(x, p=0.5)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.2)
        x = self.fc2(x)
        return x


model = CNN(K)

# move the model to GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
model.to(device)

# loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())

# training of the model
def batch_gd(model, criterion, optimizer, train_loader, test_loader, epochs):

    train_losses = np.zeros(epochs)
    test_losses = np.zeros(epochs)

    for it in range(epochs):

        model.train()  # model in train mode
        t0 = datetime.now()
        train_loss = []

        for inputs, targets in train_loader:

            # move data to GPU
            inputs, targets = inputs.to(device), targets.to(device)

            # zero the parameter gradient
            optimizer.zero_grad()

            # forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # Backward and optimize
            loss.backward()
            optimizer.step()

            train_loss.append(loss.item())

        # Get train loss
        train_loss = np.mean(train_loss)

        model.eval()  # model in evaluation mode
        test_loss = []

        for inputs, targets in test_loader:

            # move data to GPU
            inputs, targets = inputs.to(device), targets.to(device)

            # forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            test_loss.append(loss.item())

        # Get test loss
        test_loss = np.mean(test_loss)

        # Save the losses
        train_losses[it] = train_loss
        test_losses[it] = test_loss

        dt = datetime.now() - t0

        print(f"Epoch {it + 1}/{epochs}, Train Loss: {train_loss:.4f}, \
        Test Loss:{test_loss:.4f}, Duration: {dt}")

    return train_losses, test_losses


train_losses, test_losses = batch_gd(
    model, criterion, optimizer, train_loader, test_loader, epochs=15
)

# Plot the train loss and test loss per iteration
plt.plot(train_losses, label='train loss')
plt.plot(test_losses, label='test loss')
plt.legend()
plt.show()

# Accuracy

model.eval()
n_correct = 0.
n_total = 0.

for inputs, targets in train_loader:

    # move data to GPU
    inputs, targets = inputs.to(device), targets.to(device)

    # forward pass
    outputs = model(inputs)

    # Get prediction
    _, predictions = torch.max(outputs, 1)

    # update counts
    n_correct += (predictions == targets).sum().item()
    n_total += targets.shape[0]

train_acc = n_correct / n_total

n_correct = 0.
n_total = 0.

for inputs, targets in test_loader:
    # move data to GPU
    inputs, targets = inputs.to(device), targets.to(device)

    # forward pass
    outputs = model(inputs)

    # Get prediction
    _, predictions = torch.max(outputs, 1)

    # update counts
    n_correct += (predictions == targets).sum().item()
    n_total += targets.shape[0]

test_acc = n_correct / n_total
print(f"Train acc: {train_acc:.4f}, Test acc: {test_acc:.4f}")

# Confusion matrix
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
  """
  This function prints and plots the confusion matrix.
  Normalization can be applied by setting `normalize=True`.
  """
  if normalize:
      cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
      print("Normalized confusion matrix")
  else:
      print('Confusion matrix, without normalization')

  print(cm)

  plt.imshow(cm, interpolation='nearest', cmap=cmap)
  plt.title(title)
  plt.colorbar()
  tick_marks = np.arange(len(classes))
  plt.xticks(tick_marks, classes, rotation=45)
  plt.yticks(tick_marks, classes)

  fmt = '.2f' if normalize else 'd'
  thresh = cm.max() / 2.
  for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
      plt.text(j, i, format(cm[i, j], fmt),
               horizontalalignment="center",
               color="white" if cm[i, j] > thresh else "black")

  plt.tight_layout()
  plt.ylabel('True label')
  plt.xlabel('Predicted label')
  plt.show()


# get all predictions in an array and plot confusion matrix

x_test = test_dataset.data
y_test = np.array(test_dataset.targets)
p_test = np.array([])
for inputs, targets in test_loader:
    # move data to GPU
    inputs, targets = inputs.to(device), targets.to(device)

    # Forward pass
    outputs = model(inputs)

    # Get prediction
    _, predictions = torch.max(outputs, 1)

    # update p_test
    p_test = np.concatenate((p_test, predictions.cpu().numpy()))

cm = confusion_matrix(y_test, p_test)
plot_confusion_matrix(cm, list(range(10)))

# label mapping
labels = '''airplane
automobile
bird
cat
deer
dog
frog
horse
ship
truck'''.split()

# Some random misclassified examples
p_test = p_test.astype(np.uint8)
misclassified_idx = np.where(p_test != y_test)[0]
i = np.random.choice(misclassified_idx)
plt.imshow(x_test[i].reshape(32, 32, 3))
plt.title("True label: %s Predicted: %s" % (labels[y_test[i]], labels[p_test[i]]))
plt.show()

print(summary(model, (3, 32, 32)))

# Model Summary
# ----------------------------------------------------------------
#         Layer (type)               Output Shape         Param #
# ================================================================
#             Conv2d-1           [-1, 32, 32, 32]             896
#               ReLU-2           [-1, 32, 32, 32]               0
#        BatchNorm2d-3           [-1, 32, 32, 32]              64
#             Conv2d-4           [-1, 32, 32, 32]           9,248
#               ReLU-5           [-1, 32, 32, 32]               0
#        BatchNorm2d-6           [-1, 32, 32, 32]              64
#          MaxPool2d-7           [-1, 32, 16, 16]               0
#             Conv2d-8           [-1, 64, 16, 16]          18,496
#               ReLU-9           [-1, 64, 16, 16]               0
#       BatchNorm2d-10           [-1, 64, 16, 16]             128
#            Conv2d-11           [-1, 64, 16, 16]          36,928
#              ReLU-12           [-1, 64, 16, 16]               0
#       BatchNorm2d-13           [-1, 64, 16, 16]             128
#         MaxPool2d-14             [-1, 64, 8, 8]               0
#            Conv2d-15            [-1, 128, 8, 8]          73,856
#              ReLU-16            [-1, 128, 8, 8]               0
#       BatchNorm2d-17            [-1, 128, 8, 8]             256
#            Conv2d-18            [-1, 128, 8, 8]         147,584
#              ReLU-19            [-1, 128, 8, 8]               0
#       BatchNorm2d-20            [-1, 128, 8, 8]             256
#         MaxPool2d-21            [-1, 128, 4, 4]               0
#            Linear-22                 [-1, 1024]       2,098,176
#            Linear-23                   [-1, 10]          10,250
# ================================================================
# Total params: 2,396,330
# Trainable params: 2,396,330
# Non-trainable params: 0