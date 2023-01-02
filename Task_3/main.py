import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import FashionMNIST
from torchvision.transforms import ToTensor
import time
import torch
import torch_directml

dml = torch_directml.device()

start = time.time()


# CNN model
class CNN(nn.Module):
    def __init__(self, output_dim):
        super(CNN, self).__init__()

        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.relu3 = nn.ReLU()
        self.drop = nn.Dropout(p=0.2)
        self.fc2 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        x = x.view(x.size(0), -1)

        x = self.fc1(x)
        x = self.relu3(x)
        x = self.drop(x)
        x = self.fc2(x)

        return x


# Load the Fashion-MNIST dataset
trainData = FashionMNIST(root='./data', train=True, transform=ToTensor(), download=True)
testData = FashionMNIST(root='./data', train=False, transform=ToTensor(), download=True)

# Data loaders
trainLoader = DataLoader(trainData, batch_size=32, shuffle=True)
testLoader = DataLoader(testData, batch_size=32, shuffle=False)

# Model, loss function, and optimizer
model = CNN(output_dim=10)
lossCalc = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001)

# Epochs and Storage
epochs = 10
trainLosses = []
testLosses = []

# Training
model.train()
for epoch in range(epochs):

    trainLoss = 0

    for data, labels in trainLoader:
        optimizer.zero_grad()
        output = model(data)  # Forward Pass is here
        loss = lossCalc(output, labels)
        loss.backward()
        optimizer.step()
        trainLoss = trainLoss + loss.item() * data.size(0)

    trainEnd = time.time()
    trainTime = trainEnd - start
    trainLoss = trainLoss / len(trainLoader.dataset)
    trainLosses.append(trainLoss)
    print("Epoch: " + str(epoch + 1) + " Losses in training: " + str(trainLoss) + " Time in Seconds: " + str(trainTime))

# Testing
testLoss = 0
correct = 0
total = 0
model.eval()
for data, target in testLoader:
    optimizer.zero_grad()
    output = model(data)
    _, prediction = torch.max(output, 1)
    total = total + target.size(0)
    correct = correct + (prediction == target).sum().item()

end = time.time()
elapsedTime = end - start
print("Time taken overall: " + str(elapsedTime) + " Seconds")

accuracy = 100 * correct / total
print('Accuracy: ' + str(accuracy))
