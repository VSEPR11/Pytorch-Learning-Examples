import torch
from torch import nn


class CNN(nn.Module):
    def __init__(self, device=torch.device('cpu')):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 3, 1, 1, bias=True, device=device)
        self.conv2 = nn.Conv2d(6, 16, 5, 1, 0, bias=True, device=device)
        self.pool1 = nn.MaxPool2d(2)
        self.pool2 = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(400, 120, device=device)
        self.fc2 = nn.Linear(120, 84, device=device)
        self.fc3 = nn.Linear(84, 10, device=device)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = x.view(-1, 400)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)
