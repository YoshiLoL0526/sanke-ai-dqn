import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        """Initialize the neural network"""
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        """Do feed forward"""
        identity = x
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += identity
        x = self.relu(x)
        return x


class NeuralNetwork(nn.Module):
    def __init__(self, n_inputs, n_actions, hidden_dim=256, n_hidden=1):
        """Initialize the neural network"""
        super(NeuralNetwork, self).__init__()

        self.n_inputs = n_inputs
        self.n_actions = n_actions
        self.hidden_dim = hidden_dim
        self.n_hidden = n_hidden

        # Convolutional module
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=self.n_inputs, out_channels=64, kernel_size=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            *[ResidualBlock(64, 64) for _ in range(4)],
            nn.Conv2d(in_channels=64, out_channels=1, kernel_size=1),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True),
            nn.Flatten()
        )

        # Defining the layers
        # RNN Layer
        self.lstm = nn.LSTM(1 * 8 * 8, self.hidden_dim, self.n_hidden, batch_first=True)

        # Value stream
        self.value = nn.Sequential(
            nn.Linear(self.hidden_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 1),
        )

        # Advantage stream
        self.advantage = nn.Sequential(
            nn.Linear(self.hidden_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, self.n_actions),
        )

    def forward(self, x):
        """Do feed forward"""
        x = x.view(-1, 5, 8, 8)
        x = self.conv(x)

        x = x.unsqueeze(0)

        # Initializing hidden states
        h0 = torch.zeros(self.n_hidden, x.size(0), self.hidden_dim)
        c0 = torch.zeros(self.n_hidden, x.size(0), self.hidden_dim)

        # Passing in the input and hidden states into the model and obtaining outputs
        out, _ = self.lstm(x, (h0, c0))
        out = out[:, -1, :]

        v = self.value(out)
        a = self.advantage(out)
        q = v + (a - a.mean(dim=1, keepdim=True))
        return q
