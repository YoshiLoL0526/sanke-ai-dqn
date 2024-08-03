import torch.nn as nn
import torch.nn.functional as F


class NeuralNetwork(nn.Module):
    def __init__(self, n_inputs, n_actions):
        """Initialize the neural network"""
        super(NeuralNetwork, self).__init__()
        self.n_inputs = n_inputs
        self.n_actions = n_actions

        self.net = nn.Sequential(
            nn.Linear(self.n_inputs, 256),
            nn.ReLU(inplace=True),
        )

        # Value stream
        self.value = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1),
        )

        # Advantage stream
        self.advantage = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, self.n_actions),
        )

    def forward(self, x):
        """Do feed forward"""
        x = x.view(-1, self.n_inputs)
        x = self.net(x)
        v = self.value(x)
        a = self.advantage(x)
        q = v + (a - a.mean(dim=1, keepdim=True))
        return q
