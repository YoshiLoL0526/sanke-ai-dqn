import torch.nn as nn


# class ResidualBlock(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         """Initialize the neural network"""
#         super(ResidualBlock, self).__init__()
#
#         self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1)
#         # self.bn1 = nn.BatchNorm2d(out_channels)
#         self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1)
#         # self.bn2 = nn.BatchNorm2d(out_channels)
#         self.relu = nn.ReLU(inplace=True)
#
#     def forward(self, x):
#         """Do feed forward"""
#         identity = x
#         # x = self.relu(self.bn1(self.conv1(x)))
#         # x = self.bn2(self.conv2(x))
#         x = self.relu(self.conv1(x))
#         x = self.conv2(x)
#         x += identity
#         x = self.relu(x)
#         return x


class NeuralNetwork(nn.Module):
    def __init__(self, n_inputs, n_actions):
        """Initialize the neural network"""
        super(NeuralNetwork, self).__init__()

        # Convolutional module
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=n_inputs, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.Flatten(),
        )

        # Value stream
        self.value = nn.Sequential(
            nn.Linear(64 * 2 * 2, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
        )

        # Advantage stream
        self.advantage = nn.Sequential(
            nn.Linear(64 * 2 * 2, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, n_actions),
        )

    def forward(self, x):
        """Do feed forward"""
        x = x.view(-1, 5, 8, 8)
        x = self.conv(x)
        v = self.value(x)
        a = self.advantage(x)
        q = v + (a - a.mean(dim=1, keepdim=True))
        return q
