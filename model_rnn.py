import torch
import torch.nn as nn
import torch.nn.functional as F


class NeuralNetwork(nn.Module):
    def __init__(self, input_size, output_size, hidden_dim=256, n_layers=1):
        super(NeuralNetwork, self).__init__()

        # Defining some parameters
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        # Defining the layers
        # RNN Layer
        self.lstm = nn.LSTM(input_size, hidden_dim, n_layers, batch_first=True)

        # Value
        self.v_fc1 = nn.Linear(hidden_dim, 128)
        self.v_fc2 = nn.Linear(128, 1)

        # Advantage
        self.a_fc1 = nn.Linear(hidden_dim, 128)
        self.a_fc2 = nn.Linear(128, output_size)

    def forward(self, x):
        x = x.unsqueeze(0)

        # Initializing hidden states
        h0 = torch.zeros(self.n_layers, x.size(0), self.hidden_dim)
        c0 = torch.zeros(self.n_layers, x.size(0), self.hidden_dim)

        # Passing in the input and hidden states into the model and obtaining outputs
        out, _ = self.lstm(x, (h0, c0))
        out = out[:, -1, :]

        # Value
        v = F.relu(self.v_fc1(out))
        v = self.v_fc2(v)

        # Advantage
        a = F.relu(self.a_fc1(out))
        a = self.a_fc2(a)

        # Q-Value
        q = v + (a - a.mean(dim=1, keepdim=True))
        return q
