import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from memory import PrioritizedNStepReplayBuffer
# from model_crnn import NeuralNetwork
from model_cnn import NeuralNetwork
# from model import NeuralNetwork
# from model_rnn import NeuralNetwork


# Coming soon...
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Agent:
    def __init__(self, n_inputs, n_actions, batch_size=64, gamma=0.9, lr=1e-3, n_step=3, capacity=10000):
        """Initialize the agent"""
        self.BATCH_SIZE = batch_size
        self.GAMMA = gamma
        self.lr = lr
        self.epsilon = 1.0
        self.n_step = n_step

        self.n_inputs = n_inputs
        self.n_actions = n_actions

        self.policy_net = NeuralNetwork(n_inputs, n_actions).to(device)
        self.target_net = NeuralNetwork(n_inputs, n_actions).to(device)
        self.target_net.eval()
        self.update_target_network()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

        self.memory = PrioritizedNStepReplayBuffer(buffer_size=capacity, batch_size=self.BATCH_SIZE, n_step=self.n_step, gamma=self.GAMMA)

    def load_model(self, file_name='model.pth'):
        """Load last saved model"""
        model_folder_path = './model'
        if os.path.exists(model_folder_path):
            file_name = os.path.join(model_folder_path, file_name)
            self.policy_net.load_state_dict(torch.load(file_name, map_location=device))
            self.target_net.load_state_dict(self.policy_net.state_dict())

    def save_model(self, file_name='model.pth'):
        """Save model checkpoint"""
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)
        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.policy_net.state_dict(), file_name)

    def select_action(self, state):
        """Select best actions or do random actions"""
        if random.random() <= self.epsilon:
            action = random.randint(0, self.n_actions - 1)
            self.epsilon = max(self.epsilon * 0.99, 0.01)
        else:
            self.policy_net.eval()
            state = torch.tensor(state, dtype=torch.float).to(device)
            with torch.no_grad():
                prediction = self.policy_net(state)
            action = torch.argmax(prediction).item()
        return action

    def remember(self, state, action, new_state, reward, done):
        """Save experience into the memory"""
        self.memory.store(state, action, new_state, reward, done)

    def clear_memory_buffer(self):
        """Clear memory buffer"""
        self.memory.clear_buffer()

    def optimize_model(self):
        """Training of mini batch"""
        if len(self.memory) < self.BATCH_SIZE:
            return

        self.policy_net.train()

        # Get mini-batch sample
        samples, weights, indices = self.memory.sample()
        states, actions, new_states, rewards, dones = zip(*samples)

        # To tensor everybody
        states = torch.tensor(np.array(states, dtype=float), dtype=torch.float32).to(device)
        actions = torch.tensor(actions, dtype=torch.long).unsqueeze(1).to(device)
        new_states = torch.tensor(np.array(new_states, dtype=float), dtype=torch.float32).to(device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        dones = torch.tensor(dones, dtype=torch.int8).to(device)
        weights = torch.tensor(weights, dtype=torch.float32).unsqueeze(1).to(device)

        # Predict q value for the current states
        q_value = self.policy_net(states).gather(1, actions)
        next_q_value = self.target_net(new_states).max(1)[0].detach()

        # Calculate new q value
        targets = rewards + self.GAMMA ** self.n_step * (1 - dones) * next_q_value

        # Compute loss
        td_error = q_value - targets.unsqueeze(1)
        loss = (td_error.pow(2) * weights).mean().to(device)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.memory.update_priorities(indices, abs(td_error.data.cpu().numpy()))

        # LSTM
        # idxs = []
        # errors = []
        # self.optimizer.zero_grad()
        # samples, weights, indices = self.memory.sample()
        # for sample, weight, idx in zip(samples, weights, indices):
        #     # Unpack
        #     state, action, new_state, reward, done = sample
        #
        #     # Convert to tensor
        #     state = torch.tensor(np.array(state, dtype=float), dtype=torch.float32).to(device)
        #     action = torch.tensor(action, dtype=torch.long).to(device)
        #     new_state = torch.tensor(np.array(new_state, dtype=float), dtype=torch.float32).to(device)
        #     reward = torch.tensor(reward, dtype=torch.float32).to(device)
        #     done = torch.tensor(done, dtype=torch.int8).to(device)
        #     weight = torch.tensor(weight, dtype=torch.float32).to(device)
        #
        #     # Predict q value for the current states
        #     q_value = self.policy_net(state)
        #     next_q_value = self.target_net(new_state)
        #
        #     # Calculate new q value
        #     target = reward + self.GAMMA ** self.n_step * (1 - done) * torch.max(next_q_value, dim=1)[0]
        #
        #     # Compute loss
        #     td_error = q_value.squeeze(0)[action] - target[0]
        #     loss = (td_error.pow(2) * weight).mean().to(device)
        #
        #     loss.backward()
        #
        #     idxs.append(idx)
        #     errors.append(abs(td_error.data.cpu().numpy()))
        #
        # self.optimizer.step()
        #
        # self.memory.update_priorities(idxs, errors)

    def update_target_network(self):
        """Update target network"""
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def reduce_learning_rate(self):
        """Update learning rate"""
        self.lr *= .9
        optim.Adam(self.policy_net.parameters(), lr=self.lr)
