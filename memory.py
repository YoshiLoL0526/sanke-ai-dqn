from collections import deque
from random import sample

import numpy as np


class ReplayMemory:
    def __init__(self, capacity, batch_size):
        """Initialize the memory"""
        self.capacity = capacity
        self.batch_size = batch_size
        self.memory = deque(maxlen=capacity)

    def store(self, state, action, new_state, reward, done):
        """Store transition"""
        self.memory.append((state, action, new_state, reward, done))

    def sample(self):
        """Get random sample from the memory"""
        mini_batch = sample(self.memory, self.batch_size)
        states, actions, new_states, rewards, dones = zip(*mini_batch)
        return states, actions, new_states, rewards, dones

    def __len__(self):
        """Return the len of the memory"""
        return len(self.memory)


class NStepReplayMemory:
    def __init__(self, capacity, batch_size, n_step, gamma):
        """Initialize the memory"""
        self.capacity = capacity
        self.batch_size = batch_size
        self.n_step = n_step
        self.gamma = gamma
        self.memory = deque(maxlen=capacity)
        self.n_step_buffer = deque(maxlen=self.n_step)

    def store(self, state, action, new_state, reward, done):
        """Store transition"""
        self.n_step_buffer.append((state, action, new_state, reward, done))
        if len(self.n_step_buffer) < self.n_step:
            return

        # Calculate the reward from n_steps
        reward = 0.0
        for i in range(self.n_step):
            reward += self.gamma ** i * self.n_step_buffer[i][3]

        # Store the transition with new reward
        self.memory.append((state, action, self.n_step_buffer[-1][2], reward, self.n_step_buffer[-1][4]))

    def sample(self):
        """Get random sample from the memory"""
        mini_batch = sample(self.memory, self.batch_size)
        states, actions, new_states, rewards, dones = zip(*mini_batch)
        return states, actions, new_states, rewards, dones

    def clear_buffer(self):
        """Clear n_step buffer"""
        self.n_step_buffer.clear()

    def __len__(self):
        """Return the len of the memory"""
        return len(self.memory)


class PrioritizedNStepReplayMemory:
    def __init__(self, capacity, batch_size, gamma=0.9, n_step=1, alpha=0.6, beta=0.4):
        """Initialize the replay memory"""
        self.capacity = capacity
        self.batch_size = batch_size
        self.gamma = gamma
        self.n_step = n_step
        self.alpha = alpha
        self.beta = beta
        self.buffer = []
        self.pos = 0
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.n_step_buffer = deque(maxlen=self.n_step)

    def calc_multistep_return(self):
        """Calculate multi-step return"""
        reward = 0.0
        for idx in range(self.n_step):
            reward += self.gamma ** idx * self.n_step_buffer[idx][3]

        return self.n_step_buffer[0][0], self.n_step_buffer[0][1], self.n_step_buffer[-1][2], reward, \
            self.n_step_buffer[-1][4]

    def store(self, state, action, next_state, reward, done):
        self.n_step_buffer.append((state, action, next_state, reward, done))
        if len(self.n_step_buffer) >= self.n_step:
            state, action, next_state, reward, done = self.calc_multistep_return()

        max_prio = self.priorities.max() if self.buffer else 1.0  # gives max priority if buffer is not empty else 1

        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, next_state, reward, done))
        else:
            # puts the new data on the position of the oldes since it circles via pos variable
            # since if len(buffer) == capacity -> pos == 0 -> oldest memory (at least for the first round?)
            self.buffer[self.pos] = (state, action, next_state, reward, done)

        self.priorities[self.pos] = max_prio
        self.pos = (self.pos + 1) % self.capacity

    def sample(self):
        N = len(self.buffer)
        if N == self.capacity:
            prios = self.priorities
        else:
            prios = self.priorities[:self.pos]

        # calc P = p^a/sum(p^a)
        probs = prios ** self.alpha
        P = probs / probs.sum()

        # gets the indices depending on the probability p
        indices = np.random.choice(N, self.batch_size, p=P)
        samples = [self.buffer[idx] for idx in indices]

        # Compute importance-sampling weight
        weights = (N * P[indices]) ** (-self.beta)
        # normalize weights
        weights /= weights.max()
        weights = np.array(weights, dtype=np.float32)

        states, actions, next_states, rewards, dones = zip(*samples)
        return states, actions, next_states, rewards, dones, indices, weights

    def update_priorities(self, batch_indices, batch_priorities):
        for idx, prio in zip(batch_indices, batch_priorities):
            self.priorities[idx] = prio

    def clear_buffer(self):
        """Clear n_step buffer"""
        self.n_step_buffer.clear()

    def __len__(self):
        return len(self.buffer)


class PrioritizedNStepReplayBuffer:
    def __init__(self, buffer_size, batch_size, n_step=1, gamma=0.9, alpha=0.6, beta=0.4):
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.n_step = n_step
        self.gamma = gamma
        self.alpha = alpha
        self.beta = beta
        self.buffer = deque(maxlen=buffer_size)
        self.priorities = deque(maxlen=buffer_size)
        self.n_step_buffer = deque(maxlen=n_step)
        self.max_priority = 1.0

    def store(self, state, action, new_state, reward, done):
        """Store a transition"""
        self.n_step_buffer.append((state, action, new_state, reward, done))
        if len(self.n_step_buffer) < self.n_step:
            return

        # Calculate reward
        reward = 0.0
        for i in range(self.n_step):
            reward += self.gamma ** i * self.n_step_buffer[i][3]

        self.buffer.append((state, action, self.n_step_buffer[-1][2], reward, self.n_step_buffer[-1][4]))
        self.priorities.append(self.max_priority)

    def sample(self):
        """Get random sample based on priorities"""
        priorities = np.array(self.priorities)
        probabilities = priorities ** self.alpha / np.sum(priorities ** self.alpha)

        indices = np.random.choice(len(self.buffer), size=self.batch_size, replace=False, p=probabilities)
        samples = [self.buffer[i] for i in indices]
        weights = (len(self.buffer) * probabilities[indices]) ** -self.beta
        weights /= np.max(weights)

        return samples, weights, indices

    def update_priorities(self, indices, priorities):
        """Update priorities"""
        for i, priority in zip(indices, priorities):
            # self.priorities[i] = priority
            self.priorities[i] = priority[0]
            # self.max_priority = max(self.max_priority, priority)
            self.max_priority = max(self.max_priority, priority[0])

    def clear_buffer(self):
        """Clear the buffer for n steps"""
        self.n_step_buffer.clear()

    def __len__(self):
        """Return length of the memory"""
        return len(self.buffer)
