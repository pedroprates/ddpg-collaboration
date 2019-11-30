from collections import deque, namedtuple
import random

import numpy as np
import torch

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class ReplayBuffer:
    """ Fixed-size buffer to store experience tuples """

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """ Initialize ReplayBuffer parameters

        Params
        ======
            action_size (int): size of the action space
            buffer_size (int): maximum size of the buffer
            batch_size (int): size of the batch to be sampled
            seed (int): seed to be used on random functions
        """

        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=['state', 'action', 'reward', 'next_state', 'done'])
        random.seed(seed)

    def add(self, state, action, reward, next_state, done):
        """ Add a experience to memory """
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        """ Randomly sample a batch of experiences from memory """
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(DEVICE)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(DEVICE)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(DEVICE)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(DEVICE)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(DEVICE)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """ Returns the current size of the memory """
        return len(self.memory)