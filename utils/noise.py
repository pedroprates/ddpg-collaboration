import numpy as np
import torch


class OrsnteinUhlenbeck:
    """ Orsntein-Uhlenbeck noise process """
    
    def __init__(self, action_state, scale=0.1, mu=0, theta=0.15, sigma=0.2):
        self.action_state = action_state
        self.scale = scale
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(self.action_state) * self.mu
        self.reset()

    def reset(self):
        self.state = np.ones(self.action_state) * self.mu

    def noise(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        
        return torch.tensor(self.state * self.scale).float()
