import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.utils import reset_parameters

class Critic(nn.Module):
    """ Critic (value) model """

    def __init__(self, state_size, action_size, seed, l1, l2):
        """ Initialize parameters and build the model

        Params
        ======
            state_size (int): size of the environment state
            action_size (int): size of the environment action
            seed (int): seed for the random functions
            l1 (int): size of the first layer
            l2 (int): size of the second layer
        """
        super(Critic, self).__init__()

        self.fc1 = nn.Linear(state_size, l1)
        self.bn1 = nn.BatchNorm1d(l1)
        self.fc2 = nn.Linear(state_size+action_size, l2)
        self.fc3 = nn.Linear(l2, 1)

        reset_parameters(self.fc1, self.fc2, self.fc3)

    def forward(self, state, action):
        x = F.relu(self.bn1(self.fc1(state)))
        sa = torch.cat((x, action), dim=1)
        sa = F.relu(self.fc2(sa))

        return self.fc3(sa)
