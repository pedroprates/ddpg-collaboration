import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.utils import hidden_init

class Actor(nn.Module):
    """ Actor (policy) model """

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
        super(Actor, self).__init__()
        self.seed = torch.manual_seed(seed)

        self.fc1 = nn.Linear(state_size, l1)
        self.bn1 = nn.BatchNorm1d(l1)
        self.fc2 = nn.Linear(l1, l2)
        self.fc3 = nn.Linear(l2, action_size)

        self.reset_parameters()

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state):
        """ Forward propagation on the Actor (policy) network, mapping states -> actions """
        x = F.relu(self.bn1(self.fc1(state)))
        x = F.relu(self.fc2(x))

        return F.tanh(self.fc3(x))
