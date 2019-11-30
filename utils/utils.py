import numpy as np

def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)

    return -lim, lim

def reset_parameters(fc1, fc2, fc3):
        fc1.weight.data.uniform_(*hidden_init(fc1))
        fc2.weight.data.uniform_(*hidden_init(fc2))
        fc3.weight.data.uniform_(-3e-3, 3e-3)