import numpy as np
import pandas as pd

def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)

    return -lim, lim

def plot_result(scores):
    """Plot result chart"""
    fig, ax = plt.subplots(figsize=(10, 7))
    pd.DataFrame(scores).max(axis=1).plot(color='b', ax=ax)
    pd.DataFrame(scores).max(axis=1).rolling(100, min_periods=1).mean().plot(color='orange', ax=ax)
    ax.plot(np.arange(len(scores)), np.repeat(.5, len(scores)), color='r')
    ax.legend(['Episode Max Score', 'Moving Avg (100 episodes)', 'Target'])
    ax.set_ylabel('Score')
    ax.set_xlabel('Episode #')
    ax.set_title('Moving Average (100 episodes window) ')
