import random
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam

from models.actor import Actor
from models.critic import Critic
from utils.noise import OrsnteinUhlenbeck
from utils.replay import ReplayBuffer

LR_ACTOR = 5e-4
LR_CRITIC = 2e-4
WEIGHT_DECAY = 0
TAU = 2e-3
BUFFER_SIZE = int(1e5)
BATCH_SIZE = 128
LEARN_STEPS = 1
N_UPDATES = 1
GAMMA = .99
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class Agent:
    """ Interacts with and learn from the environment """

    def __init__(self, state_size, action_size, random_seed, actor_layers, critic_layers):
        """ Initialize an Agent object

        Params
        ======
            state_size (int): size of the environment state
            action_size (int): size of the environment action
            random_seed (int): seed for the random functions
            actor_layers (array[int]): array containing the size of each layer of the actor network
            critic_layers (array[int]): array containing the size of each layer of the critic network
        """
        assert len(actor_layers) == 2, 'The size of the actor network must be 2'
        assert len(critic_layers) == 2, 'The size of the critic network must be 2'

        self.state_size = state_size
        self.action_size = action_size
        self.random_seed = random_seed
        self.seed = random.seed(random_seed)
        np.random.seed(random_seed)

        # Actor
        self.actor_local = Actor(self.state_size, self.action_size, self.random_seed, *actor_layers).to(DEVICE)
        self.actor_target = Actor(self.state_size, self.action_size, self.random_seed, *actor_layers).to(DEVICE)
        self.actor_optimizer = Adam(self.actor_local.parameters(), lr=LR_ACTOR)

        # Critic
        self.critic_local = Critic(self.state_size, self.action_size, self.random_seed, *critic_layers).to(DEVICE)
        self.critic_target = Critic(self.state_size, self.action_size, self.random_seed, *critic_layers).to(DEVICE)
        self.critic_optimizer = Adam(self.critic_local.parameters(), lr=LR_CRITIC, weight_decay=WEIGHT_DECAY)

        # Noise - using Ornstein-Uhlenbeck
        self.noise = OrsnteinUhlenbeck(self.action_size, scale=1.0, sigma=0.1)

        # ReplayBuffer
        self.memory = ReplayBuffer(self.action_size, BUFFER_SIZE, BATCH_SIZE, self.random_seed)

    def step(self, states, actions, rewards, next_states, dones, time_step):
        """ Save experience in replay memory, and use random sample from buffer to learn """
        for state, action, reward, next_state, done in zip(states, actions, rewards, next_states, dones):
            self.memory.add(state, action, reward, next_state, done)

        # Learn only if there is enough samples on memory
        if len(self.memory) > BATCH_SIZE and time_step % LEARN_STEPS == 0:
            for _ in range(N_UPDATES):
                experiences = self.memory.sample()
                self.learn(experiences, GAMMA)

    def act(self, state, add_noise=True, epsilon=1.0):
        """ Returns actions for given state as per current policy """
        state = torch.from_numpy(state).float().to(DEVICE)
        self.actor_local.eval()

        with torch.no_grad():
            actions = self.actor_local(state).cpu().data.numpy()
        
        if add_noise:
            actions += self.noise.noise() * epsilon
            # actions += np.random.normal(0, .3) * epsilon
        
        return np.clip(actions, -1, 1)

    def reset(self):
        self.noise.reset()

    def learn(self, experiences, gamma):
        """ Update policy and value parameters using given batch of experience tuples
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """

        states, actions, rewards, next_states, dones = experiences

        # Critic update
        actions_next = self.actor_target(next_states)
        Q_targets_next = self.critic_target(next_states, actions_next)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        Q_expected = self.critic_local(states, actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1)
        self.critic_optimizer.step()

        # Actor update
        actions_pred = self.actor_local(states)
        actor_loss = -self.critic_local(states, actions_pred).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Update weights
        self.soft_update(self.actor_local, self.actor_target, TAU)
        self.soft_update(self.critic_local, self.critic_target, TAU)

    def soft_update(self, local_model, target_model, tau):
        """ Soft update model parameters
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will copied from
            target_model (PyTorch model): weights will copied to
            tau (float): interpolation parameter
        """

        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1 - tau) * target_param.data)
