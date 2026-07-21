from torch import nn
from numpy as np

class Actor(nn.Module):
    def __init__(self, in_layer, hidden, output, env):
        self.input = in_layer
        self.hidden = hidden
        self.output = output
        
        obs_space = env.single_observation_space
        self.observation_dim = obs_space["observation"].shape[0]
        self.goal_dim = obs_space["desired_goal"].shape[0]
        
          