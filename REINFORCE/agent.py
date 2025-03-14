import torch
from torch import optim

from model import VPG
import torch.nn.functional as F
from torch.distributions import Normal

gamma = 0.99
learning_rate = 0.001

class Agent():
    def __init__(self):
        
        self.model = VPG()
        self.optimizer = optim.Adam(params=self.model.parameters(), lr=learning_rate)
        self.rewards = []
        self.log_probs = []
        
    def decide_action(self, state):
        
        action_mean, act_std = self.model(state).chunk(2, dim=-1)
        action_std = F.softplus(act_std) + 5e-2  # increase variance to stimulate exploration
        dist = Normal(loc=action_mean, scale=action_std)
        action = dist.sample()
        prob = dist.log_prob(action)
        self.log_probs.append(prob)
        return action
    
    def update_model(self):
        step_count = len(self.rewards)
        if step_count == 0:
            return
        
        Rt = self.rewards[-1]
        exp_return = Rt* self.log_probs[-1]

        for i in range(step_count - 2, -1):
            Rt = self.rewards[i] + Rt * gamma
            exp_return += Rt* self.log_probs[i]
        
        loss = -exp_return
        self.optimizer.zero_grad()
        loss.backwards()
        self.optimizer.step() 

        self.rewards.clear()
        self.log_probs.clear()

        return

    def add_reward(self, reward):
        self.rewards.append(reward)
        
