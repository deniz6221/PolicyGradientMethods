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
        if not self.rewards:
            return
        returns = []
        Rt = 0

        for r in reversed(self.rewards):
            Rt = r + self.gamma * Rt
            returns.insert(0, Rt)

        returns = torch.tensor(returns, dtype=torch.float32)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)  

        policy_loss = torch.stack(self.log_probs) * returns.detach()
        loss = -policy_loss.mean()
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Clear memory
        self.rewards.clear()
        self.log_probs.clear()

    def add_reward(self, reward):
        self.rewards.append(reward)
        
