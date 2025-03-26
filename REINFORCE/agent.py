import torch
from torch import optim
import torch.utils

from model import VPG
import torch.nn.functional as F
from torch.distributions import Normal

gamma = 0.99
learning_rate = 5e-6

class Agent():
    def __init__(self):
        
        self.model = VPG()
        self.optimizer = optim.Adam(params=self.model.parameters(), lr=learning_rate)
        self.rewards = []
        self.log_probs = []
        
    def decide_action(self,  state):
        action_mean, act_std = self.model(state).chunk(2, dim=-1)
        action_std = act_std.exp() + 0.05
        dist = Normal(loc=action_mean, scale=action_std)
        action = dist.rsample()
        prob = dist.log_prob(action).sum(-1).unsqueeze(-1)
        self.log_probs.append(prob)
        return action.detach()
    
    def update_model(self):
        if not self.rewards:
            return
        
        loss_lst = []
        returns = []
        Rt = 0

        for r in reversed(self.rewards):
            Rt = r + gamma * Rt
            returns.insert(0, Rt)

        returns = torch.tensor(returns, dtype=torch.float32)
        #returns = (returns - returns.mean()) / (returns.std() + 1e-5)  

        for i in range(len(returns)):
            loss_lst.append(-self.log_probs[i]*returns[i])

        
        self.optimizer.zero_grad()
        loss = torch.cat(loss_lst).mean()
        loss.backward()
        torch.nn .utils.clip_grad_norm_(self.model.parameters(), 0.5)
        self.optimizer.step()

        # Clear memory
        del self.rewards[:]
        del self.log_probs[:]

    def add_reward(self, reward):
        self.rewards.append(reward)
        
