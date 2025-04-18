import torch
from torch import optim
import torch.utils

from model import VPG
import torch.nn.functional as F
from torch.distributions import Normal

gamma = 0.99
learning_rate = 3e-4

class Agent():
    def __init__(self):
        
        self.model = VPG()
        self.optimizer = optim.AdamW(params=self.model.parameters(), lr=learning_rate, weight_decay=1e-3)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=1000, gamma=0.99)
        self.rewards = []
        self.log_probs = []
        
    def decide_action(self,  state):
        action_mean, act_std = self.model(state).chunk(2, dim=-1)
        action_mean = torch.tanh(action_mean)
        act_std = torch.clamp(act_std, min=-20, max=2)
        action_std = act_std.exp() 
        dist = Normal(loc=action_mean, scale=action_std)
        action = dist.rsample()
        prob = dist.log_prob(action).sum(-1).unsqueeze(-1)
        self.log_probs.append(prob.clone())
        return action.detach()
    
    def update_model(self):
        if not self.rewards:
            return
        
        if len(self.rewards) < 2:
            self.rewards.clear()
            self.log_probs.clear()
            return
        
        loss_lst = []
        returns = []
        Rt = 0

        for r in reversed(self.rewards):
            Rt = r + gamma * Rt
            returns.insert(0, Rt)

        returns = torch.tensor(returns, dtype=torch.float32)

        """ returns = returns - returns.mean()
        returns_std = returns.std()
        if returns_std > 1e-6:
            returns = returns / (returns_std + 1e-5) """
        
        if returns.std() < 1e-6:
            returns = returns - returns.mean()  
        else:
            returns = (returns - returns.mean()) / (returns.std() + 1e-5)

        for i in range(len(returns)):
            loss_lst.append(-self.log_probs[i]*returns[i])

        
        self.optimizer.zero_grad()
        loss = torch.cat(loss_lst).sum()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        self.scheduler.step()

        # Clear memory
        del self.rewards[:]
        del self.log_probs[:]

    def add_reward(self, reward):
        self.rewards.append(reward)

    def save_checkpoint(self, path):
        torch.save({"model": self.model.state_dict(), "optim": self.optimizer.state_dict()}, path)
    
    def load_checkpoint(self, path):
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint["model"])
        self.optimizer.load_state_dict(checkpoint["optim"])
        