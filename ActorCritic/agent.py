import torch
from torch import optim

from model import *
import torch.nn.functional as F
from collections import deque
import random
gamma = 0.99


class Agent():
    def __init__(self):
        self.actor = Actor()
        self.critic = QCritic()
        self.target_critic = QCritic()
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.target_critic.eval()
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=1e-4)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=1e-4)
        self.replay_buffer = deque(maxlen=10000)
        
    def decide_action(self, state):
        action_mean, act_std = self.actor(state).chunk(2, dim=-1)
        action_std = torch.clamp(act_std, min=-20, max=2)
        action_std = torch.exp(action_std)

        dist = torch.distributions.Normal(action_mean, action_std)
        action = dist.sample()

        return action.detach()
    
    def get_action_with_probs(self, state):
        action_mean, act_std = self.actor(state).chunk(2, dim=-1)
        action_std = torch.clamp(act_std, min=-20, max=2)
        action_std = torch.exp(action_std)

        dist = torch.distributions.Normal(action_mean, action_std)
        action = dist.rsample()

        log_prob = dist.log_prob(action).sum(dim=-1)
        return action.detach(), log_prob

    
    def update_model(self):
        # Implement the soft actor critic update
        if len(self.replay_buffer) < 1000:
            return
        batch = random.sample(self.replay_buffer, 64)
        states, actions, rewards, next_states, dones = zip(*batch)
        states = torch.stack(states)
        actions = torch.stack(actions)
        rewards = torch.stack(rewards)
        next_states = torch.stack(next_states)
        dones = torch.stack(dones)

        # Update critic
        with torch.no_grad():
            next_actions, next_log_probs = self.get_action_with_probs(next_states)
            target_q = self.target_critic(next_states, next_actions)
            target_q = rewards + (1 - dones) * gamma * (target_q - next_log_probs)
        q = self.critic(states, actions)
        critic_loss = F.mse_loss(q, target_q)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Update actor
        actions, log_probs = self.get_action_with_probs(states)
        q = self.critic(states, actions)
        actor_loss = (log_probs - q).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

    def save_checkpoint(self, filename):
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'target_critic_state_dict': self.target_critic.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict()
        }, filename)


    def load_checkpoint(self, filename):
        checkpoint = torch.load(filename)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.target_critic.load_state_dict(checkpoint['target_critic_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        

    
        
