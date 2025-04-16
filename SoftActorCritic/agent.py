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
        self.critic = QCritic(input_dim=8, output_dim=1)
        self.target_critic = QCritic(input_dim=8, output_dim=1)
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.target_critic.eval()
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=3e-4)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=3e-4)
        self.replay_buffer = deque(maxlen=100000)
        self.alpha = 0.2
    def decide_action(self, state):
        action_mean, act_std = self.actor(state).chunk(2, dim=-1)
        action_std = torch.clamp(act_std, min=-20, max=2)
        action_std = torch.exp(action_std)

        dist = torch.distributions.Normal(action_mean, action_std)
        action = dist.sample()
        action = torch.tanh(action)

        return action.detach()
    
    def get_action_with_probs(self, state):
        action_mean, act_std = self.actor(state).chunk(2, dim=-1)
        action_std = torch.clamp(act_std, min=-2, max=2)
        action_std = torch.exp(action_std)

        dist = torch.distributions.Normal(action_mean, action_std)
        action_x = dist.rsample()

        action = torch.tanh(action_x)
        

        log_prob = dist.log_prob(action_x) - torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)

        return action, log_prob

    
    def update_model(self):
        # Implement the soft actor critic update
        if len(self.replay_buffer) < 1000:
            return
        batch = random.sample(self.replay_buffer, 256)
        states, actions, rewards, next_states, dones = zip(*batch)
        states = torch.stack(states)
        actions = torch.stack(actions)
        rewards = torch.stack(rewards)
        next_states = torch.stack(next_states)
        dones = torch.tensor(dones, dtype=torch.float32)



        # Update critic
        with torch.no_grad():
            next_actions, next_log_probs = self.get_action_with_probs(next_states)
            target_q1, target_q2 = self.target_critic(next_states, next_actions)
            target_q = torch.min(target_q1, target_q2)
            target_q = target_q.squeeze(-1)
            target_q = rewards + (1 - dones) * gamma * (target_q - self.alpha * next_log_probs)
        q_1, q_2 = self.critic(states, actions)
        q_1 = q_1.squeeze(-1)
        q_2 = q_2.squeeze(-1)
        critic_loss = F.mse_loss(q_1, target_q) + F.mse_loss(q_2, target_q)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Update actor
        actions, log_probs = self.get_action_with_probs(states)
        q1,q2 = self.critic(states, actions)
        q = torch.min(q1, q2)
        q = q.squeeze(-1)
        actor_loss = (self.alpha * log_probs - q).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Soft update target critic
        self.soft_update()

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
        
    def soft_update(self, tau=0.005):
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
    
        
