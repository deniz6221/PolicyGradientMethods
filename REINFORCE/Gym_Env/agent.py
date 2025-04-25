import torch
from torch.distributions import Normal
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from model import PolicyNet



class Agent:
    def __init__(self, act_dim, obs_dim):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = PolicyNet(obs_dim, act_dim).to(self.device)

        self.optimizer = Adam(self.model.parameters(), lr=3e-4)
        self.scheduler = StepLR(self.optimizer, step_size=1000, gamma=0.99)

        self.states = []
        self.rewards = []
        self.log_probs = []
        self.entropies = []

        self.gamma = 0.99
        self.entropy_coef = 0.1

        self.trajectory_returns = []
        self.baseline = None

    def decide_action(self, state):
        state = state.to(self.device)
        mu, log_std = self.model(state)
        log_std = log_std.clamp(-20, 2)
        mu = torch.tanh(mu)
        dist = Normal(mu, log_std.exp())
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)

        self.states.append(state)
        self.log_probs.append(log_prob)
        self.entropies.append(dist.entropy().mean())

        return action.detach().cpu()


    def decide_action_fixed(self, state):
        state = state.to(self.device)
        mu, _ = self.model(state)
        mu = torch.tanh(mu)
        return mu.detach().cpu()

    def add_reward(self, reward):
        self.rewards.append(reward)

    def update_model(self):
        if not self.rewards:
            return

        returns = []
        R = 0
        for r in reversed(self.rewards):
            R = r + self.gamma * R
            returns.insert(0, R)

        returns = torch.tensor(returns, dtype=torch.float32).to(self.device)
        self.trajectory_returns.append(returns.mean().item())

        if self.baseline is None:
            self.baseline = returns.mean()
        else:
            discount_factor = 0.999
            weights = torch.tensor([discount_factor ** (len(self.trajectory_returns) - i - 1) for i in range(len(self.trajectory_returns))]).to(self.device)
            weights = weights / weights.sum()
            weighted_returns = torch.tensor(self.trajectory_returns, dtype=torch.float32).to(self.device) * weights
            baseline = weighted_returns.sum()
            self.baseline = baseline * 0.05 + self.baseline * 0.95


        advantages = returns - self.baseline

        log_probs = torch.stack(self.log_probs)
        entropies = torch.stack(self.entropies)

        policy_loss = -(log_probs * advantages.detach()).mean()
        entropy_loss = -self.entropy_coef * entropies.mean()

        loss = policy_loss + entropy_loss

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        self.scheduler.step()

        self.states.clear()
        self.rewards.clear()
        self.log_probs.clear()
        self.entropies.clear()

    def save_checkpoint(self, path):
        torch.save({
            "model": self.model.state_dict(),
            "optim": self.optimizer.state_dict(),
            "baseline": self.baseline
        }, path)

    def load_checkpoint(self, path):
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint["model"])
        self.optimizer.load_state_dict(checkpoint["optim"])
        self.baseline = checkpoint["baseline"]

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path))