import torch.nn as nn
import torch
def apply_custom_init(module):
    if isinstance(module, nn.Linear):
        if module.out_features == 128:
            nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
        else:
            nn.init.xavier_normal_(module.weight, gain=0.01)
        nn.init.constant_(module.bias, 0.0)


class PolicyNet(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )
        self.mu_head = nn.Linear(128, act_dim)
        
        self.std_head = nn.Sequential(
            nn.Linear(obs_dim, 16),
            nn.ReLU(),
            nn.Linear(16, act_dim),
        )

        self.backbone.apply(apply_custom_init)
        apply_custom_init(self.mu_head)

    def forward(self, state):
        x = self.backbone(state)
        mu = self.mu_head(x)
        return mu, self.std_head(state)
