import torch.nn as nn
import torch

class Actor(nn.Module):
    def __init__(self, obs_dim=6, act_dim=2, hl=[256, 512, 256]) -> None:
        super(Actor, self).__init__()
        layers = []
        layers.append(nn.Linear(obs_dim, hl[0]))
        layers.append(nn.ReLU())
        for i in range(1, len(hl)):
            layers.append(nn.Linear(hl[i-1], hl[i]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hl[-1], act_dim*2))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
    
class QCritic(nn.Module):
    def __init__(self, input_dim, output_dim) -> None:
        super(QCritic, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim)
        )

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        return self.model(x)
    
