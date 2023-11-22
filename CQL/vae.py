"""
    Vanilla VAE: d(s,a) = ELBO
"""

import torch
import torch.nn.functional as F
from torch import nn
import torch.distributions as td


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class VAE(nn.Module):
    # Vanilla Variational Auto-Encoder

    def __init__(self, state_dim, action_dim, latent_dim, max_action, hidden_dim=750, dropout=0.0):
        super(VAE, self).__init__()
        self.e1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.e2 = nn.Linear(hidden_dim, hidden_dim)

        self.mean = nn.Linear(hidden_dim, latent_dim)
        self.log_std = nn.Linear(hidden_dim, latent_dim)

        self.d1 = nn.Linear(latent_dim, hidden_dim)
        self.d2 = nn.Linear(hidden_dim, hidden_dim)
        self.d3 = nn.Linear(hidden_dim, state_dim + action_dim)

        self.max_action = max_action
        self.latent_dim = latent_dim
        self.device = device

    def forward(self, state, action):
        mean, std = self.encode(state, action)
        z = mean + std * torch.randn_like(std)
        u = self.decode(z)
        return u, mean, std

    def elbo_loss(self, state, action, beta=0.5):
        sa = torch.cat([state, action], -1)
        mean, std = self.encode(state, action)
        z = mean + std * torch.randn_like(std)
        u = self.decode(z)
        recon_loss = ((sa - u) ** 2).mean(-1)
        KL_loss = -0.5 * (1 + torch.log(std.pow(2)) - mean.pow(2) - std.pow(2)).mean(-1)
        vae_loss = recon_loss + beta * KL_loss
        return vae_loss
    
    def encode(self, state, action):
        z = F.relu(self.e1(torch.cat([state, action], -1)))
        z = F.relu(self.e2(z))
        mean = self.mean(z)
        # Clamped for numerical stability
        log_std = self.log_std(z).clamp(-4, 15)
        std = torch.exp(log_std)
        return mean, std

    def decode(self, z):
        a = F.relu(self.d1(z))
        a = F.relu(self.d2(a))
        if self.max_action is not None:
            return self.max_action * torch.tanh(self.d3(a))
        else:
            return self.d3(a)
        

if __name__ == '__main__':
    import torch
    a = torch.rand([100, 10])
    b = torch.rand([100, 10])
    c = F.mse_loss(a, b)
    d = ((a - b) ** 2).mean(dim=1)
    print(c)
    print(d)
