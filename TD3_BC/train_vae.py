import argparse
import numpy as np
import torch
import os
import torch.nn as nn
import torch.nn.functional as F
import gym
from tqdm import tqdm
from vae import VAE
import utils
import d4rl
import wandb

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=0)
# dataset
parser.add_argument('--env', type=str, default='walker2d')
# model
parser.add_argument('--model', default='VAE', type=str)
parser.add_argument('--hidden_dim', type=int, default=750)
parser.add_argument('--beta', type=float, default=0.5)
# train
parser.add_argument('--num_iters', type=int, default=int(1e5))
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--weight_decay', default=0, type=float)
args = parser.parse_args()

utils.set_seed_everywhere(args.seed)
device = 'cuda'

# load wandb
env_name = f"{args.env}"
run = wandb.init(
    project="VAE",
    name=env_name,
    config=vars(args)
)

if  not os.path.exists("../models"):
	os.makedirs("../models")

# load data
env = gym.make(env_name)

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
max_action = float(env.action_space.high[0])
print(state_dim, action_dim, max_action)
latent_dim = action_dim * 2

replay_buffer = utils.ReplayBuffer(state_dim, action_dim)
replay_buffer.convert_D4RL(d4rl.qlearning_dataset(env))
mean, std = replay_buffer.normalize_states()
states = replay_buffer.state
actions = replay_buffer.action

# train
if args.model == 'VAE':
    vae = VAE(state_dim, action_dim, latent_dim, max_action, hidden_dim=args.hidden_dim).to(device)
else:
    raise NotImplementedError
optimizer = torch.optim.Adam(vae.parameters(), lr=args.lr, weight_decay=args.weight_decay)

total_size = states.shape[0]
batch_size = args.batch_size

for step in tqdm(range(args.num_iters + 1), desc='train'):
    idx = np.random.choice(total_size, batch_size)
    train_states = torch.from_numpy(states[idx]).to(device)
    train_actions = torch.from_numpy(actions[idx]).to(device)

    # Variational Auto-Encoder Training
    recon, mean, std = vae(train_states, train_actions)
    train_states_actions = torch.cat([train_states, train_actions], dim=-1)
    recon_loss = F.mse_loss(recon, train_states_actions)
    KL_loss = -0.5 * (1 + torch.log(std.pow(2)) - mean.pow(2) - std.pow(2)).mean()
    vae_loss = recon_loss + args.beta * KL_loss
    
    wandb.log({'train/recon_loss': recon_loss})
    wandb.log({'train/KL_loss': KL_loss})
    wandb.log({'train/vae_loss': vae_loss})

    optimizer.zero_grad()
    vae_loss.backward()
    optimizer.step()

    if step % 5000 == 0:
        torch.save(vae.state_dict(), '../models/vae_%s_%s.pt' % (args.env[:-3], step))