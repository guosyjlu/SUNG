import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from vae import VAE


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()

        self.l1 = nn.Linear(state_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, action_dim)

        self.max_action = max_action

    def forward(self, state):
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        return self.max_action * torch.tanh(self.l3(a))


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()

        # Q1 architecture
        self.l1 = nn.Linear(state_dim + action_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 1)

        # Q2 architecture
        self.l4 = nn.Linear(state_dim + action_dim, 256)
        self.l5 = nn.Linear(256, 256)
        self.l6 = nn.Linear(256, 1)

    def forward(self, state, action):
        sa = torch.cat([state, action], 1)

        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)

        q2 = F.relu(self.l4(sa))
        q2 = F.relu(self.l5(q2))
        q2 = self.l6(q2)
        return q1, q2

    def Q1(self, state, action):
        sa = torch.cat([state, action], 1)

        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)
        return q1


class TD3(object):
    def __init__(
        self,
        state_dim,
        action_dim,
        max_action,
        discount=0.99,
        tau=0.005,
        policy_noise=0.2,
        noise_clip=0.5,
        policy_freq=2,
        topk=10,
        strategy="greedy",
        constrained=True,
        filter_prob=0.1,
        candidate_num=100,
        order="qu",
        expl_noise=0.1,
        epsilon=0.9,
        alpha=2.5,
        lamda=0.1,
    ):

        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=3e-4)

        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=3e-4)

        self.vae = VAE(
            state_dim, action_dim, latent_dim=action_dim*2, max_action=max_action, hidden_dim=750
        ).to(device)
        self.vae_optimizer = torch.optim.Adam(self.vae.parameters(), lr=1e-3)

        self.max_action = max_action
        self.discount = discount
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq
        self.alpha = alpha
        self.topk = topk
        self.order = order
        self.strategy = strategy
        self.constrained = constrained
        self.filter_prob = filter_prob
        self.lamda = lamda
        self.candidate_num = candidate_num
        self.epsilon = epsilon
        self.dist = torch.distributions.Normal(0.0, max_action * expl_noise)
        self.action_dim = action_dim

        self.total_it = 0

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        return self.actor(state).cpu().data.numpy().flatten()

    def select_exploration_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        action = self.actor(state)
        states = state.repeat(self.candidate_num, 1)
        action = action.repeat(self.candidate_num, 1)
        noise = self.dist.sample([self.candidate_num, self.action_dim]).to(device)
        candidate_action = action + noise
        
        if self.order == "qu":
            Q = self.critic.Q1(states, candidate_action).flatten()
            Q, idx = torch.topk(Q, self.topk)
            states, candidate_action = states[idx], candidate_action[idx]
            density_uncertainty = self.vae.elbo_loss(states, candidate_action, beta=0.5)
        elif self.order == "uq":
            Q = self.vae.elbo_loss(states, candidate_action, beta=0.5)
            Q, idx = torch.topk(Q, self.topk)
            states, candidate_action = states[idx], candidate_action[idx]
            density_uncertainty = self.critic.Q1(states, candidate_action).flatten()
        else:
            raise NotImplementedError("The selected order is not implemented yet!")
            
        if self.strategy == "greedy":
            idx = torch.argmax(density_uncertainty)
            action = candidate_action[idx]
        elif self.strategy == "e-greedy":
            if np.random.random() > self.epsilon:
                idx = np.random.choice(self.topk)
            else:
                idx = torch.argmax(density_uncertainty)
            action = candidate_action[idx]
        elif self.strategy == "prob":
            probs = F.softmax(density_uncertainty)
            dist = torch.distributions.Categorical(probs)
            idx = dist.sample()
            action = candidate_action[idx]
        else:
            raise NotImplementedError("The selected strategy is not implemented yet!")
        
        return action.cpu().data.numpy().flatten()

    def train(self, replay_buffer, batch_size=256):
        self.total_it += 1

        # Sample replay buffer
        state, action, next_state, reward, not_done = replay_buffer.sample(
            batch_size)

        with torch.no_grad():
            # Select action according to policy and add clipped noise
            noise = (
                torch.randn_like(action) * self.policy_noise
            ).clamp(-self.noise_clip, self.noise_clip)

            next_action = (
                self.actor_target(next_state) + noise
            ).clamp(-self.max_action, self.max_action)

            # Compute the target Q value
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + not_done * self.discount * target_Q

        # Get current Q estimates
        current_Q1, current_Q2 = self.critic(state, action)

        # Compute critic loss
        critic_loss = F.mse_loss(current_Q1, target_Q) + \
            F.mse_loss(current_Q2, target_Q)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # Calculate VAE loss
        recon, mean, std = self.vae(state, action)
        train_states_actions = torch.cat([state, action], dim=-1)
        recon_loss = F.mse_loss(recon, train_states_actions)
        KL_loss = -0.5 * (1 + torch.log(std.pow(2)) - mean.pow(2) - std.pow(2)).mean()
        vae_loss = recon_loss + 0.5 * KL_loss
        
        # Optimize the VAE
        self.vae_optimizer.zero_grad()
        vae_loss.backward()
        self.vae_optimizer.step()

        training_outputs = None
        # Delayed policy updates
        if self.total_it % self.policy_freq == 0:

            # Compute actor loss
            pi = self.actor(state)
            Q = self.critic.Q1(state, pi)
            if self.constrained == "greedy":
                uncertainty = self.vae.elbo_loss(state, pi).detach()
                _, idx = torch.topk(uncertainty, k=int(batch_size*self.filter_prob))
                regularization = ((pi[idx] - action[idx])**2).mean(axis=1).mean()
            elif self.constrained == "prob":
                uncertainty = self.vae.elbo_loss(state, pi).detach()
                probs = F.softmax(uncertainty)
                dist = torch.distributions.Categorical(probs)
                idx = dist.sample([int(batch_size*self.filter_prob)])
                regularization = ((pi[idx] - action[idx])**2).mean(axis=1).mean()
            elif self.constrained == "default":
                regularization = torch.tensor(0.0)
            else:
                raise NotImplementedError(f"{self.constrained} is not implemented yet!")
            
            lmbda = self.alpha / Q.abs().mean().detach()
            actor_loss = -lmbda * Q.mean() + self.lamda * regularization

            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update the frozen target models
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(
                    self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(
                    self.tau * param.data + (1 - self.tau) * target_param.data)
            
            # Log Training Details
            training_outputs = {
                "training/Critic Loss": critic_loss,
				"training/VAE Reconstruction Loss": recon_loss,
				"training/VAE KL Loss": KL_loss,
				"training/VAE Loss": vae_loss,
				"training/Actor Loss": actor_loss,
                "training/Actor Q": Q.mean(),
                "training/Actor regularization": regularization,
                # "training/step": int(self.total_it//self.policy_freq)
            }
            
            return training_outputs
        
			

    def save(self, filename):
        torch.save(self.critic.state_dict(), filename + "_critic")
        torch.save(self.critic_optimizer.state_dict(),
                   filename + "_critic_optimizer")

        torch.save(self.actor.state_dict(), filename + "_actor")
        torch.save(self.actor_optimizer.state_dict(),
                   filename + "_actor_optimizer")

    def load(self, filename):
        self.critic.load_state_dict(torch.load(filename + "_critic"))
        self.critic_optimizer.load_state_dict(
            torch.load(filename + "_critic_optimizer"))
        self.critic_target = copy.deepcopy(self.critic)

        self.actor.load_state_dict(torch.load(filename + "_actor"))
        self.actor_optimizer.load_state_dict(
            torch.load(filename + "_actor_optimizer"))
        self.actor_target = copy.deepcopy(self.actor)
